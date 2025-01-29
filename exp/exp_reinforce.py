import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Transformer,moe,ppo,tradingenv,Informer,Reformer,Autoformer,Fedformer,Flowformer,Flashformer,itransformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual,port_visual
from utils.metrics import metric
from utils.backtest import *
from itertools import tee
import gc
import torch.nn.functional as F

import matplotlib.pyplot as plt



# import FinanceDataReader as fdr
warnings.filterwarnings('ignore')


class Exp_Reinforce(Exp_Basic):
    def __init__(self, args,setting):
        super(Exp_Reinforce, self).__init__(args,setting)
        self.temperature = self.args.temperature
        self.env = tradingenv.TradingEnvironment(self.args)


        self.buffer_size = 1  #20
        self.gamma =0.99
        self.lmbda = 0.95
        self.eps_clip = 0.1
        self.data = []
        self.policy_update_freq = 500
        self.max_clip = 10  # 적절한 범위 설정 (예: 10은 exp(10) = 22026 정도)
        self.min_clip = -10  # exp(-10) ≈ 0

    def _build_model(self):

        model = ppo.PPO(self.args.model, self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.device_ids)
        return model
    def put_data(self,item):
        self.data.append(item)
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def make_batch(self):
        batch_data = {
            "batch_x": [],
            "batch_y": [],
            "batch_x_mark": [],
            "batch_y_mark": [],
            "scores": [],
            "reward": [],
            "next_batch_x": [],
            "next_batch_y": [],
            "next_batch_x_mark": [],
            "next_batch_y_mark": [],
            "log_prob": [],
            "done": [],
            'return_data': []
        }

        for _ in range(self.buffer_size):
            rollout = self.data.pop(0)
            for transition in rollout:
                batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward, next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, log_prob, done, return_data = transition
                batch_data["batch_x"].append(batch_x)
                batch_data["batch_y"].append(batch_y)
                batch_data["batch_x_mark"].append(batch_x_mark)
                batch_data["batch_y_mark"].append(batch_y_mark)
                batch_data["scores"].append(scores)
                batch_data["reward"].append([reward])
                batch_data["next_batch_x"].append(next_batch_x)
                batch_data["next_batch_y"].append(next_batch_y)
                batch_data["next_batch_x_mark"].append(next_batch_x_mark)
                batch_data["next_batch_y_mark"].append(next_batch_y_mark)
                batch_data["log_prob"].append(log_prob)
                done_mask = 0 if done else 1
                batch_data["done"].append([done_mask])
                batch_data['return_data'].append(return_data)

        for key in batch_data:
            # if key in ["reward", "log_prob", "done"]:
            if key in ["reward", "done"]:
                batch_data[key] = torch.tensor(batch_data[key], dtype=torch.float).squeeze().to(self.device)

            else:
                batch_data[key] = torch.stack(batch_data[key], dim=0).to(self.device)

        mini_batches = []
        for i in range(0, len(batch_data["batch_x"])):
            # scalar_index = i //self.batch_size
            mini_batches.append((
                    batch_data["batch_x"][i],
                    batch_data["batch_y"][i],
                    batch_data["batch_x_mark"][i],
                    batch_data["batch_y_mark"][i],
                    batch_data["scores"][i],
                    batch_data["reward"][i],
                    batch_data["next_batch_x"][i],
                    batch_data["next_batch_y"][i],
                    batch_data["next_batch_x_mark"][i],
                    batch_data["next_batch_y_mark"][i],
                    batch_data["done"][i],
                    batch_data["log_prob"][i],
                    batch_data['return_data'][i]))


        return mini_batches
        # return batch_data
    def calc_advantage(self, data):
        data_with_adv = []
        td_target_lst = []
        delta_lst = []
        for i, transition in enumerate(data):
            batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward, next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, done_mask, old_log_prob, return_data = transition

            with torch.no_grad():
                # V(s'), V(s)
                v_s_next = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)
                v_s = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                # td_target
                td_target = reward + self.gamma * v_s_next * done_mask
                # delta
                delta = td_target - v_s
            td_target_lst.append(td_target.item())
            delta_lst.append(delta.item())
        advantage_lst = []
        running_adv = 0.0
        for delta_t in reversed(delta_lst):
            running_adv = self.gamma * self.lmbda * running_adv + delta_t
            advantage_lst.append(running_adv)
        advantage_lst.reverse()
        for i, transition in enumerate(data):
            (batch_x, batch_y, batch_x_mark, batch_y_mark,
             scores, reward,
             next_batch_x, next_batch_y, next_batch_x_mark_, next_batch_y_mark_,
             done_mask, old_log_prob, return_data) = transition

            # td_target도 텐서로
            td_target_tensor = torch.tensor(td_target_lst[i], dtype=torch.float, device=self.device)
            advantage_tensor = torch.tensor(advantage_lst[i], dtype=torch.float, device=self.device)

            data_with_adv.append((
                batch_x, batch_y, batch_x_mark, batch_y_mark,
                scores, reward,
                next_batch_x, next_batch_y, next_batch_x_mark_, next_batch_y_mark_,
                done_mask, old_log_prob,
                td_target_tensor, advantage_tensor,
                return_data
            ))

        return data_with_adv



    def train_net(self, K_epoch=3, model_optim=None):
        if len(self.data) == self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            losses = 0.0

            for _ in range(K_epoch):
                for mini_batch in data:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward,next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, done_mask, old_log_prob, td_target, advantage,return_data= mini_batch

                    scores,log_prob = self.model.pi(batch_x, batch_x_mark, batch_y,batch_y_mark)


                    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=self.min_clip, max=self.max_clip))
                    # ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))


                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    pred_loss = F.smooth_l1_loss(self.model.pred(batch_x, batch_x_mark,batch_y, batch_y_mark), return_data)
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark), td_target) + pred_loss
                    losses += loss.mean().item()
                    model_optim.zero_grad()
                    loss.mean().backward()
                    model_optim.step()
            return losses

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validate the model on the validation dataset.

        This function evaluates the model's value loss (based on TD targets) and prediction loss.
        """
        self.model.eval()  # Set model to evaluation mode
        total_loss, total_value_loss, total_pred_loss = 0.0, 0.0, 0.0
        self.env.reset()
        total_samples = 0
        prev_batch=None

        with torch.no_grad():
            for i, current_batch in enumerate(vali_loader):
                if prev_batch is None:
                    # 첫 배치는 next_state가 없으므로 skip
                    prev_batch = current_batch
                    continue
                # (1) 이전 batch = state
                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = [
                    x.squeeze(0).float().to(self.device) for x in prev_batch
                ]
                # (2) 현재 batch = next_state
                next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, next_ground_true = [
                    x.squeeze(0).float().to(self.device) for x in current_batch
                ]

                done = (i == len(vali_loader) - 1)  # 마지막 배치면 done

                # Decoder input
                dec_inp = torch.cat(
                    [batch_y[:, :self.args.label_len], torch.zeros_like(batch_y[:, -self.args.pred_len:])], dim=1
                ).to(self.device)

                # Forward pass
                scores, _ = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                value = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                next_value = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)

                # Portfolio selection
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                final_weights = torch.zeros_like(scores)
                final_weights[top_indices] = topk_weights
                returns = ground_true
                reward = self.env.step(final_weights, returns)

                # Compute TD Target
                td_target = reward + self.gamma * next_value * (0 if done else 1)

                # Compute losses
                value_loss = F.smooth_l1_loss(value, td_target)  # Value loss using TD target
                pred_loss = F.smooth_l1_loss(scores, ground_true)  # Prediction loss

                # Combine losses
                loss = value_loss + pred_loss

                # Update totals
                total_value_loss += value_loss.item()
                total_pred_loss += pred_loss.item()
                total_loss += loss.item()


                prev_batch = current_batch

        # Compute averages
        avg_loss = total_loss
        avg_value_loss = total_value_loss
        avg_pred_loss = total_pred_loss



        # Log metrics
        self.logger.info(
            f"Validation - total_Loss: {avg_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Prediction Loss: {avg_pred_loss:.4f}"
        )


        self.model.train()  # Reset model to training mode

        return avg_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)



        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        self.model.train()
        self.env.reset()


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_loss = []

            prev_batch = None
            epoch_time = time.time()
            for i, current_batch in enumerate(train_loader):
                if prev_batch is None:
                    # next_state가 없으니 skip.
                    # 또는 "첫 배치를 state로 설정, next_batch는 다음 배치에서" 라고 할 수도 있음
                    prev_batch = current_batch
                    continue
                # Unpack prev_batch => state
                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = [
                    x.squeeze(0).float().to(self.device) for x in prev_batch
                ]
                # Unpack current_batch => next_state
                next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, next_ground_true = [
                    x.squeeze(0).float().to(self.device) for x in current_batch
                ]

                done = (i == len(train_loader) - 1)  # epoch 마지막 배치면 done=True



                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                            selected_scores = scores[top_indices]
                            topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                            final_weights = torch.zeros_like(scores)
                            final_weights[top_indices] = topk_weights
                            returns = ground_true
                            reward = self.env.step(final_weights, returns)
                            self.env.rollout.append(
                                (batch_x, batch_y, batch_x_mark, batch_y_mark, scores.detach(), reward,
                                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                                 log_prob.detach(), done, returns))
                        else:
                            scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            # Select top stocks based on scores
                            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                            selected_scores = scores[top_indices]
                            topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                            final_weights = torch.zeros_like(scores)
                            final_weights[top_indices] = topk_weights
                            returns = ground_true
                            reward = self.env.step(final_weights, returns)
                            self.env.rollout.append(
                                (batch_x, batch_y, batch_x_mark, batch_y_mark, scores.detach(), reward,
                                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                                 log_prob.detach(), done, returns))




                else:
                    if self.args.output_attention:
                        scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        # Select top stocks based on scores
                        top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                        selected_scores = scores[top_indices]
                        topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                        final_weights = torch.zeros_like(scores)
                        final_weights[top_indices] = topk_weights
                        returns = ground_true
                        reward = self.env.step(final_weights, returns)
                        self.env.rollout.append((batch_x, batch_y, batch_x_mark, batch_y_mark, scores.detach(), reward,
                                                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                                                 log_prob.detach(), done, returns))
                    else:
                        scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # Select top stocks based on scores
                        top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                        selected_scores = scores[top_indices]
                        topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                        final_weights = torch.zeros_like(scores)
                        final_weights[top_indices] = topk_weights
                        returns = ground_true
                        reward = self.env.step(final_weights, returns)
                        self.env.rollout.append((batch_x, batch_y, batch_x_mark, batch_y_mark, scores.detach(), reward,
                                                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                                                 log_prob.detach(), done, returns))
                    if len(self.env.rollout) == self.env.rollout_len:
                        self.put_data(self.env.rollout)
                        self.env.rollout = []
                        loss = self.train_net(K_epoch=3, model_optim=model_optim)
                        train_loss.append(loss)

                    prev_batch = current_batch

                    gc.collect()
                    torch.cuda.empty_cache()

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break


            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def backtest(self, setting, load=False):
        """
        Backtest the trained model using reinforcement learning principles.
        """
        # Load data for testing and backtesting
        back_test_data, back_test_loader = self._get_data(flag='backtest')
        dataset = back_test_loader.dataset  # This is Dataset_Custom
        unique_dates = pd.to_datetime(dataset.unique_dates)# sorted unique dates
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len

        if dataset.use_step_sampling:
            valid_indices = dataset.indexes  # e.g. [0, step_size, 2*step_size, ...]
        else:
            valid_indices = range(len(dataset))  # 0, 1, 2, ..., (len -1)
        subset_dates = [
            unique_dates[idx + seq_len] for idx in valid_indices
            if (idx + seq_len) < len(unique_dates)  # ensure index is valid
        ]

        # Load the model checkpoint if specified
        if load:
            self.logger.info('Loading model for backtesting...')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path))

        # Initialize environment
        self.env.reset()
        portfolio_values = [1.0]
        investment_dates = [unique_dates[0]]

        # Set parameters
        fee_rate = self.args.fee_rate
        seq_len = self.args.seq_len
        pred_len = self.args.pred_len

        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) in enumerate(back_test_loader):
                # Prepare input data
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float().to(self.device)
                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)
                ground_true = ground_true.squeeze(0).float().to(self.device)
                # self.logger.info(f"Processing date: {current_date}")
                # Make predictions with the model
                scores, _ = self.model.pi(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # Select top-performing stocks and calculate weights
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                topk_weights = torch.softmax(selected_scores / self.temperature, dim=0)
                final_weights = torch.zeros_like(scores)
                final_weights[top_indices] = topk_weights
                returns = ground_true
                reward = self.env.step(final_weights,returns)
                portfolio_values.append(self.env.portfolio_value)
                investment_dates.append(subset_dates[i])


        final_pf = self.env.portfolio_value
        self.logger.info(f"[BackTest] Final Portfolio Value = {final_pf:.4f}")
        self.wandb.log({"BackTest Final Portfolio Value": final_pf})
        start_date = dataset.unique_dates[0]
        end_date = unique_dates[-1]

        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        raw_data = pd.read_csv(csv_path)
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        index_name = self._get_market_index_name()
        index_data = fetch_index_data(index_name, start_date, end_date)
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)



        # Run backtest and save results
        run_backtest(
            data=raw_data,
            index_data=index_data,
            start_date=start_date,
            end_date=end_date,
            fee_rate=fee_rate,
            external_portfolio=portfolio_values,
            external_dates=investment_dates,
            pred_len=pred_len,
            total_periods=len(dataset.unique_dates),
            folder_path=folder_path
        )

        self.logger.info("Backtest completed. Portfolio values and metrics saved.")

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0].squeeze(-1)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark).squeeze(-1)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0].squeeze(-1)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark).squeeze(-1)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

    def setting_expert(self,setting):
        """
    Train daily, weekly, and monthly expert models using the existing train method.
    Each expert has a different pred_len: daily=1, weekly=5, monthly=20.

    Args:
        base_setting_components (list): List of setting components excluding pred_len.
    """
        # Define pred_len for each expert
        experts = {
            'daily': 1,
            'weekly': 5,
            'monthly': 20
        }

        # Dictionary to store model paths
        self.expert_model_paths = {}
        for period, pred_len in experts.items():
            self.logger.info(f"Training {period} expert...")

            # Create a copy of base_setting_components
            expert_setting_components = setting.copy()

            # Replace the pl component
            pl_component = f"pl({pred_len})"
            for i, component in enumerate(expert_setting_components):
                if component.startswith("pl("):
                    expert_setting_components[i] = pl_component
                    break
            else:
                # If pl is not found, append it
                expert_setting_components.append(pl_component)

            # Create setting string
            setting = "_".join(expert_setting_components)



            # Define the model path
            model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.expert_model_paths[period] = model_path

            self.logger.info(f"{period.capitalize()} expert trained and saved at {model_path}")

        self.logger.info("Finished training all experts.")

    def train_moe(self,setting):
        """
        Train MOE using the pre-trained expert models.
        """
        self.logger.info("Loading pre-trained expert models...")

        # Ensure expert_model_paths are available
        if not hasattr(self, 'expert_model_paths') or not self.expert_model_paths:
            raise AttributeError("Expert model paths not found. Please run train_expert first.")

        # Load the pre-trained experts
        self.daily_model.load_state_dict(torch.load(self.expert_model_paths['daily']))
        self.daily_model.eval()

        self.weekly_model.load_state_dict(torch.load(self.expert_model_paths['weekly']))
        self.weekly_model.eval()

        self.monthly_model.load_state_dict(torch.load(self.expert_model_paths['monthly']))
        self.monthly_model.eval()

        self.logger.info("Experts loaded. Initializing MOE model...")

        # Initialize MOE model
        self.moe_model.to(self.device)

        # Load MOE-specific data
        self.logger.info("Loading MOE training data...")
        self.args.moe_train = True
        self.args.train_method = 'Reinfoce'
        train_data,train_loader = self._get_data(flag='train')
        train_x, train_y, train_x_mark, train_y_mark, train_gt = self.gather_entire_data_from_loader(
            train_loader, self.device
        )
        vali_data, vali_loader = self._get_data(flag='val')
        valid_x, valid_y, valid_x_mark, valid_y_mark, valid_gt = self.gather_entire_data_from_loader(
            vali_loader, self.device
        )
        test_data, test_loader = self._get_data(flag='test')
        test_x, test_y, test_x_mark, test_y_mark, test_gt = self.gather_entire_data_from_loader(
            test_loader, self.device
        )
        # shape: (stock_num, total_time, feat_x), etc.
        stock_num, total_time, feat_dim = train_x.shape
        seq_len = self.args.seq_len
        self.logger.info(f"Collected data shape: (stock_num={stock_num}, total_time={total_time}, feature={feat_dim})")

        optimizer = torch.optim.Adam(self.moe_model.parameters(), lr=self.args.learning_rate)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        criterion = self._select_criterion()

        # Define the path for saving the best model
        moe_best_model_dir = os.path.join(self.args.checkpoints,'moe_model', "_".join(setting))
        os.makedirs(moe_best_model_dir, exist_ok=True)
        moe_best_model_path = os.path.join(moe_best_model_dir, 'checkpoint.pth')


        self.logger.info("Training MOE model...")
        best_epoch = 0
        for epoch in range(self.args.train_epochs):
            epoch_start_time = time.time()
            self.moe_model.train()
            self.env.reset()
            rollout = []
            train_loss_list = []

            current_index = 0
            episode_count = 0
            while True:
                # (a) 종료 조건
                if current_index >= total_time//seq_len:
                    break

                # 현재 시점의 데이터 (이미 seq_len 길이!)
                # cat_x[current_index]: shape (seq_len, feature)?
                # 만약 shape=(N, seq_len, feature), N=각 시점 => cat_x[current_index] => (seq_len, feature)
                batch_x = train_x[current_index]
                batch_y = train_y[current_index]
                batch_x_mark = train_x_mark[current_index]
                batch_y_mark = train_y_mark[current_index]
                ground_true = train_gt[current_index]  # shape (?)

                # (b) MOE forward => gating => pred_len
                #   예: self.moe_model.pi(...) -> (scores, log_prob, gating_logits)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        scores, log_prob, gating_logits = self.moe_model.pi(
                            batch_x.unsqueeze(0),
                            batch_x_mark.unsqueeze(0),
                            batch_y.unsqueeze(0),
                            batch_y_mark.unsqueeze(0)
                        )
                else:
                    scores, log_prob, gating_logits = self.moe_model.pi(
                        batch_x.unsqueeze(0),
                        batch_x_mark.unsqueeze(0),
                        batch_y.unsqueeze(0),
                        batch_y_mark.unsqueeze(0)
                    )
                # gating_logits.shape = (1, 3)
                # argmax => 0->1, 1->5, 2->20
                action_idx = torch.argmax(gating_logits, dim=1).item()
                if action_idx == 0:
                    pred_len = 1
                elif action_idx == 1:
                    pred_len = 5
                else:
                    pred_len = 20

                # (c) reward 계산
                #   예: env.step(scores, ground_true)
                #   질문 코드처럼 top_indices => weights => reward
                top_indices = torch.topk(scores.squeeze(0), self.args.num_stocks, dim=0).indices
                selected_scores = scores.squeeze(0)[top_indices]
                selected_returns = ground_true[top_indices]  # ground_true shape?
                weights = torch.softmax(selected_scores / self.temperature, dim=0)
                reward = self.env.step(weights, selected_returns)

                # (d) rollout 추가
                done = (current_index + pred_len >= N - 1)  # 끝까지 가면 done
                next_index = current_index + pred_len
                transition = (
                    batch_x, batch_y, batch_x_mark, batch_y_mark,
                    scores.detach(), reward,
                    None, None, None, None,  # next_x, next_y ... (생략)
                    log_prob.detach(), done, ground_true
                )
                self.env.rollout.append(transition)

                # rollout_len마다 업데이트
                if len(self.env.rollout) == self.env.rollout_len:
                    self.put_data(self.env.rollout)
                    self.env.rollout = []
                    loss_val = self.train_net(K_epoch=1, model_optim=optimizer)
                    train_loss_list.append(loss_val)

                # index 이동
                current_index += pred_len
                episode_count += 1

            # epoch 마무리
            train_loss = np.mean(train_loss_list) if len(train_loss_list) else 0.0
            cost_time = time.time() - epoch_start_time

            # (e) validation
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info(
                f"Epoch: {epoch + 1}, Episodes: {episode_count}, Cost: {cost_time:.2f}, "
                f"TrainLoss: {train_loss:.5f}, ValiLoss: {vali_loss:.5f}, TestLoss: {test_loss:.5f}"
            )

            # early stopping
            early_stopping(vali_loss, self.moe_model, moe_best_model_dir)
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered")

                break

            # lr decay
            adjust_learning_rate(optimizer, epoch + 1, self.args)



        self.model.load_state_dict(torch.load(moe_best_model_path))
        self.logger.info(f"MOE model training completed and saved at {moe_best_model_path}")

        return self.moe_model

    def test(self, setting, load=False):
        """
        Evaluate the trained model in a test environment using reinforcement learning principles.
        """
        test_data, test_loader = self._get_data(flag='test')

        if load:
            self.logger.info('Loading model for testing...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.env.reset()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) in enumerate(test_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float().to(self.device)
                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)
                ground_true = ground_true.squeeze(0).float().to(self.device)

                scores, _ = self.model.pi(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # Select top stocks based on scores
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                selected_returns = ground_true[top_indices]
                weights = torch.softmax(selected_scores / self.temperature, dim=0)

                # Calculate reward and update environment
                reward = self.env.step(weights, selected_returns)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        portfolio_values = np.array(self.env.asset_memory)
        np.save(folder_path + 'test_portfolio_values.npy', portfolio_values)

        # Use run_backtest for metrics and visualization
        start_date = str(test_data.gt.index.get_level_values('date').unique()[0].date())
        end_date = str(test_data.gt.index.get_level_values('date').unique()[-1].date())
        index_data = fetch_index_data(self.args.market_index, start_date, end_date)

        run_backtest(
            data=test_data.raw_gt.reset_index(),
            index_data=index_data,
            start_date=str(start_date),
            end_date=str(end_date),
            fee_rate=self.args.fee_rate,
            external_portfolio=portfolio_values,
            external_dates=test_data.gt.index.get_level_values('date').unique(),
            folder_path=folder_path
        )

        self.logger.info("Test completed. Portfolio values and metrics saved.")



    def moe_backtest(self, setting, load=True):
        """
        Perform backtesting using the trained MOE model with reinforcement learning adjustments.
        """
        # Load test data
        test_data, test_loader = self._get_data(flag='test')
        unique_dates = test_data.data_x.index.get_level_values('date').unique()

        # Load the MOE model if specified
        if load:
            self.logger.info('Loading MOE model for backtesting...')
            moe_model_path = os.path.join(self.args.checkpoints, 'moe_model.pth')
            self.moe_model.load_state_dict(torch.load(moe_model_path))
            self.moe_model.to(self.device)
            self.moe_model.eval()

        # Initialize environment
        self.env.reset()
        portfolio_values = [1.0]
        investment_dates = []

        # Perform backtesting
        self.moe_model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) in enumerate(test_loader):
                # Prepare input data
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float().to(self.device)
                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)
                ground_true = ground_true.squeeze(0).float().to(self.device)

                # Generate predictions using MOE model
                outputs, _ = self.moe_model(batch_x, batch_x_mark, batch_y, batch_y_mark)

                # Select top-performing stocks
                pred = outputs.cpu().numpy()
                top_indices = np.argsort(-pred)[:self.args.num_stocks]
                selected_returns = ground_true[top_indices]
                weights = np.ones(self.args.num_stocks) / self.args.num_stocks

                # Update environment
                reward = self.env.step(weights, selected_returns)
                portfolio_values.append(self.env.portfolio_value)
                investment_dates.append(unique_dates[i + self.args.seq_len])

        # Save portfolio values
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        portfolio_values = np.array(self.env.asset_memory)
        np.save(os.path.join(folder_path, 'moe_backtest_portfolio_values.npy'), portfolio_values)

        # Fetch benchmark data for the backtest period
        start_date = str(unique_dates[0].date())
        end_date = str(unique_dates[-1].date())
        index_name = self._get_market_index_name()
        index_data = fetch_index_data(index_name, start_date, end_date)

        # Run backtest and save results
        run_backtest(
            data=test_data.raw_gt.reset_index(),
            index_data=index_data,
            start_date=start_date,
            end_date=end_date,
            fee_rate=self.args.fee_rate,
            external_portfolio=portfolio_values,
            external_dates=investment_dates,
            pred_len=self.args.pred_len,
            total_periods=test_data.raw_gt.index.get_level_values('date').nunique(),
            folder_path=folder_path
        )

        self.logger.info("MOE Backtest completed. Portfolio values and metrics saved.")




    def _get_market_index_name(self):
        """Returns the market index name based on the selected market."""
        market_indices = {
            'kospi': '^KS11',
            'dj30': '^DJI',
            'sp500': '^GSPC',
            'nasdaq': '^IXIC',
            'csi300': '000300.SS'
        }
        index_name = market_indices.get(self.args.market)
        if not index_name:
            raise ValueError(f"Unsupported market: {self.args.market}")
        return index_name

    def gather_entire_data_from_loader(self,loader, device):
        """
        DataLoader에서
          batch_x: (stock_num, seq_len, feature_x)
          batch_y: (stock_num, seq_len, feature_y)
          batch_x_mark: (stock_num, seq_len, feature_xm)
          batch_y_mark: (stock_num, seq_len, feature_ym)
          ground_true: (stock_num, seq_len, feature_gt) 혹은 (stock_num, seq_len) 등
        형태로 순차적으로 들어오는 텐서들을
        (stock_num, total_time, feature_*)로 이어붙여 반환.

        전제:
         - DataLoader가 shuffle=False, 시간 순서대로 batch를 내놓는다.
         - stock_num은 고정
         - seq_len은 batch마다 동일 (이어서 concat)
         - feature_*는 서로 다를 수 있으니, 각자 cat
        """
        all_x, all_y = [], []
        all_x_mark, all_y_mark = [], []
        all_gt = []

        for (batch_x, batch_y,
             batch_x_mark, batch_y_mark,
             ground_true) in loader:

            # 혹시 (1, stock_num, seq_len, feature)처럼 4차원이면 squeeze(0)
            if batch_x.dim() == 4:
                batch_x = batch_x.squeeze(0)  # -> (stock_num, seq_len, feat_x)
                batch_y = batch_y.squeeze(0)
                batch_x_mark = batch_x_mark.squeeze(0)
                batch_y_mark = batch_y_mark.squeeze(0)
                ground_true = ground_true.squeeze(0)  # -> (stock_num, seq_len, ...)

            # GPU로 이동
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_x_mark = batch_x_mark.to(device)
            batch_y_mark = batch_y_mark.to(device)
            ground_true = ground_true.to(device)

            # 리스트에 추가
            all_x.append(batch_x)
            all_y.append(batch_y)
            all_x_mark.append(batch_x_mark)
            all_y_mark.append(batch_y_mark)
            all_gt.append(ground_true)

        # 이제 각 list를 dim=1(시간 축)으로 이어 붙인다
        # => 결과 shape: (stock_num, total_time, feature)
        cat_x = torch.cat(all_x, dim=1)  # (stock_num, total_time, feat_x)
        cat_y = torch.cat(all_y, dim=1)  # (stock_num, total_time, feat_y)
        cat_xm = torch.cat(all_x_mark, dim=1)  # (stock_num, total_time, feat_xm)
        cat_ym = torch.cat(all_y_mark, dim=1)  # (stock_num, total_time, feat_ym)
        cat_gt = torch.cat(all_gt, dim=1)  # (stock_num, total_time, feat_gt) or shape

        return cat_x, cat_y, cat_xm, cat_ym, cat_gt

    def train_expert_reinforce(self):
        """
        예시) epoch를 돌 때마다 pred_len을 [1,5,20]에서 순환
        또는, 일정 epoch 구간씩 pred_len=1->5->20으로 바꿔가며 학습.
        """
        self.logger.info("Starting dynamic pred_len training for expert...")

        # 1) 전체 학습 데이터를 한 번에 로딩
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 전체 tensor로 모으기
        all_x, all_y, all_x_mark, all_y_mark, all_gt = gather_entire_data_from_loader(
            train_loader, self.device
        )

        # 2) 학습 준비
        path = os.path.join(self.args.checkpoints, "dynamic_expert")
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # RL 파라미터 등
        K_epoch = 1  # 혹은 원하는 PPO 업데이트 횟수
        rollout = []
        rollout_idx = 0

        # 3) epoch 루프
        #    매 epoch마다 pred_len을 [1,5,20] 중 하나로 세팅(간단 예시)
        possible_pred_lens = [1, 5, 20]
        for epoch in range(self.args.train_epochs):
            # pred_len 선택(예: 순환)
            self.args.pred_len = possible_pred_lens[epoch % len(possible_pred_lens)]
            self.logger.info(f"Epoch {epoch + 1}, using pred_len={self.args.pred_len}")

            # (옵션) env reset
            self.env.reset()
            self.model.train()

            # 4) 한 epoch 동안, all_x를 step-by-step으로 돌며 rollout 생성
            current_index = 0
            max_index = all_x.shape[0]  # 시점 수
            seq_len = self.args.seq_len
            pred_len = self.args.pred_len
            train_loss = []

            while current_index < max_index - 1:
                # state
                state_x = all_x[current_index].unsqueeze(0)  # (1, seq_len, d_feat)
                state_x_mark = all_x_mark[current_index].unsqueeze(0)
                # etc... 필요한 텐서들

                # 모델의 scores, log_prob
                scores, log_prob = self.model.pi(
                    state_x,
                    state_x_mark,
                    # decoder input 등 필요 시 ...
                )

                # 보상 계산(예시): pred_len만큼 앞으로 간 지점의 ground truth 비교
                next_index = current_index + pred_len
                if next_index >= max_index:
                    next_index = max_index - 1
                reward = self._calc_reward(all_gt, current_index, next_index)

                done = (next_index >= max_index - 1)

                # rollout 저장
                # next state 도 필요하면 여기서 all_x[next_index] 등
                rollout.append((
                    state_x,  # batch_x
                    # batch_y ...
                    state_x_mark,
                    scores.detach(),
                    reward,
                    # next_state...
                    log_prob.detach(),
                    done,
                    all_gt[current_index].detach()  # return_data
                ))
                rollout_idx += 1

                # current_index 업데이트
                current_index = next_index

                # rollout이 꽉 차면(또는 일정 step마다) -> put_data -> train_net
                if len(rollout) >= self.env.rollout_len:
                    self.put_data(rollout)
                    rollout = []

                    loss = self.train_net(K_epoch=K_epoch, model_optim=model_optim)
                    if loss is not None:
                        train_loss.append(loss)

            # epoch 끝난 후 validation
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            if train_loss:
                mean_train_loss = np.mean(train_loss)
            else:
                mean_train_loss = 0.0

            self.logger.info(f"[Epoch {epoch + 1}] TrainLoss={mean_train_loss:.4f}, "
                             f"ValiLoss={vali_loss:.4f}, TestLoss={test_loss:.4f}")

            # early_stopping 등
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping triggered.")
                break

        # 학습 종료, 모델 저장
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        self.logger.info("Dynamic pred_len expert training finished.")

    def train_x(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            # Duplicate the train_loader for accessing current and next batch
            loader1, loader2 = tee(train_loader)
            data_len = len(train_loader)
            next(loader2, None)  # Advance the second iterator by one step
            rollout_idx = 0
            self.env.reset()
            for i, (current_data, next_data) in enumerate(zip(loader1, loader2)):
                iter_count += 1
                rollout_idx += 1

                # Set done flag based on the iteration
                done = i == data_len

                # Prepare current batch data
                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = (
                    current_data[0].squeeze(0).float().to(self.device),
                    current_data[1].squeeze(0).float().to(self.device),
                    current_data[2].squeeze(0).float().to(self.device),
                    current_data[3].squeeze(0).float().to(self.device),
                    current_data[4].squeeze(0).float().to(self.device),
                )

                # Prepare next batch data
                next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, next_ground_true = (
                    next_data[0].squeeze(0).float().to(self.device),
                    next_data[1].squeeze(0).float().to(self.device),
                    next_data[2].squeeze(0).float().to(self.device),
                    next_data[3].squeeze(0).float().to(self.device),
                    next_data[4].squeeze(0).float().to(self.device),
                )

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                returns = ground_true.squeeze(0).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                            # Select top stocks based on scores
                            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                            selected_scores = scores[top_indices]
                            selected_returns = returns[top_indices]
                            weights = torch.softmax(selected_scores / self.temperature, dim=0)
                            reward = self.env.step(weights, selected_returns)
                        else:
                            scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            # Select top stocks based on scores
                            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                            selected_scores = scores[top_indices]
                            selected_returns = returns[top_indices]
                            weights = torch.softmax(selected_scores / self.temperature, dim=0)
                            reward = self.env.step(weights, selected_returns)

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        ground_true = ground_true.squeeze(0).float().to(self.device)


                else:
                    if self.args.output_attention:
                        scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        # Select top stocks based on scores
                        top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                        selected_scores = scores[top_indices]
                        selected_returns = returns[top_indices]
                        weights = torch.softmax(selected_scores / self.temperature, dim=0)
                        reward = self.env.step(weights, selected_returns)
                    else:
                        scores, log_prob = self.model.pi(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # Select top stocks based on scores
                        top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                        selected_scores = scores[top_indices]
                        selected_returns = returns[top_indices]
                        weights = torch.softmax(selected_scores / self.temperature, dim=0)
                        reward = self.env.step(weights, selected_returns)
                        self.env.rollout.append((batch_x, batch_y, batch_x_mark, batch_y_mark, scores.detach(), reward,
                                                 next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark,
                                                 log_prob.detach(), done, returns))
                    if len(self.env.rollout) == self.env.rollout_len:
                        self.put_data(self.env.rollout)
                        self.env.rollout = []

                    if rollout_idx != 0 and rollout_idx % self.env.rollout_len == 0:
                        loss = self.train_net(K_epoch=1, model_optim=model_optim)
                        train_loss.append(loss)

                    gc.collect()
                    torch.cuda.empty_cache()

            self.logger.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model