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
from models import Transformer,moe,ppo,tradingenv,ada_ppo
from utils.tools import EarlyStopping, adjust_learning_rate, visual,port_visual
from utils.metrics import metric
from utils.backtest import *
from itertools import tee
import gc
import torch.nn.functional as F

import matplotlib.pyplot as plt



# import FinanceDataReader as fdr
warnings.filterwarnings('ignore')



def get_sample(dataset, index, device):
    sample = dataset[index]  # __getitem__ 호출
    (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) = sample
    return (
        torch.tensor(batch_x, dtype=torch.float).to(device),
        torch.tensor(batch_y, dtype=torch.float).to(device),
        torch.tensor(batch_x_mark, dtype=torch.float).to(device),
        torch.tensor(batch_y_mark, dtype=torch.float).to(device),
        torch.tensor(ground_true, dtype=torch.float).to(device)
    )

class Exp_MOE(Exp_Basic):
    def __init__(self, args):
        super(Exp_MOE, self).__init__(args)
        self.horizons = args.horizons
        self.temperature = self.args.temperature
        self.env = tradingenv.TradingEnvironment(self.args)
        #### agent config###
        if self.args.num_stocks % 2==1:
            self.batch_size = self.args.num_stocks * self.args.select_factor +1
        else:
            self.batch_size = self.args.num_stocks * self.args.select_factor

        self.buffer_size = 1  #20
        self.gamma =0.99
        self.lmbda = 0.95
        self.eps_clip = 0.1
        self.data = []
        self.max_clip = 10  # 적절한 범위 설정 (예: 10은 exp(10) = 22026 정도)
        self.min_clip = -10  # exp(-10) ≈ 0
        ####################
        # if args.moe_train:
        #     self.daily_model = self._build_model()
        #     self.weekly_model = self._build_model()
        #     self.monthly_model = self._build_model()
        #     self.moe_model = moe.MOEModel(
        #         input_size=self.args.enc_in,
        #         experts=[self.daily_model, self.weekly_model, self.monthly_model],
        #         train_experts=False  # Freeze experts during MOE training
        #     )
        ##############################
    def _build_model(self):

        model = ada_ppo.ADA_PPO(self.args.model, self.args, deterministic=False)

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



    def train(self, setting):
        train_dataset,_ = self._get_data(flag='train')
        n_data = len(train_dataset)
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)


        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()


        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            self.logger.info(f"[Train] Epoch {epoch + 1}/{self.args.train_epochs}")
            self.model.train()
            self.env.reset()

            i = 0
            epoch_loss = []

            epoch_time = time.time()
            while i < n_data:
                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = get_sample(train_dataset, i, self.device)
                dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                # Select top stocks based on scores
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                weights = torch.softmax(selected_scores / self.temperature, dim=0)
                returns = ground_true[:,selected_period_indices]
                selected_returns = returns[top_indices]
                reward = self.env.step(weights,selected_returns)
                chosen_horizon =self.horizons[selected_period_indices]
                next_i = i+chosen_horizon
                done = (next_i >= n_data - 1)
                if not done:
                    (next_batch_x, next_batch_y,
                     next_batch_x_mark, next_batch_y_mark,
                     next_ground_true) = get_sample(train_dataset, next_i, self.device)

                else:
                    next_batch_x = torch.zeros_like(batch_x)
                    next_batch_y = torch.zeros_like(batch_y)
                    next_batch_x_mark = torch.zeros_like(batch_x_mark)
                    next_batch_y_mark = torch.zeros_like(batch_y_mark)


                transition = (
                    batch_x, batch_y,batch_x_mark, batch_y_mark,  # current state
                    scores.detach(), reward,
                    next_batch_x,next_batch_y, next_batch_x_mark, next_batch_y_mark,  # next state
                    log_prob.detach(),
                    done,
                    returns,
                )
                self.env.rollout.append(transition)

                i = next_i
                if len(self.env.rollout) >= self.env.rollout_len:
                    self.put_data(self.env.rollout)
                    self.env.rollout = []
                    loss_val = self.train_net(K_epoch=1, model_optim=model_optim)
                    if loss_val is not None:
                        epoch_loss.append(loss_val)
            vali_loss, test_loss = self.vali()
            avg_train_loss = np.mean(epoch_loss) if epoch_loss else 0.0

            self.logger.info(
                f"[Epoch {epoch + 1}] TrainLoss={avg_train_loss:.4f}, "
                f"ValiLoss={vali_loss:.4f}, TestLoss={test_loss:.4f}"
            )

            # Early Stopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopped!")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

                # Best model 로드
            best_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_path))
            self.logger.info("Training completed, best model loaded.")

    def vali(self):
        # validation
        vali_dataset,_ = self._get_data('val')
        v_loss = self._evaluate_dataset(vali_dataset)

        # test
        test_dataset,_ = self._get_data('test')
        t_loss = self._evaluate_dataset(test_dataset)

        return v_loss, t_loss

    def _evaluate_dataset(self, dataset):
        self.model.eval()
        env_temp = tradingenv.TradingEnvironment(self.args)
        env_temp.reset()
        total_loss, total_value_loss, total_pred_loss = 0.0, 0.0, 0.0
        n_data = len(dataset)
        i = 0
        total_reward = 0.0

        while i < n_data:
            (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) = get_sample(dataset, i, self.device)

            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)

            with torch.no_grad():
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                # Select top stocks based on scores
                top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
                selected_scores = scores[top_indices]
                weights = torch.softmax(selected_scores / self.temperature, dim=0)
                returns = ground_true[:, selected_period_indices]
                selected_returns = returns[top_indices]
                reward = env_temp.step(weights,selected_returns)
                chosen_horizon = self.horizons[selected_period_indices]
                i += chosen_horizon
                done = (i >= n_data - 1)
                total_reward += reward
                if not done:
                    (next_batch_x, next_batch_y,
                     next_batch_x_mark, next_batch_y_mark,
                     next_ground_true) = get_sample(dataset, i, self.device)
                else:
                    next_batch_x = torch.zeros_like(batch_x)
                    next_batch_y = torch.zeros_like(batch_y)
                    next_batch_x_mark = torch.zeros_like(batch_x_mark)
                    next_batch_y_mark = torch.zeros_like(batch_y_mark)
                value = self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark)
                next_value = self.model.value(next_batch_x, next_batch_x_mark, next_batch_y, next_batch_y_mark)
                td_target = reward + self.gamma * next_value * (0 if done else 1)
                # loss = value_loss + pred_loss
                value_loss = F.smooth_l1_loss(value, td_target)
                pred_loss = F.smooth_l1_loss(scores, returns)  # scores vs ground_true
                loss = value_loss + pred_loss

                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_pred_loss += pred_loss.item()
        avg_loss = total_loss
        self.logger.info(
            f"[Validation]total_reward={total_reward:.4f} total_Loss={avg_loss:.4f}, Value_Loss={total_value_loss:.4f}, Pred_Loss={total_pred_loss:.4f}"
        )

        self.model.train()  # reset train mode
        return avg_loss


    def backtest(self, setting, load=False):
        """
        Backtest the trained model using reinforcement learning principles.
        """
        if load:
            best_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_path))
            self.logger.info(f"Loaded best model from {best_path}")

            # backtest dataset
        backtest_dataset,_ = self._get_data('backtest')
        n_data = len(backtest_dataset)
        self.model.eval()

        env_test = tradingenv.TradingEnvironment(self.args)
        env_test.reset()

        portfolio_values = []
        portfolio_dates = []

        i = 0
        while i < n_data:
            (batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true) = get_sample(backtest_dataset, i, self.device)

            dec_zeros = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_zeros], dim=1)

            with torch.no_grad():
                scores, log_prob, entropy, selected_period_indices = self.model.pi(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
            current_date = backtest_dataset.unique_dates[i]
            # Select top stocks based on scores
            top_indices = torch.topk(scores, self.args.num_stocks, dim=0).indices
            selected_scores = scores[top_indices]
            weights = torch.softmax(selected_scores / self.temperature, dim=0)
            selected_returns = ground_true[:, selected_period_indices][top_indices]
            chosen_horizon = self.horizons[selected_period_indices]
            reward = env_test.step(weights, selected_returns)
            portfolio_values.append(env_test.portfolio_value)
            portfolio_dates.append(current_date)

            i += chosen_horizon

        final_pf = env_test.portfolio_value
        self.logger.info(f"[BackTest] Final Portfolio Value = {final_pf:.4f}")
        start_date = backtest_dataset.unique_dates[0]
        end_date = backtest_dataset.unique_dates[-1]

        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        raw_data = pd.read_csv(csv_path)
        raw_data['date'] = pd.to_datetime(raw_data['date'])
        index_name = self._get_market_index_name()
        index_data = fetch_index_data(index_name, start_date, end_date)
        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)

        run_backtest(
            data=raw_data,
            index_data=index_data,
            start_date=start_date,   # pd.Timestamp
            end_date=end_date,       # pd.Timestamp
            fee_rate=self.args.fee_rate,  # 수수료율
            external_portfolio=np.array(portfolio_values),
            external_dates=pd.to_datetime(portfolio_dates),
            pred_len=self.args.pred_len,
            total_periods=len(backtest_dataset.unique_dates),  # 혹은 다른 값
            folder_path=folder_path
        )

        return final_pf

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
            try:
                # if key in ["reward", "log_prob", "done"]:
                if key in ["reward", "done"]:
                    batch_data[key] = torch.tensor(batch_data[key], dtype=torch.float).squeeze().to(self.device)

                else:
                    batch_data[key] = torch.stack(batch_data[key], dim=0).to(self.device)
            except:
                print()

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

    def train_net(self, K_epoch=10, model_optim=None):
        if len(self.data) == self.buffer_size:
            data = self.make_batch()
            data = self.calc_advantage(data)
            losses = 0.0

            for _ in range(K_epoch):
                for mini_batch in data:
                    batch_x, batch_y, batch_x_mark, batch_y_mark, scores, reward, next_batch_x, next_batch_y, next_batch_x_mark, next_batch_y_mark, done_mask, old_log_prob, td_target, advantage, return_data = mini_batch

                    scores, log_prob,_,_ = self.model.pi(batch_x, batch_x_mark, batch_y, batch_y_mark)

                    ratio = torch.exp(torch.clamp(log_prob - old_log_prob, min=self.min_clip, max=self.max_clip))
                    # ratio = torch.exp(log_prob - old_log_prob)  # a/b == exp(log(a)-log(b))

                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                    pred_loss = F.smooth_l1_loss(self.model.pred(batch_x, batch_x_mark, batch_y, batch_y_mark),
                                                 return_data)
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(
                        self.model.value(batch_x, batch_x_mark, batch_y, batch_y_mark), td_target) + pred_loss
                    losses += loss.mean().item()
                    model_optim.zero_grad()
                    loss.mean().backward()
                    model_optim.step()
            return losses

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

