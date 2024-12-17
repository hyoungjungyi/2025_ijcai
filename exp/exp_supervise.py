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
from models import Transformer,moe
from utils.tools import EarlyStopping, adjust_learning_rate, visual,port_visual
from utils.metrics import metric
from utils.backtest import *
import matplotlib.pyplot as plt



# import FinanceDataReader as fdr
warnings.filterwarnings('ignore')


class Exp_Supervise(Exp_Basic):
    def __init__(self, args):
        super(Exp_Supervise, self).__init__(args)
        if args.moe_train:
            self.daily_model_path = os.path.join(self.args.checkpoints, f'{self.args.market}_{self.args.data}_num_stocks({args.num_stocks})_daily_sl({self.args.seq_len})_pl(1)_moe_train-{args.moe_train}/checkpoint.pth')
            self.weekly_model_path = os.path.join(self.args.checkpoints,f'{self.args.market}_{self.args.data}_num_stocks({args.num_stocks})_weekly_sl({self.args.seq_len})_pl(5)_moe_train-{args.moe_train}/checkpoint.pth')
            self.monthly_model_path = os.path.join(self.args.checkpoints, f'{self.args.market}_{self.args.data}_num_stocks({args.num_stocks})_monthly_sl({self.args.seq_len})_pl(20)_moe_train-{args.moe_train}/checkpoint.pth')
            self.daily_model = self._build_model()
            self.weekly_model = self._build_model()
            self.monthly_model = self._build_model()
            self.moe_model = moe.MOEModel(
                input_size=self.args.enc_in,
                experts=[self.daily_model, self.weekly_model, self.monthly_model],
                train_experts=False  # Freeze experts during MOE training
            )
    def _build_model(self):
        # model_dict = {
        #     'FEDformer': FEDformer,
        #     'iFEDformer': iFEDformer,
        #     'Autoformer': Autoformer,
        #     'Transformer': Transformer,
        #     'iTransformer': iTransformer,
        #     'Informer': Informer,
        #     'iInformer': iInformer,
        #     'Reformer': Reformer,
        #     'iReformer': iReformer,
        #     'Flowformer': Flowformer,
        #     'iFlowformer': iFlowformer,
        #     'Flashformer': Flashformer,
        #     'iFlashformer': iFlashformer,
        #
        # }
        model_dict = {
            'Transformer': Transformer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,ground_true) in enumerate(vali_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()

                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)

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

                # f_dim = -1 if self.args.features == 'MS' else 0
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                ground_true = ground_true.squeeze(0)
                pred = outputs.detach().cpu()
                true = ground_true

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,ground_true) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()

                batch_x = batch_x.squeeze(0).float().to(self.device)

                batch_y = batch_y.squeeze(0).float().to(self.device)
                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)

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

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        ground_true = ground_true.squeeze(0).float().to(self.device)
                        loss = criterion(outputs, ground_true)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0].squeeze(-1)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark).squeeze(-1)

                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    ground_true = ground_true.squeeze(0).float().to(self.device)
                    loss = criterion(outputs, ground_true)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     # self.logger.info.info("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     # self.logger.info.info('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            self.logger.info.info("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info.info("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,ground_true) in enumerate(test_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float().to(self.device)

                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)

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

                # f_dim = -1 if self.args.features == 'MS' else 0
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                ground_true = ground_true.squeeze(0)

                outputs = outputs.detach().cpu().numpy()
                # batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = ground_true  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        self.logger.info.info('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        self.logger.info.info('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        self.logger.info.info('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse: {:.3f}, mae: {:.3f}, rmse: {:.3f}, mape: {:.3f}, mspe: {:.3f}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

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

    def train_expert(self):
        """
              Train daily, weekly, and monthly expert models using the existing train method.
              """


        # Train daily expert
        self.logger.info.info("Training daily expert...")
        self.args.pred_len =  1
        setting = f'{self.args.data}_daily_sl({self.args.seq_len})_pl(1)'
        self.train(setting)


        # Train weekly expert
        self.logger.info.info("Training weekly expert...")
        self.args.pred_len =  5
        setting = f'{self.args.data}_weekly_sl({self.args.seq_len})_pl(5)'
        self.train(setting)


        # Train monthly expert
        self.logger.info.info("Training monthly expert...")
        self.args.pred_len =  20
        setting = f'{self.args.data}_monthly_sl({self.args.seq_len})_pl(20)'
        self.train(setting)

        self.logger.info.info("Finished training all experts.")

    def train_moe(self,setting):
        """
        Train MOE using the pre-trained expert models.
        """
        self.logger.info.info("Loading pre-trained expert models...")

        # Load the pre-trained experts
        # daily_model = self._build_model()
        self.daily_model.load_state_dict(torch.load(self.daily_model_path))
        self.daily_model.eval()

        # weekly_model = self._build_model()
        self.weekly_model.load_state_dict(torch.load(self.weekly_model_path))
        self.weekly_model.eval()

        # monthly_model = self._build_model()
        self.monthly_model.load_state_dict(torch.load(self.monthly_model_path))
        self.monthly_model.eval()

        self.logger.info.info("Experts loaded. Initializing MOE model...")

        # Initialize MOE model
        self.moe_model.to(self.device)

        # Load MOE-specific data
        self.logger.info.info("Loading MOE training data...")
        moe_data, moe_loader = self._get_data(flag='train')

        optimizer = torch.optim.Adam(self.moe_model.parameters(), lr=self.args.learning_rate)
        criterion = nn.CrossEntropyLoss()  # Classification loss for gating network

        self.logger.info.info("Training MOE model...")
        for epoch in range(self.args.train_epochs):
            self.moe_model.train()
            train_loss = []
            for batch_x, batch_y, batch_x_mark, batch_y_mark, target_period in moe_loader:
                optimizer.zero_grad()

                # Move data to the appropriate device
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float().to(self.device)
                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)
                target_period = target_period.to(self.device)

                # Create decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass through MOE
                outputs, gating_weights = self.moe_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Compute loss
                loss = criterion(gating_weights, target_period)
                loss.backward()
                optimizer.step()

                train_loss.append(loss.item())

            avg_loss = np.mean(train_loss)
            self.logger.info.info(f"Epoch {epoch + 1}, Loss: {avg_loss:.6f}")

        moe_model_path = self.args.checkpoints+'/'+'moe_model' + setting + '/'+'moe_model.pth'
        if not os.path.exists(moe_model_path):
            os.makedirs(moe_model_path)
        torch.save(self.moe_model.state_dict(), moe_model_path)
        self.logger.info.info("MOE model training completed and saved.")


    def backtest(self, setting, load=False):
        test_data, test_loader = self._get_data(flag='test')
        back_test_data, back_test_loader = self._get_data(flag='backtest')
        dates = test_data.gt.index.get_level_values('date').unique()  # date 정보 추출
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))


        portfolio_values =[1.0]
        portfolio_data = []
        fee_rate = self.args.fee_rate
        num_stocks = self.args.num_stocks
        pred_len = self.args.pred_len
        portfolio_value = 1.0
        # x축을 전체 기간으로 사용

        indices = list(range(0, len(dates) -self.args.seq_len - pred_len -1, pred_len))
        investment_dates = [dates[0]] + [dates[i + self.args.seq_len] for i in indices]

        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark,ground_true) in zip(indices,back_test_loader):
                batch_x = batch_x.squeeze(0).float().to(self.device)
                batch_y = batch_y.squeeze(0).float()
                batch_x_mark = batch_x_mark.squeeze(0).float().to(self.device)
                batch_y_mark = batch_y_mark.squeeze(0).float().to(self.device)

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

                pred = outputs.detach().cpu().numpy()
                ground_true = ground_true.squeeze(0)
                # 예측값을 기준으로 상위 num_stocks 종목 선택
                selected_indices = np.argsort(-pred)[:num_stocks]
                selected_preds = pred[selected_indices]

                # MultiIndex에서 tic과 실제 수익률 가져오기
                current_date = dates[i + self.args.seq_len]
                original_true = test_data.raw_gt.loc[current_date]
                selected_tics = original_true.index[selected_indices]
                selected_true = original_true.values[selected_indices]
                # 현재 날짜
                selected_tics_fix = test_data.gt.loc[current_date, :].index[selected_indices]
                selected_true_fix = ground_true[selected_indices]

                # 선택된 종목 정보 저장
                portfolio_data.append({
                    "date": current_date,
                    "tics": selected_tics.tolist(),
                    "pred": selected_preds.tolist(),
                    "ground_true": selected_true.tolist()
                })


                # 포트폴리오 수익률 계산
                weights = np.ones(num_stocks) / num_stocks  # 동일 비중
                portfolio_return = np.dot(weights, selected_true)

                traded_value = portfolio_value
                buy_fee = traded_value * fee_rate
                sell_fee = (portfolio_value + portfolio_value * portfolio_return) * fee_rate

                # update portoflio value
                portfolio_value = portfolio_value + (portfolio_value * portfolio_return) - buy_fee - sell_fee
                portfolio_values.append(portfolio_value)


        portfolio_values = np.array(portfolio_values)


        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv(folder_path + 'portfolio_data.csv', index=False)

        # Use run_backtest for comparisons
        start_date = str(dates[0])
        end_date = str(dates[-1])
        back_test_data = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
        back_test_data['date'] = pd.to_datetime(back_test_data['date'])
        if self.args.market == 'kospi':
            index_name = '^KS11'  # KOSPI Index
        elif self.args.market == 'dj30':
            index_name = '^DJI'  # Dow Jones Industrial Average
        elif self.args.market == 'sp500':
            index_name = '^GSPC'  # S&P 500 Index
        elif self.args.market == 'nasdaq':
            index_name = '^IXIC'  # NASDAQ Composite
        elif self.args.market == 'csi300':
            index_name = '000300.SS'  # CSI 300 Index
        else:
            raise ValueError(f"Unsupported market: {self.args.market}")

        index_data = fetch_index_data(index_name, '2012-01-01', '2024-01-01')

        run_backtest(
            data=back_test_data,  # Your dataset as DataFrame
            index_data=index_data,  # Index data (e.g., DJI)
            start_date=start_date,
            end_date=end_date,
            fee_rate=self.args.fee_rate,
            external_portfolio=portfolio_values,
            external_dates=investment_dates,
            pred_len=pred_len,
            total_periods=test_data.raw_gt.index.get_level_values('date').nunique(),
            folder_path=folder_path
        )
        # 시각화
        self.logger.info.info("Backtest completed. Results saved to:", folder_path)

    def moe_backtest(self, setting, load=True):
        """
        Perform backtesting using the trained MOE model with dynamic periods based on gating choices.

        Args:
            setting (str): Name of the model setting.
            load (bool): Whether to load a pre-trained model for backtesting.
        """
        test_data, test_loader = self._get_data(flag='test')
        unique_dates = test_data.data_x.index.get_level_values('date').unique()

        # Load the pre-trained MOE model if specified
        if load:
            model_path = os.path.join(self.args.checkpoints, setting)
            moe_model_path = model_path + '/' + 'moe_model.pth'
            self.moe_model.load_state_dict(torch.load(moe_model_path))
            self.moe_model.to(self.device)
            self.moe_model.eval()

        portfolio_values = [1.0]
        portfolio_data = []
        gating_choices = []
        fee_rate = self.args.fee_rate
        num_stocks = self.args.num_stocks
        portfolio_value = 1.0



        # Initialize investment_dates dynamically
        investment_dates = [unique_dates[0]]
        step_sizes = []
        index = 0
        self.moe_model.eval()

        with torch.no_grad():

            while index < len(unique_dates) - self.args.seq_len - self.args.pred_len -1:

                batch_x, batch_y, batch_x_mark, batch_y_mark, ground_true = test_data[index]


                # Prepare inputs for the model

                batch_x =torch.tensor(batch_x, dtype=torch.float32).to(self.device)
                batch_y = torch.tensor(batch_y, dtype=torch.float32).to(self.device)
                batch_x_mark = torch.tensor(batch_x_mark, dtype=torch.float32).float().to(self.device)
                batch_y_mark = torch.tensor(batch_y_mark, dtype=torch.float32).float().to(self.device)

                # Create decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass through MOE
                outputs, gating_weights = self.moe_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Choose expert with the highest gating weight
                gating_choice = gating_weights.argmax(dim=1).item()
                gating_choices.append(gating_choice)

                # Determine step size based on gating choice
                step_size = [1, 5, 20][gating_choice]
                step_sizes.append(step_size)

                # Add the current investment date
                current_date = unique_dates[index + self.args.seq_len]
                investment_dates.append(current_date)

                # Simulate investment using the selected expert
                pred = outputs.cpu().numpy()  # [batch, num_stocks]
                 # Average over all selected stocks

                # Sort predictions to select top num_stocks
                selected_indices = np.argsort(-pred)[:num_stocks]


                # Calculate portfolio returns
                if step_size ==1:
                    current_ground_true = test_data.daily_label[unique_dates[index+self.args.seq_len]]
                elif step_size ==5:
                    current_ground_true = test_data.weekly_label[unique_dates[index+self.args.seq_len]]
                else:
                    current_ground_true = test_data.monthly_label[unique_dates[index+self.args.seq_len]]


                selected_true = current_ground_true[selected_indices]

                weights = np.ones(num_stocks) / num_stocks
                portfolio_return = np.dot(weights, selected_true)

                traded_value = portfolio_value
                buy_fee = traded_value * fee_rate
                sell_fee = (portfolio_value + portfolio_value * portfolio_return) * fee_rate

                # Update portfolio value
                portfolio_value = portfolio_value + (portfolio_value * portfolio_return) - buy_fee - sell_fee
                portfolio_values.append(portfolio_value)

                # Record portfolio data
                portfolio_data.append({

                    "gating_choice": gating_choice,
                    "portfolio_return": portfolio_return,
                    "portfolio_value": portfolio_value
                })

                # Move index forward by step_size
                index += step_size

        # Calculate dynamic annual factor
        total_days_in_backtest = len(unique_dates) - self.args.seq_len - 20 - 1
        average_step_size = sum(step_sizes) / len(step_sizes)
        dynamic_annual_factor = total_days_in_backtest / average_step_size

        # Save results
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df.to_csv(folder_path + 'moe_portfolio_dynamic_periods.csv', index=False)

        # Backtest visualization and result comparison
        start_date = str(unique_dates[0])
        end_date = str(unique_dates[-1])
        back_test_data = pd.read_csv(os.path.join(self.args.root_path, self.args.data_path))
        back_test_data['date'] = pd.to_datetime(back_test_data['date'])
        if self.args.market == 'kospi':
            index_name = '^KS11'  # KOSPI Index
        elif self.args.market == 'dj30':
            index_name = '^DJI'  # Dow Jones Industrial Average
        elif self.args.market == 'sp500':
            index_name = '^GSPC'  # S&P 500 Index
        elif self.args.market == 'nasdaq':
            index_name = '^IXIC'  # NASDAQ Composite
        elif self.args.market == 'csi300':
            index_name = '000300.SS'  # CSI 300 Index
        else:
            raise ValueError(f"Unsupported market: {self.args.market}")

        index_data = fetch_index_data(index_name, '2012-01-01', '2024-01-01')

        run_backtest(
            data=back_test_data,  # Dataset as DataFrame
            index_data=index_data,  # Index data (e.g., DJI)
            start_date=start_date,
            end_date=end_date,
            fee_rate=self.args.fee_rate,
            external_portfolio=portfolio_values,
            external_dates=investment_dates,
            pred_len=None,
            total_periods=None,
            folder_path=folder_path,
            dynamic_annual_factor = dynamic_annual_factor
        )
        self.logger.info.info("MOE Backtest completed. Results saved to:", folder_path)






