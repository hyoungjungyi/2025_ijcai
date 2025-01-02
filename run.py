import argparse
import torch
import random
import numpy as np
import pandas as pd
from exp.exp_supervise import Exp_Supervise
from exp.exp_reinforce import Exp_Reinforce
from exp.exp_moe import Exp_MOE

import torch.multiprocessing as mp
from utils.tools import *  
import os
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

def main():
    parser = argparse.ArgumentParser(description='Transformer Family and Mixture of experts for Time Series Forecasting')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--model',type=str,default='Transformer',help='options = [Transformer, LSTM, GRU, MHA, MHA_LSTM, MHA_GRU, MoE, MHA_MoE]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--train_method',type= str,default='Reinforce',help='options = [Reinforce, Supervise]')
    parser.add_argument('--moe_train', action='store_true', help='Enable MOE training after expert training',default=True)
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature parameter for softmax')
    # data loader
    parser.add_argument('--market',type=str,default='dj30',help='options = [dj30,nasdaq,kospi,csi300,sp500]')
    parser.add_argument('--data', type=str, default='general', help='options = [general,alpha158]')
    parser.add_argument('--root_path', type=str, help='root path for the dataset')
    parser.add_argument('--data_path', type=str, help='data path for the dataset')
    # parser.add_argument('--root_path', type=str, default='./data/kospi/',help='options = [dj30,nasdaq,kospi,csi300]')
    # parser.add_argument('--data_path', type=str, default='kospi_general_data.csv',help='options = [general,alpha158]')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--valid_year', type=int, default=2020, help='select valid period')
    parser.add_argument('--test_year', type=int, default=2021, help='select test period')
    parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')  # 12
    parser.add_argument('--label_len', type=int, default=5, help='start token length')  # 5
    parser.add_argument('--pred_len', type=int, default=5, help='prediction sequence length')  # 1,5,20
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '  # d
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 20],
                        help='List of horizons for multi-horizon trading, e.g. 1 5 20')


    # model define
    parser.add_argument('--enc_in', type=int, help='encoder input size (auto-detected from data)', required=False)
    parser.add_argument('--dec_in', type=int, help='decoder input size (auto-detected from data)', required=False)
    parser.add_argument('--c_out', type=int, default=1, help='output size')  # 26
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    # parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    # parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate,S:univariate predict univariate, MS:multivariate predict univariate')
    # parser.add_argument('--target', type=str, default='close', help='target feature in S or MS task')  # Adj Close,'OT'
    #optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    #portoflio
    parser.add_argument('--fee_rate', type=float, default=0.0001, help='tax fee rate')
    parser.add_argument('--select_factor', type=int, default=2, help='select factor to determine the number of stocks in portfolio')
    parser.add_argument('--num_stocks', type=int ,help='number of stocks to include in the portfolio')
    args = parser.parse_args()

    # Automatically set root_path and data_path based on market and data arguments
    if not args.root_path:
        args.root_path = f'./data/{args.market}/'
    if not args.data_path:
        args.data_path = f'{args.market}_{args.data}_data.csv'

    ##moe setting

    if args.train_method =='Supervise':
        args.moe_train = False

    if args.moe_train:
        args.pred_len = args.horizons[-1]

    # Automatically determine `enc_in` and `dec_in` based on input data
    data_file_path = f"{args.root_path}/{args.data_path}"
    try:
        data = pd.read_csv(data_file_path)
        num_features = data.shape[1] - 2  # Exclude datetime or index column date,tic (Unnamed:0)
        args.enc_in = num_features if args.enc_in is None else args.enc_in
        args.dec_in = num_features if args.dec_in is None else args.dec_in

        # Set num_stocks based on unique tickers
        args.num_stocks = len(data['tic'].unique()) // args.select_factor

        print(f"Detected {num_features} input features and select {args.num_stocks}  among {len(data['tic'].unique())} unique stocks. Setting enc_in={args.enc_in}, dec_in={args.dec_in}.")
        print(f"Detected {num_features} input features. Setting enc_in={args.enc_in}, dec_in={args.dec_in}.")
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}")
        return
        # Setting and logging configuration
    setting_components = [f"{args.train_method}",
                          f"moe_train-{args.moe_train}",
                          args.market,
                          args.data,
                          f"num_stocks({args.num_stocks})",
                          f"sl({args.seq_len})",
                          f"pl({args.pred_len})"
                          ]
    # Combine components to form the setting string
    setting = "_".join(setting_components)

    result_dir = os.path.join("./results", setting)
    os.makedirs(result_dir, exist_ok=True)

    # Initialize logger with result directory
    global logger
    logger = initialize_logger(result_dir)

    logger.info(f"Set root_path: {args.root_path}")
    logger.info(f"Set data_path: {args.data_path}")
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.seed is None:
        args.seed = int(time.time()) % (2 ** 32)  # Generate a seed based on current time
        print(f"No seed provided. Generated random seed: {args.seed}")
    seed = args.seed
    fix_seed(seed)
    # Training or Backtesting logic


    if args.is_training:
        # Train experts
        if args.moe_train:
            exp = Exp_MOE(args)
            exp.train(setting)
            logger.info(f">>>>>>> Expert + MOE Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.backtest(setting)
        else:
            exp = Exp_Supervise(args) if args.train_method == 'Supervise' else Exp_Reinforce(args)
            exp.train(setting)
            logger.info(f">>>>>>> Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.backtest(setting)
    else:
        exp = Exp_Supervise(args) if args.train_method == 'Supervise' else Exp_Reinforce(args)

        if args.moe_train:
            logger.info(f">>>>>>> Expert + MOE Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.moe_backtest(setting,1)
        else:
            logger.info(f">>>>>>> Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.backtest(setting,1)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()