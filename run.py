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
import wandb
import uuid

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ['OPENBLAS_NUM_THREADS'] = "1"

def main():
    parser = argparse.ArgumentParser(description='Transformer Family and Mixture of experts for Time Series Forecasting')
    parser.add_argument('--wandb_project_name', '-wpn', type=str, default='KDD2025')
    parser.add_argument('--wandb_group_name', '-wgn', type=str, default='Debugging Mode')
    parser.add_argument('--wandb_session_name', '-wsn', type=str, default='Debugging Mode')


    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--model',type=str,default='Transformer',help='options = [Transformer,Reformer,Informer,Autoformer,Fedformer,Flowformer,Flashformer,itransformer,crossformer,deformer,deformableTST]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--train_method',type= str,default='Reinforce',help='options = [Reinforce, Supervise]')
    parser.add_argument('--moe_train', action='store_true', help='Enable MOE training after expert training',default=True)
    parser.add_argument('--transfer', action='store_true', help='whether to use transfer learning',default = False)
    parser.add_argument('--freeze', action='store_true', help='whether to use transfer learning', default=True)


    parser.add_argument('--temperature', type=float, default=1.0, help='temperature parameter for softmax')
    # data loader
    parser.add_argument('--market',type=str,default='dj30',help='options = [dj30,nasdaq,kospi,csi300,ftse]')
    parser.add_argument('--data', type=str, default='general', help='options = [general,alpha158]')
    parser.add_argument('--root_path', type=str, help='root path for the dataset')
    parser.add_argument('--data_path', type=str, help='data path for the dataset')
    # parser.add_argument('--root_path', type=str, default='./data/kospi/',help='options = [dj30,nasdaq,kospi,csi300]')
    # parser.add_argument('--data_path', type=str, default='kospi_general_data.csv',help='options = [general,alpha158]')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--valid_year', type=int, default=2020, help='select valid period') #2020
    parser.add_argument('--test_year', type=int, default=2021, help='select test period') #2021
    parser.add_argument('--seq_len', type=int, default=20, help='input sequence length')  # 12
    parser.add_argument('--label_len', type=int, default=5, help='start token length')  # 5
    parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')  # 1,5,20 #reinforce는 1로 고정
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '  # d
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--horizons', nargs='+', type=int, default=[1, 5, 20],
                        help='List of horizons for multi-horizon trading, e.g. 1 5 20')


    # model define
    parser.add_argument('--enc_in', type=int, help='encoder input size (auto-detected from data)', required=False)
    parser.add_argument('--dec_in', type=int, help='decoder input size (auto-detected from data)', required=False)
    parser.add_argument('--c_out', type=int, default=1, help='output size')  # 26
    parser.add_argument('--mode_select', type=str, default='low',help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=4, help='modes to be selected random 64')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
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
    ##DeformableTST parameters
    parser.add_argument('--revin', type=int, default=1, help='use RevIN; True 1 False 0')
    parser.add_argument('--revin_affine', type=int, default=0, help='use RevIN-affine; True 1 False 0')
    parser.add_argument('--revin_subtract_last', type=int, default=0,
                        help='0: subtract mean; 1: subtract the last value')
    parser.add_argument('--stem_ratio', type=int, default=3, help='down sampling ratio in stem layer')
    parser.add_argument('--down_ratio', type=int, default=2,
                        help='down sampling ratio in DownSampling layer between two stages')
    parser.add_argument('--fmap_size', type=int, default=768, help='feature series length')
    parser.add_argument('--dims', nargs='+', type=int, default=[64, 128, 256, 512], help='dims for each stage')
    parser.add_argument('--depths', nargs='+', type=int, default=[1, 1, 3, 1],
                        help='number of Transformer blocks for each stage')
    parser.add_argument('--drop_path_rate', type=float, default=0.3, help='drop path rate')
    parser.add_argument('--layer_scale_value', nargs='+', type=float, default=[-1, -1, -1, -1],
                        help='layer_scale_init_value')
    parser.add_argument('--use_pe', nargs='+', type=int, default=[1, 1, 1, 1], help='use pe; True 1 False 0')
    parser.add_argument('--use_lpu', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='use Local Perception Unit; True 1 False 0')
    parser.add_argument('--local_kernel_size', nargs='+', type=int, default=[3, 3, 3, 3], help='kernel size for LPU')
    parser.add_argument('--expansion', type=int, default=4, help='ffn ratio')
    parser.add_argument('--drop', type=float, default=0.0, help='dropout prob for FFN module')
    parser.add_argument('--use_dwc_mlp', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='use FFN with a DWConv; True 1 False 0')
    parser.add_argument('--heads', nargs='+', type=int, default=[4, 8, 16, 32], help='number of heads')
    parser.add_argument('--attn_drop', type=float, default=0.0,
                        help='dropout prob for attention map in attention module')
    parser.add_argument('--proj_drop', type=float, default=0.0, help='dropout prob for proj in attention module')
    parser.add_argument('--stage_spec', nargs='+', type=list, default=[['D'], ['D'], ['D', 'D', 'D'], ['D']],
                        help='type of blocks in each stage')
    parser.add_argument('--window_size', nargs='+', type=int, default=[3, 3, 3, 3],
                        help='kernel size for window attention')
    parser.add_argument('--nat_ksize', nargs='+', type=int, default=[3, 3, 3, 3],
                        help='kernel size for neighborhood attention')
    parser.add_argument('--ksize', nargs='+', type=int, default=[9, 7, 5, 3], help='kernel size for offset sub-network')
    parser.add_argument('--stride', nargs='+', type=int, default=[8, 4, 2, 1], help='stride for offset sub-network')
    parser.add_argument('--n_groups', nargs='+', type=int, default=[2, 4, 8, 16], help='number of offset groups')
    parser.add_argument('--offset_range_factor', nargs='+', type=float, default=[-1, -1, -1, -1],
                        help='restrict the offset value in a small range')
    parser.add_argument('--no_off', nargs='+', type=int, default=[0, 0, 0, 0], help='not use offset; True 1 False 0')
    parser.add_argument('--dwc_pe', nargs='+', type=int, default=[0, 0, 0, 0], help='use DWC-pe; True 1 False 0')
    parser.add_argument('--fixed_pe', nargs='+', type=int, default=[0, 0, 0, 0], help='use fixed pe; True 1 False 0')
    parser.add_argument('--log_cpb', nargs='+', type=int, default=[0, 0, 0, 0],
                        help='use pe of SWin-v2; True 1 False 0')
    parser.add_argument('--head_dropout', type=float, default=0.1, help='dropout prob for the head')
    parser.add_argument('--head_type', type=str, default='Flatten', help='Flatten')
    parser.add_argument('--use_head_norm', type=int, default=1, help='use final LN layer; True 1 False 0')


    #optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    #portoflio
    parser.add_argument('--fee_rate', type=float, default=0.001, help='tax fee rate')
    parser.add_argument('--complex_fee', action='store_true', default=True)
    # parser.add_argument('--select_factor', type=int, default=2, help='select factor to determine the number of stocks in portfolio')
    parser.add_argument('--num_stocks', type=int ,help='number of stocks to include in the portfolio',default=20)
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
        # args.num_stocks = len(data['tic'].unique()) //2
        # Set num_stocks based on unique tickers
        if (not args.num_stocks ) or (args.num_stocks > len(data['tic'].unique())):
            args.num_stocks = len(data['tic'].unique())

        print(f"Detected {num_features} input features and select {args.num_stocks}  among {len(data['tic'].unique())} unique stocks. Setting enc_in={args.enc_in}, dec_in={args.dec_in}.")
        print(f"Detected {num_features} input features. Setting enc_in={args.enc_in}, dec_in={args.dec_in}.")
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}")
        return
        # Setting and logging configuration
    setting_components = [f"{args.model}",f"{args.train_method}",
                          f"moe_train-{args.moe_train}",
                          args.market,
                          args.data,
                          f"num_stocks({args.num_stocks})",
                          f"sl({args.seq_len})",
                          f"pl({args.pred_len})"
                          ]
    # Combine components to form the setting string
    setting = "_".join(setting_components)
    unique_id = uuid.uuid4().hex[:8]
    unique_setting = f"{setting}_{unique_id}"
    result_dir = os.path.join("./results", unique_setting)
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



    if args.is_training:
        # Train experts
        if args.moe_train:
            if not args.transfer:
                exp = Exp_MOE(args,unique_setting)
                exp.train(unique_setting)
                logger.info(f">>>>>>> Expert + MOE Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.backtest(unique_setting)
            else:
                args.train_method = 'Supervise'
                args.moe_train = False
                exp_supervise = Exp_Supervise(args,unique_setting)
                exp_supervise.train(unique_setting)
                logger.info(f">>>>>>> Complete supervise learning <<<<<<<<<<<<<<<<<<<<<<<<<<<")
                args.train_method = 'Reinforce'
                args.moe_train = True
                exp = Exp_MOE(args,unique_setting)
                exp.train(setting)
                logger.info(f">>>>>>> Expert + MOE Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
                exp.backtest(unique_setting)
        else:
            exp = Exp_Supervise(args,unique_setting) if args.train_method == 'Supervise' else Exp_Reinforce(args,unique_setting)
            exp.train(unique_setting)
            logger.info(f">>>>>>> Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.backtest(unique_setting)
    else:
        exp = Exp_Supervise(args,unique_setting) if args.train_method == 'Supervise' else Exp_Reinforce(args,unique_setting)

        if args.moe_train:
            logger.info(f">>>>>>> Expert + MOE Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.moe_backtest(setting,1)
        else:
            logger.info(f">>>>>>> Backtesting: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.backtest(setting,1)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()