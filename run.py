import argparse
import torch
import random
import numpy as np
import pandas as pd
from exp.exp_supervise import Exp_Supervise
from exp.exp_reinforce import Exp_Reinforce




def main():
    parser = argparse.ArgumentParser(description='Transformer Family and Mixture of experts for Time Series Forecasting')
    parser.add_argument('--seed', type=int, default=2024, help='random seed')
    parser.add_argument('--model',type=str,default='Transformer',help='options = [Transformer, LSTM, GRU, MHA, MHA_LSTM, MHA_GRU, MoE, MHA_MoE]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--train_method',type= str,default='Supervise',help='options = [Reinforce, Supervise]')
    parser.add_argument('--moe_train', action='store_true', help='Enable MOE training after expert training',default=False)
    # data loader
    parser.add_argument('--data',type=str,default='custom')
    parser.add_argument('--root_path', type=str, default='./data/dj30/',help='options = [dj30, nasdaq,kospi]')
    parser.add_argument('--data_path', type=str, default='data.csv',help='options = [dj30, nasdaq,kospi]')

    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--valid_year', type=int, default=2020, help='select valid period')
    parser.add_argument('--test_year', type=int, default=2021, help='select test period')
    parser.add_argument('--seq_len', type=int, default=40, help='input sequence length')  # 12
    parser.add_argument('--label_len', type=int, default=5, help='start token length')  # 5
    parser.add_argument('--pred_len', type=int, default=20, help='prediction sequence length')  # 1,5,20
    parser.add_argument('--freq', type=str, default='d',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '  # d
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

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
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    #portoflio
    parser.add_argument('--fee_rate', type=float, default=0.0001, help='tax fee rate')
    parser.add_argument('--num_stocks', type=int, default=29, help='number of stocks to include in the portfolio')
    args = parser.parse_args()
    # Automatically determine `enc_in` and `dec_in` based on input data
    data_file_path = f"{args.root_path}/{args.data_path}"
    try:
        data = pd.read_csv(data_file_path)
        num_features = data.shape[1] - 3  # Exclude datetime or index column ,Unnamed:0,date,tic
        args.enc_in = num_features if args.enc_in is None else args.enc_in
        args.dec_in = num_features if args.dec_in is None else args.dec_in
        print(f"Detected {num_features} input features. Setting enc_in={args.enc_in}, dec_in={args.dec_in}.")
    except Exception as e:
        print(f"Error loading data from {data_file_path}: {e}")
        return

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]


    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed()
    setting = f'{args.data}_{args.model}_sl({args.seq_len})_pl({args.pred_len})'
    if args.is_training:
        if args.train_method == 'Supervise':
            if args.moe_train:
                exp = Exp_Supervise(args)
                # exp.train_expert()
                # exp.train_moe()
                print('>>>>>>>ex_pert_backtesting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.moe_backtest(setting)
            else:
                exp = Exp_Supervise(args)
                exp.train(setting)
                print('>>>>>>>backtesting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.backtest(setting)
        else:
            pass
    else:
        exp = Exp_Supervise(args)
        print('>>>>>>>backtesting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.backtest(setting,load=1)
        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()