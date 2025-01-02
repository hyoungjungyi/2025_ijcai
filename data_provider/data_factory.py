from data_provider.data_loader import  Dataset_Custom, Dataset_moe,TimeSeriesDataset
from torch.utils.data import DataLoader,Subset

def data_provider(args, flag):
    """
        Data provider function to return dataset and dataloader for a given flag.

        Args:
            args: Argument parser object containing parameters.
            flag: Data split flag ('train', 'test', 'val', 'backtest', 'moe', etc.)

        Returns:
            tuple: Dataset object and DataLoader object.
        """



    if flag in ['test', 'backtest']:
        shuffle_flag = False
        drop_last = (flag == 'test')  # test이면 True, backtest이면 False 예시
        batch_size = 1
    elif flag == 'pred':
        # 예측은 보통 batch_size=1, shuffle=False
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    else:
        # train, val
        # args.data, args.train_method 등에 따라 shuffle 여부 결정 예시
        shuffle_flag = (args.data != 'alpha158') and (args.train_method == 'Supervise')
        drop_last = True
        batch_size = args.batch_size if hasattr(args, 'batch_size') else 1

        # Dataset 생성
        # 만약 moe_train이면 use_multi_horizon=True, 아니면 False 식으로 예시
    use_multi_horizon = args.moe_train
    lookaheads = args.horizons if use_multi_horizon else None
    if args.train_method =='Supervise':
        dataset = Dataset_Custom(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            timeenc=(1 if args.embed == 'timeF' else 0),
            freq=args.freq
        )

    else:
        dataset = TimeSeriesDataset(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            valid_year=args.valid_year,
            test_year=args.test_year,
            size=[args.seq_len, args.label_len, args.pred_len],
            use_multi_horizon=use_multi_horizon,
            lookaheads=lookaheads if lookaheads else [args.pred_len],  # 단일이면 [pred_len] 가정
            scale=True,
            timeenc=(1 if args.embed == 'timeF' else 0),
            freq=args.freq,
            step_size=args.pred_len if args.train_method != 'Supervise' else None
        )

    print(flag, len(dataset))

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        persistent_workers=True
    )
    return dataset, data_loader
