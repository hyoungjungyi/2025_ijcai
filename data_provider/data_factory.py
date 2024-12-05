from data_provider.data_loader import  Dataset_Custom, Dataset_Pred,Dataset_moe
from torch.utils.data import DataLoader,Subset

data_dict = {
    'custom': Dataset_Custom,
    # 'portfolio':Dataset_Port_Pred,
}


def data_provider(args, flag):
    """
        Data provider function to return dataset and dataloader for a given flag.

        Args:
            args: Argument parser object containing parameters.
            flag: Data split flag ('train', 'test', 'val', 'backtest', 'moe', etc.)

        Returns:
            tuple: Dataset object and DataLoader object.
        """
    if args.moe_train:
        Data = Dataset_moe  # Use Dataset_moe for MOE training
    else:
        Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag in ['test', 'backtest']:  # test와 backtest 조건 통합
        shuffle_flag = False
        drop_last = False if flag == 'backtest' else True  # backtest에서는 drop_last=False
        batch_size = 1
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = 1  # args.batch_size
        freq = args.freq

    # Dataset 생성
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        valid_year=args.valid_year,
        test_year=args.test_year,
        size=[args.seq_len, args.label_len, args.pred_len],
        timeenc=timeenc,
        freq=freq
    )
    print(flag, len(data_set))

    # DataLoader 생성
    if flag == 'backtest':
        # SubsetRandomSampler를 사용하여 pred_len 간격의 인덱스를 생성
        pred_len = args.pred_len
        indices = list(range(0, len(data_set), pred_len))  # pred_len 간격으로 인덱스 생성
        subset_data = Subset(data_set, indices)

        data_loader = DataLoader(
            subset_data,
            batch_size=batch_size,
            num_workers=args.num_workers,
            drop_last=drop_last
        )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

    return data_set, data_loader
