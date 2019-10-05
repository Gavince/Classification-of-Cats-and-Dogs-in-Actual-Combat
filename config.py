import torch as t


class DefaultConfig:
    """
    设置默认参数
    """
    #0.模型选择

    model = "AlexNet"
    # 1.数据集路径
    train_root = "/home/gavin/Pytorch/Code/第六章/New/data/train"
    test_root = "/home/gavin/Pytorch/Code/第六章/New/data/test"

    # 2.数据保存
    model_path = "./checkpoints/AlexNet.pth"
    result_path = "./result.csv"

    # 3.固定参数设定
    device = t.device("cuda" if t.cuda.is_available() else "cpu")#较为灵活的设置
    lr = 0.01
    weight_decay = 0e-5
    classes_num = 2
    batch_size = 32
    num_works = 4
    epoch_nums = 60

    # 4.tensorboard参数设定
    logdir = "AlexNet"
    comment = " "


opt = DefaultConfig()
