import torch as t
import torch.nn as nn
from torch.nn.functional import softmax
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T
from data import DogCat
import models
from config import opt
import torchnet.meter as meter
import fire

classes = ["cat", "dog"]


def train():
    # 1.选择合适的model
    model = getattr(models, opt.model)()
    model.cuda()

    # 2.得到批量化的数据
    train_data = DogCat(root=opt.train_root, train=True)
    val_data = DogCat(root=opt.train_root, train=False)

    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_works)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_works)

    # 3.设置参数
    loss_meter = meter.AverageValueMeter()
    confuse_metrix = meter.ConfusionMeter(2)

    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    # optimizer = t.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    writer = SummaryWriter()
    for epoch in range(opt.epoch_nums):  # 设置迭代次数
        # 4 执行批量化的数据
        print("Epoch :", epoch)
        loss_meter.reset()
        confuse_metrix.reset()
        for ii, (data, label) in tqdm(enumerate(train_dataloader)):
            input1 = data.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()#梯度清零
            score = model(input1)
            loss = criterion(score, target)
            loss.backward()
            optimizer.step()

            # 记录数据
            confuse_metrix.add(score.detach(), target.detach())
            loss_meter.add(loss.item())
            if (ii + 1) % 20 == 0:
                print("loss_meter", loss_meter.value()[0])
                writer.add_scalar("Loss", loss_meter.value()[0])

        if epoch == 50:
            opt.lr = 0.001

    # 5. 保存网络数据
    writer.close()
    t.save(model.state_dict(), opt.model_path)


@t.no_grad()
def test():
    # 1.加载模型
    # model = AlexNet()
    model = getattr(models, opt.model)()

    model.to(opt.device)
    # 2.加载模型数据
    model.load_state_dict(t.load(opt.model_path))
    # 3.加载测试集数据
    test_data = DogCat(root=opt.test_root, train=False, test=True)
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False,
                                  num_workers=opt.num_works)

    # 4.测试集合的结果
    result = []

    # 5.测试结果
    for ii, (data, path) in tqdm(enumerate(test_data_loader)):
        input = data.to(opt.device)
        score = model(input)
        # TODO Softmax知识点
        probability = softmax(score, dim=1)[:, 0].detach().tolist()  # dog的概率

        batch_result = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        result += batch_result

    write_csv(result)


def write_csv(results):
    """
    写入csv文件
    :param results:
    :param file_name:
    :return:
    """
    import csv
    with open(opt.result_path, "w") as f:
        write = csv.writer(f)
        write.writerow(['id', 'label'])
        write.writerows(results)


@t.no_grad()
def predict():
    """
    预测单张照片
    :return:
    """
    # global input
    address = input("Image path:")
    # address = "/home/gavin/Pytorch/Code/第六章/New/data/test/1.jpg"
    img = Image.open(address)
    
    # 注意：因为全连接的缘故，所以在测试图片时我们必须要规定输图片的大小
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transforms = T.Compose([
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
    # 1.加载模型
    # model = AlexNet()
    model = getattr(models, opt.model)()

    model.to(opt.device)
    # 2.加载模型数据
    model.load_state_dict(t.load(opt.model_path))

    # 3.装换类型

    img = transforms(img)  # 装换数据格式
    img = img.to(opt.device)  # 转入GPU
    img = img.unsqueeze(0)  # 装成四维数据,数据输入必须为batch格式形式

    # 4.评估
    score = model(img)
    # print("score:", score.data)

    # 5.输出各类出现的概率
    probability = softmax(score, dim=1)
    probability = t.max(probability).item()
    # print(type(probability))
    print("probability is :", probability)

    # 6.输出相应类别
    value, predicted = t.max(score.data, 1)
    # print(value, predicted)
    print("类别:{}".format(classes[predicted.item()]))


def help_info():
    """
    输出帮助文档
    :return:
    """
    print("""
    usage:python file.py <function> [--args = value]
    <function> := train | test | help
    example:
    python {0} train 
    python {0} test
    python {0} predict
    python {0} read_csv
    python {0} help_info
    """.format(__file__))
    print("参数的设定：")

    # 获得opt类参数的源码
    from inspect import getsource
    print(getsource(opt.__class__))


def read_csv():
    import pandas as pd
    result = pd.read_csv("result.csv")
    print(result)


if __name__ == "__main__":
    fire.Fire()
