"""
target1: 实验sigmoid和Relu激活函数的本质区别
target2:  Dropout的使用
target3: 重叠池化
"""
from torch import nn
from tensorboardX import SummaryWriter
import torch as t


class AlexNet(nn.Module):
    """
    定义网络
    """

    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.model = "alexnet"
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 重叠池化
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        """
        前向传播
        :param x:输入数据
        :return: 最终结果
        """
        x = self.feature(x)
        # 卷积到全连接的Flat
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = AlexNet(num_classes=2)
    input = t.Tensor(64, 3, 224, 224)

    with SummaryWriter() as write:
        write.add_graph(model, (input,))
