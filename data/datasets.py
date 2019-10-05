# 自定义数据集
# 用来表示数据需要处
import os
import torchvision.transforms as T
from PIL import Image
from torch.utils import data


class DogCat(data.Dataset):

    def __init__(self, root, transform=None, train=True, test=False):
        """
        初始化数据集
        :param root: 根目录
        :param transform: 数据格式转换
        :param train: 什么类型的数据
        """
        super(DogCat, self).__init__()
        # 1.读取数据集
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        imgs = imgs[:2000]
        self.test = test

        # 2.数据集整理
        if self.test:  # TODO:split(为一个字符型数据)
            imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2].split("/")[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split(".")[-2]))

        # 3.数据集的划分
        num_imgs = len(imgs)
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * num_imgs)]
        else:
            self.imgs = imgs[int(0.7 * num_imgs):]

        # 4.数据处理

        if transform is None:

            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(226),
                    T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, item):
        """得到一条数据"""
        # 1.得到数据
        img_path = self.imgs[item]
        # print(img_path)

        # 2.提取label
        if self.test:
            label = int(self.imgs[item].split(".")[-2].split("/")[-1])

        else:
            label = 1 if "dog" in img_path.split("/")[-1] else 0

        data = Image.open(img_path)
        # data.show(label)
        data = self.transforms(data)
        # data = self.transforms(data)

        return data, label

    def __len__(self):
        """得到总体数据量数"""
        return len(self.imgs)


if __name__ == "__main__":
    data = DogCat(root="./test", train=False, test=True)
    print("数据量：", len(data))

    data1, label = data[100]  # obj[2]取出一张照片
    print(label)
