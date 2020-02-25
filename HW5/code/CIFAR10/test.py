# 导入必要的库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transform
import time

from model import VGG
import torchvision
import torchvision.transforms as transforms

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import VGG

# 超参数
BATCH_SIZE = 100
# 损失函数
criterion = nn.CrossEntropyLoss()
# 可以在CPU或者GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10的输入图片各channel的均值和标准差
mean = [x / 255 for x in [125.3, 23.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

def get_dataset():
    testset = dsets.CIFAR10(root='./data/',
                            train=False,
                            download=True,
                            transform=transform.Compose([
                                transform.ToTensor(),
                                transform.Normalize(mean, std)
                            ]))

    testloader = DataLoader(testset,
                            batch_size=BATCH_SIZE,
                            num_workers=0)

    return len(testset), testloader


def main():
    # 图片类别
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 加载模型
    for i in range(50):
        model = torch.load('net' + str(i+1) + '.pkl')
        if torch.cuda.is_available():
            # 使用GPU
            model.cuda()

        # get dataloader
        data_len, dataloader = get_dataset()

        # 预测正确的个数
        correct_num = 0
        for k, data in enumerate(dataloader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            # 取最大值为预测结果
            _, predicted = torch.max(outputs, 1)

            for j in range(len(predicted)):
                predicted_num = predicted[j].item()
                label_num = labels[j].item()
                # 预测值与标签值进行比较
                if predicted_num == label_num:
                    correct_num += 1

        # 计算预测准确率
        correct_rate = correct_num / data_len
        print(str(i+1), 'correct rate is {:.3f}%'.format(correct_rate * 100))


if __name__ == "__main__":
    main()


