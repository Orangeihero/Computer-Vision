# 导入必要的库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transform
from torch.autograd import Variable

from model import VGG

# 超参数
BATCH_SIZE = 100
# 可以在CPU或者GPU上运行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR10的输入图片各channel的均值和标准差
mean = [x / 255 for x in [125.3, 23.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

if __name__ == '__main__':
    # 数据增强-->训练集
    trainset = dsets.CIFAR10(root='./data/',
                              train=True,
                              download=True,
                              transform=transform.Compose([
                                  transform.RandomHorizontalFlip(),
                                  transform.RandomCrop(32, padding=4),
                                  transform.ToTensor(),
                                  transform.Normalize(mean, std)
                              ]))
    trainloader = DataLoader(trainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    vgg19 = VGG().to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(vgg19.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    print('开始训练VGG19……')
    # 图片类别
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 加载模型
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # 得到输入数据
            inputs, labels = data
            # 使用gpu
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = vgg19(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += loss.item()

            # 每2000个batch打印一次训练状态
            if i % 2000 == 1999:
                print(
                    '[{}/{}][{}/{}] loss: {:.3f}'.format(epoch + 1, 50, (i + 1) * 4, len(trainset),
                                                         running_loss / 2000))
                running_loss = 0.0

        # 保存参数文件
        torch.save(vgg19, 'net' + str(epoch + 1) + '.pkl')
        print('net_{}.pkl saved'.format(epoch + 1))

    print('Finished Training')
