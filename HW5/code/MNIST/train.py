import torchvision
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from model import Net

if __name__ == '__main__':
    # 定义对数据的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为Tensor，并归一化至[0, 1]
    ])

    # 训练集
    trainset = torchvision.datasets.MNIST(
        root='data/',
        train=True,
        download=True,
        transform=transform
    )
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=4,
        shuffle=True
    )

    # MNIST数据集中的十种标签
    classes = ('0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9')

    # 创建网络模型
    net = Net()

    # 使用GPU
    if torch.cuda.is_available():
        net.cuda()

    # 定义损失函数和优化器,使用分类交叉熵Cross-Entropy 作损失函数，动量SGD做优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

    # 训练网络
    for epoch in range(50):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            #得到输入数据
            inputs, labels = data
            #使用gpu
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)

            # 梯度清0
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            running_loss += loss.item()

            # 每2000个batch打印一次训练状态
            if i % 2000 == 1999:
                print(
                    '[{}/{}][{}/{}] loss: {:.3f}'.format(epoch + 1, 50, (i + 1) * 4, len(trainset), running_loss / 2000))
                running_loss = 0.0

        # 保存参数文件
        torch.save(net, 'net'+str(epoch+1)+'.pkl')
        print('net_{}.pkl saved'.format(epoch + 1))

    print('Finished Training')

