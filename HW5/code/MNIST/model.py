import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 卷积层
        #三个参数分别代表：input channels 输入维度，output channels 输出维度，kernel size 卷积核大小
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 全连接层
        #in_features: input vector dimensions 输入样本的大小
        #out_features: output vector dimensions 输出样本的大小
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    #定义前向传播
    def forward(self, x):
        # 卷积 --> ReLu --> 池化
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        # '-1'表示自适应
        # x = (n * 16 * 4 * 4)  其中n表示input channels
        # x.size()[0]  即n
        x = x.view(x.size()[0], -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

