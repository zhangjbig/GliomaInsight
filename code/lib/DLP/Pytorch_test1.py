import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# 定义超参数
batch_size = 64
learning_rate = 1e-2
num_epochs = 5 # 训练次数
# 判断GPU是否可用
use_gpu = torch.cuda.is_available()

# 下载训练集 MNIST 手写数字训练集
# 数据是datasets类型的
train_dataset = datasets.FashionMNIST(
    root='../datasets', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.FashionMNIST(
    root='../datasets', train=False, transform=transforms.ToTensor())
#　将数据处理成 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 选择打乱数据
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 选择不打乱数据

"""
# 基本的网络构建类模板
class net_name(nn.Module):
    def __init__(self):
        super(net_name, self).__init__()
        # 可以添加各种网络层
        self.conv1 = nn.Conv2d(3, 10, 3)
        # 具体每种层的参数可以去查看文档

    def forward(self, x):
        # 定义向前传播
        out = self.conv1(x)
        return out
"""
# 定义简单的前馈神经网络
class neuralNetwork(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(neuralNetwork, self).__init__() # super() 函数是用于调用父类(超类)的一个方法
# Sequential()表示将一个有序的模块写在一起，也就相当于将神经网络的层按顺序放在一起，这样可以方便结构显示
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True)) # 表示使用ReLU激活函数
        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True))
        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, out_dim),
            nn.ReLU(True))

# 定义向前传播
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# 图片大小是28*28，中间定义了两个隐藏层大小分别为300和100，最后输出层为10，10分类问题
model = neuralNetwork(28 * 28, 300, 100, 10)
if use_gpu:
    model = model.cuda() # 现在可以在GPU上跑代码了

criterion = nn.CrossEntropyLoss() # 定义损失函数类型，使用交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # 定义优化器，使用随机梯度下降

# 开始模型训练
for epoch in range(num_epochs):
    print('*' * 10)
    print(f'epoch {epoch + 1}')
    running_loss = 0.0  # 初始值
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):  # 枚举函数enumerate返回下标和值
        img, label = data
        img = img.view(img.size(0), -1)  # 将图片展开为28*28
        # 使用GPU？
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        # 向前传播
        out = model(img)  # 前向传播
        loss = criterion(out, label)  # 计算loss
        running_loss += loss.item()  # loss求和
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().mean()
        # 向后传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 后向传播
        optimizer.step()  # 更新参数

        if i % 300 == 0:
            print(f'[{epoch + 1}/{num_epochs}] Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
    print(f'Finish {epoch + 1} epoch, Loss: {running_loss / i:.6f}, Acc: {running_acc / i:.6f}')
"""
    ## 模型测试
    model.eval() # 让模型变成测试模式
    eval_loss = 0.
    eval_acc = 0.
    for data in test_loader:
        img, label = data
        img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        with torch.no_grad():
            out = model(img)
            loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = torch.max(out, 1)
        eval_acc += (pred == label).float().mean()
    print(f'Test Loss: {eval_loss/len(test_loader):.6f}, Acc: {eval_acc/len(test_loader):.6f}\n')
"""







