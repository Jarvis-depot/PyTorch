from pathlib import Path
import requests

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

import pickle
import gzip

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)
# 读进一个图像，也就是"5"这个数字
# 数据大小是50000 x 784
# 50000是样本数，784 = 28(长度) x 28(宽度) x 1(颜色通道)
# 此处是灰白图片，因此只有一个颜色通道

import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()

# torch.nn.functional中有很多功能，后续会常用
# 如果模型有可学习的参数，最好用nn.Module
# 其他情况用nn.functional相对简单一点
import torch.nn.functional as F

loss_func = F.cross_entropy


def model(xb):
    return xb.mm(weights) + bias


bs = 64
xb = x_train[0:bs]
yb = y_train[0:bs]
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bs = 64
bias = torch.zeros(10, requires_grad=True)

print(loss_func(model(xb), yb))

# 创建一个model来简化代码
# 必须继承nn.Model且在其构造函数中调用nn.Model的构造函数
# 无需写反向传播函数，nn.Model能够利用autograd自动实现反向传播
# Module中的可学习参数可以通过named_parameters()或者parameters()返回迭代器

from torch import nn


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x


net = Mnist_NN()
print(net)
for name, parameter in net.named_parameters():
    print(name, parameter, parameter.size())

# 使用TensorDataset和DataLoader来简化
# 帮助自动化构建数据集
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2)
    )


# 一般在训练模型时加上model.train()，这样会正常使用Batch Normalization和Dropout
# 测试的时候，一般选择model.eval()，这样就不会使用Batch Normalization和Dropout
# 训练函数
import numpy as np


# steps: 迭代次数
# model: 上方定义的model
# loss_func: function
# opt: 优化器
# train_dl: train
# valid_dl: 实际数据
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()  # 训练
        for xb, yb in train_dl:  # x batch和y batch
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()  # 测试，看一下验证集损失
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums) / np.sum(nums))
        print('当前step: ' + str(step), '验证集损失: ' + str(val_loss))


from torch import optim


def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(25, model, loss_func, opt, train_dl, valid_dl)