import torch
import numpy as np
import torch.nn as nn

# 框架最厉害的一件事就是帮我们把反向传播全部计算好了
# ============
#   demo 1  #
# ============
x = torch.ones(3, requires_grad=True)
y = 2 * x ** 2  # y = 2*x^2
gradients = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float)
y.backward(gradients)
## print(y)
## print(x.grad)

# ============
#    main 1
# ============
# 需要求导的，可以手动定义，其中require_grad=True表示该变量为自变量，可被求导
x = torch.randn(3, 4, requires_grad=True)
b = torch.randn(3, 4, requires_grad=True)
t = x + b
y = t.sum()
## print("[y] is: ", y)
# backward()是求导函数
## print(y.backward())
## print(b.grad)

# 构建一个线性回归模型
x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
## print(x_train.shape)
y_values = [2 * i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


## print(y_train.shape)
# 线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
## print(model)

# 指定好参数和损失函数
## 深度学习的优化算法，可以理解为是梯度下降的算法
## epochs：指训练过程中，数据被"轮"的次数
epochs = 1000
## 学习率：决定目标函数是否能收敛到局部最小值，以及何时能收敛到最小值
learning_rate = 0.01
## SGD(随即梯度下降法)：W <- W-η*∂L/∂W
## 其中W为需要更新的权重参数，损失函数关于W的梯度记为∂L/∂W
## η表示学习率，实际上会取0.01, 0.001这样事先决定好的值
## 式子中的"<-"表示用右边的值更新左边的值
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# 训练模型
for epoch in range(epochs):
    epoch += 1
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    optimizer.zero_grad()
    # 前向传播（将实际的x传进来）
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 更新权重参数(会基于学习率和梯度值自动完成更新)
    optimizer.step()
    if epoch % 50 == 0:
        print("epoch {}, loss {}".format(epoch, loss.item()))

# 测试模型预测结果
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
## print(predicted)
# 模型的保存与读取
torch.save(model.state_dict(), 'model.pkl')
model.load_state_dict(torch.load('model.pkl'))
# ============
#    main 2
# ============



