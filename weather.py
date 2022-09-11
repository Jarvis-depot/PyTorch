import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore")
# %matplotlib inline

# 1. 解析csv数据
# 数据表中参数解释如下：
# temp_2: 前天的最高温度值
# temp_1: 昨天的最高温度值
# average: 历史中，每年的这一天平均最高温度值
# actual: 标签值，记录当前的真实最高温度
# friend: 这个参数没实际意义，是你朋友猜测的可能值
features = pd.read_csv("temps.csv")
## print(features.head())
## print('数据维度:', features.shape)

# 2. 转换下数据格式
import datetime

years = features['year']
months = features['month']
days = features['day']

dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
## print(dates[:5])

# 3. 准备画图
# 3.1 指定默认风格
plt.style.use('fivethirtyeight')
# 3.2 设置布局
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
fig.autofmt_xdate(rotation = 45)
# 3.3 标签值
ax1.plot(dates, features['actual'])
ax1.set_xlabel(''); ax1.set_ylabel('Temperature'); ax1.set_title('Max Temp')
# 3.4 昨天
ax2.plot(dates, features['temp_1'])
ax2.set_xlabel(''); ax2.set_ylabel('Temperature'); ax2.set_title('Previous Max Temp')
# 3.5 前天
ax3.plot(dates, features['temp_2'])
ax3.set_xlabel(''); ax3.set_ylabel('Temperature'); ax3.set_title('Two Days Prior Max Temp')
# 3.6 我的逗比朋友
ax4.plot(dates, features['friend'])
ax4.set_xlabel(''); ax4.set_ylabel('Temperature'); ax4.set_title('Friend Estimate')
plt.tight_layout(pad=2)
## plt.show()
# 3.7 读热编码(将csv中的Week参数从string转换为不同表头对应的bool)
features = pd.get_dummies(features)
features.head(5)
## print(features)

# 4. 标签
labels = np.array(features['actual'])
# 4.1 在特征中去掉标签
features = features.drop('actual', axis=1)
# 4.2 保存下名字
features_list = list(features.columns)
# 4.3 转换成合适的格式
features = np.array(features)
## print(features.shape)

# 5. 构建网络模型
from sklearn import preprocessing
input_features = preprocessing.StandardScaler().fit_transform(features)
x = torch.tensor(input_features, dtype = float)
y = torch.tensor(labels, dtype = float)
# 5.1 权重参数初始化
# [5,12] x [12, 128]
# 取128个特征
weights  = torch.randn((14,128), dtype = float, requires_grad = True)
biases   = torch.randn(128, dtype = float, requires_grad = True)
weights2 = torch.randn((128,1), dtype = float, requires_grad = True)
biases2  = torch.randn(1, dtype = float, requires_grad = True)

learning_rate = 0.001
losses = []

for i in range(1000):
  # 计算隐层
  hidden = x.mm(weights) + biases
  # 加入激活函数
  hidden = torch.relu(hidden)
  # 预测结果
  predictions = hidden.mm(weights2) + biases2
  # 通计算损失
  loss = torch.mean((predictions - y) ** 2)
  losses.append(loss.data.numpy())

  # 打印损失值
  ## if i % 100 == 0:
  ##   print('loss: ', loss)
  # 反向传播计算
  loss.backward()

  # 更新参数
  weights.data.add_(- learning_rate * weights.grad.data)
  biases.data.add_(- learning_rate * biases.grad.data)
  weights2.data.add_(- learning_rate * weights2.grad.data)
  biases2.data.add_(- learning_rate * biases2.grad.data)

  # 每次迭代都记得清空
  weights.grad.data.zero_()
  biases.grad.data.zero_()
  weights2.grad.data.zero_()
  biases2.grad.data.zero_()
print(predictions.shape)

# ====================
# 一个更简单的构建网络模型
# ====================
input_size = input_features.shape[1]
hidden_size = 128
output_size = 1
batch_size = 16
my_nn = torch.nn.Sequential(
  torch.nn.Linear(input_size, hidden_size),
  torch.nn.Sigmoid(),
  torch.nn.Linear(hidden_size, output_size),
)
cost = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(my_nn.parameters(), lr=0.001)
# 训练网络
looses = []
for i in range(1000):
  batch_loss = []
  # 用MINI-Batch的方法来进行训练
  for start in range(0, len(input_features), batch_size):
    end = start + batch_size if start + batch_size < len(input_features) else len(input_features)
    xx = torch.tensor(input_features[start:end], dtype=torch.float, requires_grad=True)
    yy = torch.tensor(labels[start:end], dtype=torch.float, requires_grad=True)
    prediction = my_nn(xx)
    loss = cost(prediction, yy)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    batch_loss.append(loss.data.numpy())
  # 打印损失值
  if i % 100 == 0:
    losses.append(np.mean(batch_loss))
    print(i, np.mean(batch_loss))
# 预测训练结果
x = torch.tensor(input_features, dtype=torch.float)
predict = my_nn(x).data.numpy()
# 转换日期格式
dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in dates]
# 创建一个表格来存日期和其他对应的标签数值
true_data = pd.DataFrame(data={'date': dates, 'actual': labels})
# 同理，再创建一个来存日期和其它对应的模型预测值
months = features[:, features_list.index('month')]
days = features[:, features_list.index('day')]
years = features[:, features_list.index('year')]
test_dates = [str(int(year)) + '-' + str(int(month)) + '-' + str(int(day)) for year, month, day in zip(years, months, days)]
test_dates = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in test_dates]
predictions_data = pd.DataFrame(data={'date': test_dates, 'prediction': predict.reshape(-1)})
# 真实值
plt.plot(true_data['date'], true_data['actual'], 'b-', label='actual')
# 预测值
plt.plot(predictions_data['date'], predictions_data['prediction'], 'ro', label='prediction')
plt.xticks(rotation='60');
plt.legend()
# 图名
plt.xlabel('Date'); plt.ylabel('Maximum Temperature (F)'); plt.title('Actual and Predicted Values');
plt.show()
