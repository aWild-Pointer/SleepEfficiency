# 定义多元线性回归模型
# nn.Module可以自动梯度计算、模型保存
import torch
from torch import nn, optim
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from static.machine_learning.draw_plot import plot_loss_curve, plot_results


class LinearRegressionModel(nn.Module):
    # input_dim 输入特征的维度
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        # 定义线性层，输入维度为 input_dim，输出维度为 1
        self.linear = nn.Linear(input_dim, 1)
        # 定义ReLU激活函数缓解过拟合问题， 能够在x>0时保持梯度不衰减，从而缓解梯度消失问题
        self.relu = nn.ReLU()

    # 定义前向传播逻辑
    def forward(self, x):
        # 线性层对输入数据进行线性变换后传递给ReLU激活函数，得到非线性变换后的输出
        return self.relu(self.linear(x))


# 多元线性回归算法
def linear_regression(X_train_lr, X_test_lr, y_train_lr, y_test_lr):
    # 创建模型实例 维度为 1
    input_dim = X_train_lr.shape[1]
    model_lr = LinearRegressionModel(input_dim)

    # 定义损失函数均方误差，预测与实际之间的平均平方差
    criterion = nn.MSELoss()
    # 优化器 SGD=随机梯度下降 lr=学习率 momentum=动量加速，跳过局部最小值
    optimizer = optim.SGD(model_lr.parameters(), lr=0.01, momentum=0.9)
    # 学习率指数衰减 gamma=衰减率
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    X_train_tensor_lr = torch.tensor(X_train_lr, dtype=torch.float32)
    y_train_tensor_lr = torch.tensor(y_train_lr, dtype=torch.float32).view(-1, 1)

    train_dataset_lr = TensorDataset(X_train_tensor_lr, y_train_tensor_lr)
    train_loader_lr = DataLoader(train_dataset_lr, batch_size=32, shuffle=True)

    # 训练模型
    num_epochs = 500  # 迭代100次
    epoch_loss = []  # 存储每个 epoch 的损失
    for epoch in range(num_epochs):
        model_lr.train()
        total_loss = 0
        for inputs, targets in train_loader_lr:
            # 前向传播
            outputs = model_lr(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_lr)
        epoch_loss.append(avg_loss)

        # 更新学习率
        scheduler.step()

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    plot_loss_curve(epoch_loss, '多元线性回归 训练损失变化曲线')

    # 评估模型
    model_lr.eval()
    y_test_pred = model_lr(torch.tensor(X_test_lr, dtype=torch.float32)).detach().numpy()

    mse = mean_squared_error(y_test_lr, y_test_pred)
    r2 = r2_score(y_test_lr, y_test_pred)
    # print(f"均方误差 (MSE): {mse}")
    # print(f"决定系数 (R²): {r2}")

    plot_results(y_test_lr, y_test_pred, '多元线性回归 真实值与预测值对比')

    return mse, r2

