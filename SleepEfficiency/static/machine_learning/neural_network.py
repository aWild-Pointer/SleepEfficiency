from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from static.machine_learning.draw_plot import plot_loss_curve, plot_results


# 定义神经网络模型
class NeuralNetworkModel(nn.Module):
    # input_dim控制输入层的神经元数量
    def __init__(self, input_dim):
        super(NeuralNetworkModel, self).__init__()
        # 第一线性层 64个
        self.fc1 = nn.Linear(input_dim, 64)
        # 第二线性层 64-32
        self.fc2 = nn.Linear(64, 32)
        # 第三线性层 32-1
        self.fc3 = nn.Linear(32, 1)
        # ReLU 非线性激活函数 提高能力
        self.relu = nn.ReLU()

    # 前向传播
    def forward(self, x):
        # 输出第一线性层的激活值
        x = self.relu(self.fc1(x))
        # 输出第二线性层的激活值
        x = self.relu(self.fc2(x))
        # 第三层不用输出，已经是最终值
        x = self.fc3(x)
        return x


# 神经网络回归算法
def neural_network_regression(X_train_nn, X_test_nn, y_train_nn, y_test_nn):
    # 创建模型实例 维度为 1
    input_dim = X_train_nn.shape[1]
    model_nn = NeuralNetworkModel(input_dim)

    # 定义损失函数均方误差，预测与实际之间的平均平方差
    criterion = nn.MSELoss()
    # 优化器 Adam
    optimizer = optim.Adam(model_nn.parameters(), lr=0.01)

    # 转换为 PyTorch 张量
    X_train_tensor_nn = torch.tensor(X_train_nn, dtype=torch.float32)
    # 转换为一个二维张量，使其形状变为 [样本数量, 1]
    y_train_tensor_nn = torch.tensor(y_train_nn, dtype=torch.float32).view(-1, 1)

    # 设置数据集
    train_dataset_nn = TensorDataset(X_train_tensor_nn, y_train_tensor_nn)
    # DataLoader创建训练加载器 batch_size每个批次包含32个样本 shuffle打乱数据以提高模型的泛化能力
    train_loader_nn = DataLoader(train_dataset_nn, batch_size=32, shuffle=True)

    # 训练模型
    num_epochs = 500  # 迭代200次
    epoch_loss = []  # 存储每个 epoch 的损失
    for epoch in range(num_epochs):
        model_nn.train()
        total_loss = 0
        for inputs, targets in train_loader_nn:
            # 前向传播
            outputs = model_nn(inputs)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader_nn)
        epoch_loss.append(avg_loss)

        # if (epoch + 1) % 10 == 0:
        #     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

    plot_loss_curve(epoch_loss, '神经网络回归 训练损失变化曲线')

    # 评估模型
    model_nn.eval()
    y_test_pred_nn = model_nn(torch.tensor(X_test_nn, dtype=torch.float32)).detach().numpy()

    mse = mean_squared_error(y_test_nn, y_test_pred_nn)
    r2 = r2_score(y_test_nn, y_test_pred_nn)
    # print(f"均方误差 (MSE): {mse}")
    # print(f"决定系数 (R²): {r2}")

    plot_results(y_test_nn, y_test_pred_nn, '神经网络回归 真实值与预测值对比')

    return mse, r2
