import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    """Improved fully connected neural network for regression"""

    def __init__(self):
        self.model = None
        self.loss_list = []
        self.optimizer = None
        self.scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测 GPU
        self.build_model()

    def build_model(self):
        """Define the structure of the improved neural network"""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(26, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),

            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),

            torch.nn.Linear(32, 1)
        ).to(self.device)

    def train(self, train_x, train_y, learning_rate=0.001, n_iters=200, batch_size=64):
        # 检查 train_x 和 train_y 的样本数量是否一致
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of samples.")

        # 标准化输入特征
        train_x_flat = train_x.reshape(train_x.shape[0], -1)
        train_x_scaled = self.scaler.fit_transform(train_x_flat)

        # 标准化目标值
        train_y_reshaped = train_y.reshape(train_y.shape[0], -1)  # 将 train_y 进行 reshape，使其与模型输出匹配
        train_y_scaled = self.y_scaler.fit_transform(train_y_reshaped)

        # 将数据转换为张量并移动到设备
        x_tensor = torch.tensor(train_x_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(train_y_scaled, dtype=torch.float32).to(self.device)
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # 初始化优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = torch.nn.MSELoss()

        self.model.train()  # 设置模型为训练模式

        for epoch in range(n_iters):
            epoch_loss = 0.0
            for x_batch, y_batch in dataloader:
                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)
            epoch_loss /= len(train_x)
            self.loss_list.append(epoch_loss)
            print(f"Epoch {epoch + 1}/{n_iters}, Loss: {epoch_loss:.6f}")

    def predict(self, x):
        """Predict the output for given input"""
        x_flat = x.reshape(x.shape[0], -1)
        x_scaled = self.scaler.transform(x_flat)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(self.device)

        self.model.eval()  # 设置模型为评估模式
        with torch.no_grad():
            output = self.model(x_tensor)
        output = output.cpu().numpy()
        # 逆标准化
        output_original = self.y_scaler.inverse_transform(output)
        return output_original.reshape(-1)
