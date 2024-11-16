import numpy as np

class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # 输入到隐藏层的权重
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层的权重
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # 隐藏层到输出层的权重
        self.bh = np.zeros((hidden_size, 1))  # 隐藏层偏置
        self.by = np.zeros((output_size, 1))  # 输出层偏置

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_size, 1))  # 初始化前一时刻的隐藏状态
        hs, ys = {}, {}  # 存储隐藏状态和输出
        hs[-1] = h_prev
        for t in range(len(inputs)):
            x = np.array(inputs[t]).reshape(-1, 1)  # 将输入转换为列向量
            hs[t] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hs[t-1]) + self.bh)  # 计算隐藏状态
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # 计算输出
        return ys, hs

    def loss(self, outputs, targets):
        loss = 0
        for t in range(len(outputs)):
            loss += np.sum((outputs[t] - targets[t]) ** 2)  # 计算均方误差
        return loss / len(outputs)

    def backward(self, inputs, hs, outputs, targets, learning_rate=1e-3):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])  # 初始化下一时刻的隐藏状态梯度

        for t in reversed(range(len(inputs))):
            dy = outputs[t] - targets[t]  # 计算输出误差
            dWhy += np.dot(dy, hs[t].T)  # 计算输出层权重梯度
            dby += dy  # 计算输出层偏置梯度
            dh = np.dot(self.Why.T, dy) + dh_next  # 计算隐藏状态误差
            dh_raw = (1 - hs[t] * hs[t]) * dh  # 计算隐藏状态梯度
            dbh += dh_raw  # 计算隐藏层偏置梯度
            dWxh += np.dot(dh_raw, inputs[t].reshape(1, -1))  # 计算输入到隐藏层的权重梯度
            dWhh += np.dot(dh_raw, hs[t-1].T)  # 计算隐藏层到隐藏层的权重梯度
            dh_next = np.dot(self.Whh.T, dh_raw)  # 更新下一时刻的隐藏状态梯度

        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # 梯度裁剪

        self.Wxh -= learning_rate * dWxh  # 更新输入到隐藏层的权重
        self.Whh -= learning_rate * dWhh  # 更新隐藏层到隐藏层的权重
        self.Why -= learning_rate * dWhy  # 更新隐藏层到输出层的权重
        self.bh -= learning_rate * dbh  # 更新隐藏层偏置
        self.by -= learning_rate * dby  # 更新输出层偏置

    def train(self, inputs, targets, epochs=100, learning_rate=1e-3):
        for epoch in range(epochs):
            outputs, hs = self.forward(inputs)  # 前向传播
            loss = self.loss(outputs, targets)  # 计算损失
            self.backward(inputs, hs, outputs, targets, learning_rate)  # 反向传播
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')  # 每10个epoch打印一次损失