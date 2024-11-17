import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, fit_intercept=True, verbose=False, tol=1e-4, early_stopping=False):
        self.learning_rate = learning_rate  # 学习率
        self.n_iter = n_iter  # 训练迭代次数
        self.fit_intercept = fit_intercept  # 是否包含截距
        self.verbose = verbose  # 是否打印训练过程
        self.tol = tol  # 收敛容忍度
        self.early_stopping = early_stopping  # 是否启用提前停止
        self.weights = None  # 权重参数
        self.bias = None  # 偏置参数

    def sigmoid(self, z):
        """ Sigmoid 激活函数 """
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        """ 为 X 添加一列 1 作为截距项 """
        if self.fit_intercept:
            intercept = np.ones((X.shape[0], 1))
            X = np.concatenate([intercept, X], axis=1)
        return X

    def _initialize_weights(self, n_features):
        """ 初始化权重和偏置 """
        self.weights = np.zeros(n_features)
        self.bias = 0

    def fit(self, X, y):
        """ 使用梯度下降法训练模型 """
        X = self._add_intercept(X)  # 添加截距项
        m, n = X.shape  # m: 样本数，n: 特征数
        self._initialize_weights(n)  # 初始化权重

        prev_loss = float('inf')  # 初始化上一个损失值

        for i in range(self.n_iter):
            # 前向传播
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)

            # 计算损失函数的梯度
            dw = (1 / m) * np.dot(X.T, (y_pred - y))  # 权重的梯度
            db = (1 / m) * np.sum(y_pred - y)  # 偏置的梯度

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 计算当前损失
            loss = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

            # 输出训练进度
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

            # 提前停止：如果损失的变化小于容忍度，终止训练
            if self.early_stopping and abs(prev_loss - loss) < self.tol:
                print(f"Early stopping at iteration {i}, Loss = {loss:.4f}")
                break
            prev_loss = loss

    def predict(self, X):
        """ 使用训练好的模型进行预测 """
        X = self._add_intercept(X)  # 添加截距项
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return np.round(y_pred)  # 输出预测类别 (0 或 1)

    def predict_proba(self, X):
        """ 返回预测的概率值 """
        X = self._add_intercept(X)  # 添加截距项
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
