import numpy as np

class RidgeRegression:
    def __init__(self):
        self.trainSet = 0
        self.trainLabel = 0
        self.testSet = 0
        self.testLabel = 0
        self.lambdas = 0
        self.w = None
        self.b = None
        self.sqrLoss = 0  # 最小均方误差

    def train(self, X, y, lambdas=0.2):
        # 维度一定大于2
        if X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = X
        self.trainLabel = y
        self.lambdas = lambdas
        n_samples, n_features = self.trainSet.shape

        # 合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))

        # 初始化 w 和 b 的容器
        n_outputs = y.shape[1]
        self.w = np.zeros((n_features, n_outputs))  # 多输出维度
        self.b = np.zeros(n_outputs)  # 每个输出一个偏置

        # 逐个输出目标计算岭回归解
        for i in range(n_outputs):
            E_X = np.linalg.inv(np.dot(X2.T, X2) + self.lambdas * np.eye(X2.shape[1]))
            what = np.dot(E_X, X2.T @ y[:, i])  # 针对第 i 个输出进行岭回归
            self.w[:, i] = what[:-1]  # 提取权重
            self.b[i] = what[-1]  # 提取偏置

        # 计算平方损失
        y_predict = np.dot(X2, np.vstack((self.w, self.b)))  # 矩阵乘法预测多输出
        self.sqrLoss = np.power((y - y_predict), 2).sum()
        return

    def predict(self, testSet):
        # 添加偏置项
        n_samples = testSet.shape[0]
        X2 = np.hstack((testSet, np.ones((n_samples, 1))))
        y_predict = np.dot(X2, np.vstack((self.w, self.b)))
        return y_predict
