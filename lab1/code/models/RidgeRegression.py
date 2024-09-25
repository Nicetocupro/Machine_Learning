import numpy as np

'''
岭回归最早是用来处理特征数多于样本数的情况
现在也用于在估计中加入偏差
'''


class RidgeRegression:
    def __init__(self):
        self.trainSet = 0
        self.trainLabel = 0
        self.testSet = 0
        self.testLabel = 0
        self.lambdas = 0
        self.w = 0
        self.b = 0
        self.sqrLoss = 0  # 最小均方误差

    def train(self, X, y, lambdas=0.2):
        # 维度一定大于2
        if X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = X
        self.trainLabel = y  # 修改这里
        self.lambdas = lambdas
        n_samples, n_features = self.trainSet.shape

        # 合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))

        # 求w和b
        E_X = np.linalg.inv(np.dot(X2.T, X2) + self.lambdas * np.eye(X2.shape[1]))
        what = np.dot(E_X, X2.T @ y)  # 使用@符号进行矩阵乘法

        self.w = what[:-1]
        self.b = what[-1]

        # 计算平方损失
        self.sqrLoss = np.power((y - np.dot(X2, what).flatten()), 2).sum()
        return

    def predict(self, testSet, testLabel):
        self.testSet = testSet
        self.testLabel = testLabel
        y_predict = np.dot(self.testSet, self.w) + self.b
        MSE = np.power((y_predict - self.testLabel), 2).sum() / len(testLabel)  # 计算均方误差
        return MSE