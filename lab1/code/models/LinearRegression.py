import numpy as np


class LinearRegression:
    """ simple linear regression & multivariate linear regression """

    def __init__(self):
        self.w = 0  # 斜率
        self.b = 0  # 截距
        self.sqrLoss = 0  # 最小均方误差
        self.trainSet = 0  # 训练集特征
        self.trainLabel = 0  # 训练集标签
        self.testSet = 0  # 测试集特征
        self.testLabel = 0  # 测试集标签
        self.learning_rate = None  # 学习率
        self.n_iters = None  # 实际迭代次数
        self.lossList = []  # 梯度下降每轮迭代的误差列表

    def train(self, train_X, train_y, method, learning_rate=0.1, n_iters=1000):
        # 维度一定大于2
        if train_X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = train_X
        self.trainLabel = train_y

        if method.lower() == "formula":
            self.__train_formula()
        elif method.lower() == "matrix":
            self.__train_matrix()
        elif method.lower() == "gradient":
            self.__train_gradient(learning_rate, n_iters)
        else:
            raise ValueError("method value not found!")
        return

    # 公式求解法（仅适用于一元线性回归）
    def __train_formula(self):
        if self.trainSet.ndim == 2:
            n_samples, n_features = self.trainSet.shape
            X = self.trainSet.flatten()
            y = self.trainLabel
            Xmean = np.mean(X)
            ymean = np.mean(y)
            # 求w
            self.w = (np.dot(X, y) - n_samples * Xmean * ymean) / (np.power(X, 2).sum() - n_samples * Xmean ** 2)
            # 求b
            self.b = ymean - self.w * Xmean
            # 求误差
            self.sqrLoss = np.power((y - np.dot(X, self.w) - self.b), 2).sum()
        else:
            raise ValueError("公式求解法（仅适用于一元线性回归）")
        return

    # 矩阵求解法
    def __train_matrix(self):
        n_samples, n_features = self.trainSet.shape
        X = self.trainSet
        y = self.trainLabel
        # 合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))
        # 求w和b
        EX = np.linalg.inv(np.dot(X2.T, X2))
        what = np.dot(np.dot(EX, X2.T), y)
        self.w = what[:-1]
        self.b = what[-1]
        self.sqrLoss = np.power((y - np.dot(X2, what).flatten()), 2).sum()
        return

    def __train_gradient(self, learning_rate, n_iters, minloss=1.0e-6):
        n_samples, n_features = self.trainSet.shape
        X = self.trainSet
        y = self.trainLabel
        # 初始化迭代次数为0，初始化w0, b0都为1，初始化误差平方和以及迭代误差之差
        n = 0
        w = np.ones(n_features)
        b = 1
        sqrLoss0 = np.power((y - np.dot(X, w).flatten() - b), 2).sum()
        self.lossList.append(sqrLoss0)
        deltaLoss = np.inf

        while (n < n_iters) and (sqrLoss0 > minloss) and (abs(deltaLoss) > minloss):
            # 求w和b的梯度
            ypredict = np.dot(X, w) + b
            gradient_w = -np.sum((y - ypredict).reshape(-1, 1) * X, axis=0) / n_samples
            gradient_b = -np.sum(y - ypredict) / n_samples

            # 更新w和b的值
            w = w - learning_rate * gradient_w
            b = b - learning_rate * gradient_b

            # 求更新后的误差和更新前后的误差之差
            sqrLoss1 = np.power((y - np.dot(X, w).flatten() - b), 2).sum()
            deltaLoss = sqrLoss0 - sqrLoss1
            sqrLoss0 = sqrLoss1
            self.lossList.append(sqrLoss0)
            n += 1
            print("第{}次迭代，损失平方和为{}，损失前后差为{}".format(n, sqrLoss0, deltaLoss))
        self.w = w
        self.b = b
        self.sqrLoss = sqrLoss0
        self.learning_rate = learning_rate
        self.n_iters = n + 1
        return

    def predict(self, testSet, testLabel):
        self.testSet = testSet
        self.testLabel = testLabel
        y_predict = np.dot(self.testSet, self.w) + self.b
        MSE = np.power((y_predict - self.testLabel), 2).sum() / len(testLabel)  # 计算均方误差
        return MSE
