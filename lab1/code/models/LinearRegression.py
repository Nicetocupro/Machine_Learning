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

        print("X2 shape:", X2.shape)
        print("what shape:", what.shape)
        print("y shape:", y.shape)

        self.w = what[:-1]
        self.b = what[-1]
        self.sqrLoss = np.power((y - np.dot(X2, what)), 2).sum()
        return

    def __train_gradient(self, learning_rate, n_iters, minloss=1.0e-6):
        # 确保 trainLabel 是二维的
        if self.trainLabel.ndim == 1:
            self.trainLabel = np.expand_dims(self.trainLabel, axis=1)
        n_samples, n_features = self.trainSet.shape
        _, n_outputs = self.trainLabel.shape
        X = self.trainSet
        y = self.trainLabel
        n = 0
        w = np.ones((n_features, n_outputs))
        b = np.zeros((1, n_outputs))  # 初始化为0，并且形状调整为 (1, n_outputs)

        print("X shape:", X.shape)
        print("w shape:", w.shape)
        print("y shape:", y.shape)

        # 计算初始损失
        ypredict = np.dot(X, w) + b  # 直接相加，b会广播成 (n_samples, n_outputs)
        sqrLoss0 = np.power((y - ypredict), 2).sum()
        self.lossList.append(sqrLoss0)
        deltaLoss = np.inf

        while (n < n_iters) and (sqrLoss0 > minloss) and (abs(deltaLoss) > minloss):
            # 计算预测值
            ypredict = np.dot(X, w) + b

            # 计算梯度
            gradient_w = -np.dot(X.T, (y - ypredict)) / n_samples
            gradient_b = -np.sum(y - ypredict, axis=0, keepdims=True) / n_samples

            # 更新权重和偏置
            w = w - learning_rate * gradient_w
            b = b - learning_rate * gradient_b

            # 计算新的损失
            sqrLoss1 = np.power((y - np.dot(X, w) - b), 2).sum()
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

    def predict(self, testSet):
        self.testSet = testSet
        y_predict = np.dot(self.testSet, self.w) + self.b
        return y_predict
