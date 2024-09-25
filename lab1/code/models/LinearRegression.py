# import numpy as np

# class LinearRegression:
#     """ simple linear regression & multivariate linear regression """

#     def __init__(self):
#         self.w = 0  # 斜率
#         self.b = 0  # 截距
#         self.sqrLoss = 0  # 最小均方误差
#         self.trainSet = 0  # 训练集特征
#         self.trainLabel = 0  # 训练集标签
#         self.testSet = 0  # 测试集特征
#         self.testLabel = 0  # 测试集标签
#         self.learning_rate = None  # 学习率
#         self.n_iters = None  # 实际迭代次数
#         self.lossList = []  # 梯度下降每轮迭代的误差列表

#     def train(self, train_X, train_y, method, learning_rate=0.1, n_iters=1000):
#         # 维度一定大于2
#         if train_X.ndim < 2:
#             raise ValueError("X must be 2D array-like!")
#         self.trainSet = train_X
#         self.trainLabel = train_y

#         if method.lower() == "formula":
#             self.__train_formula()
#         elif method.lower() == "matrix":
#             self.__train_matrix()
#         elif method.lower() == "gradient":
#             self.__train_gradient(learning_rate, n_iters)
#         else:
#             raise ValueError("method value not found!")
#         return

#     # 公式求解法（仅适用于一元线性回归）
#     def __train_formula(self):
#         if self.trainSet.ndim == 2:
#             n_samples, n_features = self.trainSet.shape
#             X = self.trainSet.flatten()
#             y = self.trainLabel
#             Xmean = np.mean(X)
#             ymean = np.mean(y)
#             # 求w
#             self.w = (np.dot(X, y) - n_samples * Xmean * ymean) / (np.power(X, 2).sum() - n_samples * Xmean ** 2)
#             # 求b
#             self.b = ymean - self.w * Xmean
#             # 求误差
#             self.sqrLoss = np.power((y - np.dot(X, self.w) - self.b), 2).sum()
#         else:
#             raise ValueError("公式求解法（仅适用于一元线性回归）")
#         return

#     # 矩阵求解法
#     def __train_matrix(self):
#         n_samples, n_features = self.trainSet.shape
#         X = self.trainSet
#         y = self.trainLabel.flatten()  # 确保 y 是一个一维数组
#         # 合并 w 和 b，在 X 尾部添加一列全是 1 的特征
#         X2 = np.hstack((X, np.ones((n_samples, 1))))
        
#         # 打印调试信息
#         print(f"Shape of X2: {X2.shape}")
#         print(f"Shape of y: {y.shape}")
        
#         # 求 w 和 b
#         EX = np.linalg.inv(np.dot(X2.T, X2))
#         what = np.dot(np.dot(EX, X2.T), y)
#         self.w = what[:-1]
#         self.b = what[-1]
#         self.sqrLoss = np.power((y - np.dot(X2, what)), 2).sum()  # 确保 np.dot(X2, what) 是一个一维数组
#         return

#     def __train_gradient(self, learning_rate, n_iters, minloss=1.0e-6):
#         n_samples, n_features = self.trainSet.shape
#         X = self.trainSet
#         y = self.trainLabel
#         # 初始化迭代次数为0，初始化w0, b0都为1，初始化误差平方和以及迭代误差之差
#         n = 0
#         w = np.ones(n_features)
#         b = 1
#         sqrLoss0 = np.power((y - np.dot(X, w).flatten() - b), 2).sum()
#         self.lossList.append(sqrLoss0)
#         deltaLoss = np.inf

#         while (n < n_iters) and (sqrLoss0 > minloss) and (abs(deltaLoss) > minloss):
#             # 求w和b的梯度
#             ypredict = np.dot(X, w) + b
#             gradient_w = -np.sum((y - ypredict).reshape(-1, 1) * X, axis=0) / n_samples
#             gradient_b = -np.sum(y - ypredict) / n_samples

#             # 更新w和b的值
#             w = w - learning_rate * gradient_w
#             b = b - learning_rate * gradient_b

#             # 求更新后的误差和更新前后的误差之差
#             sqrLoss1 = np.power((y - np.dot(X, w).flatten() - b), 2).sum()
#             deltaLoss = sqrLoss0 - sqrLoss1
#             sqrLoss0 = sqrLoss1
#             self.lossList.append(sqrLoss0)
#             n += 1
#             print("第{}次迭代，损失平方和为{}，损失前后差为{}".format(n, sqrLoss0, deltaLoss))
#         self.w = w
#         self.b = b
#         self.sqrLoss = sqrLoss0
#         self.learning_rate = learning_rate
#         self.n_iters = n + 1
#         return

#     def predict(self, testSet, testLabel):
#         self.testSet = testSet
#         self.testLabel = testLabel
#         y_predict = np.dot(self.testSet, self.w) + self.b
#         MSE = np.power((y_predict - self.testLabel), 2).sum() / len(testLabel)  # 计算均方误差
#         return MSE
import numpy as np
from scipy import optimize, sparse
from sklearn.utils.validation import check_X_y, check_array, _check_sample_weight

class LinearRegression:
    def __init__(self, fit_intercept=True, positive=False):
        self.w = None
        self.b = None
        self.fit_intercept = fit_intercept
        self.positive = positive

    def _validate_data(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'], y_numeric=True, multi_output=True)
        return X, y

    def _preprocess_data(self, X, y, sample_weight=None):
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            sw_sqrt = np.sqrt(sample_weight)
            X = X * sw_sqrt[:, np.newaxis]
            y = y * sw_sqrt
        if self.fit_intercept:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            y_offset = np.average(y, axis=0, weights=sample_weight)
            X -= X_offset
            y -= y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0.0
        return X, y, X_offset, y_offset

    def train(self, X, y, method='matrix', learning_rate=0.1, n_iters=1000, sample_weight=None):
        X, y = self._validate_data(X, y)
        X, y, X_offset, y_offset = self._preprocess_data(X, y, sample_weight)

        if method.lower() == 'matrix':
            self._train_matrix(X, y)
        elif method.lower() == 'gradient':
            self._train_gradient(X, y, learning_rate, n_iters)
        else:
            raise ValueError("method value not found!")

        if self.fit_intercept:
            self.b = y_offset - np.dot(X_offset, self.w)

    def _nnls(self, X, y):#非负最小二乘法
        m, n = X.shape
        w = np.zeros(n)
        for _ in range(1000):
            for j in range(n):
                X_j = X[:, j]
                residual = y - X @ w
                w[j] = max(0, w[j] + X_j.T @ residual / (X_j.T @ X_j))
        return w
    
    def _train_matrix(self, X, y):
        if self.positive:
            if y.ndim < 2:
                self.w = self._nnls(X, y)
            else:
                self.w = np.vstack([self._nnls(X, y[:, j]) for j in range(y.shape[1])])
        else:
            self.w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    def _train_gradient(self, X, y, learning_rate, n_iters):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(n_iters):
            y_pred = np.dot(X, self.w) + self.b
            dw = -2 * np.dot(X.T, (y - y_pred)) / n_samples
            db = -2 * np.sum(y - y_pred) / n_samples
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b