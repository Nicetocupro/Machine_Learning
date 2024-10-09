import numpy as np

'''
Lasso正则化回归与岭回归非常相似
但是这里用的是L1层的正则化参数
与岭回归的L2层不相同
因此Lasso无法直接进行求导进行求解
这里有两类方法，在此都会进行演示
1. 坐标下降法
2. 最小角回归法
'''


class LassoRegression:
    def __init__(self):
        self.trainSet = 0
        self.trainLabel = 0
        self.testSet = 0
        self.testLabel = 0
        self.lambdas = 0
        self.w = 0
        self.b = 0
        self.sqrLoss = 0  # 最小均方误差
        self.n_iters = 0  # 最大迭代次数
        self.tol = 0  # 变化量容忍值
        self.lossList = []  # 梯度下降每轮迭代的误差列表

    def train(self, train_X, train_y, method='coordinate_descent', n_iters=1000, tol=1e-6):
        # 维度一定大于2
        if train_X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = train_X
        self.trainLabel = train_y
        self.n_iters = n_iters
        self.tol = tol

        if method == 'coordinate_descent':
            # 坐标下降法
            self.coordinate_descent()

        elif method == 'least_angle_regression':
            # 最小角回归法
            self.least_angle_regression()

        else:
            raise ValueError("Unknown method: choose 'coordinate_descent' or 'least_angle_regression'.")

    def coordinate_descent(self):
        def down(X, y, w, index):
            """
            cost(w) = (x1 * w1 + x2 * w2 + ... - y)^2 + ... + λ(|w1| + |w2| + ...)
            假设 w1 是变量，这时其他的值均为常数，带入上式后，其代价函数是关于 w1 的一元二次函数，可以写成下式：
            cost(w1) = (a * w1 + b)^2 + ... + λ|w1| + c (a,b,c,λ 均为常数)
            => 展开后
            cost(w1) = aa * w1^2 + 2ab * w1 + λ|w1| + c (aa,ab,c,λ 均为常数)
            """
            aa, ab = 0, 0
            for i in range(X.shape[0]):
                a = X[i][index]
                b = np.dot(X[i], w) - X[i][index] * w[index] - y[i]

                aa = aa + a * a
                ab = ab + a * b

            # 接下来就要讨论w[index]的正负，然后进行分类讨论
            '''
                假设w[index] == 0
                    当W[index]为0的时候，cost函数为常数,因此无需更改，因为和w无关
                w[index] > 0
                cost(w1) = aa * w1^2 + 2ab * w1 + λw1 + c
                最小值是 (2ab + λ)^2 - 4 * aa * c / 2 * aa = 0
                w = - (2 * ab + λ) / (2 * aa)
                w1[index] < 0 
                cost(w1) = aa * w1^2 + 2ab * w1 - λw1 + c
                最小值是 (2ab - λ)^2 - 4 * aa * c / 2 * aa = 0
                w = - (2 * ab - λ) / (2 * aa)
            '''

            New_w = - (2 * ab + self.lambdas) / (2 * aa)
            if New_w < 0:
                New_w = - (2 * ab - self.lambdas) / (2 * aa)
                if New_w > 0:
                    New_w = 0
            return New_w

        n_samples, n_features = self.trainSet.shape

        X = self.trainSet
        y = self.trainLabel

        # 合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))

        # 初始化 w 和 b 参数
        t_w = np.zeros(n_features + 1)

        # 计算MSE损失
        sqrLoss0 = np.power(y - np.dot(X2, t_w)).sum()
        self.lossList.append(sqrLoss0)
        deltaLoss = np.inf

        for i in range(self.n_iters):
            done = True
            for j in range(n_features + 1):
                # 记录上一轮的权重
                weight = t_w[j]
                # 求出当前条件下的最佳参数
                t_w[j] = down(X2, y, t_w, j)

                # 计算MSE损失
                sqrLoss1 = np.power(y - np.dot(X2, t_w)).sum()
                deltaLoss = sqrLoss0 - sqrLoss1
                sqrLoss0 = sqrLoss1
                self.lossList.append(sqrLoss0)

                if (sqrLoss0 > self.tol) and (abs(deltaLoss) > self.tol) and (np.abs(weight - t_w[j]) > self.tol):
                    done = False

                print("第{}次迭代，第w{}参数优化后，损失平方和为{}，损失前后差为{}".format(i + 1, j, sqrLoss0, deltaLoss))

            if done:
                break
        self.w = t_w[:-1]
        self.b = t_w[-1]
        self.sqrLoss = sqrLoss0
        return

    def least_angle_regression(self):
        pass