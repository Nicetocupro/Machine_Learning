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
        self.lambdas = 0
        self.w = 0
        self.b = 0
        self.sqrLoss = 0  # 最小均方误差
        self.n_iters = 0  # 最大迭代次数
        self.tol = 0  # 变化量容忍值
        self.lossList = []  # 梯度下降每轮迭代的误差列表

    def train(self, train_X, train_y, method='coordinate_descent', n_iters=1000, tol=1e-6, lambdas=0.2):
        if train_X.ndim < 2:
            raise ValueError("X must be 2D array-like!")
        self.trainSet = train_X
        self.trainLabel = np.atleast_2d(train_y)
        self.n_iters = n_iters
        self.tol = tol
        self.lambdas = lambdas

        if method == 'coordinate_descent':
            self.coordinate_descent()
        elif method == 'least_angle_regression':
            self.least_angle_regression()
        else:
            raise ValueError("Unknown method: choose 'coordinate_descent' or 'least_angle_regression'.")

    # 坐标下降法
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
            # 修改这里的条件判断
            if np.any(New_w < 0):
                New_w = - (2 * ab - self.lambdas) / (2 * aa)
                if np.any(New_w > 0):
                    New_w = 0
            return New_w

        # 确保 trainLabel 是二维的
        if self.trainLabel.ndim == 1:
            self.trainLabel = np.expand_dims(self.trainLabel, axis=1)

        n_samples, n_features = self.trainSet.shape
        _, n_outputs = self.trainLabel.shape

        X = self.trainSet
        y = self.trainLabel

        # 合并w和b，在X尾部添加一列全是1的特征
        X2 = np.hstack((X, np.ones((n_samples, 1))))

        # 初始化 w 和 b 参数
        t_w = np.ones((n_features + 1, n_outputs))

        # 计算MSE损失
        sqrLoss0 = np.power(y - np.dot(X2, t_w), 2).sum()
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
                sqrLoss1 = np.power(y - np.dot(X2, t_w), 2).sum()
                deltaLoss = sqrLoss0 - sqrLoss1
                sqrLoss0 = sqrLoss1
                self.lossList.append(sqrLoss0)

                if (sqrLoss0 > self.tol) and (np.abs(deltaLoss) > self.tol):
                    done = False

                print("第{}次迭代，第w{}参数优化后，损失平方和为{}，损失前后差为{}".format(i + 1, j, sqrLoss0, deltaLoss))

            if done:
                break
        self.w = t_w[:-1]
        self.b = t_w[-1]
        self.sqrLoss = sqrLoss0
        return

    # 最小角回归
    '''
        数学难度偏大，暂时未复现出来
        这里给出了网上的通用解法
    '''

    def least_angle_regression(self):
        # 处理多个输出变量
        n_outputs = self.trainLabel.shape[1]
        n_features = self.trainSet.shape[1]  # 更新特征数量
        self.w = np.zeros((n_features, n_outputs))  # 权重初始化
        self.b = np.zeros((n_outputs,))
        self.sqrLoss = np.zeros((n_outputs,))

        for output_index in range(n_outputs):
            X = self.trainSet
            y = self.trainLabel[:, output_index]
            n_samples, n_features = self.trainSet.shape
            # 合并w和b，在X尾部添加一列全是1的特征
            X2 = np.hstack((X, np.ones((n_samples, 1))))
            # 已被选择的特征下标
            active_set = set()
            # 当前预测向量
            cur_pred = np.zeros((n_samples,), dtype=np.float32)
            # 残差向量
            residual = y - cur_pred
            # 特征相连与残差向量的点积，即相关性
            cur_corr = X2.T.dot(residual)
            # 选取相关性最大的下标
            j = np.argmax(np.abs(cur_corr), 0)
            # 将下标添加至已被选择的特征下标集合
            active_set.add(j)
            # 初始化权重系数
            w = np.zeros((n_features + 1,), dtype=np.float32)
            # 记录上一次的权重系数
            prev_w = np.zeros((n_features + 1,), dtype=np.float32)
            # 记录特征更新的方向
            sign = np.zeros((n_features + 1,), dtype=np.int32)
            sign[j] = 1
            # 平均相关性
            lambda_hat = None
            # 记录上一次平均相关性
            prev_lambda_hat = None

            # 计算MSE损失
            sqrLoss0 = np.power(y - np.dot(X2, w), 2).sum()
            self.lossList.append(sqrLoss0)
            deltaLoss = np.inf

            # 迭代开始
            for it in range(self.n_iters):
                # 计算残差向量
                residual = y - cur_pred
                # 特征相连与残差向量的点积，即相关性
                cur_corr = X2.T.dot(residual)
                # 当前相关性最大值
                largest_abs_correlation = np.abs(cur_corr).max()
                # 计算当前平均相关性
                lambda_hat = largest_abs_correlation / n_samples
                # 当平均相关性小于λ时，提前结束迭代
                if lambda_hat <= self.lambdas:
                    if it > 0 and lambda_hat != self.lambdas:
                        ss = ((prev_lambda_hat - self.lambdas) / (prev_lambda_hat - lambda_hat))
                        # 重新计算权重系数
                        w[:] = prev_w + ss * (w - prev_w)
                    break

                # 更新上一次平均相关性
                prev_lambda_hat = lambda_hat

                # 当全部特征都被选择，结束迭代
                if len(active_set) > n_features + 1:
                    break

                # 选中的特征向量
                X_a = X2[:, list(active_set)]
                # 论文中 X_a 的计算公式 - (2.4)
                X_a *= sign[list(active_set)]
                # 论文中 G_a 的计算公式 - (2.5)
                G_a = X_a.T.dot(X_a)
                G_a_inv = np.linalg.inv(G_a)
                G_a_inv_red_cols = np.sum(G_a_inv, 1)
                # 论文中 A_a 的计算公式 - (2.5)
                A_a = 1 / np.sqrt(np.sum(G_a_inv_red_cols))
                # 论文中 ω 的计算公式 - (2.6)
                omega = A_a * G_a_inv_red_cols
                # 论文中角平分向量的计算公式 - (2.6)
                equiangular = X_a.dot(omega)
                # 论文中 a 的计算公式 - (2.11)
                cos_angle = X2.T.dot(equiangular)
                # 论文中的 γ
                gamma = None
                # 下一个选择的特征下标
                next_j = None
                # 下一个特征的方向
                next_sign = 0
                for j in range(n_features + 1):
                    if j in active_set:
                        continue
                    # 论文中 γ 的计算方法 - (2.13)
                    v0 = (largest_abs_correlation - cur_corr[j]) / (A_a - cos_angle[j]).item()
                    v1 = (largest_abs_correlation + cur_corr[j]) / (A_a + cos_angle[j]).item()
                    if v0 > 0 and (gamma is None or v0 < gamma):
                        gamma = v0
                        next_j = j
                        next_sign = 1
                    if v1 > 0 and (gamma is None or v1 < gamma):
                        gamma = v1
                        next_j = j
                        next_sign = -1
                if gamma is None:
                    # 论文中 γ 的计算方法 - (2.21)
                    gamma = largest_abs_correlation / A_a

                # 选中的特征向量
                sa = X_a
                # 角平分向量
                sb = equiangular * gamma
                # 解线性方程（sa * sx = sb）
                sx = np.linalg.lstsq(sa, sb)
                # 记录上一次的权重系数
                prev_w = w.copy()
                d_hat = np.zeros((n_features + 1,), dtype=np.int32)
                for i, j in enumerate(active_set):
                    # 更新当前的权重系数
                    w[j] += sx[0][i] * sign[j]
                    # 论文中 d_hat 的计算方法 - (3.3)
                    d_hat[j] = omega[i] * sign[j]
                # 论文中 γ_j 的计算方法 - (3.4)
                gamma_hat = -w / d_hat
                # 论文中 γ_hat 的计算方法 - (3.5)
                gamma_hat_min = float("+inf")
                # 论文中 γ_hat 的下标
                gamma_hat_min_idx = None
                for i, j in enumerate(gamma_hat):
                    if j <= 0:
                        continue
                    if gamma_hat_min > j:
                        gamma_hat_min = j
                        gamma_hat_min_idx = i
                if gamma_hat_min < gamma:
                    # 更新当前预测向量 - (3.6)
                    cur_pred = cur_pred + gamma_hat_min * equiangular
                    # 将下标移除至已被选择的特征下标集合
                    active_set.remove(gamma_hat_min_idx)
                    # 更新特征更新方向集合
                    sign[gamma_hat_min_idx] = 0
                else:
                    # 更新当前预测向量
                    cur_pred = X2.dot(w)
                    # 将下标添加至已被选择的特征下标集合
                    active_set.add(next_j)
                    # 更新特征更新方向集合
                    sign[next_j] = next_sign

                # 计算MSE损失
                sqrLoss1 = np.power(y - np.dot(X2, w), 2).sum()
                deltaLoss = sqrLoss0 - sqrLoss1
                sqrLoss0 = sqrLoss1
                self.lossList.append(sqrLoss0)

                print("第{}次迭代，损失平方和为{}，损失前后差为{}".format(it + 1, sqrLoss0, deltaLoss))

            self.w[:, output_index] = w[:-1]
            self.b[output_index] = w[-1]
            self.sqrLoss[output_index] = sqrLoss0
        return

    def predict(self, testSet):
        y_predict = np.dot(testSet, self.w) + self.b
        return y_predict