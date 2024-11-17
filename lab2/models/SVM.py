import numpy as np

class SVM:
    def __init__(self, C=1.0, kernel='linear', tol=1e-3, max_iter=1000):
        self.C = C                  # 正则化参数
        self.kernel = kernel        # 核函数类型（'linear' 或 'rbf'）
        self.tol = tol              # 容忍度（停止准则）
        self.max_iter = max_iter    # 最大迭代次数

    def compute_kernel(self, X, X_train):
        if self.kernel == 'linear':
            # 线性核函数：K(x, x') = x^T * x'
            return np.dot(X, X_train.T)
        elif self.kernel == 'rbf':
            # 高斯径向基核（RBF）：K(x, x') = exp(-gamma * ||x - x'||^2)
            gamma = 1.0 / X.shape[1]  # 默认的gamma为特征数的倒数
            sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X_train**2, axis=1) - 2 * np.dot(X, X_train.T)
            return np.exp(-gamma * sq_dists)
        else:
            raise ValueError("Unsupported kernel type")

    def fit(self, X, y):
        # 1. 数据预处理
        n_samples, n_features = X.shape
        self.n_support_vectors = 0
        
        # 2. 训练数据标签转化为 +1 或 -1
        y = np.where(y <= 0, -1, 1)

        # 3. 初始化参数
        self.alpha = np.zeros(n_samples)  # 拉格朗日乘子
        self.b = 0.0  # 偏置项

        # 4. 计算内积核矩阵
        K = self.compute_kernel(X, X)

        # 5. 使用SMO算法（Sequential Minimal Optimization）来优化问题
        # 这里简化优化过程：实际应用中可以使用现成的二次规划求解器。
        for it in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)

            for i in range(n_samples):
                # 选择第i个样本
                # 计算预测值
                yi = y[i]
                xi = X[i]
                Ei = np.dot((self.alpha * y), K[i, :]) + self.b - yi

                if (yi * Ei < -self.tol and self.alpha[i] < self.C) or (yi * Ei > self.tol and self.alpha[i] > 0):
                    j = np.random.choice([x for x in range(n_samples) if x != i])  # 随机选择一个不同的j

                    yj = y[j]
                    xj = X[j]
                    Ej = np.dot((self.alpha * y), K[j, :]) + self.b - yj

                    # 保存alpha[i]和alpha[j]之前的值
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # 计算边界
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    # 计算eta（内积的第二导数）
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]

                    if eta >= 0:
                        continue

                    # 更新alpha[j]和alpha[i]
                    self.alpha[j] -= y[j] * (Ei - Ej) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    # 计算偏置项b
                    b1 = self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

            # 检查是否收敛
            diff = np.linalg.norm(self.alpha - alpha_prev)
            if diff < self.tol:
                break

    def predict(self, X):
        # 计算测试集的核矩阵
        K = self.compute_kernel(X, X)
        # 预测：y = sign(Σα * y * K(x, x') + b)
        y_pred = np.sign(np.dot(K, self.alpha * self.y_train) + self.b)
        return y_pred

