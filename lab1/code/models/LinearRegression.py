import numpy as np
from sklearn.utils.validation import check_X_y, _check_sample_weight
import copy


class LinearRegression:
    def __init__(self, fit_intercept=True, positive=False):
        self.w = None
        self.b = None
        self.fit_intercept = fit_intercept
        self.positive = positive

    def _validate_data(self, X, y):
        X_validated, y_validated = check_X_y(
            X,
            y,
            accept_sparse=['csr', 'csc', 'coo'],
            y_numeric=True,
            multi_output=False,  # 设置为 False，如果是单输出
            copy=True
        )
        return X_validated, y_validated

    def _preprocess_data(self, X, y, sample_weight=None):
        X = copy.deepcopy(X)
        y = copy.deepcopy(y)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            sw_sqrt = np.sqrt(sample_weight)
            X = X * sw_sqrt[:, np.newaxis]
            y = y * sw_sqrt

        if self.fit_intercept:
            X_offset = np.average(X, axis=0, weights=sample_weight)
            y_offset = np.average(y, axis=0, weights=sample_weight)
            X = X - X_offset
            y = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1])
            y_offset = 0.0
        return X, y, X_offset, y_offset

    def train(self, X, y, method='matrix', learning_rate=0.1, n_iters=1000, sample_weight=None):
        X_copy = copy.deepcopy(X)
        y_copy = copy.deepcopy(y)

        X_copy, y_copy = self._validate_data(X_copy, y_copy)
        X_processed, y_processed, X_offset, y_offset = self._preprocess_data(X_copy, y_copy, sample_weight)

        if method.lower() == 'matrix':
            self._train_matrix(X_processed, y_processed)
        elif method.lower() == 'gradient':
            self._train_gradient(X_processed, y_processed, learning_rate, n_iters)
        else:
            raise ValueError("method value not found!")

        if self.fit_intercept:
            self.b = y_offset - np.dot(X_offset, self.w)

    def _nnls(self, X, y):  # 非负最小二乘法
        m, n = X.shape
        w = np.zeros(n)
        for _ in range(1000):
            for j in range(n):
                X_j = X[:, j]
                residual = y - X @ w
                denom = X_j.T @ X_j
                if denom != 0:
                    w_j_update = X_j.T @ residual / denom
                else:
                    w_j_update = 0
                w[j] = max(0, w[j] + w_j_update)
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
            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            # 参数更新
            self.w -= learning_rate * dw
            self.b -= learning_rate * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b
