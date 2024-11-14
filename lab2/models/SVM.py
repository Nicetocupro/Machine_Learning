import numpy as np

class SVM:
    def __init__(self, kernel='linear', degree=3, gamma=0.1, learning_rate=0.001, lambda_param=0.01, n_iters=1000, C=1.0):
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.C = C
        self.w = None
        self.b = None

    def linear_kernel(self, x, y):
        return np.dot(x, y)

    def polynomial_kernel(self, x, y):
        return (1 + np.dot(x, y)) ** self.degree

    def tanh_kernel(self, x, y):
        return np.tanh(self.gamma * np.dot(x, y))

    def rbf_kernel(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)

    def compute_kernel(self, x, y):
        if self.kernel == 'linear':
            return self.linear_kernel(x, y)
        elif self.kernel == 'polynomial':
            return self.polynomial_kernel(x, y)
        elif self.kernel == 'tanh':
            return self.tanh_kernel(x, y)
        elif self.kernel == 'rbf':
            return self.rbf_kernel(x, y)
        else:
            raise ValueError("Unsupported kernel")

    def train(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (self.compute_kernel(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
    
    def predict(self, X):
        y_pred = []
        for x in X:
            prediction = np.sign(self.compute_kernel(x, self.w) - self.b)
            y_pred.append(prediction)
        return np.array(y_pred)