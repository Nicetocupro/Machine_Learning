import numpy as np
from sklearn.tree import DecisionTreeRegressor


class RandomForest:
    def __init__(self, n_estimators=100, random_state=0):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []

    def _validate_data(self, X, y):
        # 数据预处理
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
        return X, y

    def train(self, X, y):
        # 拟合模型
        X, y = self._validate_data(X, y)
        n_samples = X.shape[0]
        rs = np.random.RandomState(self.random_state)

        self.trees = []

        for _ in range(self.n_estimators):
            dt = DecisionTreeRegressor(
                max_features="sqrt",
                random_state=self.random_state  # 使用相同的随机状态，避免每棵树不同
            )
            sample_indices = rs.choice(n_samples, n_samples, replace=True)
            dt.fit(X[sample_indices], y[sample_indices])
            self.trees.append(dt)

    def predict(self, X):
        # 初始化预测结果为与输入样本数量一致的零数组
        X = np.array(X)  # 确保输入是一个NumPy数组
        predictions = np.zeros(X.shape[0])
        for tree in self.trees:
            predictions += tree.predict(X)
        # 对所有树的预测结果取平均
        predictions /= self.n_estimators
        return predictions
