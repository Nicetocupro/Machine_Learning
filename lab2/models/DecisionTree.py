import numpy as np
import pandas as pd
from collections import Counter

class DecisionTreeID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def entropy(self, y):
        # 计算熵
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def information_gain(self, X_column, y, threshold):
        # 计算信息增益
        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        # 计算左右子节点的熵
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        entropy_parent = self.entropy(y)
        entropy_left, entropy_right = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        # 计算信息增益
        ig = entropy_parent - (n_left / n) * entropy_left - (n_right / n) * entropy_right
        return ig

    def best_split(self, X, y):
        # 查找最佳分裂点
        best_gain = -1
        split_idx, split_threshold = None, None
        for i in range(X.shape[1]):
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_threshold = threshold
        return split_idx, split_threshold

    def build_tree(self, X, y, depth=0):
        # 检查是否满足停止条件
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # 添加深度限制的停止条件
        if depth >= self.max_depth or num_labels == 1 or num_samples < 2:
            most_common_label = Counter(y).most_common(1)[0][0]
            return most_common_label

        # 找到最佳分裂点
        feature_idx, threshold = self.best_split(X, y)
        if feature_idx is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return most_common_label

        # 递归构建左右子树，深度 +1
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = X[:, feature_idx] > threshold
        left = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return (feature_idx, threshold, left, right)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_idx, threshold, left, right = tree
        if x[feature_idx] <= threshold:
            return self.predict_sample(x, left)
        else:
            return self.predict_sample(x, right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])
class DecisionTreeCART:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def gini(self, y):
        # 计算基尼指数
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum([p ** 2 for p in probabilities if p > 0])

    def gini_gain(self, X_column, y, threshold):
        # 计算基尼增益
        left_idxs = X_column <= threshold
        right_idxs = X_column > threshold
        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0
        n = len(y)
        n_left, n_right = len(y[left_idxs]), len(y[right_idxs])
        gini_parent = self.gini(y)
        gini_left, gini_right = self.gini(y[left_idxs]), self.gini(y[right_idxs])
        # 计算加权基尼指数
        return gini_parent - (n_left / n) * gini_left - (n_right / n) * gini_right

    def best_split(self, X, y):
        # 查找最佳分裂点
        best_gain = -1
        split_idx, split_threshold = None, None
        for i in range(X.shape[1]):
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.gini_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_threshold = threshold
        return split_idx, split_threshold

    def build_tree(self, X, y, depth=0):
        # 检查是否满足停止条件
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # 添加深度限制的停止条件
        if depth >= self.max_depth or num_labels == 1 or num_samples < 2:
            most_common_label = Counter(y).most_common(1)[0][0]
            return most_common_label

        # 找到最佳分裂点
        feature_idx, threshold = self.best_split(X, y)
        if feature_idx is None:
            most_common_label = Counter(y).most_common(1)[0][0]
            return most_common_label

        # 递归构建左右子树，深度 +1
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = X[:, feature_idx] > threshold
        left = self.build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self.build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return (feature_idx, threshold, left, right)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_idx, threshold, left, right = tree
        if x[feature_idx] <= threshold:
            return self.predict_sample(x, left)
        else:
            return self.predict_sample(x, right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])
