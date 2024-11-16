import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

# 定义一个决策树节点类
class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=10):
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = False
        self.prediction = None
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def fit(self, X, y):
        # 检查停止条件
        if self.depth >= self.max_depth or len(set(y)) == 1:
            self.is_leaf = True
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        # 找到最优分裂
        best_gini, best_feature, best_threshold = float('inf'), None, None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = X[:, feature_index] <= threshold
                right_indices = X[:, feature_index] > threshold
                gini = self._gini_index(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        # 如果无法分裂，设为叶节点
        if best_feature is None:
            self.is_leaf = True
            self.prediction = Counter(y).most_common(1)[0][0]
            return

        # 分裂数据
        self.feature_index = best_feature
        self.threshold = best_threshold
        left_indices = X[:, self.feature_index] <= self.threshold
        right_indices = X[:, self.feature_index] > self.threshold

        # 创建子节点
        self.left = DecisionTreeNode(self.depth + 1, self.max_depth)
        self.right = DecisionTreeNode(self.depth + 1, self.max_depth)
        self.left.fit(X[left_indices], y[left_indices])
        self.right.fit(X[right_indices], y[right_indices])

    def predict(self, X):
        if self.is_leaf:
            return self.prediction
        if X[self.feature_index] <= self.threshold:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

    def _gini_index(self, left_y, right_y):
        def gini(y):
            counts = np.bincount(y)
            probabilities = counts / len(y)
            return 1 - np.sum(probabilities ** 2)

        left_gini = gini(left_y) * (len(left_y) / (len(left_y) + len(right_y)))
        right_gini = gini(right_y) * (len(right_y) / (len(left_y) + len(right_y)))
        return left_gini + right_gini

# 定义随机森林类
class RandomForest:
    def __init__(self, n_trees=5, max_depth=3, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def fit(self, X, y):
        # 将 X 和 y 转换成 numpy 数组，以便在索引选择中兼容 numpy 索引
        X_np = np.array(X)
        y_np = np.array(y)
        n_samples = int(self.sample_size * X.shape[0])
        for _ in range(self.n_trees):
            indices = np.random.choice(X.shape[0], n_samples, replace=True)
            sample_X, sample_y = X_np[indices], y_np[indices]
            tree = DecisionTreeNode(max_depth=self.max_depth)
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)


    def predict(self, X):
        # 确保 X 是 numpy 数组
        X = np.array(X)
        tree_predictions = np.array([tree.predict(x) for x in X for tree in self.trees])
        tree_predictions = tree_predictions.reshape(len(X), self.n_trees)
        return [Counter(tree_predictions[i]).most_common(1)[0][0] for i in range(len(X))]
    def predict_proba(self, X):
        X = np.array(X)
        predictions = np.array([tree.predict(x) for x in X for tree in self.trees])
        predictions = predictions.reshape(len(X), self.n_trees)
        probas = np.mean(predictions == 1, axis=1)  # 假设二分类问题中的正类标签为 1
        return np.vstack([1 - probas, probas]).T  # 返回每类的概率
    def feature_importance(self, X, y):
        # 基于特征在树中的使用次数来计算重要性
        importance = np.zeros(X.shape[1])
        for tree in self.trees:
            if tree.is_leaf:
                continue
            importance[tree.feature_index] += 1
        importance /= self.n_trees
        return importance


def train_random_forest(X_train, y_train, X_test, y_test):
    # 初始化并训练随机森林
    rf_model = RandomForest(n_trees=5, max_depth=3)
    rf_model.fit(X_train, y_train)

    # 预测并评估
    y_pred = rf_model.predict(X_test)
    y_pred_proba = [p[1] for p in rf_model.predict_proba(X_test)]

    print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    
    
    # 计算 AUC 和 AP
    auc_score = roc_auc_score(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    print(f"AUC Score: {auc_score}")
    print(f"AP Score: {ap_score}")

    return rf_model

