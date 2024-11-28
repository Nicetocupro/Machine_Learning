import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. 实现层次聚类
def hierarchical_clustering(X, threshold=1.0):
    """
    实现凝聚型层次聚类
    参数：
    X : 数据矩阵 (n_samples, n_features)
    threshold : 形成簇的距离阈值
    """
    # 计算样本之间的距离矩阵
    dist_matrix = pdist(X)
    dist_matrix = squareform(dist_matrix)  # 转换为方阵形式

    # 初始化每个数据点为一个簇
    clusters = [[i] for i in range(len(X))]
    
    # 不断合并最近的簇
    while len(clusters) > 1:
        # 找到距离最小的两个簇
        min_dist = np.inf
        merge_clusters = (0, 0)
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                cluster_i = clusters[i]
                cluster_j = clusters[j]
                # 计算两个簇之间的距离
                dist = np.mean(dist_matrix[np.ix_(cluster_i, cluster_j)])
                if dist < min_dist:
                    min_dist = dist
                    merge_clusters = (i, j)

        # 如果距离小于阈值，则停止合并
        if min_dist > threshold:
            break

        # 合并这两个簇
        i, j = merge_clusters
        new_cluster = clusters[i] + clusters[j]
        clusters = [clusters[k] for k in range(len(clusters)) if k != i and k != j]  # 删除合并的簇
        clusters.append(new_cluster)  # 添加新簇

    return clusters

# 2. 对每个簇进行回归分析
def regression_on_clusters(X, y, clusters):
    """
    在每个聚类上进行回归分析
    参数：
    X : 特征矩阵
    y : 目标变量
    clusters : 聚类结果
    """
    models = []
    predictions = []

    # 遍历每个簇
    for cluster in clusters:
        X_cluster = X[cluster]
        y_cluster = y[cluster]
        
        # 训练线性回归模型
        model = LinearRegression()
        model.fit(X_cluster, y_cluster)
        models.append(model)
        
        # 对当前簇进行预测
        y_pred = model.predict(X_cluster)
        predictions.append(y_pred)

    # 返回回归模型和预测结果
    return models, np.concatenate(predictions)

# 3. 可视化聚类结果（可选）
def plot_dendrogram(X):
    """
    可视化层次聚类的树状图
    """
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    # 生成树状图的链接矩阵
    linked = linkage(X, 'single')
    
    # 绘制树状图
    dendrogram(linked)
    plt.show()

# 示例数据
# 假设 X 是特征矩阵，y 是目标变量
# 例如使用随机数据
np.random.seed(42)
X = np.random.rand(100, 2)  # 100个样本，2个特征
y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1  # 目标变量是X的两列之和，并添加一些噪声

# 1. 执行层次聚类
clusters = hierarchical_clustering(X, threshold=0.5)

# 2. 对每个簇进行回归分析
models, y_pred = regression_on_clusters(X, y, clusters)

# 3. 评估回归模型
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")

# 4. 可视化聚类结果
plot_dendrogram(X)
