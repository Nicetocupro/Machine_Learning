import numpy as np

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Parameters:
    - X: 输入数据，应该是一个 NumPy 数组或类似的结构
    - eps: 邻域的最大距离（即DBSCAN的eps参数）
    - min_samples: 核心点的最小样本数（即DBSCAN的min_samples参数）

    Returns:
    - labels: 聚类标签，噪声点标记为-1
    """
    # 初始化
    n_samples = X.shape[0]
    labels = -1 * np.ones(n_samples)  # 默认所有点都为噪声点
    visited = np.zeros(n_samples, dtype=bool)  # 记录每个点是否被访问过
    cluster_id = 0  # 簇的编号

    # 遍历每个点
    for i in range(n_samples):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = _region_query(X, i, eps)

        if len(neighbors) < min_samples:
            # 如果邻居少于 min_samples，标记为噪声点
            labels[i] = -1
        else:
            # 如果邻居数大于等于 min_samples，开始扩展簇
            cluster_id += 1
            _expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_samples, visited)

    return labels


def _region_query(X, point_idx, eps):
    """
    查找给定点的邻居，返回邻居点的索引
    """
    distances = np.linalg.norm(X - X[point_idx], axis=1)
    neighbors = np.where(distances <= eps)[0]
    return neighbors


def _expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples, visited):
    """
    从核心点开始，扩展簇，包括给定点以及它的邻居点
    """
    # 将当前点标记为属于当前簇
    labels[point_idx] = cluster_id

    # 使用队列来扩展簇
    queue = list(neighbors)

    while queue:
        current_point = queue.pop(0)

        if not visited[current_point]:
            visited[current_point] = True
            current_neighbors = _region_query(X, current_point, eps)

            if len(current_neighbors) >= min_samples:
                # 如果该点的邻居数达到min_samples，则继续扩展
                queue.extend(current_neighbors)

        # 如果该点还没有被标记为簇，则将其标记为当前簇
        if labels[current_point] == -1:
            labels[current_point] = cluster_id
