dbscan:
```
# DBSCAN聚类
# k-距离图自动选择eps
# 数据预处理
scaler = StandardScaler()
PCA_ds_scaled = scaler.fit_transform(PCA_ds)

nbrs = NearestNeighbors(n_neighbors=4).fit(PCA_ds_scaled)
distances, _ = nbrs.kneighbors(PCA_ds_scaled)
distances = sorted(distances[:, -1], reverse=True)
knee_locator = KneeLocator(range(len(distances)), distances, curve="convex", direction="decreasing")
optimal_eps = distances[knee_locator.knee]

# 动态网格搜索
eps_values = np.linspace(0.1, 1.2 * optimal_eps, 10)
min_samples_values = range(2, 10)
best_score, best_eps, best_min_samples = -1, None, None

for eps in eps_values:
    for min_samples in min_samples_values:
        labels = run_dbscan(PCA_ds_scaled, eps=eps, min_samples=min_samples)
        if len(set(labels)) > 1:
            score = silhouette_score(PCA_ds_scaled, labels)
            if score > best_score:
                best_score, best_eps, best_min_samples = score, eps, min_samples

# 使用最佳参数重新聚类
labels_dbscan = run_dbscan(PCA_ds_scaled, eps=best_eps, min_samples=best_min_samples)
PCA_ds['DBSCAN_Clusters'] = labels_dbscan

# 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(PCA_ds['col1'], PCA_ds['col2'], c=PCA_ds['DBSCAN_Clusters'], cmap='viridis', marker='o')
plt.title("Optimized DBSCAN Clustering Results")
plt.colorbar(label="Cluster Label")
plt.show()

# 聚类效果评估
if len(set(labels_dbscan)) > 1:
    silhouette_avg = silhouette_score(PCA_ds_scaled, labels_dbscan)
    print(f"Optimized DBSCAN Silhouette Score: {silhouette_avg}")
else:
    print("DBSCAN found only one cluster or all points are noise.")
```

DEC聚类
```
# DEC聚类
# 数据预处理
# scaler = StandardScaler()
# 使用MinMaxScaler重新标准化数据
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(PCA_ds)  # 标准化数据

# 初始化 DEC 模型
n_clusters = 3  # 设置簇数
dec_model = DEC(n_clusters=n_clusters)

# 构建自编码器
dec_model.build_autoencoder(input_dim=X_scaled.shape[1])

# 训练自编码器
dec_model.train_autoencoder(X_scaled, epochs=50, batch_size=256)

# 进行聚类（同时优化自编码器）
dec_model.clustering(X_scaled, epochs=100, batch_size=256)

# 获取聚类标签
labels_dec = np.argmax(dec_model._calculate_cluster_probabilities(X_scaled), axis=1)

# 可视化 DEC 聚类结果
plt.figure(figsize=(10, 8))
plt.scatter(PCA_ds['col1'], PCA_ds['col2'], c=labels_dec, cmap='viridis', marker='o')
plt.title("DEC Clustering Results")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster Label")
plt.show()

# 评估 DEC 聚类效果（轮廓系数）
silhouette_avg = silhouette_score(X_scaled, labels_dec)
print(f"DEC Silhouette Score: {silhouette_avg}")
```