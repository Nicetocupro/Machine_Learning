import numpy as np
import pandas as pd

class KMeans:
    def __init__(self, X, n_clusters=3, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.labels = None  # This will store the cluster labels
        
        # Fit the model during initialization (optional, but fits the use case)
        self.fit(X)

    def _assign_labels(self, X):
        # Calculate distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X):
        # Recompute centroids based on the labels
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def fit(self, X):
        """Fit the KMeans model to the data X."""
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X

        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # Assign labels based on the current centroids
            self.labels = self._assign_labels(X)
            
            # Compute new centroids
            new_centroids = self._compute_centroids(X)
            
            # Check for convergence
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"Convergence reached at iteration {_}")
                break
            
            # Update centroids for next iteration
            self.centroids = new_centroids

    def predict(self, X):
        """Predict the closest cluster for each sample in X."""
        X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
        return self._assign_labels(X)



"""KMeans Clustering,簇的数量由上述代码验证为4,并且使用自定义的KMeans类,放在juoyter notebook中的代码"""
"""""
# Using custom KMeans class
kmeans = KMeans.KMeans(n_clusters=4,X=PCA_ds)
kmeans.fit(PCA_ds)
PCA_ds["Clusters"] = kmeans.predict(PCA_ds)
# Adding the Clusters feature to the original dataframe.
data["Clusters"] = PCA_ds["Clusters"]

#对KMeans聚类结果使用silhouette_score进行评估
silhouette = silhouette_score(PCA_ds, PCA_ds["Clusters"])
print("The silhouette score of the KMeans clustering model is:", silhouette)

# Plotting the clusters
fig = plt.figure(figsize=(10, 8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap=cmap)
ax.set_title("The Plot Of The Clusters")
plt.show()
"""""