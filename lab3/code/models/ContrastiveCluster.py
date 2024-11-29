import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the ContrastiveClustering class
class ContrastiveCluster(nn.Module):
    def __init__(self, input_dim, n_clusters=3, n_features=2, alpha=1.0, hidden_dim=128):
        super(ContrastiveCluster, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.randn(n_clusters, input_dim))  # Centroids to be learned
        self.linear1 = nn.Linear(input_dim, hidden_dim)  # First linear transformation
        self.linear2 = nn.Linear(hidden_dim, n_features)  # Second linear transformation for feature extraction
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)  # Apply ReLU activation
        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize feature vectors
        centroids = F.normalize(self.centroids, p=2, dim=1)  # Normalize centroids
        return x, centroids
    
    def loss(self, x):
        x, centroids = self.forward(x)
        
        # Compute squared Euclidean distance between features and centroids using matrix operations
        dist = torch.cdist(x, centroids, p=2) ** 2
        
        # Contrastive loss function
        q = 1.0 / (1.0 + (dist / self.alpha))  # Similarity measure
        q = q**((self.alpha + 1.0) / 2.0)  # Sharpen similarity
        q = q / q.sum(dim=1, keepdim=True)  # Normalize
        
        # Inter-class contrast
        inter_class_dist = torch.cdist(centroids, centroids, p=2)
        inter_class_loss = torch.mean(inter_class_dist)
        
        # Intra-class contrast
        intra_class_loss = torch.mean(dist)
        
        # Regularization term to prevent overfitting
        reg_loss = torch.mean(torch.norm(self.centroids, p=2, dim=1))#reg_loss是对中心点的正则化
        
        return torch.mean(q) + inter_class_loss + intra_class_loss + 0.1 * reg_loss

    def predict(self, x):
        x, centroids = self.forward(x)
        dist = torch.cdist(x, centroids, p=2)
        return torch.argmin(dist, dim=1)  # Assign data points to the nearest centroid
