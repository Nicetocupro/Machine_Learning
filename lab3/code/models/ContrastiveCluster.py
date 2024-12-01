import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the ContrastiveClustering class
class ContrastiveCluster(nn.Module):
    def __init__(self, input_dim, n_clusters=3, n_features=2, alpha=1.0, hidden_dim=128):
        """
        ContrastiveCluster类的初始化方法

        参数:
        - input_dim: 输入数据的特征维度
        - n_clusters: 聚类的数量，默认为3
        - n_features: 原本的特征数量（这里经过修改，实际使用时更多是配合input_dim来设置线性层输出维度），默认为2
        - alpha: 用于计算相似性度量等的缩放因子，默认为1.0
        - hidden_dim: 第一个线性变换层的隐藏层维度，默认为128
        """
        super(ContrastiveCluster, self).__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.alpha = alpha
        self.centroids = nn.Parameter(torch.randn(n_clusters, input_dim))  # Centroids to be learned
        self.linear1 = nn.Linear(input_dim, hidden_dim)  # First linear transformation
        self.linear2 = nn.Linear(hidden_dim, input_dim)  # 保证输出特征维度和centroids一致，用于特征提取
   
    def forward(self, x):
        """
        模型的前向传播方法

        参数:
        - x: 输入的张量数据

        返回:
        - x: 经过线性变换、激活函数和归一化后的特征向量
        - centroids: 经过归一化后的聚类中心点
        """
        x = self.linear1(x)
        x = F.relu(x)  # Apply ReLU activation
        x = self.linear2(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize feature vectors
        centroids = F.normalize(self.centroids, p=2, dim=1)  # Normalize centroids
        return x, centroids
    
    def loss(self, x):
        """
        计算模型的损失函数

        参数:
        - x: 输入的张量数据

        返回:
        - loss: 计算得到的总的损失值，包含对比损失、类间对比损失、类内对比损失以及正则化项
        """
        x, centroids = self.forward(x)
        
        # Compute squared Euclidean distance between features and centroids using matrix operations
        dist = torch.cdist(x, centroids, p=2) ** 2
        
        # 为了数值稳定性添加的小正数
        eps = 1e-12  
        # Contrastive loss function
        q = 1.0 / (1.0 + (dist / self.alpha))  # Similarity measure
        q = q**((self.alpha + 1.0) / 2.0)  # Sharpen similarity
        q_sum = q.sum(dim=1, keepdim=True)
        q = q / (q_sum + eps)  # 归一化，添加eps避免数值问题
        
        # Inter-class contrast
        inter_class_dist = torch.cdist(centroids, centroids, p=2)
        inter_class_loss = torch.mean(inter_class_dist)
        
        # Intra-class contrast
        intra_class_loss = torch.mean(dist)
        
        # Regularization term to prevent overfitting
        reg_loss = torch.mean(torch.norm(self.centroids, p=2, dim=1))
        return torch.mean(q) + inter_class_loss + intra_class_loss + 0.1 * reg_loss

    def predict(self, x):
        """
        使用模型进行预测，将数据点分配到最近的聚类中心点

        参数:
        - x: 输入的张量数据

        返回:
        - predictions: 每个数据点对应的聚类标签，通过找到最近的聚类中心点确定
        """
        x, centroids = self.forward(x)
        dist = torch.cdist(x, centroids, p=2)
        return torch.argmin(dist, dim=1)  # Assign data points to the nearest centroid