import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_feats, out_feats) * 0.01)  # Xavier初始化
        self.bias = nn.Parameter(torch.zeros(out_feats))

    def forward(self, g, features):
        h = torch.matmul(features, self.weight)
        # 聚合邻居节点的信息
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        return h + self.bias


class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_feats, hidden_feats)
        self.layer2 = GCNLayer(hidden_feats, hidden_feats)
        self.layer3 = GCNLayer(hidden_feats, out_feats)
        self.dropout_rate = dropout_rate

    def forward(self, g, features):
        x = features
        x = F.relu(self.layer1(g, x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.layer2(g, x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.layer3(g, x)
        return x
