# 写一个简单的GNN神经网络接受Data(x=torch.tensor(x, dtype=torch.float32),edge_index=edge_index,edge_attr=edge_weight,y=label)
# 最后一层是图池化数据数据是(batch, 2)的形状
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, SAGEConv

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNClassifier, self).__init__()
        # 图卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x

class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
        super(GATClassifier, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        self.fc = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.gat1(x, edge_index)
        x = torch.relu(x)
        x = self.gat2(x, edge_index)
        x = torch.relu(x)
        x = self.gat3(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x


class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEClassifier, self).__init__()
        # GraphSAGE卷积层
        self.sage1 = SAGEConv(in_channels, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.sage3 = SAGEConv(hidden_channels, hidden_channels)

        # 分类输出层
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.sage1(x, edge_index)
        x = torch.relu(x)
        x = self.sage2(x, edge_index)
        x = torch.relu(x)
        x = self.sage3(x, edge_index)
        x = torch.relu(x)
        x = global_mean_pool(x, data.batch)
        x = self.fc(x)
        return x