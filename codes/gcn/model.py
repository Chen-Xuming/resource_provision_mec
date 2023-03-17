import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gmp

class GCN4OPTIMAL(nn.Module):
    def __init__(self, num_feature, embedding_size, num_class):
        super(GCN4OPTIMAL, self).__init__()

        self.num_feature = num_feature
        self.embedding_size = embedding_size

        self.num_class = num_class

        self.conv1 = GCNConv(num_feature, embedding_size)
        # self.conv2 = GCNConv(embedding_size, embedding_size)
        # self.conv3 = GCNConv(embedding_size, embedding_size)
        self.fc = nn.Linear(embedding_size, self.num_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, graph_data):
        x, edge_index, edge_weight, batch_index = graph_data.x, graph_data.edge_index, graph_data.edge_attr, graph_data.batch

        # x = (x - x.mean(dim=0)) / x.std(dim=0)
        edge_index = torch.Tensor(edge_index)


        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=0.5)

        x = self.fc(x)
        out = self.softmax(x)

        return out


