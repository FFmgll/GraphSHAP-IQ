import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv


class GCN_2layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, graph_bias=False, node_bias=False):
        super(GCN_2layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,bias=node_bias)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, bias=node_bias)
        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=graph_bias)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, node_bias=False, graph_bias=False):
        super().__init__()

        self.conv_layers = {}
        self.conv_layers[0] = GCNConv(in_channels, hidden_channels, bias=node_bias)
        for i in range(1,n_layers):
            self.conv_layers[i] = GCNConv(in_channels, hidden_channels, bias=node_bias)

        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=graph_bias)

    def forward(self, x, edge_index, batch):
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            if i == self.n_layers-1:
                x = global_mean_pool(x, batch)
            else:
                x = x.relu()

        x = self.lin(x)

        return x

