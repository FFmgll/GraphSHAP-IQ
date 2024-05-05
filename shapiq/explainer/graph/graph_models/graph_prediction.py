""" This module stores all model architectures for graph prediction tasks."""

import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv, GINConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, node_bias=True, graph_bias=True):
        super().__init__()
        self.n_layers = n_layers
        self.conv_layers = {}
        self.conv_layers[0] = GCNConv(in_channels, hidden_channels, bias=node_bias)
        for i in range(1,n_layers):
            self.conv_layers[i] = GCNConv(hidden_channels, hidden_channels, bias=node_bias)
        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=graph_bias)
    def forward(self, x, edge_index, batch):
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = x.relu()

            if i == self.n_layers-1:
                x = global_mean_pool(x, batch)
            else:
                x = x.relu()

        x = self.lin(x)
        return x



class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, node_bias=True,graph_bias=True):
        super().__init__()
        self.n_layers = n_layers
        self.mlp_layers = {}
        self.conv_layers = {}
        self.mlp_layers[0] = torch.nn.Linear(in_channels, hidden_channels)
        self.conv_layers[0] = GINConv(self.mlp_layers[0])
        for i in range(1,n_layers):
            self.mlp_layers[i] = torch.nn.Linear(hidden_channels, hidden_channels, bias=node_bias)
            self.conv_layers[i] = GINConv(self.mlp_layers[i])

        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=graph_bias)

    def forward(self, x, edge_index, batch):
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            if i == self.n_layers -1:
                x = global_mean_pool(x, batch)
            else:
                x = x.relu()

        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
