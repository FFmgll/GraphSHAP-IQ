""" This module stores all model architectures for graph prediction tasks."""

from collections import OrderedDict

import torch
import torch_geometric
from torch.nn import Dropout, Module, ReLU
from torch.nn import Sequential as Seq
from torch_geometric.nn import (
	GCNConv, GINConv, JumpingKnowledge, Linear, Sequential, global_mean_pool,
	MessagePassing
)
from torch_geometric.nn.norm import BatchNorm
import torch.nn.functional as F

class GCN(Module):
	def __init__(self, in_channels, hidden_channels, out_channels, n_layers, node_bias=True, graph_bias=True,
	             dropout=True, batch_norm=True, jumping_knowledge=False):
		super(GCN, self).__init__()
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.n_layers = n_layers
		self.node_bias = node_bias
		self.graph_bias = graph_bias
		self.dropout = dropout
		self.p = 0.5 if dropout else 0
		self.batch_norm = batch_norm
		self.jumping_knowledge = jumping_knowledge

		# Build model architecture
		self.layers = OrderedDict()
		self.intermediate_outputs = tuple(f"x{i}" for i in range(n_layers))
		assert n_layers > 0, "Number of layers must be greater than 0"
		for i in range(n_layers):
			if i == 0:
				self.layers[f"conv_{i}"] = (
				GCNConv(in_channels, hidden_channels, bias=node_bias), f"x, edge_index -> x{i}")
			else:
				self.layers[f"conv_{i}"] = (
				GCNConv(hidden_channels, hidden_channels, bias=node_bias), f"x{i - 1}, edge_index -> x{i}")

			if batch_norm:
				self.layers[f"batch_norm_{i}"] = BatchNorm(hidden_channels)

			self.layers[f"relu_{i}"] = ReLU()
			self.layers[f"dropout_{i}"] = Dropout(p=self.p)

		if jumping_knowledge:
			self.layers["concat"] = (lambda *x: [*x], f"{', '.join(self.intermediate_outputs)} -> x")
			self.jk = JumpingKnowledge(mode='cat')
			self.lin = Linear(hidden_channels * n_layers, out_channels, bias=graph_bias)
		else:
			self.lin = Linear(hidden_channels, out_channels, bias=graph_bias)

		self.node_model = Sequential('x, edge_index', self.layers)

		# Initialize parameters
		self.reset_parameters()

	def reset_parameters(self):
		for layer in self.layers:
			if hasattr(layer, 'reset_parameters'):
				layer.reset_parameters()

		self.lin.reset_parameters()

	def forward(self, x, edge_index, batch):
		x = self.node_model(x=x, edge_index=edge_index)
		if hasattr(self, 'jk'):
			x = self.jk(x)
		x = global_mean_pool(x, batch)
		x = self.lin(x)
		return x


class GIN(torch.nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels, n_layers, node_bias=True, graph_bias=True):
		super().__init__()
		self.n_layers = n_layers
		self.mlp_layers = {}
		self.conv_layers = {}
		self.mlp_layers[0] = torch.nn.Linear(in_channels, hidden_channels)
		self.conv_layers[0] = GINConv(self.mlp_layers[0])
		for i in range(1, n_layers):
			self.mlp_layers[i] = torch.nn.Linear(hidden_channels, hidden_channels, bias=node_bias)
			self.conv_layers[i] = GINConv(self.mlp_layers[i])

		self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=graph_bias)

	def forward(self, x, edge_index, batch):
		for i in range(self.n_layers):
			x = self.conv_layers[i](x, edge_index)
			if i == self.n_layers - 1:
				x = global_mean_pool(x, batch)
			else:
				x = x.relu()

		# x = F.dropout(x, p=0.5, training=self.training)
		x = self.lin(x)

		return x

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=False):
        super().__init__(aggr='mean') #  "Max" aggregation.
        self.mlp1 = Seq(
			Linear(2 * in_channels, hidden_channels, bias=bias),
            ReLU(),
			Linear(hidden_channels, hidden_channels, bias=bias)
		)
        self.mlp2 = Seq(Linear(hidden_channels, out_channels, bias=False))

    def forward(self, x, edge_index, edge_features):
        # x: [N, in_channels]
        # edge_index: [2, n_edges]
        x = self.propagate(edge_index, x=x, edge_features=edge_features)
        return self.mlp2(x)

    def message(self, x_j, edge_features):
        # x_j: [n_edges, in_channels]
        # edge_features: [n_edges, in_channels]
        tmp = torch.cat([x_j, edge_features], dim=-1)
        return self.mlp1(tmp)

class QualityModel(Module):
    def __init__(self, in_channels, hidden_channels, out_channels, bias=False, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.node_enc = Linear(in_channels, hidden_channels, bias=bias)
        self.edge_enc = Linear(in_channels, hidden_channels, bias=bias)
        self._layers = torch.nn.ModuleList([
            EdgeConv(hidden_channels, hidden_channels, hidden_channels, bias=bias)
            for _ in range(self.n_layers)
        ])
        self.out = Seq(Linear(hidden_channels, out_channels, bias=False))

    def forward(self, x, edge_index, edge_features, batch):
        x = self.node_enc(x)
        edge_features = self.edge_enc(edge_features)
        for layer in self._layers:
            x += layer(x, edge_index, edge_features)
        x = global_mean_pool(x, batch)
        return self.out(x)

if __name__ == "__main__":
	model = GCN(128, 64, 2, 3, node_bias=True, graph_bias=True, dropout=False, batch_norm=False,
	            jumping_knowledge=False)
	x = torch.randn(100, 128)
	edge_index = torch.randint(100, size=(2, 20))
	data = torch_geometric.data.Data(x=x, edge_index=edge_index)
	dataloader = torch_geometric.loader.DataLoader([data], batch_size=1)
	for data in dataloader:
		print(model(data.x, data.edge_index, data.batch))
		break