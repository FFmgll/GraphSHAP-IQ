import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import GCNConv
class GCN_3Layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, bias=False, normalize=True)
        self.conv2 = GCNConv(
            hidden_channels, hidden_channels, bias=False, normalize=True
        )
        self.conv3 = GCNConv(
            hidden_channels, hidden_channels, bias=False, normalize=True
        )
        self.lin = torch.nn.Linear(hidden_channels, out_channels, bias=True)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # x = global_mean_pool(x, batch)
        x = global_mean_pool(x, batch)

        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


def get_models(dataset_name,model_name):
    if dataset_name == "MUTAG":
        if model_name == "GCN_3Layer":
            return GCN_3Layer