import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler

"""
references :
- https://github.com/khuangaf/PyTorch-Geometric-YooChoose/
- https://github.com/rusty1s/pytorch_geometric
- https://github.com/snap-stanford/ogb/blob/9cbd598a31cb7934e69827eaccccc69135b8ef69/examples/linkproppred/ddi/gnn.py
- (positive/negative sampler) https://github.com/snap-stanford/ogb/blob/5a12524a1fd2e948d0b086972876cd4f68aefb25/examples/linkproppred/citation/sampler.py
"""


class SAGE(nn.Module):
    def __init__(self, num_layers, in_size, hidden_size, out_size):
        super(SAGE, self).__init__()
        assert (num_layers > 0)

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()

        # TODO : currently only mean aggregation
        self.convs.append(SAGEConv(in_size, hidden_size))
        # inner layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        # last layer
        self.convs.append(SAGEConv(hidden_size, out_size))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def inference(self, x_all, subgraph_loader, device):
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                xs.append(x.cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all
