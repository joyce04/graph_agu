import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv

"""
references :
- https://github.com/khuangaf/PyTorch-Geometric-YooChoose/
- https://github.com/rusty1s/pytorch_geometric
- https://github.com/snap-stanford/ogb/blob/9cbd598a31cb7934e69827eaccccc69135b8ef69/examples/linkproppred/ddi/gnn.py
- (positive/negative sampler) https://github.com/snap-stanford/ogb/blob/5a12524a1fd2e948d0b086972876cd4f68aefb25/examples/linkproppred/citation/sampler.py
"""


# class SAGE(nn.Module):
#     def __init__(self, num_layers, in_size, hidden_size, out_size):
#         super(SAGE, self).__init__()
#         assert (num_layers > 0)
#
#         self.num_layers = num_layers
#         self.convs = torch.nn.ModuleList()
#
#         # TODO : currently only mean aggregation
#         self.convs.append(SAGEConv(in_size, hidden_size))
#         # inner layers
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_size, hidden_size))
#         # last layer
#         self.convs.append(SAGEConv(hidden_size, out_size))
#
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#
#     def forward(self, x, adjs):
#         for i, (edge_index, _, size) in enumerate(adjs):
#             x_target = x[:size[1]]  # Target nodes are always placed first.
#             x = self.convs[i]((x, x_target), edge_index)
#             if i != self.num_layers - 1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x
#
#     def inference(self, x_all, subgraph_loader, device):
#         for i in range(self.num_layers):
#             xs = []
#             for batch_size, n_id, adj in subgraph_loader:
#                 edge_index, _, size = adj.to(device)
#                 x = x_all[n_id].to(device)
#                 x_target = x[:size[1]]
#                 x = self.convs[i]((x, x_target), edge_index)
#                 if i != self.num_layers - 1:
#                     x = F.relu(x)
#                 xs.append(x.cpu())
#
#             x_all = torch.cat(xs, dim=0)
#
#         return x_all
#
#
# def generate_sage_loader(edge_index, train_mask, num_nodes, batch_size):
#     train_loader = NeighborSampler(edge_index, node_idx=train_mask, num_nodes=num_nodes,
#                                    sizes=[25, 10], batch_size=batch_size, shuffle=True)
#     # full graph for evaluation
#     subgraph_loader = NeighborSampler(edge_index, node_idx=None, sizes=[-1],
#                                       batch_size=batch_size, shuffle=False)
#     return train_loader, subgraph_loader
#
#
# def get_train_unlabeled_adj(data):
#     unlabeled_nodes = ((data.val_mask | data.test_mask) == False) #TODO include train_mask?
#     train_loader = NeighborSampler(data.train_index, node_idx=unlabeled_nodes, num_nodes=data.num_nodes,
#                                    sizes=[25, 10], batch_size=unlabeled_nodes.shape[0], shuffle=True)
#     return [u_adjs for _, _, u_adjs in train_loader][0]

# reference : https://github.com/snap-stanford/ogb/blob/68a303f320220cda859e83e3a8660f2b9debedf6/examples/linkproppred/citation2/gnn.py#L208

class SAGE(torch.nn.Module):
    def __init__(self, num_layers, in_size, hidden_size, out_size, dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_size, hidden_size))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.convs.append(SAGEConv(hidden_size, out_size))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj_t)
