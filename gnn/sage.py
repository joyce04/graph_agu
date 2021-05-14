import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

'''
References :  https://github.com/snap-stanford/ogb
@article{hu2020ogb,
  title={Open Graph Benchmark: Datasets for Machine Learning on Graphs},
  author={Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec},
  journal={arXiv preprint arXiv:2005.00687},
  year={2020}
}
'''


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
