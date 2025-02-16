import torch
from torch import nn
import torch.nn.functional as F
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb


class Bern_prop(MessagePassing):
    def __init__(self, K, is_source_domain=True, bias=True, **kwargs):
        super(Bern_prop, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.is_source_domain = is_source_domain
        self.cached_terms = None
        self.cached_coefs = None
        self.temp = nn.Parameter(torch.Tensor(self.K + 1), requires_grad=is_source_domain)
        self.reset_parameters()

    def reset_parameters(self):
        if self.is_source_domain:
            self.temp.data.fill_(1)
        else:
            self.temp.data = torch.linspace(1, 0, self.K + 1)
            # self.temp.data = torch.linspace(0, 1, self.K + 1)
            # self.temp.data.fill_(1)

    def get_filter(self):
        TEMP = F.relu(self.temp)
        H = 0

        for k in range(self.K + 1):
            H = H + TEMP[k] * self.cached_coefs[k] * self.cached_terms[k]

        return H

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = F.relu(self.temp)

        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        edge_index2, norm2 = add_self_loops(edge_index1, -norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp = []
        tmp.append(x)
        for i in range(self.K):
            x = self.propagate(edge_index2, x=x, norm=norm2, size=None)
            tmp.append(x)

        out = (comb(self.K, 0) / (2 ** self.K)) * TEMP[0] * tmp[self.K]

        for i in range(self.K):
            x = tmp[self.K - i - 1]
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            for j in range(i):
                x = self.propagate(edge_index1, x=x, norm=norm1, size=None)

            out = out + (comb(self.K, i + 1) / (2 ** self.K)) * TEMP[i + 1] * x
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class BernNet(torch.nn.Module):
    def __init__(self, features, hidden, classes, dropout, dprate=0.0, K=15):
        super(BernNet, self).__init__()
        self.lin1 = nn.Linear(features, hidden)
        self.lin2 = nn.Linear(hidden, classes)
        # self.m = torch.nn.BatchNorm1d(dataset.num_classes)
        self.prop1 = Bern_prop(K)
        self.prop2 = Bern_prop(K)
        self.prop3 = Bern_prop(K)
        self.prop4 = Bern_prop(K, False)

        self.dprate = dprate
        self.dropout = dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data, is_source_domain=True):
        x, edge_index = data.x, data.edge_index

        x = self.get_props(x, edge_index, is_source_domain)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop3(x, edge_index)
        return x

    def get_props(self, x, edge_index, is_source_domain=True):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))

        x = F.dropout(x, p=self.dprate, training=self.training)
        if is_source_domain:
            x = self.prop1(x, edge_index)
        else:
            x = self.prop2(x, edge_index)
        return x