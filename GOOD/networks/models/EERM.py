r"""
The implementation of `Handling Distribution Shifts on Graphs: An Invariance Perspective <https://arxiv.org/abs/2202.02466>`_.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, subgraph

from GOOD import register
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor


@register.model_register
class EERMGCN(GNNBasic):
    r"""
    EERM implementation adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, config):
        super(EERMGCN, self).__init__(config)
        self.gnn = GCNFeatExtractor(config)
        self.p = 0.2
        self.K = config.ood.extra_param[0]
        self.T = config.ood.extra_param[1]
        self.num_sample = config.ood.extra_param[2]
        self.classifier = Classifier(config)

        self.gl = Graph_Editer(self.K, config.dataset.num_train_nodes, config.device)
        self.gl.reset_parameters()
        self.gl_optimizer = torch.optim.Adam(self.gl.parameters(), lr=config.ood.extra_param[3])

    def reset_parameters(self):
        self.gnn.reset_parameters()
        if hasattr(self, 'graph_est'):
            self.gl.reset_parameters()

    def forward(self, *args, **kwargs):
        data = kwargs.get('data')
        loss_func = self.config.metric.loss_func

        # --- K fold ---
        if self.training:
            edge_index, _ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
            x = data.x[data.train_mask]
            y = data.y[data.train_mask]

            # --- check will orig_edge_index change? ---
            orig_edge_index = edge_index
            for t in range(self.T):
                Loss, Log_p = [], 0
                for k in range(self.K):
                    edge_index, log_p = self.gl(orig_edge_index, self.num_sample, k)
                    raw_pred = self.classifier(self.gnn(data=Data(x=x, edge_index=edge_index, y=y)))

                    loss = loss_func(raw_pred, y)
                    Loss.append(loss.view(-1))
                    Log_p += log_p
                Var, Mean = torch.var_mean(torch.cat(Loss, dim=0))
                reward = Var.detach()
                inner_loss = - reward * Log_p
                self.gl_optimizer.zero_grad()
                inner_loss.backward()
                self.gl_optimizer.step()
            return Var, Mean
        else:
            out = self.classifier(self.gnn(data=data))
            return out


class Graph_Editer(nn.Module):
    r"""
    EERM's graph editer adapted from https://github.com/qitianwu/GraphOOD-EERM.
    """
    def __init__(self, K, n, device):
        super(Graph_Editer, self).__init__()
        self.B = nn.Parameter(torch.FloatTensor(K, n, n))
        self.n = n
        self.device = device

    def reset_parameters(self):
        nn.init.uniform_(self.B)

    def forward(self, edge_index, num_sample, k):
        n = self.n
        Bk = self.B[k]
        A = to_dense_adj(edge_index, max_num_nodes=n)[0].to(torch.int)
        A_c = torch.ones(n, n, dtype=torch.int).to(self.device) - A
        P = torch.softmax(Bk, dim=0)
        S = torch.multinomial(P, num_samples=num_sample)  # [n, s]
        M = torch.zeros(n, n, dtype=torch.float).to(self.device)
        col_idx = torch.arange(0, n).unsqueeze(1).repeat(1, num_sample)
        M[S, col_idx] = 1.
        C = A + M * (A_c - A)
        edge_index = dense_to_sparse(C)[0]

        log_p = torch.sum(
            torch.sum(Bk[S, col_idx], dim=1) - torch.logsumexp(Bk, dim=0)
        )

        return edge_index, log_p
