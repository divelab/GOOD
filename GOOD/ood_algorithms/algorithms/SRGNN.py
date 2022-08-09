
import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Batch
import numpy as np
from typing import Tuple
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg
from torch_geometric.utils import subgraph


def KMM(X, Xtest, config: Union[CommonArgs, Munch], _A=None, _sigma=1e1, beta=0.2):
    H = torch.exp(- 1e0 * pairwise_distances(X)) + torch.exp(- 1e-1 * pairwise_distances(X)) + torch.exp(
        - 1e-3 * pairwise_distances(X))
    f = torch.exp(- 1e0 * pairwise_distances(X, Xtest)) + torch.exp(- 1e-1 * pairwise_distances(X, Xtest)) + torch.exp(
        - 1e-3 * pairwise_distances(X, Xtest))
    z = torch.exp(- 1e0 * pairwise_distances(Xtest, Xtest)) + torch.exp(
        - 1e-1 * pairwise_distances(Xtest, Xtest)) + torch.exp(- 1e-3 * pairwise_distances(Xtest, Xtest))
    H /= 3
    f /= 3
    MMD_dist = H.mean() - 2 * f.mean() + z.mean()

    nsamples = X.shape[0]
    f = - X.shape[0] / Xtest.shape[0] * f.matmul(torch.ones((Xtest.shape[0], 1), device=config.device))
    G = - np.eye(nsamples)
    _A = _A[~np.all(_A == 0, axis=1)]
    b = _A.sum(1)
    h = - beta * np.ones((nsamples, 1))

    from cvxopt import matrix, solvers
    solvers.options['show_progress'] = False
    sol = solvers.qp(matrix(H.cpu().numpy().astype(np.double)), matrix(f.cpu().numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
    return np.array(sol['x']), MMD_dist.item()


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def cmd(X, X_test, K=5):
    """
    central moment discrepancy (cmd)
    objective function for keras models (theano or tensorflow backend)

    - Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.",
    - Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.
    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i + 2))
        # scms+=moment_diff(sx1,sx2,1)
    return sum(scms)


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    # ss1 = sx1.mean(0)
    # ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


@register.ood_alg_register
class SRGNN(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SRGNN, self).__init__(config)
        self.feat = None
        self.kmm_weight = None

    def input_preprocess(self,
                         data: Batch,
                         targets: Tensor,
                         mask: Tensor,
                         node_norm: Tensor,
                         training: bool,
                         config: Union[CommonArgs, Munch],
                         **kwargs
                         ) -> Tuple[Batch, Tensor, Tensor, Tensor]:
        r"""
        Set input data and mask format to prepare for mixup

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns:
            - data (Batch) - Processed input data.
            - targets (Tensor) - Processed input labels.
            - mask (Tensor) - Processed NAN masks for data formats.
            - node_norm (Tensor) - Processed node weights for normalization.

        """
        if training:
            # Z_train = torch.FloatTensor(adj[idx_train.tolist(), :].todense())
            # Z_test = torch.FloatTensor(adj[iid_train.tolist(), :].todense())
            # # embed()
            # label_balance_constraints = np.zeros((labels.max().item() + 1, len(idx_train)))
            # for i, idx in enumerate(idx_train):
            #     label_balance_constraints[labels[idx], i] = 1
            # # embed()
            # kmm_weight, MMD_dist = KMM(Z_train, Z_test, label_balance_constraints, beta=0.2)
            self.kmm_weight = torch.zeros(data.y.shape[0], device=config.device)
            Z_all = torch_geometric.utils.to_dense_adj(
                data.edge_index[:, data.train_mask[data.edge_index[0]] | data.train_mask[data.edge_index[1]]],
                torch.zeros(data.y.shape[0], dtype=torch.long, device=config.device)).squeeze()

            # a = int(data.y.shape[0]/2)
            # b = data.y.shape[0] - a
            # Z_train = Z_all
            # Z_test = Z_train[torch.cat((torch.randperm(b)+a, torch.randperm(a)))]
            # label_balance_constraints = np.zeros((config.dataset.num_classes, data.y.shape[0]))
            # for i, y in enumerate(data.y):
            #     label_balance_constraints[y, i] = 1
            # kmm_weight_env, MMD_dist = KMM(Z_train, Z_test, config, label_balance_constraints, beta=0.2)
            # self.kmm_weight = torch.from_numpy(kmm_weight_env).float().cuda(
            #     device=config.device).squeeze()

            # for i in range(config.dataset.num_envs):
            #     env_idx = (data.env_id == i).clone().detach()
            #     if data.y[env_idx].shape[0] > 0:
            #         Z_train = Z_all[env_idx]
            #         Z_test = Z_train[torch.randperm(data.y[env_idx].shape[0])]
            #         # Z_train = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]])
            #         # edge_env_mask = env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]
            #         # Z_test = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx_2[data.edge_index[0]] | env_idx_2[data.edge_index[1]]])
            #         label_balance_constraints = np.zeros((config.dataset.num_classes, data.y[env_idx].shape[0]))
            #         for i, y in enumerate(data.y[env_idx]):
            #             label_balance_constraints[y, i] = 1
            #         kmm_weight_env, MMD_dist = KMM(Z_train, Z_test, config, label_balance_constraints, beta=0.2)
            #         self.kmm_weight[[env_idx.nonzero().squeeze()]] = torch.from_numpy(kmm_weight_env).float().cuda(device=config.device).squeeze()

            for i in range(config.dataset.num_envs):
                env_idx = (data.env_id == i).clone().detach()
                if data.y[env_idx].shape[0] > 0:
                    Z_train = Z_all[env_idx]
                    Z_test = Z_all[~env_idx][:(data.y[env_idx].shape[0])]
                    # Z_train = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]])
                    # edge_env_mask = env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]
                    # Z_test = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx_2[data.edge_index[0]] | env_idx_2[data.edge_index[1]]])
                    label_balance_constraints = np.zeros((config.dataset.num_classes, data.y[env_idx].shape[0]))
                    for i, y in enumerate(data.y[env_idx]):
                        label_balance_constraints[y, i] = 1
                    kmm_weight_env, MMD_dist = KMM(Z_train, Z_test, config, label_balance_constraints, beta=0.2)
                    self.kmm_weight[[env_idx.nonzero().squeeze()]] = torch.from_numpy(kmm_weight_env).float().cuda(device=config.device).squeeze()

        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:

        self.feat = model_output[1]
        return model_output[0]

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:

        SRloss_list = []

        for i in range(config.dataset.num_envs):
            env_idx_1 = data.env_id == i
            env_feat_1 = self.feat[env_idx_1]
            if env_feat_1.shape[0] > 1:
                env_feat_2 = env_feat_1[torch.randperm(env_feat_1.shape[0])]
                shift_robust_loss = cmd(env_feat_1, env_feat_2)
                SRloss_list.append(shift_robust_loss)

        if len(SRloss_list) == 0:
            SRloss = torch.tensor(0)
        else:
            SRloss = sum(SRloss_list) / len(SRloss_list)
        spec_loss = config.ood.ood_param * SRloss
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = (self.kmm_weight * loss).sum() / mask.sum()
        loss = mean_loss + spec_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss
