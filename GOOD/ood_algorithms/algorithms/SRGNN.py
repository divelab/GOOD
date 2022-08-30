"""
Implementation of the SRGNN algorithm from `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
<https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper
"""
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
    r"""
    Kernel mean matching (KMM) to compute the weight for each training instance

    Args:
        X (Tensor): training instances to be matched
        Xtest (Tensor): IID samples to match the training instances
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`)
        _A (numpy array): one hot matrix of the training instance labels
        _sigma (float): normalization term
        beta (float): regularization weight

    Returns:
        - KMM_weight (numpy array) - KMM_weight to match each training instance
        - MMD_dist (Tensor) - MMD distance

    """
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
    r"""
    computation tool for pairwise distances

    Args:
        x (Tensor): a Nxd matrix
        y (Tensor): an optional Mxd matirx

    Returns (Tensor):
        dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
        if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2

    """
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
    r"""
    central moment discrepancy (cmd). objective function for keras models (theano or tensorflow backend). Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.

    Args:
        X (Tensor): training instances
        X_test (Tensor): IID samples
        K (int): number of approximation degrees

    Returns (Tensor):
         central moment discrepancy

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
    r"""
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    r"""
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    # ss1 = sx1.mean(0)
    # ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


@register.ood_alg_register
class SRGNN(BaseOODAlg):
    r"""
    Implementation of the SRGNN algorithm from `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
    <https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(SRGNN, self).__init__(config)
        self.feat = None
        self.kmm_weight = None
        self.z_test_idx = None

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
        Set input data and mask format to prepare for SRGNN

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

            # for i in range(config.dataset.num_envs):
            #     env_idx = (data.env_id == i).clone().detach()
            #     if data.y[env_idx].shape[0] > 0:
            #         Z_train = Z_all[env_idx]
            #         Z_test = Z_all[~env_idx][:(data.y[env_idx].shape[0])]
            #         # Z_train = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]])
            #         # edge_env_mask = env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]
            #         # Z_test = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx_2[data.edge_index[0]] | env_idx_2[data.edge_index[1]]])
            #         label_balance_constraints = np.zeros((config.dataset.num_classes, data.y[env_idx].shape[0]))
            #         label_balance_constraints = torch.eye(config.dataset.num_classes)[data.y[env_idx]].T
            #         # torch.stack([torch.arange(config.dataset.num_classes), data.y[env_idx]], dim=1)
            #         for i, y in enumerate(data.y[env_idx]):
            #             label_balance_constraints[y, i] = 1
            #         kmm_weight_env, MMD_dist = KMM(Z_train, Z_test, config, label_balance_constraints, beta=0.2)
            #         self.kmm_weight[[env_idx.nonzero().squeeze()]] = torch.from_numpy(kmm_weight_env).float().cuda(device=config.device).squeeze()

            Z_val = torch_geometric.utils.to_dense_adj(
                data.edge_index[:, data.val_mask[data.edge_index[0]] | data.val_mask[data.edge_index[1]]],
                torch.zeros(data.y.shape[0], dtype=torch.long, device=config.device)).squeeze()


            for i in range(config.dataset.num_envs):
                env_idx = (data.env_id == i).clone().detach()
                if data.y[env_idx].shape[0] > 0:
                    Z_train = Z_all[env_idx]
                    if data.val_mask.sum() >= data.y[env_idx].shape[0]:

                        Z_test = Z_val[data.val_mask][torch.randperm(data.val_mask.sum())][:(data.y[env_idx].shape[0])]
                    else:

                        Z_test = Z_val[torch.randperm(Z_val.shape[0])][:(data.y[env_idx].shape[0])]
                    # Z_train = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]])
                    # edge_env_mask = env_idx[data.edge_index[0]] | env_idx[data.edge_index[1]]
                    # Z_test = torch_geometric.utils.to_dense_adj(data.edge_index[:,env_idx_2[data.edge_index[0]] | env_idx_2[data.edge_index[1]]])
                    if config.dataset.num_classes == 1:
                        config.dataset.num_classes = 2
                    label_balance_constraints = torch.eye(config.dataset.num_classes)[data.y[env_idx].long().squeeze()].T.double().cpu().detach().numpy()  #data.y.unique().shape[0]
                    kmm_weight_env, MMD_dist = KMM(Z_train, Z_test, config, label_balance_constraints, beta=0.2)
                    self.kmm_weight[[env_idx.nonzero().squeeze()]] = torch.from_numpy(kmm_weight_env).float().to(device=config.device).squeeze()

        return data, targets, mask, node_norm

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get feature representations

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        self.feat = model_output[1]
        return model_output[0]

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on SRGNN algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({device: torch.device('cuda'),
                                   dataset: {num_envs: int(10)},
                                   ood: {ood_param: float(0.1)}
                                   })


        Returns (Tensor):
            loss based on SRGNN algorithm

        """
        SRloss_list = []

        for i in range(config.dataset.num_envs):
            env_idx_1 = data.env_id == i
            env_feat_1 = self.feat[env_idx_1]
            if env_feat_1.shape[0] > 1:
                if data.val_mask.sum() >= data.y[env_idx_1].shape[0]:
                    env_feat_2 = self.feat[data.val_mask][torch.randperm(data.val_mask.sum())][:(data.y[env_idx_1].shape[0])]
                else:
                    env_feat_2 = self.feat[torch.randperm(self.feat.shape[0])][:(data.y[env_idx_1].shape[0])]

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
