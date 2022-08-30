"""
GCN implementation of the SRGNN algorithm from `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
<https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function
import numpy as np
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor


@register.model_register
class SRGCN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data"
    <https://proceedings.neurips.cc/paper/2021/hash/eb55e369affa90f77dd7dc9e2cd33b16-Abstract.html>`_ paper and `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The SRGCN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, features]

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out, out_readout



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