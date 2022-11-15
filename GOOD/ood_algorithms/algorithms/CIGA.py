"""
Implementation of the CIGA algorithm from `"Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs"
<https://arxiv.org/abs/2202.05441>`_ paper

Copied from "https://github.com/LFhase/GOOD".
"""
from re import M
import torch
from torch.autograd import grad
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class CIGA(BaseOODAlg):
    r"""
    Implementation of the CIGA algorithm from `"Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs"
    <https://arxiv.org/abs/2202.05441>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIGA, self).__init__(config)
        self.rep_out = None
        self.causal_out = None
        self.spu_out = None
        self.step=0

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; apply the linear classifier

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions with the linear classifier applied

        """
        if isinstance(model_output, tuple):
            self.rep_out, self.causal_out, self.spu_out = model_output
        else:
            self.causal_out = model_output
            self.rep_out, self.spu_out = None, None
        return self.causal_out
    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on Mixup algorithm

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func()}
                                   })


        Returns (Tensor):
            loss based on IRM algorithm

        """
        self.step += 1
        if self.rep_out is not None:
            # print(mask.sum(),self.rep_out.size(),targets.size(),mask.size())
            # print(self.rep_out[mask.view(-1),:].size(),targets[mask].size())
            causal_loss = config.metric.loss_func(raw_pred, targets, reduction='none') 
            spu_loss = config.metric.loss_func(self.spu_out, targets, reduction='none')
            # print(causal_loss.sum(),spu_loss.sum())
            assert self.rep_out.size(0)==targets[mask].size(0), print(mask.sum(),self.rep_out.size(),targets.size(),mask.size())
                # exit()
            cls_loss = (causal_loss * mask).sum() / mask.sum()
            contrast_loss = get_contrast_loss(self.rep_out[mask.view(-1),:],targets[mask.view(-1)].view(-1))
            if len(config.ood.extra_param)>1:
                # hinge loss
                spu_loss_weight = torch.zeros(spu_loss.size()).to(raw_pred.device)
                spu_loss_weight[spu_loss > causal_loss] = 1.0
                spu_loss_weight = spu_loss_weight * mask
                spu_loss = (spu_loss * spu_loss_weight).sum() / (spu_loss_weight.sum() + 1e-6)
                hinge_loss = spu_loss
            else:
                hinge_loss = 0
            # print(cls_loss, contrast_loss)
            if self.step <= -1:
                loss = cls_loss
            else:
                loss = cls_loss + config.ood.extra_param[0] * contrast_loss + \
                            (config.ood.extra_param[1] if len(config.ood.extra_param)>1 else 0) * hinge_loss
            self.mean_loss = cls_loss
            self.spec_loss = contrast_loss + hinge_loss
        else:
            cls_loss = (config.metric.loss_func(raw_pred, targets, reduction='none') * mask).sum() / mask.sum()

            loss = cls_loss
            self.mean_loss = cls_loss

        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        return loss




import copy
from email.policy import default
from enum import Enum
import torch
import argparse
from torch_geometric import data
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F


def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
    loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
    grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    irm_loss = torch.sum(grad_0 * grad_1)

    return irm_loss


def get_contrast_loss(causal_rep, labels, norm=F.normalize, contrast_t=1.0, sampling='mul'):
    if norm != None:
        causal_rep = norm(causal_rep)
    if sampling.lower() in ['mul', 'var']:
        # imitate https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        # print(torch.ones_like(mask).size())
        # print(torch.arange(batch_size * anchor_count).view(-1, 1).to(device).size())
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        # print(graph.y)
        # print(causal_rep)
        # print(logits_mask)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)
        # print(mask.sum(1))
        # compute mean of log-likelihood over positive
        is_valid = mask.sum(1) != 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0
        # print(mean_log_prob_pos)
        # print(mask.sum(1))
        # exit()
        # loss
        # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
        # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
        contrast_loss = -mean_log_prob_pos.mean()
    elif sampling.lower() == 'single':
        N = causal_rep.size(0)
        pos_idx = torch.arange(N)
        neg_idx = torch.randperm(N)
        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    pos_idx[i] = j
                else:
                    neg_idx[i] = j
        contrast_loss = -torch.mean(
            torch.bmm(causal_rep.unsqueeze(1), causal_rep[pos_idx].unsqueeze(1).transpose(1, 2)) -
            torch.matmul(causal_rep.unsqueeze(1), causal_rep[neg_idx].unsqueeze(1).transpose(1, 2)))
        raise Exception("Not implmented contrasting method")
    return contrast_loss
