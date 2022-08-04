"""
Implementation of the IRM algorithm from `"Invariant Risk Minimization"
<https://arxiv.org/abs/1907.02893>`_ paper
"""
from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.data import Batch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.train import at_stage
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class DIR(BaseOODAlg):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIR, self).__init__(config)



    def stage_control(self, config: Union[CommonArgs, Munch]):
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
        config.train.alpha = config.ood.ood_param * (config.train.epoch ** 1.6)

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
        if isinstance(raw_pred, tuple):
            rep_out, causal_out, conf_out = raw_pred
            causal_loss = (config.metric.loss_func(causal_out, targets, reduction='none') * mask).sum() / mask.sum()
            conf_loss = (config.metric.loss_func(conf_out, targets, reduction='none') * mask).sum() / mask.sum()

            env_loss = torch.tensor([]).to(config.device)
            for rep in rep_out:
                tmp = (config.metric.loss_func(rep, targets, reduction='none') * mask).sum() / mask.sum()
                env_loss = torch.cat([env_loss, (tmp.sum() / mask.sum()).unsqueeze(0)])
            causal_loss += config.train.alpha * env_loss.mean()
            env_loss = config.train.alpha * torch.var(env_loss * rep_out.size(0))

            loss = causal_loss + env_loss + conf_loss
            self.mean_loss = causal_loss
            self.spec_loss = env_loss + conf_loss
        else:
            causal_out = raw_pred
            causal_loss = (config.metric.loss_func(causal_out, targets, reduction='none') * mask).sum() / mask.sum()

            loss = causal_loss
            self.mean_loss = causal_loss

        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        return loss
