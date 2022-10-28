"""
Implementation of the GSAT algorithm from `"Interpretable and Generalizable Graph Learning via Stochastic Attention Mechanism" <https://arxiv.org/abs/2201.12987>`_ paper
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
from collections import OrderedDict


@register.ood_alg_register
class GEI(BaseOODAlg):
    r"""
    Implementation of the GSAT algorithm from `"Interpretable and Generalizable Graph Learning via Stochastic Attention
    Mechanism" <https://arxiv.org/abs/2201.12987>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GEI, self).__init__(config)
        self.la_out, self.ec_out, self.ea_out, self.ef_out = None, None, None, None
        self.att = None
        self.edge_att = None
        self.targets = None

        self.IF = config.ood.ood_param
        self.LA = config.ood.extra_param[0]
        self.EC = config.ood.extra_param[1]
        self.EA = config.ood.extra_param[2]
        self.decay_r = 0.1
        self.decay_interval = config.ood.extra_param[3]
        self.EF = config.ood.extra_param[4]
        # self.final_r = config.ood.extra_param[2]      # 0.5 or 0.7

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        (raw_out, self.la_out, self.ec_out, self.ea_out, self.ef_out), self.att, self.edge_att = model_output
        return raw_out


    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss

        Args:
            raw_pred (Tensor): model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func: Accuracy}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        self.targets = targets
        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss based on GSAT algorithm

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
            loss based on DIR algorithm

        """
        if config.dataset.dataset_name == 'GOODHIV' and getattr(data, 'domain_id') is not None:
            data.env_id = data.domain_id

        self.spec_loss = OrderedDict()
        if self.EF:
            self.spec_loss['EF'] = config.metric.cross_entropy_with_logit(self.ef_out, data.env_id, reduction='mean')

        if self.LA:
            self.spec_loss['LA'] = (config.metric.loss_func(self.la_out, self.targets,
                                                            reduction='none') * mask).sum() / mask.sum()
        if self.EC:
            self.spec_loss['EC'] = self.EC * config.metric.cross_entropy_with_logit(self.ec_out, data.env_id, reduction='mean')

        if self.EA:
            self.spec_loss['EA'] = config.metric.cross_entropy_with_logit(self.ea_out, data.env_id, reduction='mean')

        if self.IF:
            att = self.att
            eps = 1e-6
            r = self.get_r(self.decay_interval, self.decay_r, config.train.epoch)
            info_loss = (att * torch.log(att / r + eps) +
                         (1 - att) * torch.log((1 - att) / (1 - r + eps) + eps)).mean()
            self.spec_loss['IF'] = self.IF * info_loss
        self.mean_loss = loss.sum() / mask.sum()
        loss = self.mean_loss + sum(self.spec_loss.values())
        return loss

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r



