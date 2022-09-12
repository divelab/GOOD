"""
Implementation of the DIR algorithm from `"Discovering Invariant Rationales for Graph Neural Networks" <https://openreview.net/pdf?id=hGXij5rfiHw>`_ paper
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
    r"""
    Implementation of the DIR algorithm from `"Discovering Invariant Rationales for Graph Neural Networks"
    <https://openreview.net/pdf?id=hGXij5rfiHw>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DIR, self).__init__(config)
        self.rep_out = None
        self.causal_out = None
        self.conf_out = None

    def stage_control(self, config: Union[CommonArgs, Munch]):
        r"""
        Set valuables before each epoch. Largely used for controlling multi-stage training and epoch related parameter
        settings.

        Args:
            config: munchified dictionary of args.

        """
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
        config.train.alpha = config.ood.extra_param[0] * (config.train.epoch ** 1.6)

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        if isinstance(model_output, tuple):
            self.rep_out, self.causal_out, self.conf_out = model_output
        else:
            self.causal_out = model_output
            self.rep_out, self.conf_out = None, None
        return self.causal_out

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor,
                       config: Union[CommonArgs, Munch]) -> Tensor:
        r"""
        Calculate loss based on DIR algorithm

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
            loss based on DIR algorithm

        """

        if self.rep_out is not None:
            causal_loss = (config.metric.loss_func(raw_pred, targets, reduction='none') * mask).sum() / mask.sum()
            conf_loss = (config.metric.loss_func(self.conf_out, targets, reduction='none') * mask).sum() / mask.sum()

            env_loss = torch.tensor([]).to(config.device)
            for rep in self.rep_out:
                tmp = (config.metric.loss_func(rep, targets, reduction='none') * mask).sum() / mask.sum()
                env_loss = torch.cat([env_loss, (tmp.sum() / mask.sum()).unsqueeze(0)])
            causal_loss += config.train.alpha * env_loss.mean()
            env_loss = config.train.alpha * torch.var(env_loss * self.rep_out.size(0))

            loss = causal_loss + env_loss + conf_loss
            self.mean_loss = causal_loss
            self.spec_loss = env_loss + conf_loss
        else:
            causal_loss = (config.metric.loss_func(raw_pred, targets, reduction='none') * mask).sum() / mask.sum()

            loss = causal_loss
            self.mean_loss = causal_loss

        return loss

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss based on DIR algorithm

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
        return loss
