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


@register.ood_alg_register
class GSAT(BaseOODAlg):
    r"""
    Implementation of the GSAT algorithm from `"Interpretable and Generalizable Graph Learning via Stochastic Attention
    Mechanism" <https://arxiv.org/abs/2201.12987>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GSAT, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.decay_r = 0.1
        self.decay_interval = config.ood.extra_param[1]
        self.final_r = config.ood.extra_param[2]      # 0.5 or 0.7

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

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions.

        """
        raw_out, self.att, self.edge_att = model_output
        return raw_out

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
        att = self.att
        eps = 1e-6
        r = self.get_r(self.decay_interval, self.decay_r, config.train.epoch, final_r=self.final_r)
        info_loss = (att * torch.log(att / r + eps) +
                     (1 - att) * torch.log((1 - att) / (1 - r + eps) + eps)).mean()

        self.mean_loss = loss.mean()
        self.spec_loss = config.ood.ood_param * info_loss
        loss = self.mean_loss + self.spec_loss
        return loss

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
