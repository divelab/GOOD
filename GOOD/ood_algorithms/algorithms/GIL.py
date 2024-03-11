"""
Implementation of the GIL algorithm from `"Learning Invariant Graph Representations for Out-of-Distribution Generalization" <https://openreview.net/forum?id=acKK8MQe2xc>`_ paper
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
class GIL(BaseOODAlg):
    r"""
    Implementation of the GIL algorithm from `"Learning Invariant Graph Representations for Out-of-Distribution Generalization" <https://openreview.net/forum?id=acKK8MQe2xc>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GIL, self).__init__(config)
        self.E_infer = None
        self.edge_att = None

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
        raw_out, self.E_infer, self.edge_att = model_output
        return raw_out

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch],
                         **kwargs) -> Tensor:
        r"""
        Process loss based on IGA algorithm

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

        env_grads = []
        for i in range(config.dataset.num_envs):
            env_idx = self.E_infer == i
            if loss[env_idx].shape[0] > 0:
                grad_all = torch.autograd.grad(loss[env_idx].sum() / mask[env_idx].sum(), self.model.parameters(), create_graph=True, allow_unused=True)
                env_grads.append(grad_all)

        self.mean_loss = loss.mean()
        mean_grad = torch.autograd.grad(self.mean_loss, self.model.parameters(), create_graph=True, allow_unused=True)
        # compute trace penalty
        penalty_value = 0
        for grad in env_grads:
            for g, mean_g in zip(grad, mean_grad):
                if g is not None:
                    penalty_value += (g - mean_g).pow(2).sum()


        self.spec_loss = OrderedDict()
        self.spec_loss['IGA'] = config.ood.ood_param * penalty_value
        loss = self.mean_loss + sum(self.spec_loss.values())
        return loss

