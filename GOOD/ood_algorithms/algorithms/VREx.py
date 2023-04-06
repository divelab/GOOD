"""
Implementation of the VREx algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
<http://proceedings.mlr.press/v139/krueger21a.html>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch


@register.ood_alg_register
class VREx(BaseOODAlg):
    r"""
    Implementation of the VREx algorithm from `"Out-of-Distribution Generalization via Risk Extrapolation (REx)"
    <http://proceedings.mlr.press/v139/krueger21a.html>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(VREx, self).__init__(config)

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on VREx algorithm

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
            loss based on VREx algorithm

        """
        loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
        spec_loss = config.ood.ood_param * torch.var(torch.stack(loss_list))
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss
