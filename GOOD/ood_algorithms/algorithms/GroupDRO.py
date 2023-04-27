"""
Implementation of the GroupDRO algorithm from `"Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization"
<https://arxiv.org/abs/1911.08731>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class GroupDRO(BaseOODAlg):
    r"""
    Implementation of the GroupDRO algorithm from `"Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization"
    <https://arxiv.org/abs/1911.08731>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.device`, :obj:`config.dataset.num_envs`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GroupDRO, self).__init__(config)

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on GroupDRO algorithm

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
            loss based on GroupDRO algorithm

        """
        loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0 and mask[env_idx].sum() > 0:
                loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
        losses = torch.stack(loss_list)
        group_weights = torch.ones(losses.shape[0], device=config.device)
        group_weights *= torch.exp(config.ood.ood_param * losses.data)
        group_weights /= group_weights.sum()
        loss = losses @ group_weights
        self.mean_loss = loss
        return loss
