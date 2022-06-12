import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class GroupDRO(BaseOODAlg):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GroupDRO, self).__init__(config)

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                loss_list.append(loss[env_idx].sum() / mask.sum())
        losses = torch.stack(loss_list)
        group_weights = torch.ones(losses.shape[0], device=config.device)
        group_weights *= torch.exp(config.ood.ood_param * losses)
        group_weights /= group_weights.sum()
        loss = losses @ group_weights
        self.mean_loss = loss
        return loss
