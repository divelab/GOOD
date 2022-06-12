import torch

from GOOD import register
from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.config_reader import Union, CommonArgs, Munch


@register.ood_alg_register
class VREx(BaseOODAlg):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(VREx, self).__init__(config)

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                loss_list.append(loss[env_idx].sum() / mask[env_idx].sum())
        spec_loss = config.ood.ood_param * torch.var(torch.tensor(loss_list, device=config.device))
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss
