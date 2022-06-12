import torch
from torch.autograd import grad

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class IRM(BaseOODAlg):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(IRM, self).__init__(config)
        self.dummy_w = torch.nn.Parameter(torch.Tensor([1.0])).to(config.device)

    def output_postprocess(self, model_output, **kwargs):
        raw_pred = self.dummy_w * model_output
        return raw_pred

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        spec_loss_list = []
        for i in range(config.dataset.num_envs):
            env_idx = data.env_id == i
            if loss[env_idx].shape[0] > 0:
                grad_all = torch.sum(
                    grad(loss[env_idx].sum() / mask[env_idx].sum(), self.dummy_w, create_graph=True)[0].pow(2))
                spec_loss_list.append(grad_all)
        spec_loss = config.ood.ood_param * sum(spec_loss_list) / len(spec_loss_list)
        if torch.isnan(spec_loss):
            spec_loss = 0
        mean_loss = loss.sum() / mask.sum()
        loss = spec_loss + mean_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss
