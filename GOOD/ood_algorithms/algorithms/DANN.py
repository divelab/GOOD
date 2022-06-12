import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class DANN(BaseOODAlg):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DANN, self).__init__(config)
        self.dc_pred = None

    def output_postprocess(self, model_output, **kwargs):
        self.dc_pred = model_output[1]
        return model_output[0]

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        if config.model.model_level == 'node':
            dc_loss: torch.Tensor = config.metric.cross_entropy_with_logit(self.dc_pred[data.train_mask],
                                                             data.env_id[data.train_mask], reduction='none')
        else:
            dc_loss: torch.Tensor = config.metric.cross_entropy_with_logit(self.dc_pred, data.env_id, reduction='none')
        # else:
        # dc_loss: torch.Tensor = binary_cross_entropy_with_logits(dc_pred, torch.nn.functional.one_hot(data.env_id % config.dataset.num_envs, num_classes=config.dataset.num_envs).float(), reduction='none') * mask
        spec_loss = config.ood.ood_param * dc_loss.mean()
        mean_loss = loss.sum() / mask.sum()
        loss = mean_loss + spec_loss
        self.mean_loss = mean_loss
        self.spec_loss = spec_loss
        return loss
