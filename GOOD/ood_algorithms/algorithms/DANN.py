"""
Implementation of the DANN algorithm from `"Domain-Adversarial Training of Neural Networks"
<https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper
"""
import torch
from torch import Tensor
from torch_geometric.data import Batch
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseOOD import BaseOODAlg


@register.ood_alg_register
class DANN(BaseOODAlg):
    r"""
    Implementation of the DANN algorithm from `"Domain-Adversarial Training of Neural Networks"
    <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.model_level`, :obj:`config.metric.cross_entropy_with_logit()`, :obj:`config.ood.ood_param`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(DANN, self).__init__(config)
        self.dc_pred = None

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tensor:
        r"""
        Process the raw output of model; get domain classifier predictions

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        self.dc_pred = model_output[1]
        return model_output[0]

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: Union[CommonArgs, Munch], **kwargs) -> Tensor:
        r"""
        Process loss based on DANN algorithm

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.model_level`, :obj:`config.metric.cross_entropy_with_logit()`, :obj:`config.ood.ood_param`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {cross_entropy_with_logit()},
                                   ood: {ood_param: float(0.1)}
                                   })

        Returns (Tensor):
            loss based on DANN algorithm

        """
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
