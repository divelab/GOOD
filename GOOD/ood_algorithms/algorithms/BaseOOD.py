from abc import ABC

from GOOD.utils.config_reader import Union, CommonArgs, Munch


class BaseOODAlg(ABC):
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.mean_loss = None
        self.spec_loss = None

    def input_preprocess(self, data, targets, mask, node_norm, training, config: Union[CommonArgs, Munch], **kwargs):
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output, **kwargs):
        return model_output

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config: Union[CommonArgs, Munch]):
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
        return loss

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss
