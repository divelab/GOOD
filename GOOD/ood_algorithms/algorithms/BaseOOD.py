"""
Base class for OOD algorithms
"""
from abc import ABC

from GOOD.utils.config_reader import Union, CommonArgs, Munch


class BaseOODAlg(ABC):
    r"""
    Base class for OOD algorithms

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(BaseOODAlg, self).__init__()
        self.mean_loss = None
        self.spec_loss = None

    def input_preprocess(self, data, targets, mask, node_norm, training, config: Union[CommonArgs, Munch], **kwargs):
        r"""
        Set input data format and preparations

        Args:
            data (Batch): input data
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            training (bool): whether the task is training
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns:
            [data (Batch) - processed input data,
            targets (Tensor) - processed input labels,
            mask (Tensor) - processed NAN masks for data formats,
            node_norm (Tensor) - processed node weights for normalization]

        """
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output, **kwargs):
        r"""
        Process the raw output of model

        Args:
            model_output (Tensor): model raw output

        Returns (Tensor):
            model raw predictions

        """
        return model_output

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config: Union[CommonArgs, Munch]):
        r"""
        Calculate loss

        Args:
            raw_pred: model predictions
            targets (Tensor): input labels
            mask (Tensor): NAN masks for data formats
            node_norm (Tensor): node weights for normalization (for node prediction only)
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.metric.loss_func()`, :obj:`config.model.model_level`)

        .. code-block:: python

            config = munchify({model: {model_level: str('graph')},
                                   metric: {loss_func()}
                                   })


        Returns (Tensor):
            cross entropy loss

        """
        loss = config.metric.loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config.model.model_level == 'node' else loss
        return loss

    def loss_postprocess(self, loss, data, mask, config: Union[CommonArgs, Munch], **kwargs):
        r"""
        Process loss

        Args:
            loss (Tensor): base loss between model predictions and input labels
            data (Batch): input data
            mask (Tensor): NAN masks for data formats
            config (Union[CommonArgs, Munch]): munchified dictionary of args

        Returns (float):
            processed loss

        """
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss
