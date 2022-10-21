"""
GIN and GIN-virtual implementation of the DANN algorithm from `"Domain-Adversarial Training of Neural Networks"
<https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper
"""
import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from typing import Tuple


@register.model_register
class DANN_GIN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Domain-Adversarial Training of Neural Networks"
    <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper and `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.num_envs`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)

        self.dc = nn.Linear(config.model.dim_hidden, config.dataset.num_envs)

        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.graph_repr = None
        self.config = config

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The DANN-GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, domain predictions]

        """
        out_readout = self.encoder(*args, **kwargs)
        self.graph_repr = out_readout

        dc_out = GradientReverseLayerF.apply(out_readout, self.config.train.alpha)
        dc_out = self.dc(dc_out)

        out = self.classifier(out_readout)
        return out, dc_out


@register.model_register
class DANN_vGIN(DANN_GIN):
    r"""
        The Graph Neural Network modified from the `"Domain-Adversarial Training of Neural Networks"
        <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper and `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_envs`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.encoder = vGINFeatExtractor(config)


class GradientReverseLayerF(Function):
    r"""
    Gradient reverse layer for DANN algorithm.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        r"""
        gradient forward propagation

        Args:
            ctx (object): object of the GradientReverseLayerF class
            x (Tensor): feature representations
            alpha (float): the GRL learning rate

        Returns (Tensor):
            feature representations

        """
        ctx.alpha = alpha
        return x.view_as(x)  # * alpha

    @staticmethod
    def backward(ctx, grad_output):
        r"""
        gradient backpropagation step

        Args:
            ctx (object): object of the GradientReverseLayerF class
            grad_output (Tensor): raw backpropagation gradient

        Returns (Tensor):
            backpropagation gradient

        """
        output = grad_output.neg() * ctx.alpha
        return output, None
