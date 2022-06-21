"""
GCN implementation of the DANN algorithm from `"Domain-Adversarial Training of Neural Networks"
<https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper
"""
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GCNs import GCNFeatExtractor


@register.model_register
class DANN_GCN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Domain-Adversarial Training of Neural Networks"
    <https://www.jmlr.org/papers/volume17/15-239/15-239.pdf>`_ paper and `"Semi-supervised Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.num_envs`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

        self.dc = nn.Linear(config.model.dim_hidden, config.dataset.num_envs)
        self.config = config

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        The DANN-GCN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, domain predictions]

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        dc_out = GradientReverseLayerF.apply(out_readout, self.config.train.alpha)
        dc_out = self.dc(dc_out)

        out = self.classifier(out_readout)
        return out, dc_out


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
