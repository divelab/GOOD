"""
GIN and GIN-virtual implementation of the Deep Coral algorithm from `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
<https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper
"""
from typing import Tuple

import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor


@register.model_register
class Coral_GIN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
    <https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper and `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral_GIN, self).__init__(config)
        self.encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The Deep Coral-GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            [label predictions, features]

        """
        out_readout = self.encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out, out_readout


@register.model_register
class Coral_vGIN(Coral_GIN):
    r"""
        The Graph Neural Network modified from the `"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"
        <https://link.springer.com/chapter/10.1007/978-3-319-49409-8_35>`_ paper and `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral_vGIN, self).__init__(config)
        self.encoder = vGINFeatExtractor(config)
