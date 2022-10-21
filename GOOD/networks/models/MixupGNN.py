"""
GIN and GIN-virtual implementation of the Mixup algorithm from `"Mixup for Node and Graph Classification"
<https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper
"""
import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor


@register.model_register
class Mixup_GIN(GNNBasic):
    r"""
    The Graph Neural Network modified from the `"Mixup for Node and Graph Classification"
    <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper and `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Mixup_GIN, self).__init__(config)
        self.encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The Mixup-GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): (1) dictionary of OOD args (:obj:`kwargs.ood_algorithm`) (2) key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        ood_algorithm = kwargs.get('ood_algorithm')
        out_readout = self.encoder(*args, **kwargs)

        if self.training:
            lam = ood_algorithm.lam
            out_readout = lam * out_readout + (1 - lam) * out_readout[ood_algorithm.id_a2b]

        out = self.classifier(out_readout)
        return out


@register.model_register
class Mixup_vGIN(Mixup_GIN):
    r"""
        The Graph Neural Network modified from the `"Mixup for Node and Graph Classification"
        <https://dl.acm.org/doi/abs/10.1145/3442381.3449796>`_ paper and `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Mixup_vGIN, self).__init__(config)
        self.encoder = vGINFeatExtractor(config)
