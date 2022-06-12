import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor


@register.model_register
class Mixup_vGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Mixup_vGIN, self).__init__(config)
        self.encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        ood_algorithm = kwargs.get('ood_algorithm')
        out_readout = self.encoder(*args, **kwargs)

        if self.training:
            lam = ood_algorithm.lam
            out_readout = lam * out_readout + (1 - lam) * out_readout[ood_algorithm.id_a2b]

        out = self.classifier(out_readout)
        return out


@register.model_register
class Mixup_GIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Mixup_GIN, self).__init__(config)
        self.encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :param Required[data]: Batch - input data
        :return:
        """
        ood_algorithm = kwargs.get('ood_algorithm')
        out_readout = self.encoder(*args, **kwargs)

        if self.training:
            lam = ood_algorithm.lam
            out_readout = lam * out_readout + (1 - lam) * out_readout[ood_algorithm.id_a2b]

        out = self.classifier(out_readout)
        return out
