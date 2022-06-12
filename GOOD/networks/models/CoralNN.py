import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor


@register.model_register
class Coral_vGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral_vGIN, self).__init__(config)
        self.encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :return:
        """
        out_readout = self.encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out, out_readout


@register.model_register
class Coral_GIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(Coral_GIN, self).__init__(config)
        self.encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        :return:
        """
        out_readout = self.encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out, out_readout
