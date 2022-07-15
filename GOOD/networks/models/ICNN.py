import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .ICGINs import GINFeatExtractor
from .ICGINvirtualnode import vGINFeatExtractor
from typing import Tuple

@register.model_register
class ICNN_vGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = vGINFeatExtractor(config)
        self.classifier = Classifier(config)

        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.graph_repr: Tensor
        self.config = config

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        out_readout = self.feature_extractor(*args, **kwargs)
        self.config.ood.ood_alg.layer_feat = self.feature_extractor.encoder.layer_feat
        self.graph_repr = out_readout

        out = self.classifier(out_readout)
        return out