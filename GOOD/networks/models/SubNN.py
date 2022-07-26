import itertools

import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from torch.nn.functional import gumbel_softmax
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .ICBaseGNN import GNNBasic
from .SubGINs import GINFeatExtractor
from .SubvGINs import vGINFeatExtractor
from typing import Tuple

@register.model_register
class Sub_vGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = vGINFeatExtractor(config)
        num_feat = 1 * config.model.dim_hidden
        self.classifier = nn.Sequential(*(
            [nn.Linear(num_feat, config.dataset.num_classes)]
        ))

        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.config = config

    def forward(self, *args, **kwargs) -> Tuple:
        layer_feat, out_readout, virtual_node_feat = self.feature_extractor(*args, **kwargs)

        out = self.classifier(out_readout)
        return out, layer_feat, out_readout, virtual_node_feat

@register.model_register
class Sub_GIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = GINFeatExtractor(config)

        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, config.dataset.num_classes)]
        ))

        self.config = config

    def forward(self, *args, **kwargs) -> Tuple:
        layer_feat, out_readout = self.feature_extractor(*args, **kwargs)

        out = self.classifier(out_readout)
        return out, layer_feat, out_readout



