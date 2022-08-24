import itertools

import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from torch.nn.functional import gumbel_softmax
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .ICBaseGNN import GNNBasic
from .SGGINs import GINFeatExtractor
from .SGvGINs import vGINFeatExtractor
from typing import Tuple
from GOOD.utils.train import at_stage

@register.model_register
class SG_vGIN(GNNBasic):

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
class SG_GIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = GINFeatExtractor(config)

        self.classifier = nn.Sequential(*(
            [nn.Linear(config.model.dim_hidden, config.dataset.num_classes)]
        ))

        self.config = config

    def forward(self, *args, **kwargs) -> Tuple:
        layer_feat, out_readout = self.feature_extractor(*args, **kwargs)

        if at_stage(2, self.config) or at_stage(3, self.config):
            part_outs = []
            for part_readout in out_readout:
                part_outs.append(self.classifier(part_readout))

        out = self.classifier(out_readout)
        return out, layer_feat, out_readout



