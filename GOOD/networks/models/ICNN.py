import itertools

import torch
import torch.nn as nn
from torch.autograd import Function
from torch import Tensor
from torch.nn.functional import gumbel_softmax
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .ICBaseGNN import GNNBasic
from .ICGINs import GINFeatExtractor
from .ICGINvirtualnode import vGINFeatExtractor
from typing import Tuple

@register.model_register
class ICNN_vGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = vGINFeatExtractor(config)
        # self.classifier = nn.Sequential(*(
        #     [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
        #      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(), nn.Dropout(config.model.dropout_rate)]
        #      + list(itertools.chain(*[[nn.Linear(2 * config.model.dim_hidden, 2 * config.model.dim_hidden),
        #      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(), nn.Dropout(config.model.dropout_rate)]
        #      for _ in range(2)]))
        #      + [nn.Linear(2 * config.model.dim_hidden, config.dataset.num_classes)]
        # ))
        num_feat = 2 * config.model.dim_hidden
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
class ICNN_GIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feature_extractor = GINFeatExtractor(config)
        # self.classifier = nn.Sequential(*(
        #     [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
        #      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(), nn.Dropout(config.model.dropout_rate)]
        #      + list(itertools.chain(*[[nn.Linear(2 * config.model.dim_hidden, 2 * config.model.dim_hidden),
        #      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(), nn.Dropout(config.model.dropout_rate)]
        #      for _ in range(2)]))
        #      + [nn.Linear(2 * config.model.dim_hidden, config.dataset.num_classes)]
        # ))
        num_feat = 1 * config.model.dim_hidden
        self.feature_selection_logits = nn.Parameter(torch.Tensor(num_feat))
        nn.init.constant_(self.feature_selection_logits, val=3.)
        self.feature_selection_mask = None

        self.classifier = nn.Sequential(*(
            [nn.Linear(num_feat, config.dataset.num_classes)]
        ))

        self.dropout = nn.Dropout(config.model.dropout_rate)
        self.config = config

    def forward(self, *args, **kwargs) -> Tuple:
        layer_feat, out_readout = self.feature_extractor(*args, **kwargs)

        if self.config.train.epoch >= self.config.train.stage_stones[0]:
            # Gumbel-sigmoid trick
            self.feature_selection_mask = gumbel_softmax(torch.stack([self.feature_selection_logits,
                                                                 - self.feature_selection_logits]), dim=0)[0]
            out_readout = self.feature_selection_mask * out_readout

        out = self.classifier(out_readout)
        return out, layer_feat, out_readout