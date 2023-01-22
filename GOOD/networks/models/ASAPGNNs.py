r"""
The implementation of `Discovering Invariant Rationales for Graph Neural Networks <https://openreview.net/pdf?id=hGXij5rfiHw>`_.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import ASAPooling
from torch_geometric.nn.conv import MessagePassing

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor


@register.model_register
class ASAPGIN(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(ASAPGIN, self).__init__()
        self.pool = ASAPooling(config.model.dim_hidden, config.ood.ood_param, dropout=config.model.dropout_rate)

        orig_model_layer = config.model.model_layer
        config.model.model_layer = orig_model_layer - 2
        self.sub_encoder = GINFeatExtractor(config)

        config.model.model_layer = 2
        self.gnn = GINFeatExtractor(config, without_embed=True)
        config.model.model_layer = orig_model_layer

        self.classifier = Classifier(config)
        self.config = config

        self.edge_mask = None

    def forward(self, *args, **kwargs):
        h = self.sub_encoder.get_node_repr(*args, **kwargs)
        data = kwargs.get('data')
        pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch, perm = self.pool(h, data.edge_index,
                                                                                        batch=data.batch)
        # col, row = data.edge_index
        # node_mask = torch.zeros(data.x.size(0)).to(self.config.device)
        # node_mask[perm] = 1
        # edge_mask = node_mask[col] * node_mask[row]

        pooled_data = Batch(x=pooled_x,
                            edge_index=pooled_edge_index,
                            edge_attr=None,
                            batch=pooled_batch)
        set_masks(pooled_edge_weight, self.gnn)
        out_readout = self.gnn(data=pooled_data)
        clear_masks(self.gnn)
        self.edge_mask = pooled_edge_index
        pred = self.classifier(out_readout)
        return pred


@register.model_register
class ASAPvGIN(ASAPGIN):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(ASAPvGIN, self).__init__(config)
        orig_model_layer = config.model.model_layer
        config.model.model_layer = orig_model_layer - 2
        self.sub_encoder = vGINFeatExtractor(config)

        config.model.model_layer = 2
        self.gnn = vGINFeatExtractor(config, without_embed=True)
        config.model.model_layer = orig_model_layer


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Adopted from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None
