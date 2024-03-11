r"""
Implementation of the GIL algorithm from `"Learning Invariant Graph Representations for Out-of-Distribution Generalization" <https://openreview.net/forum?id=acKK8MQe2xc>`_.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import InstanceNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import is_undirected
from torch_sparse import transpose

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from GOOD.utils.fast_pytorch_kmeans import KMeans


@register.model_register
class GILGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GILGIN, self).__init__(config)
        self.gnn = GINFeatExtractor(config)
        self.gnn_i = GINFeatExtractor(config)
        self.gnn_v = GINFeatExtractor(config)

        self.classifier_i = Classifier(config)
        self.config = config

        self.top_t = self.config.ood.extra_param[0]
        self.num_env = self.config.ood.extra_param[1]

        self.edge_mask = None

    def forward(self, *args, **kwargs):
        r"""
        The GIL model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            Label predictions and other results for loss calculations.

        """
        data = kwargs.get('data')
        emb = self.gnn(*args, without_readout=True, **kwargs)
        col, row = data.edge_index
        f1, f2 = emb[col], emb[row]
        edge_att = (f1 * f2).sum(-1)
        hard_edge_att = self.control_sparsity(edge_att, top_t=self.top_t)

        set_masks(hard_edge_att, self)
        logits = self.classifier_i(self.gnn_i(*args, **kwargs))
        clear_masks(self)

        set_masks(1 - hard_edge_att, self)
        H = self.gnn_v(*args, **kwargs)
        clear_masks(self)

        kmeans = KMeans(n_clusters=self.num_env, n_init=10, device=H.device).fit(H)
        E_infer = kmeans.labels_
        self.edge_mask = edge_att
        return logits, E_infer, edge_att

    def control_sparsity(self, mask, top_t=None):
        r"""

        :param mask: mask that need to transform
        :param top_t: sparsity we need to control i.e. 0.7, 0.5
        :return: transformed mask where top 1 - sparsity values are set to inf.
        """
        _, indices = torch.sort(mask, descending=True)
        mask_len = mask.shape[0]
        split_point = int(top_t * mask_len)
        important_indices = indices[: split_point]
        unimportant_indices = indices[split_point:]
        trans_mask = mask.clone()
        trans_mask[important_indices] = 1.
        trans_mask[unimportant_indices] = 0.

        return trans_mask

    def sampling(self, att_log_logits, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att

    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern


@register.model_register
class GILvGIN(GILGIN):
    r"""
    The GIN virtual node version of GSAT.
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GILvGIN, self).__init__(config)
        self.gnn = vGINFeatExtractor(config)
        self.gnn_i = vGINFeatExtractor(config)
        self.gnn_v = vGINFeatExtractor(config)

        self.classifier_i = Classifier(config)
        self.classifier_v = Classifier(config)


class ExtractorMLP(nn.Module):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__()
        hidden_size = config.model.dim_hidden
        self.learn_edge_att = config.ood.extra_param[0]  # learn_edge_att
        dropout_p = config.model.dropout_rate

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits


class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                # m.append(nn.BatchNorm1d(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)


def set_masks(mask: Tensor, model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module._apply_sigmoid = False
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    r"""
    Modified from https://github.com/wuyxin/dir-gnn.
    """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None
