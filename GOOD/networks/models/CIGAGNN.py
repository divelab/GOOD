"""
Implementation of the CIGA algorithm from `"Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs"
<https://arxiv.org/abs/2202.05441>`_ paper

Copied from https://github.com/LFhase/GOOD.
"""

import copy
import math
from GOOD.networks.models.Pooling import GlobalAddPool

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .GINvirtualnode import vGINFeatExtractor
from .GINs import GINFeatExtractor
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_geometric.nn import global_add_pool


@register.model_register
class CIGAGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIGAGIN, self).__init__(config)
        self.att_net = GAEAttNet(config.ood.ood_param, config)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = GINFeatExtractor(config_fe, without_embed=True)

        self.num_tasks = config.dataset.num_classes
        self.causal_lin = torch.nn.Linear(config.model.dim_hidden, self.num_tasks)
        self.spu_lin = torch.nn.Linear(config.model.dim_hidden, self.num_tasks)

        self.contrast_rep = "feat"
        if type(config.ood.extra_param[-1]) == str:
            self.contrast_rep = config.ood.extra_param[-1]

    def forward(self, *args, **kwargs):
        r"""
        The CIGA model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        data = kwargs.get('data')
        batch_size = data.batch[-1].item() + 1
        # data.edge_index, data.edge_attr = add_self_loops(*remove_self_loops(data.edge_index, data.edge_attr), num_nodes=data.x.shape[0])

        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
        (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch), \
        pred_edge_weight, node_h, orig_x = self.att_net(*args, **kwargs)


        if self.contrast_rep == "raw":
            causal_x, _, __, ___ = relabel(orig_x, causal_edge_index, data.batch)
            spu_x, _, __, ___ = relabel(orig_x, spu_edge_index, data.batch)

        # --- Causal repr ---
        set_masks(causal_edge_weight, self)
        causal_rep = self.get_graph_rep(
            data=Data(x=causal_x, edge_index=causal_edge_index,
                      edge_attr=causal_edge_attr, batch=causal_batch),
            batch_size=batch_size
        )
        causal_out = self.get_causal_pred(causal_rep)
        clear_masks(self)

        
        if self.training:
            # --- Conf repr ---
            set_masks(spu_edge_weight, self)
            spu_rep = self.get_graph_rep(
                data=Data(x=spu_x, edge_index=spu_edge_index,
                          edge_attr=spu_edge_attr, batch=spu_batch),
                batch_size=batch_size
            )#.detach()
            spu_out = self.get_spu_pred(spu_rep)
            clear_masks(self)
            # if self.contrast_rep == "feat":
            #     causal_h, _, __, ___ = relabel(node_h, causal_edge_index, data.batch)
            # if self.contrast_rep == "raw":
            #     causal_x, _, __, ___ = relabel(orig_x, causal_edge_index, data.batch)
            causal_rep_out = global_add_pool(causal_x, batch=causal_batch, size=batch_size)
            # print(data)
            # print(causal_x.size(),causal_edge_index.size(),causal_rep.size())
            # print(spu_x.size(),spu_edge_index.size(),spu_rep.size())
            # print(causal_h.size(),causal_rep_out.size())
            # print("+++++++++++++++++++++++++++=")
            return causal_rep_out, causal_out, spu_out
        else:

            return causal_out

    def get_graph_rep(self, *args, **kwargs):
        return self.feat_encoder(*args, **kwargs)

    def get_causal_pred(self, h_graph):
        return self.causal_lin(h_graph)

    def get_spu_pred(self, spu_graph_x):
        return self.spu_lin(spu_graph_x)

    def get_comb_pred(self, causal_graph_x, spu_graph_x):
        causal_pred = self.causal_lin(causal_graph_x)
        spu_pred = self.spu_lin(spu_graph_x).detach()
        return torch.sigmoid(spu_pred) * causal_pred

@register.model_register
class CIGAvGINNC(CIGAGIN):
    """
    using a simple GNN to encode spurious subgraph
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIGAvGINNB, self).__init__(config)
        self.att_net = GAEAttNet(config.ood.ood_param, config, virtual_node=True, no_bn=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vGINFeatExtractor(config_fe, without_embed=True)
        spu_gnn_config = copy.deepcopy(config_fe)
        spu_gnn_config.model.model_layer = 1
        self.spu_gnn = vGINFeatExtractor(spu_gnn_config, without_embed=True)
    def get_spu_pred(self, spu_graph_x):
        return self.spu_gnn(spu_graph_x)

@register.model_register
class CIGAvGIN(CIGAGIN):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIGAvGIN, self).__init__(config)
        self.att_net = GAEAttNet(config.ood.ood_param, config, virtual_node=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vGINFeatExtractor(config_fe, without_embed=True)
    
@register.model_register
class CIGAvGINNB(CIGAGIN):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(CIGAvGINNB, self).__init__(config)
        self.att_net = GAEAttNet(config.ood.ood_param, config, virtual_node=True, no_bn=True)
        config_fe = copy.deepcopy(config)
        config_fe.model.model_layer = config.model.model_layer - 2
        self.feat_encoder = vGINFeatExtractor(config_fe, without_embed=True)


class GAEAttNet(nn.Module):

    def __init__(self, causal_ratio, config, **kwargs):
        super(GAEAttNet, self).__init__()
        config_catt = copy.deepcopy(config)
        config_catt.model.model_layer = 2
        config_catt.model.dropout_rate = 0
        if kwargs.get('virtual_node'):
            self.gnn_node = vGINFeatExtractor(config_catt, without_readout=True, **kwargs)
        else:
            self.gnn_node = GINFeatExtractor(config_catt, without_readout=True, **kwargs)
        self.linear = nn.Linear(config_catt.model.dim_hidden * 2, 1)
        self.ratio = causal_ratio

    def forward(self, *args, **kwargs):
        data = kwargs.get('data') or None

        # x are last layer node representations
        node_h = self.gnn_node(*args, **kwargs)

        row, col = data.edge_index
        edge_rep = torch.cat([node_h[row], node_h[col]], dim=-1)
        edge_score = self.linear(edge_rep).view(-1)

        if data.edge_index.shape[1] != 0:
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
            (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(data, edge_score, self.ratio)

            causal_x, causal_edge_index, causal_batch, _ = relabel(node_h, causal_edge_index, data.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(node_h, spu_edge_index, data.batch)
        else:
            causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch = \
                node_h, data.edge_index, data.edge_attr, \
                float('inf') * torch.ones(data.edge_index.shape[1], device=data.x.device), \
                data.batch
            spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch = None, None, None, None, None

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
               (spu_x, spu_edge_index, spu_edge_attr, spu_edge_weight, spu_batch), \
               edge_score, node_h, data.x


def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module._explain = True
            module.__edge_mask__ = mask
            module._edge_mask = mask


def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module._explain = False
            module.__edge_mask__ = None
            module._edge_mask = None


def split_graph(data, edge_score, ratio):
    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None


    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_spu_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_spu_edge_weight = - edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_spu_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_spu_edge_attr = None

    return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
           (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)



def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges


def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos


def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    r'''
    Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
    '''
    f_src = src.float()
    f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
    perm = norm.argsort(dim=dim, descending=descending)

    return src[perm], perm


def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
    rank, perm = sparse_sort(src, index, dim, descending, eps)
    num_nodes = degree(index, dtype=torch.long)
    k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
    start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
    mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
    mask = torch.cat(mask, dim=0)
    mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
    topk_perm = perm[mask]
    exc_perm = perm[~mask]

    return topk_perm, exc_perm, rank, perm, mask
