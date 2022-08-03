
import torch
import torch.nn as nn
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINEncoder, GINMolEncoder
from .Pooling import GlobalAddPool, GlobalMaxPool
from GOOD.utils.args import args_parser
from GOOD import config_summoner
from torch_geometric.typing import OptTensor
from torch_geometric.nn.conv import MessagePassing

@register.model_register
class DIRvGIN(GNNBasic):

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(vGIN, self).__init__(config)
        self.feat_encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

        self.num_layers = config.model.model_layer
        self.dropout = config.model.dropout_rate
        self.emb_dim = config.model.dim_hidden
        self.num_tasks = config.dataset.num_classes
        self.d_out = self.num_tasks

        # if self.num_layers < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        # GNN to generate node embeddings
        # self.gnn_node = GINVirtual_node(num_layers, emb_dim, dropout=dropout, encode_node=encode_node)

        # Pooling function to generate whole-graph embeddings
        self.pool = GlobalMaxPool
        self.causal_lin = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.conf_lin = torch.nn.Linear(self.emb_dim, self.num_tasks)
        self.cq = torch.nn.Linear(self.num_tasks, self.num_tasks)

        self.att_net = CausalAttNet()

    # def forward(self, *args, **kwargs) -> torch.Tensor:
    #     r"""
    #     The vGIN model implementation.
    #
    #     Args:
    #         *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
    #         **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
    #
    #     Returns (Tensor):
    #         label predictions
    #
    #     """
    #     out_readout = self.feat_encoder(*args, **kwargs)
    #
    #     out = self.classifier(out_readout)
    #     return out

    def forward(self, *args, **kwargs):
        if self.training:

            (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), pred_edge_weight = self.att_net(data)

            set_masks(causal_edge_weight, self)
            causal_rep = self.get_graph_rep(
                x=causal_x, edge_index=causal_edge_index,
                edge_attr=causal_edge_attr, batch=causal_batch)
            causal_out = self.get_causal_pred(causal_rep)
            clear_masks(self)
            set_masks(conf_edge_weight, self)
            conf_rep = self.get_graph_rep(
                x=conf_x, edge_index=conf_edge_index,
                edge_attr=conf_edge_attr, batch=conf_batch).detach()
            conf_out = self.get_conf_pred(conf_rep)

            rep_out = torch.zeros(conf_rep.shape[0], self.num_tasks)
            i = 0
            for conf in conf_rep:
                rep_out[i] = self.get_comb_pred(causal_rep, conf)
                i = i+1

            return rep_out, causal_out, conf_out
        else:
            h_graph = self.get_graph_rep(self, *args, **kwargs)
            return self.get_causal_pred(h_graph)

    def get_graph_rep(self, *args, **kwargs):
        h_node = self.feat_encoder(*args, **kwargs)
        h_graph = self.pool(h_node, batch)
        return h_graph

    def get_causal_pred(self, h_graph):
        return self.causal_lin(h_graph)

    def get_conf_pred(self, conf_graph_x):
        return self.conf_lin(conf_graph_x)

    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_lin(causal_graph_x)
        conf_pred = self.conf_lin(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred



class CausalAttNet(nn.Module):

    def __init__(self):
        super(CausalAttNet, self).__init__()
        args = args_parser()
        config_catt = config_summoner(args)
        config_catt.model.model_layer = 2
        config_catt.model.dropout_rate = 0
        self.gnn_node = vGINFeatExtractor(config_catt)
        self.linear = nn.Linear(config.model.dim_hidden * 2, 1)
        self.ratio = config.train.causal_ratio

    def forward(self, data):
        x = self.gnn_node(data)

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        pred_edge_weight = self.linear(edge_rep).view(-1)
        causal_edge_index = torch.LongTensor([[], []]).to(x.device)
        causal_edge_weight = torch.tensor([]).to(x.device)
        causal_edge_attr = torch.LongTensor([]).to(x.device)
        conf_edge_index = torch.LongTensor([[], []]).to(x.device)
        conf_edge_weight = torch.tensor([]).to(x.device)
        conf_edge_attr = torch.LongTensor([]).to(x.device)

        edge_indices, _, _, num_edges, cum_edges = split_batch(data)
        for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
            n_reserve = int(self.ratio * N)
            edge_attr = data.edge_attr[C:C + N]
            single_mask = pred_edge_weight[C:C + N]
            single_mask_detach = pred_edge_weight[C:C + N].detach().cpu().numpy()
            rank = np.argpartition(-single_mask_detach, n_reserve)
            idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

            causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
            conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)
            causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
            conf_edge_weight = torch.cat([conf_edge_weight, -1 * single_mask[idx_drop]])
            causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
            conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
               (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), \
               pred_edge_weight


def set_masks(mask: Tensor, model: nn.Module):
        for module in model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = mask


def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = torch_geometric.utils.degree(g.batch, dtype=torch.long)
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


class vGINFeatExtractor(GNNBasic):
    r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(vGINFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = vGINMolEncoder(config)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config)
            self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch = self.arguments_read(*args, **kwargs)
            out_readout = self.encoder(x, edge_index, edge_attr, batch)
        else:
            x, edge_index, batch = self.arguments_read(*args, **kwargs)
            out_readout = self.encoder(x, edge_index, batch)
        return out_readout


class VirtualNodeEncoder(torch.nn.Module):
    r"""
        The virtual node feature encoder for vGIN.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.dropout_rate`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super(VirtualNodeEncoder, self).__init__()
        self.virtual_node_embedding = nn.Embedding(1, config.model.dim_hidden)
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                 nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU()] +
                [nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden),
                 nn.BatchNorm1d(config.model.dim_hidden), nn.ReLU(),
                 nn.Dropout(config.model.dropout_rate)]
        ))
        self.virtual_pool = GlobalAddPool()


class vGINEncoder(GINEncoder, VirtualNodeEncoder):
    r"""
    The vGIN encoder for non-molecule data, using the :class:`~vGINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(vGINEncoder, self).__init__(config)
        self.config = config

    def forward(self, x, edge_index, batch):
        r"""
        The vGIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch)
        return out_readout


class vGINMolEncoder(GINMolEncoder, VirtualNodeEncoder):
    r"""The vGIN encoder for molecule data, using the :class:`~vGINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(vGINMolEncoder, self).__init__(config)
        self.config = config

    def forward(self, x, edge_index, edge_attr, batch):
        r"""
        The vGIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))

        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = post_conv + virtual_node_feat[batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch) + virtual_node_feat)

        out_readout = self.readout(post_conv, batch)
        return out_readout
