r"""
The Graph Neural Network from the `"Neural Message Passing for Quantum Chemistry"
<https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.
"""
import torch
import torch.nn as nn
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .ICBaseGNN import GNNBasic
from .Pooling import GlobalMeanPool, GlobalMaxPool
from .ICGINs import GINEncoder, GINMolEncoder
from .Pooling import GlobalAddPool



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
            return self.encoder(x, edge_index, edge_attr, batch)
        else:
            x, edge_index, batch = self.arguments_read(*args, **kwargs)
            return self.encoder(x, edge_index, batch)


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

        virtual_node_feat = [self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))]

        layer_feat = [x]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            if i > 0:
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
            else:
                post_conv = layer_feat[-1]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
            # --- update global info ---
            if 0 < i < len(self.convs) - 1:
                virtual_node_feat.append(self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch) + virtual_node_feat[-1]))

        # out_readout = torch.cat([GlobalMeanPool()(layer_feat[-1], batch),
        #                          GlobalMaxPool()(layer_feat[-1], batch)], dim=1)
        out_readout = self.readout(layer_feat[-1], batch)

        return layer_feat, out_readout, virtual_node_feat


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
        virtual_node_feat = [self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config.device, dtype=torch.long))]

        layer_feat = [self.atom_encoder(x)]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
            # --- update global info ---
            if i < len(self.convs) - 1:
                virtual_node_feat.append(self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch) + virtual_node_feat[-1]))

        out_readout = self.readout(layer_feat[-1], batch)
        return layer_feat, out_readout, virtual_node_feat
