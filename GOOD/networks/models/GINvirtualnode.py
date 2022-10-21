r"""
The Graph Neural Network from the `"Neural Message Passing for Quantum Chemistry"
<https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.
"""
import torch
import torch.nn as nn
from torch import Tensor
from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINEncoder, GINMolEncoder, GINFeatExtractor
from .Pooling import GlobalAddPool


@register.model_register
class vGIN(GNNBasic):
    r"""
        The Graph Neural Network from the `"Neural Message Passing for Quantum Chemistry"
        <https://proceedings.mlr.press/v70/gilmer17a.html>`_ paper.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(vGIN, self).__init__(config)
        self.feat_encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The vGIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class vGINFeatExtractor(GINFeatExtractor):
    r"""
        vGIN feature extractor using the :class:`~vGINEncoder` or :class:`~vGINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`, :obj:`config.model.dropout_rate`)
            **kwargs: `without_readout` will output node features instead of graph features.
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(vGINFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = vGINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config, **kwargs)
            self.edge_feat = False


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

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(vGINEncoder, self).__init__(config, **kwargs)
        self.config = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The vGIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long))

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
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout



class vGINMolEncoder(GINMolEncoder, VirtualNodeEncoder):
    r"""The vGIN encoder for molecule data, using the :class:`~vGINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.model.dropout_rate`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(vGINMolEncoder, self).__init__(config, **kwargs)
        self.config: Union[CommonArgs, Munch] = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The vGIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        virtual_node_feat = self.virtual_node_embedding(
            torch.zeros(batch_size, device=self.config.device, dtype=torch.long))

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
                virtual_node_feat = self.virtual_mlp(self.virtual_pool(post_conv, batch, batch_size) + virtual_node_feat)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout
