r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.
"""
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, Size
from torch_sparse import SparseTensor

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier


@register.model_register
class GCN(GNNBasic):
    r"""
    The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super().__init__(config)
        self.feat_encoder = GCNFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GCN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class GCNFeatExtractor(GNNBasic):
    r"""
        GCN feature extractor using the :class:`~GCNEncoder` .

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """
    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GCNFeatExtractor, self).__init__(config)
        self.encoder = GCNEncoder(config)
        self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GCN feature extractor using the :class:`~GCNEncoder` .

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        x, edge_index, edge_weight, batch = self.arguments_read(*args, **kwargs)
        out_readout = self.encoder(x, edge_index, edge_weight, batch)
        return out_readout


# class GCNEncoderNoBN(BasicEncoder):
#
#     def __init__(self, config: Union[CommonArgs, Munch]):
#         super(GCNEncoderNoBN, self).__init__(config)
#         num_layer = config.model.model_layer
#
#         self.conv1 = GCNConv(config.dataset.dim_node, config.model.dim_hidden)
#         self.convs = nn.ModuleList(
#             [
#                 GCNConv(config.model.dim_hidden, config.model.dim_hidden)
#                 for _ in range(num_layer - 1)
#             ]
#         )
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(config.model.dropout_rate)
#
#     def forward(self, x, edge_index, edge_weight, batch):
#         post_conv = self.conv1(x, edge_index, edge_weight)
#         post_conv = self.dropout1(self.relu1(post_conv))
#         for i, conv in enumerate(self.convs):
#             post_conv = conv(post_conv, edge_index, edge_weight)
#             if i < len(self.convs) - 1:
#                 post_conv = self.relu(post_conv)
#             post_conv = self.dropout(post_conv)
#
#         out_readout = self.readout(post_conv, batch)
#         return out_readout


class GCNEncoder(BasicEncoder):
    r"""
    The GCN encoder using the :class:`~GCNConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):
        super(GCNEncoder, self).__init__(config)
        num_layer = config.model.model_layer

        self.conv1 = GCNConv(config.dataset.dim_node, config.model.dim_hidden)
        self.convs = nn.ModuleList(
            [
                GCNConv(config.model.dim_hidden, config.model.dim_hidden)
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_weight, batch):
        r"""
        The GCN encoder.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights
            batch (Tensor): batch indicator

        Returns (Tensor):
            node feature representations
        """
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_weight))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_weight))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        out_readout = self.readout(post_conv, batch)
        return out_readout


class GCNConv(gnn.GCNConv):
    r"""The graph convolutional operator from the `"Semi-supervised
        Classification with Graph Convolutional Networks"
        <https://arxiv.org/abs/1609.02907>`_ paper

    Args:
        *args (list): argument list for the use of arguments_read.
        **kwargs (dict): Additional key word arguments for the use of arguments_read.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__explain_flow__ = False
        self.edge_weight = None
        self.layer_edge_mask = None
        self.__explain__ = False
        self.__edge_mask__ = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        r"""
        The GCN graph convolutional operator.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_weight (Tensor): edge weights

        Returns (Tensor):
            node feature representations

        """

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gnn.conv.gcn_conv.gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # --- add require_grad ---
        edge_weight.requires_grad_(True)

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        # --- My: record edge_weight ---
        self.edge_weight = edge_weight

        return out

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
                :obj:`torch_sparse.SparseTensor` that defines the underlying
                graph connectivity/message passing flow.
                :obj:`edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
                If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
                shape must be defined as :obj:`[2, num_messages]`, where
                messages from nodes in :obj:`edge_index[0]` are sent to
                nodes in :obj:`edge_index[1]`
                (in case :obj:`flow="source_to_target"`).
                If :obj:`edge_index` is of type
                :obj:`torch_sparse.SparseTensor`, its sparse indices
                :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
                and :obj:`col = edge_index[0]`.
                The major difference between both formats is that we need to
                input the *transposed* sparse adjacency matrix into
                :func:`propagate`.
            size (tuple, optional): The size :obj:`(N, M)` of the assignment
                matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
                If set to :obj:`None`, the size will be automatically inferred
                and assumed to be quadratic.
                This argument is ignored in case :obj:`edge_index` is a
                :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)

        # Run "fused" message and aggregation (if applicable).
        if (isinstance(edge_index, SparseTensor) and self.fuse
                and not self.__explain__):
            coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
                                         size, kwargs)

            msg_aggr_kwargs = self.inspector.distribute(
                'message_and_aggregate', coll_dict)
            out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

        # Otherwise, run both functions in separation.
        elif isinstance(edge_index, Tensor) or not self.fuse:
            coll_dict = self.__collect__(self.__user_args__, edge_index, size,
                                         kwargs)

            msg_kwargs = self.inspector.distribute('message', coll_dict)
            out = self.message(**msg_kwargs)

            # For `GNNExplainer`, we require a separate message and aggregate
            # procedure since this allows us to inject the `edge_mask` into the
            # message passing computation scheme.
            if self.__explain__:
                edge_mask = self.__edge_mask__.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
            elif self.__explain_flow__:

                edge_mask = self.layer_edge_mask.sigmoid()
                # Some ops add self-loops to `edge_index`. We need to do the
                # same for `edge_mask` (but do not train those).
                if out.size(self.node_dim) != edge_mask.size(0):
                    loop = edge_mask.new_ones(size[0])
                    edge_mask = torch.cat([edge_mask, loop], dim=0)
                assert out.size(self.node_dim) == edge_mask.size(0)
                out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))

            aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
            out = self.aggregate(out, **aggr_kwargs)

            update_kwargs = self.inspector.distribute('update', coll_dict)
            return self.update(out, **update_kwargs)

