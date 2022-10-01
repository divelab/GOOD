r"""
The Graph Neural Network from the `"How Powerful are Graph Neural Networks?"
<https://arxiv.org/abs/1810.00826>`_ paper.
"""
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .MolEncoders import AtomEncoder, BondEncoder
from torch.nn import Identity


@register.model_register
class GIN(GNNBasic):
    r"""
    The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.num_classes`, :obj:`config.dataset.dataset_type`)
    """

    def __init__(self, config: Union[CommonArgs, Munch]):

        super().__init__(config)
        self.feat_encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs) -> torch.Tensor:
        r"""
        The GIN model implementation.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            label predictions

        """
        out_readout = self.feat_encoder(*args, **kwargs)

        out = self.classifier(out_readout)
        return out


class GINFeatExtractor(GNNBasic):
    r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`, :obj:`config.dataset.dataset_type`)
    """
    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINFeatExtractor, self).__init__(config)
        num_layer = config.model.model_layer
        if config.dataset.dataset_type == 'mol':
            self.encoder = GINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = GINEncoder(config, **kwargs)
            self.edge_feat = False

    def forward(self, *args, **kwargs):
        r"""
        GIN feature extractor using the :class:`~GINEncoder` or :class:`~GINMolEncoder`.

        Args:
            *args (list): argument list for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`
            **kwargs (dict): key word arguments for the use of arguments_read. Refer to :func:`arguments_read <GOOD.networks.models.BaseGNN.GNNBasic.arguments_read>`

        Returns (Tensor):
            node feature representations
        """
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout


class GINEncoder(BasicEncoder):
    r"""
    The GIN encoder for non-molecule data, using the :class:`~GINConv` operator for message passing.

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.dataset.dim_node`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):

        super(GINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config.model.model_layer
        self.without_readout = kwargs.get('without_readout')

        # self.atom_encoder = AtomEncoder(config.model.dim_hidden)

        if kwargs.get('without_embed'):
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
        else:
            self.conv1 = gnn.GINConv(nn.Sequential(nn.Linear(config.dataset.dim_node, 2 * config.model.dim_hidden),
                                               nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                               nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                gnn.GINConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                      nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                      nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for non-molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            batch (Tensor): batch indicator
            batch_size (int): batch size

        Returns (Tensor):
            node feature representations
        """

        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class GINMolEncoder(BasicEncoder):
    r"""The GIN encoder for molecule data, using the :class:`~GINEConv` operator for message passing.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`)
    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        super(GINMolEncoder, self).__init__(config, **kwargs)
        self.without_readout = kwargs.get('without_readout')
        num_layer = config.model.model_layer
        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config.model.dim_hidden)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                            nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                            nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))

        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(nn.Linear(config.model.dim_hidden, 2 * config.model.dim_hidden),
                                       nn.BatchNorm1d(2 * config.model.dim_hidden), nn.ReLU(),
                                       nn.Linear(2 * config.model.dim_hidden, config.model.dim_hidden)))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        r"""
        The GIN encoder for molecule data.

        Args:
            x (Tensor): node features
            edge_index (Tensor): edge indices
            edge_attr (Tensor): edge attributes
            batch (Tensor): batch indicator
            batch_size (int): Batch size.

        Returns (Tensor):
            node feature representations
        """
        x = self.atom_encoder(x)
        post_conv = self.dropout1(self.relu1(self.batch_norm1(self.conv1(x, edge_index, edge_attr))))
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(post_conv, edge_index, edge_attr))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)

        if self.without_readout or kwargs.get('without_readout'):
            return post_conv
        out_readout = self.readout(post_conv, batch, batch_size)
        return out_readout


class GINEConv(gnn.MessagePassing):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
                 edge_dim: Optional[int] = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        self.bone_encoder = BondEncoder(in_channels)
        # if edge_dim is not None:
        #     self.lin = Linear(edge_dim, in_channels)
        #     # self.lin = Linear(edge_dim, config.model.dim_hidden)
        # else:
        #     self.lin = None
        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if self.bone_encoder:
            edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)

        return (x_j + edge_attr).relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


# class GINConv(gnn.GINConv):
#     r"""The graph isomorphism operator from the `"How Powerful are
#     Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper
#
#     .. math::
#         \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
#         \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)
#
#     or
#
#     .. math::
#         \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
#         (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),
#
#     here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.
#
#     Args:
#         nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
#             maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
#             shape :obj:`[-1, out_channels]`, *e.g.*, defined by
#             :class:`torch.nn.Sequential`.
#         eps (float, optional): (Initial) :math:`\epsilon`-value.
#             (default: :obj:`0.`)
#         train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
#             will be a trainable parameter. (default: :obj:`False`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#
#     Shapes:
#         - **input:**
#           node features :math:`(|\mathcal{V}|, F_{in})` or
#           :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
#           if bipartite,
#           edge indices :math:`(2, |\mathcal{E}|)`
#         - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
#           :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
#     """
#
#     def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False,
#                  **kwargs):
#         super().__init__(nn, eps, train_eps, **kwargs)
#         self.edge_weight = None
#         self.fc_steps = None
#         self.reweight = None
#         self.__explain_flow__ = None
#         self.__explain__ = False
#         self.__edge_mask__ = None
#
#     def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
#                 edge_weight: OptTensor = None, **kwargs) -> Tensor:
#         """"""
#         self.num_nodes = x.shape[0]
#         if isinstance(x, Tensor):
#             x: OptPairTensor = (x, x)
#
#         # propagate_type: (x: OptPairTensor)
#         if edge_weight is not None:
#             self.edge_weight = edge_weight
#             assert edge_weight.shape[0] == edge_index.shape[1]
#             self.reweight = False
#         else:
#             edge_index, _ = remove_self_loops(edge_index)
#             self_loop_edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
#             if self_loop_edge_index.shape[1] != edge_index.shape[1]:
#                 edge_index = self_loop_edge_index
#             self.reweight = True
#         out = self.propagate(edge_index, x=x[0], size=None)
#
#         nn_out = self.nn(out)
#
#         return nn_out
#
#     def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
#         r"""The initial call to start propagating messages.
#
#         Args:
#             edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
#                 :obj:`torch_sparse.SparseTensor` that defines the underlying
#                 graph connectivity/message passing flow.
#                 :obj:`edge_index` holds the indices of a general (sparse)
#                 assignment matrix of shape :obj:`[N, M]`.
#                 If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its
#                 shape must be defined as :obj:`[2, num_messages]`, where
#                 messages from nodes in :obj:`edge_index[0]` are sent to
#                 nodes in :obj:`edge_index[1]`
#                 (in case :obj:`flow="source_to_target"`).
#                 If :obj:`edge_index` is of type
#                 :obj:`torch_sparse.SparseTensor`, its sparse indices
#                 :obj:`(row, col)` should relate to :obj:`row = edge_index[1]`
#                 and :obj:`col = edge_index[0]`.
#                 The major difference between both formats is that we need to
#                 input the *transposed* sparse adjacency matrix into
#                 :func:`propagate`.
#             size (tuple, optional): The size :obj:`(N, M)` of the assignment
#                 matrix in case :obj:`edge_index` is a :obj:`LongTensor`.
#                 If set to :obj:`None`, the size will be automatically inferred
#                 and assumed to be quadratic.
#                 This argument is ignored in case :obj:`edge_index` is a
#                 :obj:`torch_sparse.SparseTensor`. (default: :obj:`None`)
#             **kwargs: Any additional data which is needed to construct and
#                 aggregate messages, and to update node embeddings.
#         """
#         size = self.__check_input__(edge_index, size)
#
#         # Run "fused" message and aggregation (if applicable).
#         if (isinstance(edge_index, SparseTensor) and self.fuse
#                 and not self.__explain__):
#             coll_dict = self.__collect__(self.__fused_user_args__, edge_index,
#                                          size, kwargs)
#
#             msg_aggr_kwargs = self.inspector.distribute(
#                 'message_and_aggregate', coll_dict)
#             out = self.message_and_aggregate(edge_index, **msg_aggr_kwargs)
#
#             update_kwargs = self.inspector.distribute('update', coll_dict)
#             return self.update(out, **update_kwargs)
#
#         # Otherwise, run both functions in separation.
#         elif isinstance(edge_index, Tensor) or not self.fuse:
#             coll_dict = self.__collect__(self.__user_args__, edge_index, size,
#                                          kwargs)
#
#             msg_kwargs = self.inspector.distribute('message', coll_dict)
#             out = self.message(**msg_kwargs)
#
#             # For `GNNExplainer`, we require a separate message and aggregate
#             # procedure since this allows us to inject the `edge_mask` into the
#             # message passing computation scheme.
#             if self.__explain__:
#                 edge_mask = self.__edge_mask__.sigmoid()
#                 # Some ops add self-loops to `edge_index`. We need to do the
#                 # same for `edge_mask` (but do not train those).
#                 if out.size(self.node_dim) != edge_mask.size(0):
#                     loop = edge_mask.new_ones(size[0])
#                     edge_mask = torch.cat([edge_mask, loop], dim=0)
#                 assert out.size(self.node_dim) == edge_mask.size(0)
#                 out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
#             elif self.__explain_flow__:
#
#                 edge_mask = self.layer_edge_mask.sigmoid()
#                 # Some ops add self-loops to `edge_index`. We need to do the
#                 # same for `edge_mask` (but do not train those).
#                 if out.size(self.node_dim) != edge_mask.size(0):
#                     loop = edge_mask.new_ones(size[0])
#                     edge_mask = torch.cat([edge_mask, loop], dim=0)
#                 assert out.size(self.node_dim) == edge_mask.size(0)
#                 out = out * edge_mask.view([-1] + [1] * (out.dim() - 1))
#
#             aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
#             out = self.aggregate(out, **aggr_kwargs)
#
#             update_kwargs = self.inspector.distribute('update', coll_dict)
#             return self.update(out, **update_kwargs)
#
#     def message(self, x_j: Tensor) -> Tensor:
#         if self.reweight:
#             edge_weight = torch.ones(x_j.shape[0], device=x_j.device)
#             edge_weight.data[-self.num_nodes:] += self.eps
#             edge_weight = edge_weight.detach().clone()
#             edge_weight.requires_grad_(True)
#             self.edge_weight = edge_weight
#         return x_j * self.edge_weight.view(-1, 1)
