"""
Base classes for Graph Neural Networks
"""
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch import Tensor
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from torch.nn import Identity


class GNNBasic(torch.nn.Module):
    r"""
    Base class for graph neural networks

    Args:
        *args (list): argument list for the use of :func:`~arguments_read`
        **kwargs (dict): key word arguments for the use of :func:`~arguments_read`

    """
    def __init__(self, config: Union[CommonArgs, Munch], *args, **kwargs):
        super(GNNBasic, self).__init__()
        self.config = config

    def arguments_read(self, *args, **kwargs):
        r"""
        It is an argument reading function for diverse model input formats.
        Support formats are:
        ``model(x, edge_index)``
        ``model(x, edge_index, batch)``
        ``model(data=data)``.

        Notes:
            edge_weight is optional for node prediction tasks.

        Args:
            *args: [x, edge_index, [batch]]
            **kwargs: data, [edge_weight]

        Returns:
            Unpacked node features, sparse adjacency matrices, batch indicators, and optional edge weights.
        """

        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.config.model.model_level != 'node':
            # --- Maybe batch size --- Reason: some method may filter graphs leading inconsistent of batch size
            batch_size: int = kwargs.get('batch_size') or (batch[-1].item() + 1)

        if self.config.model.model_level == 'node':
            edge_weight = kwargs.get('edge_weight')
            return x, edge_index, edge_weight, batch
        elif self.config.dataset.dim_edge:
            edge_attr = data.edge_attr
            return x, edge_index, edge_attr, batch, batch_size

        return x, edge_index, batch, batch_size

    def probs(self, *args, **kwargs):
        # nodes x classes
        return self(*args, **kwargs).softmax(dim=1)


class BasicEncoder(torch.nn.Module):
    r"""
        Base GNN feature encoder.

        Args:
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.model.dim_hidden`, :obj:`config.model.model_layer`, :obj:`config.model.model_level`, :obj:`config.model.global_pool`, :obj:`config.model.dropout_rate`)

        .. code-block:: python

            config = munchify({model: {dim_hidden: int(300),
                               model_layer: int(5),
                               model_level: str('node'),
                               global_pool: str('mean'),
                               dropout_rate: float(0.5),}
                               })


    """

    def __init__(self, config: Union[CommonArgs, Munch], **kwargs):
        if type(self).mro()[type(self).mro().index(__class__) + 1] is torch.nn.Module:
            super(BasicEncoder, self).__init__()
        else:
            super(BasicEncoder, self).__init__(config)
        num_layer = config.model.model_layer

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer - 1)
            ]
        )
        if kwargs.get('no_bn'):
            self.batch_norm1 = Identity()
            self.batch_norms = [
                Identity()
                for _ in range(num_layer - 1)
            ]
        else:
            self.batch_norm1 = nn.BatchNorm1d(config.model.dim_hidden)
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(config.model.dim_hidden)
                for _ in range(num_layer - 1)
            ])
        self.dropout1 = nn.Dropout(config.model.dropout_rate)
        self.dropouts = nn.ModuleList([
            nn.Dropout(config.model.dropout_rate)
            for _ in range(num_layer - 1)
        ])
        if config.model.model_level == 'node':
            self.readout = IdenticalPool()
        elif config.model.global_pool == 'mean':
            self.readout = GlobalMeanPool()
        else:
            self.readout = GlobalMaxPool()
