"""
Atom (node) and bond (edge) feature encoding specified for molecule data.
"""
import torch
from torch import Tensor
from GOOD.utils.data import x_map, e_map


class AtomEncoder(torch.nn.Module):
    r"""
    atom (node) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        feat_dims = list(map(len, x_map.values()))

        for i, dim in enumerate(feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        r"""
        atom (node) feature encoding specified for molecule data.

        Args:
            x (Tensor): node features

        Returns (Tensor):
            atom (node) embeddings
        """
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""
    bond (edge) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        edge_feat_dims = list(map(len, e_map.values()))

        for i, dim in enumerate(edge_feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        r"""
        bond (edge) feature encoding specified for molecule data.

        Args:
            edge_attr (Tensor): edge attributes

        Returns (Tensor):
            bond (edge) embeddings

        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding
