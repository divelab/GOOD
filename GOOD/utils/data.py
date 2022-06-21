r"""
Some data process utils including construction of molecule PyG graph from smile (for compatibility).
"""

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils.num_nodes import maybe_num_nodes


def batch_input(G: Data, batch_size: int, num_nodes: int = None, node_attrs: list =['color']):
    r"""
    Repeat a graph ``batch_size`` times and pack into a Batch.

    Args:
        G (Data): The given graph G.
        batch_size (int): Batch size.
        num_nodes (int): The number of node of the graph. If :obj:`None`, it will use maybe_numb_nodes.
        node_attrs (list): The preserved node attributes.

    Returns:
        Repeated graph batch.
    """
    x = G.x
    edge_index = G.edge_index
    device = edge_index.device
    num_edges = edge_index.shape[1]
    num_nodes = maybe_num_nodes(edge_index, num_nodes=num_nodes)
    batch_x = x.repeat(batch_size, 1)
    batch_batch = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_nodes).view(-1)
    batch_edge_batch = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_edges).view(-1)
    batch_edge_index = edge_index.repeat(1, batch_size) + batch_edge_batch * num_nodes
    batch = Batch(x=batch_x, edge_index=batch_edge_index, batch=batch_batch)
    if node_attrs:
        for node_attr in node_attrs:
            if hasattr(G, node_attr) and getattr(G, node_attr) is not None:
                batch.__setattr__(node_attr, getattr(G, node_attr).repeat(batch_size, 1))
    return batch


x_map = {
    'atomic_num':
        list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
        list(range(0, 11)),
    'formal_charge':
        list(range(-5, 7)),
    'num_hs':
        list(range(0, 9)),
    'num_radical_electrons':
        list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}


def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False):
    r"""Converts a SMILES string to a `torch_geometric.data.data.Data`
    instance.

    Args:
        smiles (string, optional): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        mol = Chem.Kekulize(mol)

    xs = []
    for atom in mol.GetAtoms():
        x = []
        x.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        x.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        x.append(x_map['degree'].index(atom.GetTotalDegree()))
        x.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        x.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        x.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        x.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        x.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        x.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(x)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []
        e.append(e_map['bond_type'].index(str(bond.GetBondType())))
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles), mol
