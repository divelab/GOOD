"""
The original 250k ZINC dataset from the `ZINC database
<https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
`"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"
<https://arxiv.org/abs/1610.02415>`_ paper
"""
import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url
from tqdm import tqdm

from GOOD.utils.data import from_smiles


class ZINC(InMemoryDataset):
    r"""The ZINC dataset from the `ZINC database
    <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00559>`_ and the
    `"Automatic Chemical Design Using a Data-Driven Continuous Representation
    of Molecules" <https://arxiv.org/abs/1610.02415>`_ paper, containing about
    250,000 molecular graphs with up to 38 heavy atoms.
    The task is to regress a synthetic computed property dubbed as the
    constrained solubility.

    Args:
        root (string): Root directory where the dataset should be saved.
        subset (boolean, optional): If set to :obj:`True`, will only load a
            subset of the dataset (12,000 molecular graphs), following the
            `"Benchmarking Graph Neural Networks"
            <https://arxiv.org/abs/2003.00982>`_ paper. (default: :obj:`False`)
        split (string, optional): If :obj:`"train"`, loads the training
            dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset.
            (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv'

    def __init__(self, root, name, transform=None,
                 pre_transform=None, pre_filter=None, subset=False):
        self.name = 'zinc'
        self.subset = subset
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self):
        return ['zinc.csv']

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(self.url, osp.join(self.root, self.name))
        # extract_zip(path, self.root)
        os.mkdir(self.raw_dir)
        os.rename(osp.join(self.root, self.name, '250k_rndm_zinc_drugs_clean_3.csv'), self.raw_paths[0])

    def process(self):
        data_list = []
        import csv
        with open(self.raw_paths[0], "r") as csv_fp:
            csv_reader = csv.reader(csv_fp)
            fields = {field: i for i, field in enumerate(next(csv_reader))}
            for i, raw_data in tqdm(enumerate(csv_reader)):
                smiles = raw_data[fields['smiles']]
                data, mol = from_smiles(smiles)
                cycles = mol.GetRingInfo().AtomRings()
                num_cycles_le6 = 0
                for cycle in cycles:
                    if cycle.__len__() >= 6:
                        num_cycles_le6 += 1
                data.y = float(raw_data[fields['logP']]) - float(raw_data[fields['SAS']]) - float(num_cycles_le6)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

                if self.subset and i >= 1000:
                    break

        torch.save(self.collate(data_list), self.processed_paths[0])
