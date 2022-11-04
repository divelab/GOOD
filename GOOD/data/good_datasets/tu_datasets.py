"""
The GOOD-HIV dataset adapted from `MoleculeNet
<https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.
"""
import copy
import itertools
import os
import os.path as osp

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
import numpy as np
import random


class DomainGetter():
    r"""
    A class containing methods for data domain extraction.
    """
    def __init__(self):
        pass

    def get_nodesize(self, data: Data) -> int:
        """
        Args:
            data (Data): A smile string for a molecule.
        Returns:
            The number of node in the molecule.
        """
        return data.x.shape[0]


from GOOD import register


class _TU(InMemoryDataset):
    r"""
    The LBAPcore dataset. Adapted from `XXX
    <XXX>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        # self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'MCC'
        self.task = 'Binary classification'

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        shift_mode = {'covariate': 0}
        mode = {'train': 0, 'val': 1, 'test': 2}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices, self.max_x_feat, self.min_x_feat = torch.load(self.processed_paths[subset_pt])

    @property
    def raw_dir(self):
        return osp.join(self.root)

    def _download(self):
        if os.path.exists(osp.join(self.raw_dir, self.name)) or self.generate:
            return
        if not os.path.exists(self.raw_dir):
            os.makedirs(self.raw_dir)
        self.download()

    def download(self):
        path = gdown.download(self.url, output=osp.join(self.raw_dir, self.name + '.zip'), fuzzy=True)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['covariate_train.pt', 'covariate_val.pt', 'covariate_test.pt']

    def split_envs(self, sorted_data_list, fix_env=None):

        num_env_train = 10
        num_per_env = len(sorted_data_list) // num_env_train
        cur_env_id = -1
        cur_domain_id = None
        for i, data in enumerate(sorted_data_list):
            if cur_env_id < num_per_env - 1 and i >= (cur_env_id + 1) * num_per_env and data.domain_id != cur_domain_id:
                cur_env_id += 1
            cur_domain_id = data.domain_id
            data.env_id = fix_env if fix_env else cur_env_id
        return sorted_data_list

    def get_domain_sorted_list(self, data_list, domain='scaffold'):

        if domain == 'size':
            domain = 'nodesize'
        domain_getter = DomainGetter()
        for data in tqdm(data_list):
            data.__setattr__(domain, getattr(domain_getter, f'get_{domain}')(data))

        sorted_data_list = sorted(data_list, key=lambda data: getattr(data, domain))

        # Assign domain id
        cur_domain_id = -1
        cur_domain = None
        sorted_domain_split_data_list = []
        for data in sorted_data_list:
            if getattr(data, domain) != cur_domain:
                cur_domain = getattr(data, domain)
                cur_domain_id += 1
                sorted_domain_split_data_list.append([])
            data.domain_id = torch.LongTensor([cur_domain_id])
            sorted_domain_split_data_list[data.domain_id].append(data)

        return sorted_data_list, sorted_domain_split_data_list

    def process(self):
        if not os.path.exists(os.path.join(self.root, "TU_ids")):
            path = gdown.download('https://drive.google.com/file/d/1BYQIbi9pLdgE2eNMrbKJQeY1sskV5JRS/view?usp=sharing', os.path.join(self.root, "TU_ids.zip"), fuzzy=True)
            extract_zip(path, self.root)
            os.unlink(path)
        covariate_shift_list = []
        dataset = TUDataset(os.path.join(self.root, "TU"), name=self.name)
        dataset.data.x = dataset.data.x.long()
        dataset.data.y = dataset.data.y[:, None].float()
        for subset in tqdm(['train', 'val', 'test']):
            data_ids = np.loadtxt(os.path.join(self.root, "TU_ids", self.name, f'{subset}_idx.txt'), dtype=np.int64)
            data_list = []
            for data in dataset[data_ids]:
                data_list.append(data)
            sorted_data_list, _ = self.get_domain_sorted_list(data_list, self.domain)
            data_list = self.split_envs(copy.deepcopy(sorted_data_list), fix_env=-1)
            covariate_shift_list.append(data_list)

        all_data_list = covariate_shift_list
        all_data, all_slices = self.collate(list(itertools.chain(*all_data_list)))
        min_x_feat = all_data.x.min(0).values

        max_x_feat = all_data.x.max(0).values - min_x_feat + 1
        for i, final_data_list in enumerate(all_data_list):
            if final_data_list:
                data, slices = self.collate(final_data_list)
                data.x = data.x - min_x_feat[None, :]
                torch.save((data, slices, max_x_feat, min_x_feat),
                           self.processed_paths[i])

    @classmethod
    def load(cls, dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False):
        r"""
        A staticmethod for dataset loading. This method instantiates dataset class, constructing train, id_val, id_test,
        ood_val (val), and ood_test (test) splits. Besides, it collects several dataset meta information for further
        utilization.

        Args:
            dataset_root (str): The dataset saving root.
            domain (str): The domain selection. Allowed: 'degree' and 'time'.
            shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
            generate (bool): The flag for regenerating dataset. True: regenerate. False: download.

        Returns:
            dataset or dataset splits.
            dataset meta info.
        """
        meta_info = Munch()
        meta_info.dataset_type = 'mol'
        meta_info.model_level = 'graph'

        train_dataset = cls(root=dataset_root,
                                 domain=domain, shift=shift, subset='train', generate=generate)
        id_val_dataset = None
        id_test_dataset = None
        val_dataset = cls(root=dataset_root,
                               domain=domain, shift=shift, subset='val', generate=generate)
        test_dataset = cls(root=dataset_root,
                                domain=domain, shift=shift, subset='test', generate=generate)

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

        meta_info.feat_dims = train_dataset.max_x_feat  # torch.stack(max_x_feat).max(0).values - torch.stack(min_x_feat).min(0).values + 1

        meta_info.num_envs = torch.unique(train_dataset.data.env_id).shape[0]

        # Define networks' output shape.
        if train_dataset.task == 'Binary classification':
            meta_info.num_classes = train_dataset.data.y.shape[1]
        elif train_dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif train_dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(train_dataset.data.y).shape[0]

        # --- clear buffer dataset._data_list ---
        train_dataset._data_list = None
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': train_dataset.task,
                'metric': train_dataset.metric}, meta_info


@register.dataset_register
class DD(_TU):
    r"""
    The LBAPcore dataset. Adapted from `XXX
    <XXX>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.url = 'https://drive.google.com/file/d/1XDbqvWvzPMoQKocXd8iAeloggndmUs9P/view?usp=sharing'
        super(DD, self).__init__(root, domain, shift, subset, transform, pre_transform, generate)

        shift_mode = {'covariate': 0}
        mode = {'train': 0, 'val': 1, 'test': 2}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices, self.max_x_feat, self.min_x_feat = torch.load(self.processed_paths[subset_pt])

    def _download(self):
        super(DD, self)._download()

    def download(self):
        super(DD, self).download()

    def process(self):
        super(DD, self).process()


@register.dataset_register
class PROTEINS(_TU):
    r"""
    The LBAPcore dataset. Adapted from `XXX
    <XXX>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.url = 'https://drive.google.com/file/d/1hd1AgdSIqtmXY6ZU_HrhTykGe6t5F_po/view?usp=sharing'
        super(PROTEINS, self).__init__(root, domain, shift, subset, transform, pre_transform, generate)

        shift_mode = {'covariate': 0}
        mode = {'train': 0, 'val': 1, 'test': 2}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices, self.max_x_feat, self.min_x_feat = torch.load(self.processed_paths[subset_pt])

    def _download(self):
        super(PROTEINS, self)._download()

    def download(self):
        super(PROTEINS, self).download()

    def process(self):
        super(PROTEINS, self).process()


@register.dataset_register
class NCI1(_TU):
    r"""
    The LBAPcore dataset. Adapted from `XXX
    <XXX>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.url = 'https://drive.google.com/file/d/17gxhWMu5OIwkN7y7rIWRgQdfn3FTfVyC/view?usp=sharing'
        super(NCI1, self).__init__(root, domain, shift, subset, transform, pre_transform, generate)

        shift_mode = {'covariate': 0}
        mode = {'train': 0, 'val': 1, 'test': 2}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices, self.max_x_feat, self.min_x_feat = torch.load(self.processed_paths[subset_pt])

    def _download(self):
        super(NCI1, self)._download()

    def download(self):
        super(NCI1, self).download()

    def process(self):
        super(NCI1, self).process()


@register.dataset_register
class NCI109(_TU):
    r"""
    The LBAPcore dataset. Adapted from `XXX
    <XXX>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.url = 'https://drive.google.com/file/d/1bra4QcH7S8SQKyTKILt8sxQh5HCOSdRG/view?usp=sharing'
        super(NCI109, self).__init__(root, domain, shift, subset, transform, pre_transform, generate)

        shift_mode = {'covariate': 0}
        mode = {'train': 0, 'val': 1, 'test': 2}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices, self.max_x_feat, self.min_x_feat = torch.load(self.processed_paths[subset_pt])

    def _download(self):
        super(NCI109, self)._download()

    def download(self):
        super(NCI109, self).download()

    def process(self):
        super(NCI109, self).process()

