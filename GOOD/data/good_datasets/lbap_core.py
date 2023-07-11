"""
The GOOD-HIV dataset adapted from `MoleculeNet
<https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.
"""
import itertools
import os
import os.path as osp

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from tqdm import tqdm


class DummyDataset(InMemoryDataset):

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):
        super().__init__(root, transform, pre_transform)


from GOOD import register


@register.dataset_register
class LBAPcore(InMemoryDataset):
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
        self.mol_name = 'LBAPcore'
        self.domain = domain
        self.metric = 'ROC-AUC'
        self.task = 'Binary classification'
        self.url = 'https://drive.google.com/file/d/106u6ryPikpy_M-Ub8BFM2Lzd09i1FCln/view?usp=sharing'

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        shift_mode = {'covariate': 0}
        mode = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices, self.max_x_feat, self.max_edge_feat, self.min_x_feat, self.min_edge_feat = torch.load(
            self.processed_paths[subset_pt])

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
        return ['covariate_train.pt', 'covariate_val.pt', 'covariate_test.pt', 'covariate_id_val.pt',
                'covariate_id_test.pt']

    def process(self):
        covariate_shift_list = []
        max_x_feat = []
        max_edge_feat = []
        for subset in tqdm(['train', 'ood_val', 'ood_test', 'iid_val', 'iid_test']):
            temp_dataset = DummyDataset('dummy_root', 'dummy_name')
            temp_dataset.data, temp_dataset.slices = torch.load(
                os.path.join(self.root, 'ginv', 'data', 'DrugOOD', f'drugood_lbap_core_ic50_{self.domain}_{subset}.pt'))
            data_list = []
            for data in temp_dataset:
                data_list.append(Data(x=data.x.long(),
                                      edge_index=data.edge_index,
                                      edge_attr=data.edge_attr.long(),
                                      y=data.y[:, None].float(),
                                      env_id=data.group))
            covariate_shift_list.append(data_list)

        all_data_list = covariate_shift_list
        all_data, all_slices = self.collate(list(itertools.chain(*all_data_list)))
        min_x_feat = all_data.x.min(0).values
        min_edge_feat = all_data.edge_attr.min(0).values

        max_x_feat = all_data.x.max(0).values - min_x_feat + 1
        max_edge_feat = all_data.edge_attr.max(0).values - min_edge_feat + 1
        for i, final_data_list in enumerate(all_data_list):
            if final_data_list:
                data, slices = self.collate(final_data_list)
                data.x = data.x - min_x_feat[None, :]
                data.edge_attr = data.edge_attr - min_edge_feat[None, :]
                torch.save((data, slices, max_x_feat, max_edge_feat, min_x_feat, min_edge_feat),
                           self.processed_paths[i])

    @staticmethod
    def load(dataset_root: str, domain: str, shift: str = 'no_shift', generate: bool = False):
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

        train_dataset = LBAPcore(root=dataset_root,
                                 domain=domain, shift=shift, subset='train', generate=generate)
        id_val_dataset = LBAPcore(root=dataset_root,
                                  domain=domain, shift=shift, subset='id_val',
                                  generate=generate) if shift != 'no_shift' else None
        id_test_dataset = LBAPcore(root=dataset_root,
                                   domain=domain, shift=shift, subset='id_test',
                                   generate=generate) if shift != 'no_shift' else None
        val_dataset = LBAPcore(root=dataset_root,
                               domain=domain, shift=shift, subset='val', generate=generate)
        test_dataset = LBAPcore(root=dataset_root,
                                domain=domain, shift=shift, subset='test', generate=generate)

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

        meta_info.feat_dims = train_dataset.max_x_feat  # torch.stack(max_x_feat).max(0).values - torch.stack(min_x_feat).min(0).values + 1
        meta_info.edge_feat_dims = train_dataset.max_edge_feat  # torch.stack(max_edge_feat).max(0).values - torch.stack(min_edge_feat).min(0).values + 1

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
