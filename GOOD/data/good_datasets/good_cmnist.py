"""
The GOOD-CMNIST dataset following `IRM
<https://arxiv.org/abs/1907.02893>`_ paper.
"""
import itertools
import os
import os.path as osp
import random
from copy import deepcopy

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.datasets import MNISTSuperpixels
from tqdm import tqdm

from GOOD import register


@register.dataset_register
class GOODCMNIST(InMemoryDataset):
    r"""
    The GOOD-CMNIST dataset following `IRM
    <https://arxiv.org/abs/1907.02893>`_ paper.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'color', 'background'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = 'https://drive.google.com/file/d/1F2r2kVmA0X07AXyap9Y_rOM6LipDzwhq/view?usp=sharing'

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        shift_mode = {'no_shift': 0, 'covariate': 3, 'concept': 8}
        mode = {'train': 0, 'val': 1, 'test': 2, 'id_val': 3, 'id_test': 4}
        subset_pt = shift_mode[shift] + mode[subset]

        self.data, self.slices = torch.load(self.processed_paths[subset_pt])

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
        return ['no_shift_train.pt', 'no_shift_val.pt', 'no_shift_test.pt',
                'covariate_train.pt', 'covariate_val.pt', 'covariate_test.pt', 'covariate_id_val.pt',
                'covariate_id_test.pt',
                'concept_train.pt', 'concept_val.pt', 'concept_test.pt', 'concept_id_val.pt', 'concept_id_test.pt']

    def get_no_shift_list(self, data_list):
        random.shuffle(data_list)

        num_data = data_list.__len__()
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))
        train_list, val_list, test_list = data_list[: train_split], data_list[train_split: val_split], data_list[
                                                                                                       val_split:]
        num_env_train = 5
        num_per_env = train_split // num_env_train
        train_env_list = []
        for i in range(num_env_train):
            train_env_list.append(train_list[i * num_per_env: (i + 1) * num_per_env])

        all_env_list = [env_list for env_list in train_env_list] + [val_list, test_list]
        covariate_color = [0, 1, 2, 3, 4, 5, 6]

        pure_colors = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 0.5, 0.5],
                       [0.5, 1, 0.5],
                       [0.5, 0.5, 1],
                       [1, 1, 0.5]]
        for env_id, env_list in enumerate(all_env_list):
            for data in tqdm(env_list):
                # environment feature
                data.color = torch.LongTensor([0])

                # create features
                data.x = data.x * torch.FloatTensor([1, 1, 1])
                # background_idx = data.x[:, 0] < 0.001
                # data.x[background_idx] = torch.FloatTensor([0, 0, 0])

                data.env_id = torch.LongTensor([env_id])

        tmp = []
        for env_list in all_env_list[: num_env_train]:
            tmp += env_list
        all_env_list = [tmp] + [all_env_list[num_env_train]] + \
                       [all_env_list[num_env_train + 1]]

        return all_env_list

    def get_covariate_shift_list(self, data_list):
        random.shuffle(data_list)

        num_data = data_list.__len__()
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))
        train_list, val_list, test_list = data_list[: train_split], data_list[train_split: val_split], data_list[
                                                                                                       val_split:]
        num_env_train = 5
        num_per_env = train_split // num_env_train
        train_env_list = []
        for i in range(num_env_train):
            train_env_list.append(train_list[i * num_per_env: (i + 1) * num_per_env])

        all_env_list = [env_list for env_list in train_env_list] + [val_list, test_list]
        covariate_color = [0, 1, 2, 3, 4, 5, 6]

        pure_colors = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1],
                       [1, 0.5, 0.5],
                       [1, 0, 1],
                       [0.5, 1, 0.5],
                       [0.5, 0.5, 1],
                       [1, 1, 0.5]]
        for env_id, env_list in enumerate(all_env_list):
            for data in tqdm(env_list):
                # environment feature
                data.color = torch.LongTensor([covariate_color[env_id]])

                # create features
                if self.domain == 'color':
                    data.x = data.x * torch.FloatTensor(pure_colors[data.color])
                elif self.domain == 'background':
                    data.x = data.x.repeat(1, 3)
                    background_idx = data.x[:, 0] < 0.001
                    data.x[background_idx] = torch.FloatTensor(pure_colors[data.color])
                else:
                    raise ValueError(f'The domain is expected to be background or digit, but got {self.domain}.')

                data.env_id = torch.LongTensor([env_id])

        train_list, ood_val_list, ood_test_list = list(itertools.chain(*all_env_list[: num_env_train])), \
                                                  all_env_list[num_env_train], \
                                                  all_env_list[num_env_train + 1]
        num_id_test = int(num_data * test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]

        all_env_list = [train_list, ood_val_list, ood_test_list, id_val_list, id_test_list]

        return all_env_list

    def get_concept_shift_list(self, data_list):
        random.shuffle(data_list)
        num_data = data_list.__len__()
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))
        train_list, val_list, test_list = data_list[: train_split], data_list[train_split: val_split], data_list[
                                                                                                       val_split:]
        num_env_train = 5
        num_per_env = train_split // num_env_train
        train_env_list = []
        for i in range(num_env_train):
            train_env_list.append(train_list[i * num_per_env: (i + 1) * num_per_env])

        all_env_list = [env_list for env_list in train_env_list] + [val_list, test_list]
        spurious_ratio = [0.95, 0.9, 0.85, 0.8, 0.75, 0.3, 0.0]

        pure_colors = [[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1],
                       [1, 0, 1],
                       [1, 0.5, 0.5],
                       [0.5, 1, 0.5],
                       [0.5, 0.5, 1],
                       [1, 1, 0.5]]
        for env_id, env_list in enumerate(all_env_list):
            for data in tqdm(env_list):

                # Concept shift: spurious connection
                rand_color = random.randint(0, 9)

                spurious_connect = True if random.random() < spurious_ratio[env_id] else False
                if spurious_connect:
                    data.color = data.y
                else:
                    data.color = torch.LongTensor([rand_color])

                # domain features
                if self.domain == 'color':
                    data.x = data.x * torch.FloatTensor(pure_colors[data.color])
                elif self.domain == 'background':
                    data.x = data.x.repeat(1, 3)
                    background_idx = data.x[:, 0] < 0.001
                    data.x[background_idx] = torch.FloatTensor(pure_colors[data.color])
                else:
                    raise ValueError(f'The domain is expected to be background or digit, but got {self.domain}.')

                data.env_id = torch.LongTensor([env_id])

        train_list, ood_val_list, ood_test_list = list(itertools.chain(*all_env_list[: num_env_train])), \
                                                  all_env_list[num_env_train], \
                                                  all_env_list[num_env_train + 1]
        id_test_ratio = 0.15
        num_id_test = int(len(train_list) * id_test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]

        all_env_list = [train_list, ood_val_list, ood_test_list, id_val_list, id_test_list]

        return all_env_list

    def process(self):
        train_dataset = MNISTSuperpixels(root=self.root, train=True)
        test_dataset = MNISTSuperpixels(root=self.root, train=False)
        data_list = [data for data in train_dataset] + [data for data in test_dataset]
        print('Extract data done!')

        no_shift_list = self.get_no_shift_list(deepcopy(data_list))
        covariate_shift_list = self.get_covariate_shift_list(deepcopy(data_list))
        concept_shift_list = self.get_concept_shift_list(deepcopy(data_list))

        all_data_list = no_shift_list + covariate_shift_list + concept_shift_list
        for i, final_data_list in enumerate(all_data_list):
            data, slices = self.collate(final_data_list)
            torch.save((data, slices), self.processed_paths[i])

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
        meta_info.dataset_type = 'syn'
        meta_info.model_level = 'graph'

        train_dataset = GOODCMNIST(root=dataset_root, domain=domain, shift=shift, subset='train', generate=generate)
        id_val_dataset = GOODCMNIST(root=dataset_root, domain=domain, shift=shift,
                                    subset='id_val') if shift != 'no_shift' else None
        id_test_dataset = GOODCMNIST(root=dataset_root, domain=domain, shift=shift,
                                     subset='id_test') if shift != 'no_shift' else None
        val_dataset = GOODCMNIST(root=dataset_root, domain=domain, shift=shift, subset='val', generate=generate)
        test_dataset = GOODCMNIST(root=dataset_root, domain=domain, shift=shift, subset='test', generate=generate)

        meta_info.dim_node = train_dataset.num_node_features
        meta_info.dim_edge = train_dataset.num_edge_features

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

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': train_dataset.task,
                'metric': train_dataset.metric}, meta_info
