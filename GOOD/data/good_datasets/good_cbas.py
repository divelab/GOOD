"""
The GOOD-CBAS dataset modified from `BA-Shapes
<https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html>`_.
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
from torch_geometric.datasets import BAShapes
from torch_geometric.utils import to_undirected
from tqdm import tqdm


class DataInfo(object):
    r"""
    The class for data point storage. This enables tackling node data point like graph data point, facilitating data splits.
    """
    def __init__(self, idx, y, x):
        super(DataInfo, self).__init__()
        self.storage = []
        self.idx = idx
        self.y = y
        self.x = x

    def __repr__(self):
        s = [f'{key}={self.__getattribute__(key)}' for key in self.storage]
        s = ', '.join(s)
        return f"DataInfo({s})"

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key != 'storage':
            self.storage.append(key)


from GOOD import register


@register.dataset_register
class GOODCBAS(InMemoryDataset):
    r"""
    The GOOD-CBAS dataset. Modified from `BA-Shapes
    <https://proceedings.neurips.cc/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'color'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', transform=None, pre_transform=None,
                 generate: bool = False):
        self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = 'https://drive.google.com/file/d/11DoWXHiic3meNRJiUmEMKDjIHYSSVJ4w/view?usp=sharing'

        self.generate = generate

        super().__init__(root, transform, pre_transform)
        shift_mode = {'no_shift': 0, 'covariate': 1, 'concept': 2}
        subset_pt = shift_mode[shift]

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
        return ['no_shift.pt', 'covariate.pt', 'concept.pt']

    def assign_no_shift_masks(self, train_list, val_list, test_list, graph):
        num_data = self.num_data
        train_mask, val_mask, test_mask = (torch.zeros((num_data,), dtype=torch.bool) for _ in range(3))
        env_id = - torch.ones((num_data,), dtype=torch.long)
        x = torch.ones((num_data, 4), dtype=torch.float)
        for data in train_list:
            train_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            x[data.idx] = data.x

        for data in val_list:
            val_mask[data.idx] = True
            x[data.idx] = data.x

        for data in test_list:
            test_mask[data.idx] = True
            x[data.idx] = data.x
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        graph.env_id = env_id
        graph.x = x
        return graph

    def assign_masks(self, train_list, val_list, test_list, id_val_list, id_test_list, graph):
        num_data = self.num_data
        train_mask, val_mask, test_mask, id_val_mask, id_test_mask = (torch.zeros((num_data,), dtype=torch.bool) for _
                                                                      in range(5))
        env_id = - torch.ones((num_data,), dtype=torch.long)
        domain_id = - torch.ones((num_data,), dtype=torch.long)
        x = torch.ones((num_data, 4), dtype=torch.float)
        for data in train_list:
            train_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            domain_id[data.idx] = data.color
            x[data.idx] = data.x

        for data in val_list:
            val_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            domain_id[data.idx] = data.color
            x[data.idx] = data.x

        for data in test_list:
            test_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            domain_id[data.idx] = data.color
            x[data.idx] = data.x

        for data in id_val_list:
            id_val_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            domain_id[data.idx] = data.color
            x[data.idx] = data.x

        for data in id_test_list:
            id_test_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            domain_id[data.idx] = data.color
            x[data.idx] = data.x

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        graph.id_val_mask = id_val_mask
        graph.id_test_mask = id_test_mask
        graph.env_id = env_id
        graph.domain_id = domain_id
        graph.x = x
        return graph

    def get_no_shift_graph(self, data_list, graph):

        num_data = self.num_data

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

        for env_id, env_list in enumerate(all_env_list):
            for data in tqdm(env_list):
                # environment feature
                data.color = torch.LongTensor([0])

                # create features
                data.x = torch.FloatTensor([1, 1, 1, 1])

                data.env_id = torch.LongTensor([env_id])

        tmp = []
        for env_list in all_env_list[: num_env_train]:
            tmp += env_list
        train_list, val_list, test_list = tmp, all_env_list[num_env_train], all_env_list[num_env_train + 1]

        return self.assign_no_shift_masks(train_list, val_list, test_list, graph)

    def get_covariate_shift_graph(self, data_list, graph):

        num_data = self.num_data

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

        pure_colors = [[1, 0, 0, 0.5],
                       [0, 1, 0, 0.7],
                       [0, 0, 1, 0.3],
                       [1, 1, 0, 0.4],
                       [0, 1, 1, 0.6],
                       [1, 0, 1, 1.0],
                       [0, 0, 0, 0.1]]
        for env_id, env_list in enumerate(all_env_list):
            for data in tqdm(env_list):
                # environment feature
                data.color = torch.LongTensor([covariate_color[env_id]])

                # create features
                data.x = torch.FloatTensor(pure_colors[data.color])

                data.env_id = torch.LongTensor([env_id])

        train_list, ood_val_list, ood_test_list = list(itertools.chain(*all_env_list[: num_env_train])), \
                                                  all_env_list[num_env_train], \
                                                  all_env_list[num_env_train + 1]
        num_id_test = int(num_data * test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]

        return self.assign_masks(train_list, ood_val_list, ood_test_list, id_val_list, id_test_list, graph)

    def get_concept_shift_graph(self, data_list, graph):

        num_data = self.num_data
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

        pure_colors = [[1, 0, 0, 0.5],
                       [0, 1, 0, 0.7],
                       [0, 0, 1, 0.3],
                       [1, 1, 0, 0.2],
                       [0, 1, 1, 0.6],
                       [1, 0, 1, 1.0],
                       [1, 1, 1, 0.0]]
        for env_id, env_list in enumerate(all_env_list):
            for data in tqdm(env_list):

                # Concept shift: spurious connection
                rand_color = random.randint(0, 3)

                spurious_connect = True if random.random() < spurious_ratio[env_id] else False
                if spurious_connect:
                    data.color = data.y
                else:
                    data.color = torch.LongTensor([rand_color])

                # create features
                data.x = torch.FloatTensor(pure_colors[data.color])

                data.env_id = torch.LongTensor([env_id])

        train_list, ood_val_list, ood_test_list = list(itertools.chain(*all_env_list[: num_env_train])), \
                                                  all_env_list[num_env_train], \
                                                  all_env_list[num_env_train + 1]
        num_id_test = int(num_data * test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]

        return self.assign_masks(train_list, ood_val_list, ood_test_list, id_val_list, id_test_list, graph)

    def get_peudo_data_list(self, graph):

        data_list = []
        for i in tqdm(range(self.num_data)):
            data_info = DataInfo(idx=i, y=graph.y[i], x=graph.x[i])
            data_list.append(data_info)

        random.shuffle(data_list)

        return data_list

    def process(self):
        dataset = BAShapes()
        graph = dataset[0]
        graph.x = graph.x[:, :4]
        graph.edge_index = to_undirected(graph.edge_index, graph.num_nodes)
        graph.y = graph.y.squeeze()
        print('Load data done!')
        self.num_data = graph.x.shape[0]
        print('Extract data done!')

        data_list = self.get_peudo_data_list(graph)

        no_shift_graph = self.get_no_shift_graph(deepcopy(data_list), deepcopy(graph))
        print('#IN#No shift dataset done!')
        covariate_shift_graph = self.get_covariate_shift_graph(deepcopy(data_list), deepcopy(graph))
        print('#IN#\nCovariate shift dataset done!')
        concept_shift_graph = self.get_concept_shift_graph(deepcopy(data_list), deepcopy(graph))
        print('#IN#\nConcept shift dataset done!')

        all_split_graph = [no_shift_graph, covariate_shift_graph, concept_shift_graph]
        for i, final_graph in enumerate(all_split_graph):
            data, slices = self.collate([final_graph])
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
        meta_info.model_level = 'node'

        dataset = GOODCBAS(root=dataset_root, domain=domain, shift=shift, generate=generate)
        dataset.data.x = dataset.data.x.to(torch.float32)
        meta_info.dim_node = dataset.num_node_features
        meta_info.dim_edge = dataset.num_edge_features

        meta_info.num_envs = (torch.unique(dataset.data.env_id) >= 0).sum()
        meta_info.num_train_nodes = dataset[0].train_mask.sum()

        # Define networks' output shape.
        if dataset.task == 'Binary classification':
            meta_info.num_classes = dataset.data.y.shape[1]
        elif dataset.task == 'Regression':
            meta_info.num_classes = 1
        elif dataset.task == 'Multi-label classification':
            meta_info.num_classes = torch.unique(dataset.data.y).shape[0]

        # --- clear buffer dataset._data_list ---
        dataset._data_list = None

        return dataset, meta_info
