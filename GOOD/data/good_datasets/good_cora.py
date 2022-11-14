"""
The GOOD-Cora dataset adapted from the `full Cora dataset
<https://arxiv.org/abs/1707.03815>`_.
"""
import itertools
import os
import os.path as osp
import random
from copy import deepcopy

import gdown
import numpy as np
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip, Data
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import degree
from tqdm import tqdm


class DomainGetter(object):
    r"""
    A class containing methods for data domain extraction.
    """
    def __init__(self):
        pass

    def get_degree(self, graph: Data) -> int:
        """
        Args:
            graph (Data): The PyG Data object.
        Returns:
            The degrees of the given graph.
        """
        try:
            node_degree = degree(graph.edge_index[0], graph.num_nodes)
            return node_degree
        except ValueError as e:
            print('#E#Get degree error.')
            raise e

    def get_word(self, graph: Data) -> int:
        """
        Args:
            graph (Data): The PyG Data object.
        Returns:
            The word diversity value of the graph.
        """
        num_word = graph.x.sum(1)
        return num_word


class DataInfo(object):
    r"""
    The class for data point storage. This enables tackling node data point like graph data point, facilitating data splits.
    """
    def __init__(self, idx, y):
        super(DataInfo, self).__init__()
        self.storage = []
        self.idx = idx
        self.y = y

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
class GOODCora(InMemoryDataset):
    r"""
    The GOOD-Cora dataset. Adapted from the `full Cora dataset
    <https://arxiv.org/abs/1707.03815>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'degree' and 'word'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', transform=None, pre_transform=None,
                 generate: bool = False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = 'https://drive.google.com/file/d/1OyMOwT4bn_4fLdpl5B3ie18OmGsUNQxS/view?usp=sharing'

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

    def assign_masks(self, train_list, val_list, test_list, id_val_list, id_test_list, graph):
        num_data = self.num_data
        train_mask, val_mask, test_mask, id_val_mask, id_test_mask = (torch.zeros((num_data,), dtype=torch.bool) for _
                                                                      in range(5))
        env_id = - torch.ones((num_data,), dtype=torch.long)
        domain = [None for _ in range(num_data)]
        domain_id = - torch.ones((num_data,), dtype=torch.long)
        for data in train_list:
            train_mask[data.idx] = True
            env_id[data.idx] = data.env_id
            domain[data.idx] = data.__getattribute__(self.domain)
            domain_id[data.idx] = data.domain_id

        for data in val_list:
            val_mask[data.idx] = True
            domain[data.idx] = data.__getattribute__(self.domain)
            domain_id[data.idx] = data.domain_id

        for data in test_list:
            test_mask[data.idx] = True
            domain[data.idx] = data.__getattribute__(self.domain)
            domain_id[data.idx] = data.domain_id

        for data in id_val_list:
            id_val_mask[data.idx] = True
            domain[data.idx] = data.__getattribute__(self.domain)
            domain_id[data.idx] = data.domain_id

        for data in id_test_list:
            id_test_mask[data.idx] = True
            domain[data.idx] = data.__getattribute__(self.domain)
            domain_id[data.idx] = data.domain_id

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        graph.id_val_mask = id_val_mask
        graph.id_test_mask = id_test_mask
        graph.env_id = env_id
        graph.domain = self.domain
        # graph.__setattr__(self.domain, domain)
        graph.domain_id = domain_id
        return graph

    def get_no_shift_graph(self, graph):
        num_data = self.num_data
        node_indices = torch.randperm(num_data)

        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2

        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))
        train_indices, val_indices, test_indices = node_indices[: train_split], node_indices[
                                                                                train_split: val_split], node_indices[
                                                                                                         val_split:]
        train_mask, val_mask, test_mask = (torch.zeros((num_data,), dtype=torch.bool) for _ in range(3))
        env_id = - torch.ones((num_data,), dtype=torch.long)
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        env_id[train_indices] = torch.randint(0, 9, (train_indices.shape[0],))

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        graph.env_id = env_id
        graph.domain = self.domain

        return graph

    def get_covariate_shift_graph(self, sorted_data_list, graph):

        num_data = self.num_data
        if self.domain == 'degree':
            sorted_data_list = sorted_data_list[::-1]
            train_ratio = 0.6
            val_ratio = 0.2
            id_test_ratio = 0.1
        else:
            sorted_data_list = sorted_data_list[::-1]
            train_ratio = 0.6
            val_ratio = 0.2
            id_test_ratio = 0.1

        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))

        train_val_test_split = [0, train_split, val_split]
        train_val_test_list = [[], [], []]
        cur_env_id = -1
        cur_domain_id = None
        for i, data in enumerate(sorted_data_list):
            if cur_env_id < 2 and i >= train_val_test_split[cur_env_id + 1] and data.domain_id != cur_domain_id:
                # if i >= (cur_env_id + 1) * num_per_env:
                cur_env_id += 1
            cur_domain_id = data.domain_id
            train_val_test_list[cur_env_id].append(data)

        train_list, ood_val_list, ood_test_list = train_val_test_list

        # Compose domains to environments
        num_env_train = 10
        num_per_env = len(train_list) // num_env_train
        cur_env_id = -1
        cur_domain_id = None
        for i, data in enumerate(train_list):
            if cur_env_id < 9 and i >= (cur_env_id + 1) * num_per_env and data.domain_id != cur_domain_id:
                # if i >= (cur_env_id + 1) * num_per_env:
                cur_env_id += 1
            cur_domain_id = data.domain_id
            data.env_id = cur_env_id

        num_id_test = int(num_data * id_test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], train_list[
                                                                                -2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]

        return self.assign_masks(train_list, ood_val_list, ood_test_list, id_val_list, id_test_list, graph)

    def get_concept_shift_graph(self, sorted_domain_split_data_list, graph):

        # Calculate concept probability for each domain
        global_pyx = []
        for each_domain_datas in tqdm(sorted_domain_split_data_list):
            pyx = []
            for data in each_domain_datas:
                data.pyx = torch.tensor(np.nanmean(data.y).item())
                if torch.isnan(data.pyx):
                    data.pyx = torch.tensor(0.)
                pyx.append(data.pyx.item())
                global_pyx.append(data.pyx.item())
            pyx = sum(pyx) / each_domain_datas.__len__()
            each_domain_datas.append(pyx)

        global_mean_pyx = np.mean(global_pyx)
        global_mid_pyx = np.sort(global_pyx)[len(global_pyx) // 2]

        # sorted_domain_split_data_list = sorted(sorted_domain_split_data_list, key=lambda domain_data: domain_data[-1], reverse=)

        bias_connect = [0.95, 0.95, 0.9, 0.85, 0.5]
        is_train_split = [True, False, True, True, False]
        is_val_split = [False if i < len(is_train_split) - 1 else True for i in range(len(is_train_split))]
        is_test_split = [not (tr_sp or val_sp) for tr_sp, val_sp in zip(is_train_split, is_val_split)]

        split_picking_ratio = [0.4, 0.6, 0.5, 1, 1]

        order_connect = [[] for _ in range(len(bias_connect))]
        cur_num = 0
        for i in range(len(sorted_domain_split_data_list)):
            randc = 1 if cur_num < self.num_data / 2 else - 1
            cur_num += sorted_domain_split_data_list[i].__len__() - 1
            for j in range(len(order_connect)):
                order_connect[j].append(randc if is_train_split[j] else - randc)

        env_list = [[] for _ in range(len(bias_connect))]
        cur_split = 0
        env_id = -1
        while cur_split < len(env_list):
            if is_train_split[cur_split]:
                env_id += 1
            next_split = False

            # domain_ids = np.random.permutation(len(sorted_domain_split_data_list))
            # for domain_id in domain_ids:
            #     each_domain_datas = sorted_domain_split_data_list[domain_id]
            for domain_id, each_domain_datas in enumerate(sorted_domain_split_data_list):
                pyx_mean = each_domain_datas[-1]
                pop_items = []
                both_label_domain = [False, False]
                label_data_candidate = [None, None]
                both_label_include = [False, False]
                for i in range(len(each_domain_datas) - 1):
                    data = each_domain_datas[i]
                    picking_rand = random.random()
                    data_rand = random.random()  # random num for data point
                    if cur_split == len(env_list) - 1:
                        data.env_id = env_id
                        env_list[cur_split].append(data)
                        pop_items.append(data)
                    else:
                        # if order_connect[cur_split][domain_id] * (pyx_mean - global_mean_pyx) * (
                        #         data.pyx - pyx_mean) > 0:  # same signal test
                        if order_connect[cur_split][domain_id] * (data.pyx - global_mean_pyx) > 0:
                            both_label_domain[0] = True
                            if data_rand < bias_connect[cur_split] and picking_rand < split_picking_ratio[cur_split]:
                                both_label_include[0] = True
                                data.env_id = env_id
                                env_list[cur_split].append(data)
                                pop_items.append(data)
                            else:
                                label_data_candidate[0] = data
                        else:
                            both_label_domain[1] = True
                            if data_rand > bias_connect[cur_split] and picking_rand < split_picking_ratio[cur_split]:
                                both_label_include[1] = True
                                data.env_id = env_id
                                env_list[cur_split].append(data)
                                pop_items.append(data)
                            else:
                                label_data_candidate[1] = data
                    # if env_list[cur_split].__len__() >= num_split[cur_split]:
                    #     next_split = True
                # --- Add extra data: avoid extreme label imbalance ---
                if both_label_domain[0] and both_label_domain[1] and (both_label_include[0] or both_label_include[1]):
                    extra_data = None
                    if not both_label_include[0]:
                        extra_data = label_data_candidate[0]
                    if not both_label_include[1]:
                        extra_data = label_data_candidate[1]
                    if extra_data:
                        extra_data.env_id = env_id
                        env_list[cur_split].append(extra_data)
                        pop_items.append(extra_data)
                for pop_item in pop_items:
                    each_domain_datas.remove(pop_item)

            cur_split += 1
            num_train = sum([len(env) for i, env in enumerate(env_list) if is_train_split[i]])
            num_val = sum([len(env) for i, env in enumerate(env_list) if is_val_split[i]])
            num_test = sum([len(env) for i, env in enumerate(env_list) if is_test_split[i]])
            print("#D#train: %d, val: %d, test: %d" % (num_train, num_val, num_test))

        train_list, ood_val_list, ood_test_list = list(
            itertools.chain(*[env for i, env in enumerate(env_list) if is_train_split[i]])), \
                                                  list(itertools.chain(
                                                      *[env for i, env in enumerate(env_list) if is_val_split[i]])), \
                                                  list(itertools.chain(
                                                      *[env for i, env in enumerate(env_list) if is_test_split[i]]))
        id_test_ratio = 0.15
        num_id_test = int(len(train_list) * id_test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]
        all_env_list = [train_list, ood_val_list, ood_test_list, id_val_list, id_test_list]

        return self.assign_masks(train_list, ood_val_list, ood_test_list, id_val_list, id_test_list, graph)

    def get_domain_sorted_indices(self, graph, domain='degree'):
        domain_getter = DomainGetter()
        graph.__setattr__(domain, getattr(domain_getter, f'get_{domain}')(graph))

        data_list = []
        for i in range(self.num_data):
            data_info = DataInfo(idx=i, y=graph.y[i])
            data_info.__setattr__(domain, graph.__getattr__(domain)[i])
            data_list.append(data_info)

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
        dataset = CitationFull(root=self.root, name='Cora')
        graph = dataset[0]
        print('Load data done!')
        self.num_data = graph.x.shape[0]
        print('Extract data done!')

        no_shift_graph = self.get_no_shift_graph(deepcopy(graph))
        print('#IN#No shift dataset done!')
        sorted_data_list, sorted_domain_split_data_list = self.get_domain_sorted_indices(graph, domain=self.domain)
        covariate_shift_graph = self.get_covariate_shift_graph(deepcopy(sorted_data_list), deepcopy(graph))
        print()
        print('#IN#Covariate shift dataset done!')
        concept_shift_graph = self.get_concept_shift_graph(deepcopy(sorted_domain_split_data_list), deepcopy(graph))
        print()
        print('#IN#Concept shift dataset done!')

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
        meta_info.dataset_type = 'real'
        meta_info.model_level = 'node'

        dataset = GOODCora(root=dataset_root, domain=domain, shift=shift, generate=generate)
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
