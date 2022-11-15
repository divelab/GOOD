"""
The GOOD-PCBA dataset adapted from `MoleculeNet
<https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.
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
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.datasets import MoleculeNet
from tqdm import tqdm


class DomainGetter():
    r"""
    A class containing methods for data domain extraction.
    """
    def __init__(self):
        pass

    def get_scaffold(self, smile: str) -> str:
        """
        Args:
            smile (str): A smile string for a molecule.
        Returns:
            The scaffold string of the smile.
        """
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smile), includeChirality=False)
            return scaffold
        except ValueError as e:
            print('Get scaffold error.')
            raise e

    def get_nodesize(self, smile: str) -> int:
        """
        Args:
            smile (str): A smile string for a molecule.
        Returns:
            The number of node in the molecule.
        """
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            print('GetNumAtoms error, smiles:{}'.format(smile))
            return len(smile)
        number_atom = mol.GetNumAtoms()
        return number_atom


from GOOD import register


@register.dataset_register
class GOODPCBA(InMemoryDataset):
    r"""
    The GOOD-PCBA dataset. Adapted from `MoleculeNet
    <https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.

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
        self.mol_name = 'PCBA'
        self.domain = domain
        self.metric = 'Average Precision'
        self.task = 'Binary classification'
        self.url = 'https://drive.google.com/file/d/1WGieOjtgNXtGoO6o1EGhKrZj0zWU7AJl/view?usp=sharing'

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
        for data in train_list:
            data.env_id = random.randint(0, 9)

        all_env_list = [train_list, val_list, test_list]

        return all_env_list

    def get_covariate_shift_list(self, sorted_data_list):

        if self.domain == 'size':
            sorted_data_list = sorted_data_list[::-1]
        num_data = sorted_data_list.__len__()
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
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

        num_id_test = int(num_data * test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], train_list[
                                                                                -2 * num_id_test: - num_id_test], \
                                                train_list[- num_id_test:]

        all_env_list = [train_list, ood_val_list, ood_test_list, id_val_list, id_test_list]

        return all_env_list

    def get_concept_shift_list(self, sorted_domain_split_data_list):

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

        split_picking_ratio = [0.3, 0.5, 0.6, 1, 1]

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

        return all_env_list

    def get_domain_sorted_list(self, data_list, domain='scaffold'):
        if domain == 'size':
            domain = 'nodesize'

        domain_getter = DomainGetter()
        for data in tqdm(data_list):
            smile = data.smiles
            data.__setattr__(domain, getattr(domain_getter, f'get_{domain}')(smile))

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
        dataset = MoleculeNet(root=self.root, name=self.mol_name)
        print('Load data done!')
        data_list = []
        for i, data in enumerate(dataset):
            data.idx = i
            data_list.append(data)
        self.num_data = data_list.__len__()
        print('Extract data done!')

        no_shift_list = self.get_no_shift_list(deepcopy(data_list))
        print('#IN#No shift dataset done!')
        sorted_data_list, sorted_domain_split_data_list = self.get_domain_sorted_list(data_list, domain=self.domain)
        covariate_shift_list = self.get_covariate_shift_list(deepcopy(sorted_data_list))
        print()
        print('#IN#Covariate shift dataset done!')
        concept_shift_list = self.get_concept_shift_list(deepcopy(sorted_domain_split_data_list))
        print()
        print('#IN#Concept shift dataset done!')

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
        meta_info.dataset_type = 'mol'
        meta_info.model_level = 'graph'

        train_dataset = GOODPCBA(root=dataset_root,
                                 domain=domain, shift=shift, subset='train', generate=generate)
        id_val_dataset = GOODPCBA(root=dataset_root,
                                  domain=domain, shift=shift, subset='id_val', generate=generate) if shift != 'no_shift' else None
        id_test_dataset = GOODPCBA(root=dataset_root,
                                   domain=domain, shift=shift, subset='id_test', generate=generate) if shift != 'no_shift' else None
        val_dataset = GOODPCBA(root=dataset_root,
                               domain=domain, shift=shift, subset='val', generate=generate)
        test_dataset = GOODPCBA(root=dataset_root,
                                domain=domain, shift=shift, subset='test', generate=generate)

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
        if id_val_dataset:
            id_val_dataset._data_list = None
            id_test_dataset._data_list = None
        val_dataset._data_list = None
        test_dataset._data_list = None

        return {'train': train_dataset, 'id_val': id_val_dataset, 'id_test': id_test_dataset,
                'val': val_dataset, 'test': test_dataset, 'task': train_dataset.task,
                'metric': train_dataset.metric}, meta_info
