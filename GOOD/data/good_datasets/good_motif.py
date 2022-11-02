"""
The GOOD-Motif dataset motivated by `Spurious-Motif
<https://arxiv.org/abs/2201.12872>`_.
"""
import math
import os
import os.path as osp
import random

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *
from GOOD.utils.synthetic_data import synthetic_structsim


@register.dataset_register
class GOODMotif(InMemoryDataset):
    r"""
    The GOOD-Motif dataset motivated by `Spurious-Motif
    <https://arxiv.org/abs/2201.12872>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'basis' and 'size'.
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
        self.url = 'https://drive.google.com/file/d/15YRuZG6wI4HF7QgrLI52POKjuObsOyvb/view?usp=sharing'

        self.generate = generate

        self.all_basis = ["wheel", "tree", "ladder", "star", "path"]
        self.basis_role_end = {'wheel': 0, 'tree': 0, 'ladder': 0, 'star': 1, 'path': 1}
        self.all_motifs = [[["house"]], [["dircycle"]], [["crane"]]]
        self.num_data = 30000
        self.train_spurious_ratio = [0.99, 0.97, 0.95]

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

    def gen_data(self, basis_id, width_basis, motif_id):
        basis_type = self.all_basis[basis_id]
        if basis_type == 'tree':
            width_basis = int(math.log2(width_basis)) - 1
            if width_basis <= 0:
                width_basis = 1
        list_shapes = self.all_motifs[motif_id]
        G, role_id, _ = synthetic_structsim.build_graph(
            width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
        )
        G = perturb([G], 0.05, id=role_id)[0]
        # from GOOD.causal_engine.graph_visualize import plot_graph
        # print(G.edges())
        # plot_graph(G, colors=[1 for _ in G.nodes()])

        # --- Convert networkx graph into pyg data ---
        data = from_networkx(G)
        data.x = torch.ones((data.num_nodes, 1))
        role_id = torch.tensor(role_id, dtype=torch.long)
        role_id[role_id <= self.basis_role_end[basis_type]] = 0
        role_id[role_id != 0] = 1

        edge_gt = torch.stack([role_id[data.edge_index[0]], role_id[data.edge_index[1]]]).sum(0) > 1.5

        data.node_gt = role_id
        data.edge_gt = edge_gt
        data.basis_id = basis_id
        data.motif_id = motif_id

        # --- noisy labels ---
        if random.random() < 0.1:
            data.y = random.randint(0, 2)
        else:
            data.y = motif_id

        return data

    def get_no_shift_list(self, num_data=60000):
        data_list = []
        for motif_id in tqdm(range(3)):
            for _ in range(num_data // 3):
                basis_id = np.random.choice([0, 1, 2, 3, 4], p=[1. / 5.] * 5)
                width_basis = 10 + np.random.random_integers(-5, 5)
                data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
                data_list.append(data)

        random.shuffle(data_list)

        num_data = data_list.__len__()
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        train_split = int(num_data * train_ratio)
        val_split = int(num_data * (train_ratio + val_ratio))
        train_list, val_list, test_list = data_list[: train_split], data_list[train_split: val_split], data_list[
                                                                                                       val_split:]
        num_env_train = 3
        num_per_env = train_split // num_env_train
        train_env_list = []
        for i in range(num_env_train):
            train_env_list.append(train_list[i * num_per_env: (i + 1) * num_per_env])

        all_env_list = [env_list for env_list in train_env_list] + [val_list, test_list]

        for env_id, env_list in enumerate(all_env_list):
            for data in env_list:
                data.env_id = torch.LongTensor([env_id])

        tmp = []
        for env_list in all_env_list[: num_env_train]:
            tmp += env_list
        all_env_list = [tmp] + [all_env_list[num_env_train]] + \
                       [all_env_list[num_env_train + 1]]

        return all_env_list

    def get_basis_covariate_shift_list(self, num_data=60000):
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        train_num = int(num_data * train_ratio)
        val_num = int(num_data * val_ratio)
        test_num = int(num_data * test_ratio)
        split_num = [train_num, val_num, test_num]
        all_split_list = [[] for _ in range(3)]
        for split_id in range(3):
            for _ in range(split_num[split_id]):
                motif_id = random.randint(0, 2)
                if split_id == 0:
                    basis_id = random.randint(0, 2)
                else:
                    basis_id = split_id + 2
                width_basis = 10 + np.random.random_integers(-5, 5)
                data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
                data.env_id = torch.LongTensor([basis_id])
                all_split_list[split_id].append(data)

        train_list = all_split_list[0]
        num_id_test = int(num_data * test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], train_list[- num_id_test:]

        ood_val_list = all_split_list[1]
        ood_test_list = all_split_list[2]

        all_env_list = [train_list, ood_val_list, ood_test_list, id_val_list, id_test_list]

        return all_env_list

    def get_basis_concept_shift_list(self, num_data=60000):
        # data_list = []
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        num_train = int(num_data * train_ratio)
        num_val = int(num_data * val_ratio)
        num_test = int(num_data * test_ratio)
        train_spurious_ratio = self.train_spurious_ratio
        val_spurious_ratio = [0.3]
        test_spurious_ratio = [0.0]
        train_list = []
        for env_id in tqdm(range(len(train_spurious_ratio))):
            for i in range(num_train // len(train_spurious_ratio)):
                motif_id = random.randint(0, 2)
                width_basis = 10 + np.random.random_integers(-5, 5)
                if random.random() < train_spurious_ratio[env_id]:
                    basis_id = motif_id
                else:
                    basis_id = random.randint(0, 2)
                data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
                data.env_id = torch.LongTensor([env_id])
                train_list.append(data)

        val_list = []
        for i in range(num_val):
            motif_id = random.randint(0, 2)
            width_basis = 10 + np.random.random_integers(-5, 5)
            if random.random() < val_spurious_ratio[0]:
                basis_id = motif_id
            else:
                basis_id = random.randint(0, 2)
            data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
            val_list.append(data)

        test_list = []
        for i in range(num_test):
            motif_id = random.randint(0, 2)
            width_basis = 10 + np.random.random_integers(-5, 5)
            if random.random() < test_spurious_ratio[0]:
                basis_id = motif_id
            else:
                basis_id = random.randint(0, 2)
            data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
            test_list.append(data)

        id_test_ratio = 0.15
        num_id_test = int(len(train_list) * id_test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], train_list[- num_id_test:]

        all_env_list = [train_list, val_list, test_list, id_val_list, id_test_list]

        return all_env_list

    def get_size_covariate_shift_list(self, num_data=60000):
        # data_list = []
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1
        train_num = int(num_data * train_ratio)
        val_num = int(num_data * val_ratio)
        test_num = int(num_data * test_ratio)
        split_num = [train_num, val_num, test_num]
        all_width_basis = [6, 10, 15, 30, 70]
        all_split_list = [[] for _ in range(3)]
        for split_id in range(3):
            for _ in range(split_num[split_id]):
                if split_id == 0:
                    width_id = random.randint(0, 2)
                else:
                    width_id = split_id + 2
                basis_id = random.randint(0, 4)
                motif_id = random.randint(0, 2)
                width_basis = all_width_basis[width_id] + random.randint(-5, 5)
                data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
                data.width_id = width_id
                data.env_id = torch.LongTensor([width_id])
                all_split_list[split_id].append(data)

        train_list = all_split_list[0]
        num_id_test = int(num_data * test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], train_list[- num_id_test:]

        ood_val_list = all_split_list[1]
        ood_test_list = all_split_list[2]

        all_env_list = [train_list, ood_val_list, ood_test_list, id_val_list, id_test_list]

        return all_env_list

    def get_size_concept_shift_list(self, num_data=60000):
        # data_list = []
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        num_train = int(num_data * train_ratio)
        num_val = int(num_data * val_ratio)
        num_test = int(num_data * test_ratio)
        all_width_basis = [10, 40, 70]
        train_spurious_ratio = self.train_spurious_ratio
        val_spurious_ratio = [0.3]
        test_spurious_ratio = [0.0]
        train_list = []
        for env_id in tqdm(range(len(train_spurious_ratio))):
            for i in range(num_train // len(train_spurious_ratio)):
                basis_id = np.random.choice([0, 1, 2, 3, 4], p=[1. / 5.] * 5)
                motif_id = random.randint(0, 2)
                if random.random() < train_spurious_ratio[env_id]:
                    width_id = motif_id
                else:
                    width_id = random.randint(0, 2)
                width_basis = all_width_basis[width_id] + random.randint(-5, 5)
                data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
                data.width_id = width_id
                data.env_id = torch.LongTensor([env_id])
                train_list.append(data)

        val_list = []
        for i in range(num_val):
            basis_id = np.random.choice([0, 1, 2, 3, 4], p=[1. / 5.] * 5)
            motif_id = random.randint(0, 2)
            if random.random() < val_spurious_ratio[0]:
                width_id = motif_id
            else:
                width_id = random.randint(0, 2)
            width_basis = all_width_basis[width_id] + random.randint(-5, 5)
            data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
            data.width_id = width_id
            val_list.append(data)

        test_list = []
        for i in range(num_test):
            basis_id = np.random.choice([0, 1, 2, 3, 4], p=[1. / 5.] * 5)
            motif_id = random.randint(0, 2)
            if random.random() < test_spurious_ratio[0]:
                width_id = motif_id
            else:
                width_id = random.randint(0, 2)
            width_basis = all_width_basis[width_id] + random.randint(-5, 5)
            data = self.gen_data(basis_id=basis_id, width_basis=width_basis, motif_id=motif_id)
            data.width_id = width_id
            test_list.append(data)

        id_test_ratio = 0.15
        num_id_test = int(len(train_list) * id_test_ratio)
        random.shuffle(train_list)
        train_list, id_val_list, id_test_list = train_list[: -2 * num_id_test], \
                                                train_list[-2 * num_id_test: - num_id_test], train_list[- num_id_test:]

        all_env_list = [train_list, val_list, test_list, id_val_list, id_test_list]

        return all_env_list

    def process(self):

        no_shift_list = self.get_no_shift_list(self.num_data)
        print("#IN#No shift done!")
        if self.domain == 'basis':
            covariate_shift_list = self.get_basis_covariate_shift_list(self.num_data)
            print("#IN#Covariate shift done!")
            concept_shift_list = self.get_basis_concept_shift_list(self.num_data)
            print("#IN#Concept shift done!")
        elif self.domain == 'size':
            covariate_shift_list = self.get_size_covariate_shift_list(self.num_data)
            print("#IN#Covariate shift done!")
            concept_shift_list = self.get_size_concept_shift_list(self.num_data)
            print("#IN#Concept shift done!")
        else:
            raise ValueError(f'Dataset domain cannot be "{self.domain}"')

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

        train_dataset = GOODMotif(root=dataset_root,
                                  domain=domain, shift=shift, subset='train', generate=generate)
        id_val_dataset = GOODMotif(root=dataset_root,
                                   domain=domain, shift=shift, subset='id_val', generate=generate) if shift != 'no_shift' else None
        id_test_dataset = GOODMotif(root=dataset_root,
                                    domain=domain, shift=shift, subset='id_test', generate=generate) if shift != 'no_shift' else None
        val_dataset = GOODMotif(root=dataset_root,
                                domain=domain, shift=shift, subset='val', generate=generate)
        test_dataset = GOODMotif(root=dataset_root,
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
