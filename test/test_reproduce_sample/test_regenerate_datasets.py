import os
import shutil
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch_geometric.datasets import MoleculeNet
from GOOD.data.good_datasets.orig_zinc import ZINC

from GOOD import config_summoner, args_parser
from GOOD.definitions import ROOT_DIR, STORAGE_DIR
from GOOD.kernel.main import load_dataset, reset_random_seed


class Regenerator(object):
    def __init__(self, config_path):
        self.args = args_parser(['--config_path', config_path])
        self.config = config_summoner(self.args)

    def __call__(self, *args, **kwargs):
        reset_random_seed(self.config)
        download_dataset = load_dataset(self.config.dataset.dataset_name, config=self.config)

        # --- regenerate ---
        self.config.dataset.generate = True
        self.config.dataset.dataset_root = os.path.join(STORAGE_DIR, 'regenerate_datasets',
                                                        self.config.dataset.dataset_name)

        if self.config.dataset.dataset_name in ['GOODPCBA', 'GOODZINC']:
            dataset_instance = download_dataset['train']
            raw_dataset = MoleculeNet(root=self.config.dataset.dataset_root, name=dataset_instance.mol_name) \
                if self.config.dataset.dataset_name == 'GOODPCBA' \
                else ZINC(root=self.config.dataset.dataset_root, name=dataset_instance.mol_name, subset=True)
            data_list = []
            for i, data in enumerate(raw_dataset):
                data.idx = i
                data_list.append(data)
                if i >= 1000:
                    break
            dataset_instance.num_data = data_list.__len__()

            no_shift_list = dataset_instance.get_no_shift_list(deepcopy(data_list))
            sorted_data_list, sorted_domain_split_data_list = dataset_instance.get_domain_sorted_list(data_list, domain=dataset_instance.domain)
            covariate_shift_list = dataset_instance.get_covariate_shift_list(deepcopy(sorted_data_list))
            concept_shift_list = dataset_instance.get_concept_shift_list(deepcopy(sorted_domain_split_data_list))

            # --- The validity of these function is verified in GOODHIV ---
            assert no_shift_list is not None
            assert covariate_shift_list is not None
            assert concept_shift_list is not None

            return download_dataset, None, self.config.model.model_level

        reset_random_seed(self.config)
        generate_dataset = load_dataset(self.config.dataset.dataset_name, config=self.config)

        return download_dataset, generate_dataset, self.config.model.model_level


config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'GOOD_configs')
for dataset_path in config_root.iterdir():
    if not dataset_path.is_dir():
        continue
    single_dataset_paths = []
    for domain_path in dataset_path.iterdir():
        if not domain_path.is_dir():
            continue
        for shift_path in domain_path.iterdir():
            if not shift_path.is_dir():
                continue
            for ood_config_path in shift_path.iterdir():
                if 'ERM' in ood_config_path.name:
                    single_dataset_paths.append(str(ood_config_path))
    config_paths.append(single_dataset_paths)


# @pytest.mark.skip(reason="It will be tested offline and will be skipped in CI.")
@pytest.mark.parametrize('dataset_paths', config_paths)
def test_regenerate(dataset_paths):
    def regenerate_dataset(config_path):
        regenerator = Regenerator(config_path)
        download_dataset, generate_dataset, graph_node = regenerator()
        if generate_dataset is None and regenerator.config.dataset.dataset_name in ['GOODPCBA', 'GOODZINC']:
            return regenerator.config.dataset.dataset_name
        if graph_node == 'graph':
            if regenerator.config.dataset.dataset_name in ['GOODMotif', 'GOODCMNIST']:
                assert torch.equal(download_dataset['train'].data.y, generate_dataset['train'].data.y)
                assert torch.equal(download_dataset['val'].data.y, generate_dataset['val'].data.y)
                assert torch.equal(download_dataset['test'].data.y, generate_dataset['test'].data.y)
            else:
                assert torch.equal(download_dataset['train'].data.idx, generate_dataset['train'].data.idx)
                assert torch.equal(download_dataset['val'].data.idx, generate_dataset['val'].data.idx)
                assert torch.equal(download_dataset['test'].data.idx, generate_dataset['test'].data.idx)
        else:
            assert torch.equal(download_dataset.data.train_mask, generate_dataset.data.train_mask)
            assert torch.equal(download_dataset.data.val_mask, generate_dataset.data.val_mask)
            assert torch.equal(download_dataset.data.test_mask, generate_dataset.data.test_mask)
        return regenerator.config.dataset.dataset_name

    for dataset_path in dataset_paths:
        if 'GOODSST2' in dataset_path or 'GOODArxiv' in dataset_path:
            return
        dataset_name = regenerate_dataset(dataset_path)
    # release regenerate datasets space
    shutil.rmtree(os.path.join(STORAGE_DIR, 'regenerate_datasets', dataset_name))
