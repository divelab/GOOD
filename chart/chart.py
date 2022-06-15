from GOOD import config_summoner, args_parser
from GOOD.kernel.pipeline import initialize_model_dataset, load_ood_alg, \
    load_task, load_logger, config_model, evaluate, load_dataset, init
from torch_geometric.data import extract_zip
from pathlib import Path
from GOOD.definitions import ROOT_DIR, STORAGE_DIR
import os
import gdown
import shutil
import torch
import pytest

class Charter(object):
    def __init__(self, config_path):
        self.args = args_parser(['--config_path', config_path])
        self.config = config_summoner(self.args)

    def __call__(self, *args, **kwargs):
        init(self.config)
        dataset = load_dataset(self.config.dataset.dataset_name, config=self.config)
        print(f'{self.config.dataset.dataset_name}, {self.config.dataset.shift_type}, {self.config.dataset.domain}: ')
        for set_name in ['train', 'id_val', 'id_test', 'val', 'test']:
            if self.config.model.model_level == 'graph':
                if dataset[set_name] is None:
                    print(f"& - ", end='')
                    continue
                print(f"& {dataset[set_name].data.y.shape[0]} ", end='')
            else:
                if not hasattr(dataset[0], set_name + '_mask'):
                    print(f"& - ", end='')
                    continue
                print(f"& {getattr(dataset[0], set_name + '_mask').sum()}", end='')
        print()



config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'GOOD_configs')
for dataset_path in config_root.iterdir():
    if not dataset_path.is_dir():
        continue
    for domain_path in dataset_path.iterdir():
        if not domain_path.is_dir():
            continue
        for shift_path in domain_path.iterdir():
            if not shift_path.is_dir():
                continue
            for ood_config_path in shift_path.iterdir():
                if 'ERM' in ood_config_path.name:
                    config_paths.append(str(ood_config_path))

for config_path in config_paths:
    charter = Charter(config_path)
    charter()