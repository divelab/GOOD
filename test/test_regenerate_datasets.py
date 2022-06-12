from GOOD import config_summoner, args_parser
from GOOD.kernel.pipeline import initialize_model_dataset, load_ood_alg, \
    load_task, load_logger, config_model, evaluate, load_dataset, init
from torch_geometric.data import extract_zip
from pathlib import Path
from GOOD.definitions import ROOT_DIR, STORAGE_DIR
import os
import gdown
import torch
import pytest

class Regenerator(object):
    def __init__(self, config_path):
        self.args = args_parser(['--config_path', config_path])
        self.config = config_summoner(self.args)

    def __call__(self, *args, **kwargs):
        init(self.config)
        download_dataset = load_dataset(self.config.dataset.dataset_name, config=self.config)

        self.config.dataset.generate = True
        self.config.dataset.dataset_root = os.path.join(STORAGE_DIR, 'regenerate_datasets')
        init(self.config)
        generate_dataset = load_dataset(self.config.dataset.dataset_name, config=self.config)

        return download_dataset, generate_dataset, self.config.model.model_level


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


@pytest.mark.parametrize('config_path', config_paths)
def test_regenerate(config_path):
    regenerator = Regenerator(config_path)
    download_dataset, generate_dataset, graph_node = regenerator()
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