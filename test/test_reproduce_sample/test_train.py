import os
from pathlib import Path

import pytest

from GOOD import config_summoner, args_parser
from GOOD.definitions import ROOT_DIR, STORAGE_DIR
from GOOD.kernel.main import initialize_model_dataset, load_ood_alg, load_logger
from GOOD.kernel.pipeline_manager import load_pipeline


class Rerunner(object):
    def __init__(self, config_path):
        self.args = args_parser(['--config_path', config_path,
                                 '--ckpt_root', os.path.join(STORAGE_DIR, 'retrain_ckpts'),
                                 '--exp_round', '1',
                                 '--max_epoch', '1',
                                 '--dim_hidden', '2',
                                 '--dim_ffn', '2',
                                 '--model_layer', '2',
                                 '--train_bs', '32'])
        self.config = config_summoner(self.args)

    def __call__(self, *args, **kwargs):
        config = self.config
        load_logger(config, sub_print=False)
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
        pipeline.load_task()

        return 0


config_paths = []
config_root = Path(ROOT_DIR, 'configs', 'GOOD_configs')
for dataset_path in config_root.iterdir():
    if not dataset_path.is_dir():
        continue
    if dataset_path.name not in ['GOODMotif', 'GOODWebKB']:
        continue
    for domain_path in dataset_path.iterdir():
        if not domain_path.is_dir():
            continue
        if domain_path.name not in ['university', 'basis']:
            continue
        for shift_path in domain_path.iterdir():
            if not shift_path.is_dir():
                continue
            if 'covariate' not in shift_path.name:
                continue
            for ood_config_path in shift_path.iterdir():
                if 'base' in ood_config_path.name:
                    continue
                config_paths.append(str(ood_config_path))


@pytest.mark.parametrize("config_path", config_paths)
def test_train(config_path):
    rerunner = Rerunner(config_path)
    assert rerunner() == 0
