r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import os
import time
from typing import Tuple, Union

import torch.nn
from torch.utils.data import DataLoader

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger
from GOOD.definitions import OOM_CODE


def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader




def main():
    args = args_parser()
    config = config_summoner(args)
    load_logger(config)

    model, loader = initialize_model_dataset(config)
    ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

    pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
    pipeline.load_task()

    if config.task == 'train':
        pipeline.task = 'test'
        pipeline.load_task()


def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


if __name__ == '__main__':
    main()
