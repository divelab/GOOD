import random

from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from typing import List, Iterator
from torch.utils.data.sampler import Sampler
from torch_geometric.data.dataset import Dataset
import numpy as np
import torch

@register.dataloader_register
class BaseDataLoader(Munch):

    def __init__(self, *args, **kwargs):
        super(BaseDataLoader, self).__init__(*args, **kwargs)

    @classmethod
    def setup(cls, dataset, config: Union[CommonArgs, Munch]):
        r"""
        Create a PyG data loader.

        Args:
            dataset: A GOOD dataset.
            config: Required configs:
                ``config.train.train_bs``
                ``config.train.val_bs``
                ``config.train.test_bs``
                ``config.model.model_layer``
                ``config.train.num_steps(for node prediction)``

        Returns:
            A PyG dataset loader.

        """
        reset_random_seed(config)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        g = torch.Generator()
        g.manual_seed(config.random_seed)

        if config.model.model_level == 'node':
            graph = dataset[0]
            loader = GraphSAINTRandomWalkSampler(graph, batch_size=config.train.train_bs,
                                                 walk_length=config.model.model_layer,
                                                 num_steps=config.train.num_steps, sample_coverage=100,
                                                 save_dir=dataset.processed_dir)
            if config.ood.ood_alg == 'EERM':
                loader = {'train': [graph], 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                          'test': [graph]}
            else:
                loader = {'train': loader, 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                          'test': [graph]}
        else:
            loader = {'train': DataLoader(dataset['train'], batch_size=config.train.train_bs, shuffle=True, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g),
                      'eval_train': DataLoader(dataset['train'], batch_size=config.train.val_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g),
                      'id_val': DataLoader(dataset['id_val'], batch_size=config.train.val_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g) if dataset.get(
                          'id_val') else None,
                      'id_test': DataLoader(dataset['id_test'], batch_size=config.train.test_bs,
                                            shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g) if dataset.get(
                          'id_test') else None,
                      'val': DataLoader(dataset['val'], batch_size=config.train.val_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g),
                      'test': DataLoader(dataset['test'], batch_size=config.train.test_bs, shuffle=False, num_workers=config.num_workers, worker_init_fn=seed_worker, generator=g)}

        return cls(loader)