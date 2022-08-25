import random

from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from typing import List, Iterator
from torch.utils.data.sampler import Sampler
from torch_geometric.data.dataset import Dataset
import torch

class PairBatchSampler(Sampler[List[int]]):
    r"""Surrogated sampler for PairL to yield a mini-batch of indices.

        Args:
            sampler (Sampler or Iterable): Base sampler. Can be any iterable object
            batch_size (int): Size of mini-batch.
            drop_last (bool): If ``True``, the sampler will drop the last batch if
                its size would be less than ``batch_size``

        Example:
            # >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
            # >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
            [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """

    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        targets = dataset.data.y.long().reshape(-1)
        num_label = targets.max() + 1
        num_env = dataset.data.env_id.unique().shape[0]
        # self.env_ids = torch.zeros((config.dataset.num_envs, len(dataset)), dtype=torch.bool, device=config.device)
        # self.target_ids = torch.zeros((num_label, len(dataset)), dtype=torch.bool, device=config.device)
        # for env_id in range(config.dataset.num_envs):
        #     self.env_ids[env_id] = dataset.data.env_id == env_id
        # for target in range(num_label):
        #     self.target_ids[target] = targets == target

        self.y_env_ids = []
        for target in range(num_label):
            self.y_env_ids.append([])
            for env_id in range(num_env):
                self.y_env_ids[target].append(torch.where((targets == target) & (dataset.data.env_id == env_id))[0])

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_label = num_label
        self.num_env = num_env
        self.dataset = dataset

    def __iter__(self) -> Iterator[List[int]]:
        num_data = len(self.dataset)
        random_y = torch.randint(self.num_label, (num_data, ))

        # --- Find different environment pair ---
        random_env_a = torch.randint(self.num_env, (num_data * 2, ))
        random_env_b = torch.randint(self.num_env, (num_data * 2, ))
        diff_env_ids = random_env_a != random_env_b
        random_env_a, random_env_b = random_env_a[diff_env_ids], random_env_b[diff_env_ids]
        while random_env_a.shape[0] < num_data:
            temp_env_a = torch.randint(self.num_env, (num_data, ))
            temp_env_b = torch.randint(self.num_env, (num_data, ))
            diff_env_ids = temp_env_a != temp_env_b
            temp_env_a, temp_env_b = temp_env_a[diff_env_ids], temp_env_b[diff_env_ids]
            random_env_a = torch.cat([random_env_a, temp_env_a], 0)
            random_env_b = torch.cat([random_env_b, temp_env_b], 0)
        random_env_a, random_env_b = random_env_a[:num_data], random_env_b[:num_data]

        batch_a = []
        batch_b = []
        for idx in range(num_data):
            y = random_y[idx]
            env_a = random_env_a[idx]
            env_b = random_env_b[idx]
            ids_a = self.y_env_ids[y][env_a]
            ids_b = self.y_env_ids[y][env_b]
            batch_a.append(ids_a[torch.randint(ids_a.shape[0], (1, ))].item())
            batch_b.append(ids_b[torch.randint(ids_b.shape[0], (1, ))].item())
            if len(batch_a) == self.batch_size:
                yield batch_a + batch_b
                batch_a = []
                batch_b = []
        if len(batch_a) > 0 and not self.drop_last:
            yield batch_a + batch_b

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.dataset) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


@register.dataloader_register
class PairDataLoader(Munch):

    def __init__(self, *args, **kwargs):
        super(PairDataLoader, self).__init__(*args, **kwargs)

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
        loader = {'train': DataLoader(dataset['train'], batch_sampler=PairBatchSampler(dataset=dataset['train'],
                                                                                       batch_size=config.train.train_bs,
                                                                                       drop_last=True)),
                  'eval_train': DataLoader(dataset['train'], batch_size=config.train.val_bs, shuffle=False),
                  'id_val': DataLoader(dataset['id_val'], batch_size=config.train.val_bs, shuffle=False) if dataset.get(
                      'id_val') else None,
                  'id_test': DataLoader(dataset['id_test'], batch_size=config.train.test_bs,
                                        shuffle=False) if dataset.get(
                      'id_test') else None,
                  'val': DataLoader(dataset['val'], batch_size=config.train.val_bs, shuffle=False),
                  'test': DataLoader(dataset['test'], batch_size=config.train.test_bs, shuffle=False)}

        return cls(loader)