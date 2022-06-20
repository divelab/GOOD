r"""A module that is consist of a dataset loading function and a PyTorch dataloader loading function.
"""

from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch


def read_meta_info(meta_info, config: Union[CommonArgs, Munch]):
    config.dataset.dataset_type = meta_info.dataset_type
    config.model.model_level = meta_info.model_level
    config.dataset.dim_node = meta_info.dim_node
    config.dataset.dim_edge = meta_info.dim_edge
    config.dataset.num_envs = meta_info.num_envs
    config.dataset.num_classes = meta_info.num_classes


def load_dataset(name: str, config: Union[CommonArgs, Munch]) -> dir:
    r"""
    Load a dataset given the dataset name.

    Args:
        name (str): Dataset name.
        config (Union[CommonArgs, Munch]): Required configs:
            ``config.dataset.dataset_root``
            ``config.dataset.domain``
            ``config.dataset.shift_type``
            ``config.dataset.generate``

    Returns:
        A dataset object and new configs
            - config.dataset.dataset_type
            - config.model.model_level
            - config.dataset.dim_node
            - config.dataset.dim_edge
            - config.dataset.num_envs
            - config.dataset.num_classes

    """
    try:
        dataset, meta_info = register.datasets[name].load(dataset_root=config.dataset.dataset_root,
                                                          domain=config.dataset.domain,
                                                          shift=config.dataset.shift_type,
                                                          generate=config.dataset.generate)
    except KeyError as e:
        print('Dataset not found.')
        raise e
    read_meta_info(meta_info, config)

    config.metric.set_score_func(dataset['metric'] if type(dataset) is dict else getattr(dataset, 'metric'))
    config.metric.set_loss_func(dataset['task'] if type(dataset) is dict else getattr(dataset, 'task'))

    return dataset


def create_dataloader(dataset, config: Union[CommonArgs, Munch]):
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
    if config.model.model_level == 'node':
        graph = dataset[0]
        loader = GraphSAINTRandomWalkSampler(graph, batch_size=config.train.train_bs,
                                             walk_length=config.model.model_layer,
                                             num_steps=config.train.num_steps, sample_coverage=100,
                                             save_dir=dataset.processed_dir)
        loader = {'train': loader, 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                  'test': [graph]}
    else:
        loader = {'train': DataLoader(dataset['train'], batch_size=config.train.train_bs, shuffle=True),
                  'eval_train': DataLoader(dataset['train'], batch_size=config.train.val_bs, shuffle=False),
                  'id_val': DataLoader(dataset['id_val'], batch_size=config.train.val_bs, shuffle=False) if dataset.get(
                      'id_val') else None,
                  'id_test': DataLoader(dataset['id_test'], batch_size=config.train.test_bs,
                                        shuffle=False) if dataset.get(
                      'id_test') else None,
                  'val': DataLoader(dataset['val'], batch_size=config.train.val_bs, shuffle=False),
                  'test': DataLoader(dataset['test'], batch_size=config.train.test_bs, shuffle=False)}

    return loader
