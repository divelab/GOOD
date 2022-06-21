r"""A module that is consist of a GNN model loader and model configuration function.
"""
import os

import torch

from GOOD import register
from GOOD.utils.config_reader import Union, CommonArgs, Munch


def load_model(name: str, config: Union[CommonArgs, Munch]) -> torch.nn.Module:
    r"""
    A model loader.
    Args:
        name (str): Name of the chosen GNN.
        config (Union[CommonArgs, Munch]): Please refer to specific GNNs for required configs and formats.

    Returns:
        A instantiated GNN model.

    """
    try:
        model = register.models[name](config)
    except KeyError as e:
        print(f'#E#Model {name} dose not exist.')
        raise e
    return model


from GOOD.utils.config_reader import Union, CommonArgs, Munch


def config_model(model: torch.nn.Module, mode: str, config: Union[CommonArgs, Munch], load_param=False):
    r"""
    A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
    Args:
        model (torch.nn.Module): The GNN object.
        mode (str): 'train' or 'test'.
        config (Union[CommonArgs, Munch]): Only for project use. Please resort to the source code for required arguments.
        load_param: When True, loading test checkpoint will load parameters to the GNN model.

    Returns:
        Test score and loss if mode=='test'.
    """
    model.to(config.device)
    model.train()

    # load checkpoint
    if mode == 'train' and config.train.tr_ctn:
        ckpt = torch.load(os.path.join(config.ckpt_dir, f'last.ckpt'))
        model.load_state_dict(ckpt['state_dict'])
        best_ckpt = torch.load(os.path.join(config.ckpt_dir, f'best.ckpt'))
        config.metric.best_stat['score'] = best_ckpt['val_score']
        config.metric.best_stat['loss'] = best_ckpt['val_loss']
        config.train.ctn_epoch = ckpt['epoch'] + 1
        print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

    if mode == 'test':
        try:
            ckpt = torch.load(config.test_ckpt, map_location=config.device)
        except FileNotFoundError:
            print(f'#E#Checkpoint not found at {os.path.abspath(config.test_ckpt)}')
            exit(1)
        if os.path.exists(config.id_test_ckpt):
            id_ckpt = torch.load(config.id_test_ckpt, map_location=config.device)
            # model.load_state_dict(id_ckpt['state_dict'])
            print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
            print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                  f'Train {config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                  f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
                  f'ID Validation {config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                  f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                  f'ID Test {config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                  f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                  f'OOD Validation {config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                  f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
                  f'OOD Test {config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                  f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
            print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
            print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                  f'Train {config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                  f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                  f'ID Validation {config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                  f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                  f'ID Test {config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                  f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                  f'OOD Validation {config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                  f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                  f'OOD Test {config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                  f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')

            print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                  f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

        else:
            print(f'#IN#No In-Domain checkpoint.')
            # model.load_state_dict(ckpt['state_dict'])
            print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
            print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                  f'Train {config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                  f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                  f'Validation {config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                  f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                  f'Test {config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                  f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

            print(
                f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
        if load_param:
            model.load_state_dict(ckpt['state_dict'])
        return ckpt["test_score"], ckpt["test_loss"]
