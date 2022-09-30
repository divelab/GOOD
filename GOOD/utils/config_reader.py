r"""A project configuration module that reads config argument from a file; set automatic generated arguments; and
overwrite configuration arguments by command arguments.
"""

import copy
import warnings
from os.path import join as opj
from pathlib import Path
from typing import Union

import torch
from munch import Munch
from munch import munchify
from ruamel.yaml import YAML
from tap import Tap

from GOOD.definitions import STORAGE_DIR
from GOOD.utils.args import CommonArgs
from GOOD.utils.metric import Metric


def merge_dicts(dict1: dict, dict2: dict):
    """Recursively merge two dictionaries.
    Values in dict2 override values in dict1. If dict1 and dict2 contain a dictionary as a
    value, this will call itself recursively to merge these dictionaries.
    This does not modify the input dictionaries (creates an internal copy).
    Additionally returns a list of detected duplicates.
    Adapted from https://github.com/TUM-DAML/seml/blob/master/seml/utils.py
    Parameters
    ----------
    dict1: dict
        First dict.
    dict2: dict
        Second dict. Values in dict2 will override values from dict1 in case they share the same key.
    Returns
    -------
    return_dict: dict
        Merged dictionaries.
    """
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def load_config(path: str, previous_includes: list = [], skip_include=False) -> dict:
    r"""Config loader.
    Loading configs from a config file.

    Args:
        path (str): The path to your yaml configuration file.
        previous_includes (list): Included configurations. It is for the :obj:`include` configs used for recursion.
            Please leave it blank when call this function outside.

    Returns:
        config (dict): config dictionary loaded from the given yaml file.
    """
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    yaml = YAML(typ='safe')
    direct_config = yaml.load(open(path, "r"))
    if skip_include:
        return direct_config, None, None
    # direct_config = yaml.safe_load(open(path, "r"))

    # Load config from included files.
    if "includes" in direct_config:
        includes = direct_config.pop("includes")
    else:
        includes = []
    if not isinstance(includes, list):
        raise AttributeError(
            "Includes must be a list, '{}' provided".format(type(includes))
        )

    config = {}
    duplicates_warning = []
    duplicates_error = []

    for include in includes:
        include = path.parent / include
        include_config, inc_dup_warning, inc_dup_error = load_config(
            include, previous_includes
        )
        duplicates_warning += inc_dup_warning
        duplicates_error += inc_dup_error

        # Duplicates between includes causes an error
        config, merge_dup_error = merge_dicts(config, include_config)
        duplicates_error += merge_dup_error

    # Duplicates between included and main file causes warnings
    config, merge_dup_warning = merge_dicts(config, direct_config)
    duplicates_warning += merge_dup_warning

    return config, duplicates_warning, duplicates_error


def search_tap_args(args: CommonArgs, query: str):
    r"""
    Search a key in command line arguments.

    Args:
        args (CommonArgs): Command line arguments.
        query (str): The query for the target argument.

    Returns:
        A found or not flag and the target value if found.
    """
    found = False
    value = None
    for key in args.class_variables.keys():
        if query == key:
            found = True
            value = getattr(args, key)
        elif issubclass(type(getattr(args, key)), Tap):
            found, value = search_tap_args(getattr(args, key), query)
        if found:
            break
    return found, value


def args2config(config: Union[CommonArgs, Munch], args: CommonArgs):
    r"""
    Overwrite config by assigned arguments.
    If an argument is not :obj:`None`, this argument has the highest priority; thus, it will overwrite the corresponding
    config.

    Args:
        config (Union[CommonArgs, Munch]): Loaded configs.
        args (CommonArgs): Command line arguments.

    Returns:
        Overwritten configs.
    """
    for key in config.keys():
        if type(config[key]) is dict:
            args2config(config[key], args)
        else:
            found, value = search_tap_args(args, key)
            if found:
                if value is not None:
                    config[key] = value
            else:
                warnings.warn(f'Argument {key} in the chosen config yaml file are not defined in command arguments, '
                              f'which will lead to incomplete code detection and the lack of argument temporary '
                              f'modification by adding command arguments.')


def process_configs(config: Union[CommonArgs, Munch]):
    r"""
    Process loaded configs.
    This process includes setting storage places for datasets, tensorboard logs, logs, and checkpoints. In addition,
    we also set random seed for each experiment round, checkpoint saving gap, and gpu device. Finally, we connect the
    config with two components :class:`GOOD.utils.metric.Metric` and :class:`GOOD.utils.train.TrainHelper` for easy and
    unified accesses.

    Args:
        config (Union[CommonArgs, Munch]): Loaded configs.

    Returns:
        Configs after setting.
    """
    # --- Dataset setting ---
    if config.dataset.dataset_root is None:
        config.dataset.dataset_root = opj(STORAGE_DIR, 'datasets')

    # --- tensorboard directory setting ---
    config.tensorboard_logdir = opj(STORAGE_DIR, 'tensorboard', f'{config.dataset.dataset_name}')
    if config.dataset.shift_type:
        config.tensorboard_logdir = opj(config.tensorboard_logdir, config.dataset.shift_type, config.ood.ood_alg,
                                        str(config.ood.ood_param))

    # --- Round setting ---
    if config.exp_round:
        config.random_seed = config.exp_round * 97 + 13

    # --- Directory name definitions ---
    dataset_dirname = config.dataset.dataset_name + '_' + config.dataset.domain
    if config.dataset.shift_type:
        dataset_dirname += '_' + config.dataset.shift_type
    model_dirname = f'{config.model.model_name}_{config.model.model_layer}l_{config.model.global_pool}pool_{config.model.dropout_rate}dp'
    train_dirname = f'{config.train.lr}lr_{config.train.weight_decay}wd'
    ood_dirname = config.ood.ood_alg
    if config.ood.ood_param is not None and config.ood.ood_param >= 0:
        ood_dirname += f'_{config.ood.ood_param}'
    else:
        ood_dirname += '_no_param'
    if config.ood.extra_param is not None:
        for i, param in enumerate(config.ood.extra_param):
            ood_dirname += f'_{param}'

    # --- Log setting ---
    log_dir_root = opj(STORAGE_DIR, 'log', 'round' + str(config.exp_round))
    log_dirs = opj(log_dir_root, dataset_dirname, model_dirname, train_dirname, ood_dirname)
    if config.save_tag:
        log_dirs = opj(log_dirs, config.save_tag)
    config.log_path = opj(log_dirs, config.log_file + '.log')

    # --- Checkpoint setting ---
    if config.ckpt_root is None:
        config.ckpt_root = opj(STORAGE_DIR, 'checkpoints')
    if config.ckpt_dir is None:
        config.ckpt_dir = opj(config.ckpt_root, 'round' + str(config.exp_round))
        config.ckpt_dir = opj(config.ckpt_dir, dataset_dirname, model_dirname, train_dirname, ood_dirname)
        if config.save_tag:
            config.ckpt_dir = opj(config.ckpt_dir, config.save_tag)
    config.test_ckpt = opj(config.ckpt_dir, f'best.ckpt')
    config.id_test_ckpt = opj(config.ckpt_dir, f'id_best.ckpt')

    # --- Other settings ---
    if config.train.max_epoch > 100:
        config.train.save_gap = config.train.max_epoch // 10
    config.device = torch.device(f'cuda:{config.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    config.train.stage_stones.append(100000)

    # --- Attach train_helper and metric modules ---
    config.metric = Metric()


def config_summoner(args: CommonArgs) -> Union[CommonArgs, Munch]:
    r"""
    A config loading and postprocessing function.

    Args:
        args (CommonArgs): Command line arguments.

    Returns:
        Processed configs.
    """
    config, duplicate_warnings, duplicate_errors = load_config(args.config_path)
    args2config(config, args)
    config = munchify(config)
    process_configs(config)
    return config
