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
from GOOD.utils.args import args_parser
from GOOD.utils.metric import Metric
from GOOD.utils.train import TrainHelper


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


def load_config(path: str, previous_includes: list = []):
    path = Path(path)
    if path in previous_includes:
        raise ValueError(
            f"Cyclic config include detected. {path} included in sequence {previous_includes}."
        )
    previous_includes = previous_includes + [path]

    yaml = YAML(typ='safe')
    direct_config = yaml.load(open(path, "r"))
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


def search_tap_args(args: CommonArgs, query):
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


def args2config(config: Union[CommonArgs, Munch], args):
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
    if config.dataset.dataset_root is None:
        config.dataset.dataset_root = opj(STORAGE_DIR, 'datasets')
    config.device = torch.device(f'cuda:{config.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    # --- tensorboard directory setting ---
    config.tensorboard_logdir = opj(STORAGE_DIR, 'tensorboard', f'{config.dataset.dataset_name}')
    if config.dataset.shift_type:
        config.tensorboard_logdir = opj(config.tensorboard_logdir, config.dataset.shift_type, config.ood.ood_alg,
                                        str(config.ood.ood_param))

    # --- Round setting ---
    if config.exp_round:
        config.random_seed = config.exp_round * 97 + 13

    log_dir_root = opj(STORAGE_DIR, 'log', 'round' + str(config.exp_round))
    log_dirs = opj(log_dir_root, config.dataset.dataset_name, config.dataset.domain)
    if config.dataset.shift_type:
        log_dirs = opj(log_dirs, config.dataset.shift_type)
    log_dirs = opj(log_dirs, config.ood.ood_alg)
    config.log_path = opj(log_dirs, config.log_file + '.log')

    if config.ckpt_root is None:
        config.ckpt_root = opj(STORAGE_DIR, 'checkpoints')
    if config.ckpt_dir is None:
        config.ckpt_dir = opj(config.ckpt_root, 'round' + str(config.exp_round),
                              config.dataset.dataset_name, config.dataset.domain,
                              f'{config.model.model_name}_{config.model.model_layer}l')
        if config.dataset.shift_type:
            config.ckpt_dir = opj(config.ckpt_dir, config.dataset.shift_type)
        config.ckpt_dir = opj(config.ckpt_dir, config.ood.ood_alg, config.model.global_pool)
        if config.ood.ood_param is not None and config.ood.ood_param >= 0:
            config.ckpt_dir = opj(config.ckpt_dir, str(config.ood.ood_param))
        else:
            config.ckpt_dir = opj(config.ckpt_dir, 'no_param')
        if config.save_tag:
            config.ckpt_dir = opj(config.ckpt_dir, config.save_tag)
    config.test_ckpt = opj(config.ckpt_dir, f'best.ckpt')
    config.id_test_ckpt = opj(config.ckpt_dir, f'id_best.ckpt')

    if config.train.max_epoch > 100:
        config.train.save_gap = config.train.max_epoch // 10

    # Attach train_helper and metric modules
    config.metric = Metric()
    config.train_helper = TrainHelper()


def config_summoner(args: CommonArgs) -> Union[CommonArgs, Munch]:
    config, duplicate_warnings, duplicate_errors = load_config(args.config_path)
    args2config(config, args)
    config = munchify(config)
    process_configs(config)
    return config