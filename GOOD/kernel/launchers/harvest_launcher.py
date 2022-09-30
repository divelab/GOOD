import copy
import os
import shlex
import shutil
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from tqdm import tqdm

from GOOD import config_summoner
from GOOD import register
from GOOD.definitions import ROOT_DIR
from GOOD.utils.args import AutoArgs
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import load_config, args2config, merge_dicts
from .basic_launcher import Launcher


@register.launcher_register
class HarvestLauncher(Launcher):
    def __init__(self):
        super(HarvestLauncher, self).__init__()

    def __call__(self, jobs_group, auto_args: AutoArgs):
        result_dict = self.harvest_all_fruits(jobs_group)
        best_fruits = self.picky_farmer(result_dict)

        if auto_args.sweep_root:
            self.process_final_root(auto_args)
            self.update_best_config(auto_args, best_fruits)

    def update_best_config(self, auto_args, best_fruits):
        for ddsa_key in best_fruits.keys():
            final_path = (auto_args.final_root / '/'.join(ddsa_key.split(' '))).with_suffix('.yaml')
            top_config, _, _ = load_config(final_path, skip_include=True)
            whole_config, _, _ = load_config(final_path)
            args_list = shlex.split(best_fruits[ddsa_key][0])
            args = args_parser(['--config_path', str(final_path)] + args_list)
            args2config(whole_config, args)
            args_keys = [item[2:] for item in args_list if item.startswith('--')]
            # print(args_keys)
            modified_config = self.filter_config(whole_config, args_keys)
            # print(top_config)
            # print(modified_config)
            final_top_config, _ = merge_dicts(top_config, modified_config)
            # print(final_top_config)
            yaml = YAML()
            yaml.indent(offset=2)
            # yaml.dump(final_top_config, sys.stdout)
            yaml.dump(final_top_config, final_path)

    def process_final_root(self, auto_args):
        if auto_args.final_root is None:
            auto_args.final_root = auto_args.config_root
        else:
            if os.path.isabs(auto_args.final_root):
                auto_args.final_root = Path(auto_args.final_root)
            else:
                auto_args.final_root = Path(ROOT_DIR, 'configs', auto_args.final_root)
        if auto_args.final_root.exists():
            ans = input(f'Overwrite {auto_args.final_root} by {auto_args.config_root}? [y/n]')
            while ans != 'y' and ans != 'n':
                ans = input(f'Invalid input: {ans}. Please answer y or n.')
            if ans == 'y':
                shutil.copytree(auto_args.config_root, auto_args.final_root)
            elif ans == 'n':
                pass
            else:
                raise ValueError(f'Unexpected value {ans}.')
        else:
            shutil.copytree(auto_args.config_root, auto_args.final_root)

    def picky_farmer(self, result_dict):
        best_fruits = dict()
        for ddsa_key in result_dict.keys():
            for key, value in result_dict[ddsa_key].items():
                result_dict[ddsa_key][key] = np.stack([np.mean(value, axis=1), np.std(value, axis=1)], axis=1)
            best_fruits[ddsa_key] = max(list(result_dict[ddsa_key].items()), key=lambda x: x[1][-1, 0])
        print(best_fruits)
        return best_fruits

    def filter_config(self, config: dict, target_keys):
        new_config = copy.deepcopy(config)
        for key in config.keys():
            if type(config[key]) is dict:
                new_config[key] = self.filter_config(config[key], target_keys)
                if not new_config[key]:
                    new_config.pop(key)
            else:
                if key not in target_keys:
                    new_config.pop(key)
        return new_config

    def harvest_all_fruits(self, jobs_group):
        all_finished = True
        result_dict = dict()
        for cmd_args in tqdm(jobs_group, desc='Harvesting ^_^'):
            args = args_parser(shlex.split(cmd_args)[1:])
            config = config_summoner(args)
            last_line = self.harvest(config.log_path)

            if not last_line.startswith('INFO: ChartInfo'):
                print(cmd_args, 'Unfinished')
                all_finished = False
            result = last_line.split(' ')[2:]
            key_args = shlex.split(cmd_args)[1:]
            round_index = key_args.index('--exp_round')
            key_args = key_args[:round_index] + key_args[round_index + 2:]

            config_path_index = key_args.index('--config_path')
            key_args.pop(config_path_index)  # Remove --config_path
            config_path = Path(key_args.pop(config_path_index))  # Remove and save its value
            config_path_parents = config_path.parents
            dataset, domain, shift, algorithm = config_path_parents[2].stem, config_path_parents[1].stem, \
                                                config_path_parents[0].stem, config_path.stem
            ddsa_key = ' '.join([dataset, domain, shift, algorithm])
            if ddsa_key not in result_dict.keys():
                result_dict[ddsa_key] = dict()

            key_str = ' '.join(key_args)
            if key_str not in result_dict[ddsa_key].keys():
                result_dict[ddsa_key][key_str] = [[] for _ in range(5)]
            result_dict[ddsa_key][key_str] = [r + [eval(result[i])] for i, r in
                                              enumerate(result_dict[ddsa_key][key_str])]
        if not all_finished:
            print('Please launch unfinished jobs using other launchers before harvesting.')
            exit(1)
        return result_dict
