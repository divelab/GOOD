import copy
import os
import shlex
import shutil
from pathlib import Path
import distutils.dir_util

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
from typing import Literal
from cilog import create_logger


@register.launcher_register
class HarvestLauncher(Launcher):
    def __init__(self):
        super(HarvestLauncher, self).__init__()
        self.watch = False
        self.pick_reference = [-1]
        self.test_index = -2
        self.logger = create_logger('Harvest', file='result_table.md', use_color=False)

    def __call__(self, jobs_group, auto_args: AutoArgs):
        result_dict = self.harvest_all_fruits(jobs_group)
        best_fruits = self.picky_farmer(result_dict)

        if auto_args.sweep_root:
            self.process_final_root(auto_args)
            self.update_best_config(auto_args, best_fruits)
        else:
            self.show_your_fruits(best_fruits)

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
                distutils.dir_util.copy_tree(str(auto_args.config_root), str(auto_args.final_root))
            elif ans == 'n':
                pass
            else:
                raise ValueError(f'Unexpected value {ans}.')
        else:
            shutil.copytree(auto_args.config_root, auto_args.final_root)

    def picky_farmer(self, result_dict):
        best_fruits = dict()
        sorted_fruits = dict()
        for ddsa_key in result_dict.keys():
            dataset, domain, shift, algorithm = ddsa_key.split(' ')
            for key, value in result_dict[ddsa_key].items():
                result_dict[ddsa_key][key] = np.stack([np.mean(value, axis=1), np.std(value, axis=1)], axis=1)
            # lambda x: x[1][?, 0]  - ? denotes the result used to choose the best setting.
            if self.watch:
                sorted_fruits[ddsa_key] = sorted(list(result_dict[ddsa_key].items()), key=lambda x: sum(x[1][i, 0] for i in self.pick_reference), reverse=True if 'ZINC' not in dataset else False)
            else:
                if 'ZINC' in dataset:
                    best_fruits[ddsa_key] = min(list(result_dict[ddsa_key].items()), key=lambda x: sum(x[1][i, 0] for i in self.pick_reference))
                else:
                    best_fruits[ddsa_key] = max(list(result_dict[ddsa_key].items()), key=lambda x: sum(x[1][i, 0] for i in self.pick_reference))
            # best_fruits[ddsa_key] = sorted_fruits[ddsa_key][0]
        if self.watch:
            print(sorted_fruits)
            exit(0)
        # print(best_fruits)
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
                continue
            result = last_line.split(' ')[2:]
            num_result = len(result)
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
                result_dict[ddsa_key][key_str] = [[] for _ in range(num_result)]
            # print(f'{ddsa_key}_{key_str}: {result}')
            result_dict[ddsa_key][key_str] = [r + [eval(result[i])] for i, r in
                                              enumerate(result_dict[ddsa_key][key_str])]
        # if not all_finished:
        #     print('Please launch unfinished jobs using other launchers before harvesting.')
        #     exit(1)
        return result_dict

    @staticmethod
    def new_container(key, dictionary, container: Literal['dict', 'list'] = 'dict'):
        if key not in dictionary:
            dictionary[key] = eval(container)()

    def show_your_fruits(self, best_fruits):
        format_best_fruits = dict()
        for ddsa_key, result in best_fruits.items():
            dataset, domain, shift, algorithm = ddsa_key.split(' ')
            # self.new_container(dataset, format_best_fruits)
            # self.new_container(domain, format_best_fruits[dataset])
            # self.new_container(shift, format_best_fruits[dataset][domain])
            self.new_container(algorithm, format_best_fruits)
            dds_key = f'{dataset} {domain} {shift}'

            result = result[1][self.test_index]
            if 'ZINC' not in dataset:
                result = f'{result[0] * 100:.2f}({result[1] * 100:.2f})'
            else:
                result = f'{result[0]:.4f}({result[1]:.4f})'
            # format_best_fruits[dataset][domain][shift][algorithm] = result
            format_best_fruits[algorithm][dds_key] = result

        # for dataset, domain_shift_algorithm in format_best_fruits.items():
        #     for domain, shift_algorithm in domain_shift_algorithm.items():
        #         for shift, algorithm_result in shift_algorithm.items():
        #             self.logger.table_fromlist([[dataset, f'{domain}-{shift}']] + list(algorithm_result.items()))
        for algorithm, dds_key_result in format_best_fruits.items():
            headers = ['Method', *list(dds_key_result.keys())]
            data_row = [algorithm, *list(dds_key_result.values())]
            self.logger.table_fromlist([headers, data_row])


