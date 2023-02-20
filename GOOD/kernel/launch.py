import itertools
import os
import os.path
import sys
from pathlib import Path

from GOOD.definitions import ROOT_DIR
from GOOD.kernel.launcher_manager import load_launcher
from GOOD.utils.args import AutoArgs
from GOOD.utils.config_reader import load_config


def launch():
    conda_interpreter = sys.executable
    conda_goodtg = os.path.join(sys.exec_prefix, 'bin', 'goodtg')
    auto_args = AutoArgs().parse_args(known_only=True)
    auto_args.config_root = get_config_root(auto_args)

    jobs_group = make_list_cmds(auto_args, conda_goodtg)
    launcher = load_launcher(auto_args.launcher)
    launcher(jobs_group, auto_args)


def get_config_root(auto_args):
    if auto_args.config_root:
        if os.path.isabs(auto_args.config_root):
            config_root = Path(auto_args.config_root)
        else:
            config_root = Path(ROOT_DIR, 'configs', auto_args.config_root)
    else:
        config_root = Path(ROOT_DIR, 'configs', 'GOOD_configs')
    return config_root


def make_list_cmds(auto_args, conda_goodtg):
    args_group = []
    for dataset_path in auto_args.config_root.iterdir():
        if not dataset_path.is_dir() or dataset_path.name not in auto_args.allow_datasets:
            continue
        for domain_path in dataset_path.iterdir():
            if not domain_path.is_dir() or (
                    auto_args.allow_domains and domain_path.name not in auto_args.allow_domains):
                continue
            for shift_path in domain_path.iterdir():
                if not shift_path.is_dir() or (
                        auto_args.allow_shifts and shift_path.name not in auto_args.allow_shifts):
                    continue
                for ood_config_path in shift_path.iterdir():
                    if 'base' in ood_config_path.name:
                        continue
                    allowed = False
                    if auto_args.allow_algs:
                        for allowed_alg in auto_args.allow_algs:
                            if allowed_alg in ood_config_path.name:
                                allowed = True
                    else:
                        allowed = True
                    if not allowed:
                        continue

                    # args = args_parser(['--config_path', str(ood_config_path)])
                    # config = config_summoner(args)

                    if auto_args.sweep_root:
                        cmd_args = [f'{conda_goodtg} --config_path \"{ood_config_path}\" --log_file default']

                        if os.path.isabs(auto_args.sweep_root):
                            sweep_root = Path(auto_args.sweep_root)
                        else:
                            sweep_root = Path(ROOT_DIR, 'configs', auto_args.sweep_root)
                        sweep_path = sweep_root / ood_config_path.stem / dataset_path.name / domain_path.name / shift_path.name / 'base.yaml'
                        sweep_config, _, _ = load_config(str(sweep_path))
                        if 'extra_param' in sweep_config.keys():
                            sweep_config['extra_param'] = list(itertools.product(*sweep_config['extra_param']))
                        sweep_keys = list(sweep_config.keys())
                        sweep_values = list(itertools.product(*sweep_config.values()))
                        sweep_args = []
                        for value_set in sweep_values:
                            sweep_arg = []
                            for key, value in zip(sweep_keys, value_set):
                                if key == 'extra_param':
                                    sweep_arg.append(f'--{key} ' + ' '.join([str(v) for v in value]))
                                else:
                                    sweep_arg.append(f'--{key} {value}')
                            sweep_args.append(' '.join(sweep_arg))

                        cmd_args_product = list(itertools.product(cmd_args, sweep_args))
                        cmd_args = [' '.join(args_set) for args_set in cmd_args_product]
                    else:
                        cmd_args = [
                            f'{conda_goodtg} --exp_round {round} --config_path \"{ood_config_path}\"'
                            for round in auto_args.allow_rounds]

                    args_group += cmd_args

    return args_group


if __name__ == '__main__':
    launch()
