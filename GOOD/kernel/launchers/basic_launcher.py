import os
import shlex

from GOOD import config_summoner
from GOOD import register
from GOOD.utils.args import args_parser
from multiprocessing import Pool


@register.launcher_register
class Launcher:
    def __init__(self):
        super(Launcher, self).__init__()

    def __call__(self, jobs_group, auto_args):
        ready_jobs_group = []
        with Pool(20) as pool:
            read_results = pool.map(self.log_reader, jobs_group)
        for cmd_args, last_line in read_results:
            if last_line.startswith('INFO: ChartInfo'):
                print(cmd_args, '\033[1;32m[DONE]\033[0m')
            else:
                print(cmd_args, '\033[1;33m[READY]\033[0m')
                ready_jobs_group.append(cmd_args)

        ans = input(f'Launch unfinished {len(ready_jobs_group)} jobs or all {len(jobs_group)} jobs? [u/a]')
        while ans != 'u' and ans != 'a':
            ans = input(f'Invalid input: {ans}. Please answer u or a.')
        if ans == 'u':
            jobs_group = ready_jobs_group
        elif ans == 'a':
            pass
        else:
            raise ValueError(f'Unexpected value {ans}.')

        ans = input(f'Sure to launch {len(jobs_group)} jobs? [y/n]')
        while ans != 'y' and ans != 'n':
            ans = input(f'Invalid input: {ans}. Please answer y or n.')
        if ans == 'y':
            return jobs_group
        elif ans == 'n':
            print(f'See you later. :)')
            exit(0)
        else:
            raise ValueError(f'Unexpected value {ans}.')

    def log_reader(self, cmd_args):
        args = args_parser(shlex.split(cmd_args)[1:])
        config = config_summoner(args)
        last_line = self.harvest(config.log_path)
        return cmd_args, last_line

    def harvest(self, log_path):
        try:
            with open(log_path, 'rb') as f:
                try:  # catch OSError in case of a one line file
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
                return last_line
        except FileNotFoundError:
            return 'FileNotFoundError'
