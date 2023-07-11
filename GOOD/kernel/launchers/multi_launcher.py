import shlex
import os
import subprocess
import time

from GOOD import register
from .basic_launcher import Launcher


@register.launcher_register
class MultiLauncher(Launcher):
    def __init__(self):
        super(MultiLauncher, self).__init__()

    def __call__(self, jobs_group, auto_args):
        jobs_group = super(MultiLauncher, self).__call__(jobs_group, auto_args)
        procs_by_gpu = [None] * len(auto_args.allow_devices)

        while len(jobs_group) > 0:
            for idx, gpu_idx in enumerate(auto_args.allow_devices):
                proc = procs_by_gpu[idx]
                if (proc is None) or (proc.poll() is not None):
                    # No process or a process has just finished
                    cmd_args = jobs_group.pop(0)
                    new_proc = subprocess.Popen(shlex.split(cmd_args) + ['--gpu_idx', f'{gpu_idx}'],
                                                close_fds=True,
                                                stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'),
                                                start_new_session=False)
                    procs_by_gpu[idx] = new_proc
                    break
            time.sleep(1)
            print(f'\rWaiting jobs: {len(jobs_group)}', end='')

        for p in procs_by_gpu:
            if p is not None:
                p.wait()
