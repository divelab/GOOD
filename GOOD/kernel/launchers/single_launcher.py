import shlex
import subprocess

from tqdm import tqdm

from GOOD import register
from .basic_launcher import Launcher


@register.launcher_register
class SingleLauncher(Launcher):
    def __init__(self):
        super(SingleLauncher, self).__init__()

    def __call__(self, jobs_group, auto_args):
        jobs_group = super(SingleLauncher, self).__call__(jobs_group, auto_args)
        for cmd_args in tqdm(jobs_group):
            subprocess.run(shlex.split(cmd_args) + ['--gpu_idx', f'{auto_args.allow_devices[0]}'], close_fds=True,
                           stdout=open('/dev/null', 'a'), stderr=open('/dev/null', 'a'),
                           start_new_session=False)
