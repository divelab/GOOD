from GOOD import register


@register.launcher_register
class Launcher:
    def __init__(self):
        super(Launcher, self).__init__()

    def __call__(self, jobs_group, auto_args):
        for cmd_args in jobs_group:
            print(cmd_args)
        ans = input(f'Sure to launch {len(jobs_group)} jobs? [y/n]')
        while ans != 'y' and ans != 'n':
            ans = input(f'Invalid input: {ans}. Please answer y or n.')
        if ans == 'y':
            return
        elif ans == 'n':
            print(f'See you later. :)')
            exit(0)
        else:
            raise ValueError(f'Unexpected value {ans}.')
