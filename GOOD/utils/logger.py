r"""A logger related utils file: tqdm style, logger loader.
"""
import os
from datetime import datetime

from cilog import create_logger
from torch.utils.tensorboard import SummaryWriter

pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                'dynamic_ncols': True, 'ascii': '░▒█'}

from GOOD.utils.config_reader import Union, CommonArgs, Munch


def load_logger(config: Union[CommonArgs, Munch], sub_print=True):
    r"""
    logger loader

    Args:
        config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.log_path`, :obj:`config.tensorboard_logdir`, :obj:`config.log_file`)
        sub_print (bool): whether the logger substitutes general print function

    Returns:
        [cilog logger, tensorboard summary writer]

    """
    if sub_print:
        print("This logger will substitute general print function")
    logger = create_logger(name='GNN_log',
                           file=config.log_path,
                           enable_mail=False,
                           sub_print=sub_print)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    writer = SummaryWriter(
        log_dir=os.path.join(config.tensorboard_logdir, f'{config.log_file}_{current_time}'))
    return logger, writer
