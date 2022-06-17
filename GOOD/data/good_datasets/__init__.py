r"""
This module includes 8 GOOD datasets.

- Graph prediction datasets: GOOD-HIV, GOOD-PCBA, GOOD-ZINC, GOOD-CMNIST, GOOD-Motif.
- Node prediction datasets: GOOD-Cora, GOOD-Arxiv, GOOD-CBAS.
"""

import glob
from os.path import dirname, basename, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *

