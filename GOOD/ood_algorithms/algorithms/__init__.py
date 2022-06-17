r"""
This module is for OOD algorithms. It is a good idea to inherit the BaseOOD class for new OOD algorithm designs.
"""

import glob
from os.path import dirname, basename, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *
