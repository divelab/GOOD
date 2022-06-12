import glob
from os.path import dirname, basename, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from . import *

# from . import Arxiv
#
# __all__ = [
#     'Arxiv'
# ]
# from .Arxiv import GOODArxiv
# from .BAShapes import GOODCBAS
# from .HIV import GOODHIV
#
# __all__ = [
#     'GOODHIV',
#     'GOODCBAS',
#     'GOODArxiv'
# ]
