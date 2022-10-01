r"""
Only for project usage. This module includes a preset project root and a storage root.
"""
import os

# root outside the GOOD
STORAGE_DIR = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'storage')  #: :str - Storage root=<Project root>/storage/.
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  #: str - Project root or Repo root.
OOM_CODE = 88
