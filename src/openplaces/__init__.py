"""A Python package for place-based geospatial analysis"""

__version__ = '0.1.0'
__author__ = 'Christoph Nolte'
__email__ = 'chrnolte@bu.edu'

# Make `openplaces.cfg` available.
# Also ensures that configuration setup is triggered upon first import
from openplaces.api import *  # noqa: F401, F403

from .config import cfg as cfg
