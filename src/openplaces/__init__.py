"""A Python package for place-based geospatial analysis"""

import openplaces.api as _api

# Make `openplaces.cfg` available and trigger configuration setup on first
# import. Public API names (get_entities, curate, ...) resolve lazily via
# __getattr__ below so importing openplaces does not pull the whole pipeline.
from .config import cfg as cfg
from .core.constants import VERSION

__version__ = VERSION

__author__ = 'Christoph Nolte'
__email__ = 'chrnolte@bu.edu'

__all__ = ['cfg', *_api.__all__]


def __getattr__(name: str):
    if name in _api.__all__:
        return getattr(_api, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return [*globals(), *_api.__all__]
