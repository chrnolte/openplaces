"""A Python package for place-based geospatial analysis"""

import sys as _sys


def _harden_console_encoding():
    """Keep progress output from aborting a run on a narrow console.

    Windows consoles and redirected files default to cp1252, which
    cannot encode characters that progress messages legitimately
    contain (download arrows, check marks, non-Latin place names), so a
    long run used to die with UnicodeEncodeError after the expensive
    work was already done - unless the user remembered to set
    PYTHONIOENCODING=utf-8. Switching the stream's error handling to
    'replace' keeps every run alive on any encoding; an unencodable
    symbol degrades to '?', which is the right trade for a log line.
    """
    for _stream in (_sys.stdout, _sys.stderr):
        _encoding = (getattr(_stream, 'encoding', '') or '').lower()
        if 'utf' not in _encoding:
            try:
                _stream.reconfigure(errors='replace')
            except Exception:
                pass


_harden_console_encoding()

import openplaces.api as _api  # noqa: E402

# Make `openplaces.cfg` available and trigger configuration setup on first
# import. Public API names (get_entities, curate, ...) resolve lazily via
# __getattr__ below so importing openplaces does not pull the whole pipeline.
from .config import cfg as cfg  # noqa: E402
from .core.constants import VERSION  # noqa: E402

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
