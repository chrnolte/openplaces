"""
vector.py

Re-exports all public symbols from the geo submodules for backwards compatibility.
Import directly from the submodules for new code:
  - openplaces.geo          — get_crs and other universal helpers
  - openplaces.geo.polygon  — core geometry operations
  - openplaces.geo.overlay  — spatial overlay and admin-ID joins
  - openplaces.geo.link     — entity linking
"""

import warnings

from openplaces.geo import get_crs  # noqa: F401
from openplaces.geo.link import *  # noqa: F401, F403
from openplaces.geo.overlay import *  # noqa: F401, F403
from openplaces.geo.polygon import *  # noqa: F401, F403

warnings.warn(
    'openplaces.geo.vector is deprecated. Import directly from '
    'openplaces.geo, openplaces.geo.polygon, openplaces.geo.overlay, '
    'or openplaces.geo.link instead.',
    DeprecationWarning,
    stacklevel=2,
)
