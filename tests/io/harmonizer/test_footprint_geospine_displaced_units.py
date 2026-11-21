"""Units whose Microsoft layer is displaced build their spine without it.

A patch per unit discards the Microsoft source from resolve_spine; the
test also expands discovery, so that the layer cannot return through it.
"""

import pytest

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.spine import _expand_auto_discover
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

GEOSPINE = 'US_footprint-geospine-2026'
MICROSOFT = 'US_footprint-microsoft-v2'


def _spine_sources(admin_id):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(GEOSPINE), AdminId(admin_id))
    step = next(s for s in recipe['pipeline'] if s['step'] == 'resolve_spine')
    state = HarmonizeState(
        recipe=recipe, admin_id=AdminId(admin_id), verbose=False, timer=None
    )
    return [s['recipe_id'] for s in _expand_auto_discover(step['sources'], state)]


DISPLACED = [
    'US-NC-FOR',
    'US-NC-STO',
    'US-NC-DAE',
    'US-NC-ROC',
    'US-NC-CUR',
    'US-NC-PES',
]


@pytest.mark.parametrize('admin_id', DISPLACED)
def test_a_displaced_unit_reads_no_microsoft_outlines(admin_id):
    sources = _spine_sources(admin_id)
    assert MICROSOFT not in sources
    assert sources[0] == 'footprint-obm-2025'
    assert 'US-NC_footprint-ncdps-2023' in sources


def test_a_neighboring_unit_still_reads_them():
    assert MICROSOFT in _spine_sources('US-NC-GUI')
