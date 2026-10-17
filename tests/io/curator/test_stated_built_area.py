"""The building ground area a curated parcel's own record states."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator.evidence import _stated_built_area


def _parcels():
    return pd.DataFrame(
        {
            'parcel_id': ['TOWER', 'NO_STORIES', 'NO_FLOOR'],
            'gross_floor_area_sqft': [107639.0, 10763.9, float('nan')],
            'living_area_sqft': [float('nan'), float('nan'), 0.0],
            'n_stories': [10.0, float('nan'), float('nan')],
            'area_ha': [0.2, 0.1, 1.0],
        }
    )


def _area(**kwargs):
    return _stated_built_area(
        _parcels(),
        'parcel_id',
        ['gross_floor_area_sqft', 'living_area_sqft'],
        'n_stories',
        **kwargs,
    )


def test_floor_area_over_stories_and_nothing_else_by_default():
    area = _area()
    assert list(area.index) == ['TOWER']
    assert round(area['TOWER']) == 1000


def test_lot_area_stands_in_where_no_floor_area_is_stated():
    # A zero floor area states nothing; floor area without a story
    # count is still a stated building of unknown ground area.
    area = _area(lot_area_refs={'TOWER', 'NO_STORIES', 'NO_FLOOR'})
    assert round(area['TOWER']) == 1000
    assert 'NO_STORIES' not in area.index
    assert area['NO_FLOOR'] == 10000.0


def test_lot_area_only_for_the_selected_references():
    area = _area(lot_area_refs=set())
    assert 'NO_FLOOR' not in area.index
