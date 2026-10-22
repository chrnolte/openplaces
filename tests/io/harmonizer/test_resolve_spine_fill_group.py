"""`resolve_spine`'s `fill_group`: an older cadastre vintage fills a unit
only where newer sources loaded nothing, and its urban and rural layers
fill it together.

Fabricated geometry. The old vintage's parcel is the newer parcels'
parent, subdivided since: under plain IoU deduplication it would overlap
each child too little and be added on top of them.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_mod
from openplaces.io.harmonizer import HarmonizeState


def _frame(ids, geoms):
    return gpd.GeoDataFrame(
        {'parcel_id_local': ids},
        geometry=geoms,
        crs='EPSG:4326',
        index=pd.Index(ids, name='parcel_id'),
    )


def _run(monkeypatch, frames, sources):
    monkeypatch.setattr(spine_mod, '_drop_out_of_unit_rows', lambda gdf, *a, **k: gdf)
    monkeypatch.setattr(
        spine_mod, 'get_entities', lambda recipe_id, *a, **k: frames[recipe_id]
    )
    state = HarmonizeState(
        recipe={'admin_id': 'CO'}, admin_id='CO-XX-YY', verbose=False, timer=None
    )
    return spine_mod.resolve_spine(
        state, sources=sources, keep_columns=['parcel_id_local']
    )


SOURCES = [
    {'recipe_id': 'new', 'label': 'new'},
    {'recipe_id': 'old_urban', 'label': 'old_urban', 'fill_group': 'old'},
    {'recipe_id': 'old_rural', 'label': 'old_rural', 'fill_group': 'old'},
]


def test_an_older_vintage_is_skipped_where_a_newer_one_has_rows(monkeypatch):
    frames = {
        # Two children of a parcel subdivided since the old vintage.
        'new': _frame(['n1', 'n2'], [box(0, 0, 4, 10), box(6, 0, 10, 10)]),
        'old_urban': _frame(['o1'], [box(0, 0, 10, 10)]),
        'old_rural': _frame(['o2'], [box(50, 50, 60, 60)]),
    }
    state = _run(monkeypatch, frames, SOURCES)
    assert sorted(state.spine['parcel_id_local']) == ['n1', 'n2']


def test_an_older_vintage_fills_a_unit_whole(monkeypatch):
    frames = {
        'new': _frame([], []),
        'old_urban': _frame(['o1'], [box(0, 0, 10, 10)]),
        'old_rural': _frame(['o2'], [box(50, 50, 60, 60)]),
    }
    state = _run(monkeypatch, frames, SOURCES)
    assert sorted(state.spine['parcel_id_local']) == ['o1', 'o2']
