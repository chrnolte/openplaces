"""Inherited geographic ids are what the call per parcel gave before.

plans/slow-steps-same-output-faster.md, S2: `_inherit_geographic_ids`
rolls up a linked recipe's ids without a Python call per group and
column, and lists conflicts in one pass. The previous implementation
is kept below; both run on the same rows and are compared exactly:
values, dtypes, index, the kind of missing value, and the conflict
records with their types. Every id is fabricated
(docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_module
from openplaces.geo.polygon import points_from_coords
from openplaces.io.harmonizer.spine import _inherit_geographic_ids, _pick_unanimous


def _old_inherit_geographic_ids(state, output_columns, inherit_from):
    """`_inherit_geographic_ids` as it stood before S2."""
    recipe_id = inherit_from['recipe_id']
    group_by = inherit_from.get('group_by', 'parcel_id')
    spine = state.spine

    linked = spine_module.get_entities(
        recipe_id, state.admin_id, geom=True, missing='ignore'
    )
    empty = {
        col: pd.Series(pd.NA, index=spine.index, dtype=object) for col in output_columns
    }
    if linked is None or len(linked) == 0:
        return empty, []

    available = [c for c in output_columns if c in linked.columns]
    if not available:
        return empty, []

    if group_by in linked.columns:
        grouped = linked.groupby(group_by)
    elif {'lat', 'long'}.issubset(linked.columns):
        points = points_from_coords(
            linked[['lat', 'long'] + available].copy(), x='long', y='lat'
        )
        joined = gpd.sjoin(points, spine[['geometry']], how='inner', predicate='within')
        joined = spine_module._rename_right_index(joined, spine.index.name, '_spine_id')
        grouped = joined.groupby('_spine_id')
    else:
        return empty, []

    rollup = grouped[available].agg(_pick_unanimous)

    conflicts: list[dict] = []
    for col in available:
        counts = grouped[col].nunique(dropna=True)
        for key in counts.index[counts > 1]:
            values = pd.unique(grouped.get_group(key)[col].dropna())
            conflicts.append(
                {'output_column': col, 'group_key': key, 'values': list(values)}
            )

    inherited = {
        col: (
            rollup[col].reindex(spine.index)
            if col in rollup.columns
            else pd.Series(pd.NA, index=spine.index, dtype=object)
        )
        for col in output_columns
    }
    return inherited, conflicts


def _types(values) -> list:
    return [type(v) for v in values]


def _compare(monkeypatch, linked, spine, output_columns, group_by='parcel_id'):
    monkeypatch.setattr(
        spine_module, 'get_entities', lambda *args, **kwargs: linked.copy()
    )
    state = SimpleNamespace(spine=spine, admin_id='US-ZZ-AAA')
    inherit_from = {'recipe_id': 'US_building-spine-2026', 'group_by': group_by}
    new, new_conflicts = _inherit_geographic_ids(state, output_columns, inherit_from)
    old, old_conflicts = _old_inherit_geographic_ids(
        state, output_columns, inherit_from
    )
    assert list(new) == list(old)
    for column in old:
        pd.testing.assert_series_equal(
            new[column], old[column], check_dtype=True, check_exact=True
        )
        assert _types(new[column].to_numpy(dtype=object)) == _types(
            old[column].to_numpy(dtype=object)
        )
    assert new_conflicts == old_conflicts
    for new_record, old_record in zip(new_conflicts, old_conflicts, strict=True):
        assert type(new_record['group_key']) is type(old_record['group_key'])
        assert _types(new_record['values']) == _types(old_record['values'])
    return new, new_conflicts


# Parcel P1 one building; P2 two agreeing; P3 two disagreeing; P4 one
# value and one missing; P5 all missing; P6 repeated and new values; a
# building with no parcel; P9 on the spine only.
PARCELS = ['P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4', 'P5', 'P5', 'P6', 'P6', 'P6', None]
TRACTS = ['t1', 't2', 't2', 't3', 't4', 't5', None, None, None, 't6', 't7', 't6', 't8']
NUMBERS = [1, 2, 2, 3, 4, 5, None, None, None, 6, 7, 6, 8]

DTYPES = {
    'obj': (object, TRACTS),
    'str': ('str', TRACTS),
    'string': ('string', TRACTS),
    'cat': ('category', TRACTS),
    'f64': ('float64', NUMBERS),
    'Int64': ('Int64', NUMBERS),
    'Float64': ('Float64', NUMBERS),
    'boolean': ('boolean', [None if n is None else n % 2 == 0 for n in NUMBERS]),
    # Not rolled up by kind: kept on the call per group.
    'datetime': (
        'datetime64[ns]',
        [None if n is None else pd.Timestamp(2020, 1, n) for n in NUMBERS],
    ),
    'obj_int': (object, NUMBERS),
    'all_missing': (object, [None] * len(PARCELS)),
}


def _linked(parcels, columns: dict, parcel_dtype='str') -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            name: pd.Series(values, dtype=dtype)
            for name, (dtype, values) in columns.items()
        }
    )
    frame['parcel_id'] = pd.Series(parcels, dtype=parcel_dtype)
    return frame


def _spine(ids) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {'lat': 0.0, 'long': 0.0},
        geometry=[box(i, 0, i + 1, 1) for i in range(len(ids))],
        index=pd.Index(ids, name='spine_id'),
        crs=4326,
    )


SPINE_IDS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P9']


@pytest.mark.parametrize('column', list(DTYPES))
def test_each_dtype_matches_the_previous_code(monkeypatch, column):
    linked = _linked(PARCELS, {column: DTYPES[column]})
    _compare(monkeypatch, linked, _spine(SPINE_IDS), [column, 'not_linked'])


def test_all_columns_together(monkeypatch):
    linked = _linked(PARCELS, DTYPES, parcel_dtype=object)
    _, conflicts = _compare(monkeypatch, linked, _spine(SPINE_IDS), list(DTYPES))
    assert conflicts


@pytest.mark.parametrize(
    'values',
    [
        # Every group agrees: numbers come back numeric.
        (['P1', 'P2', 'P2'], [1, 2, 2], 'int64'),
        (['P1', 'P2', 'P2'], [1.5, 2.5, 2.5], 'float64'),
        (['P1', 'P2', 'P2'], [True, False, False], 'bool'),
        (['P1', 'P2', 'P2'], [1, 2, 2], object),
        (['P1', 'P2', 'P2'], ['a', 'b', 'b'], object),
        # Every group conflicts.
        (['P1', 'P1', 'P2', 'P2'], ['a', 'b', 'c', 'd'], 'str'),
        (['P1', 'P1', 'P2', 'P2'], ['a', 'b', 'c', 'd'], 'string'),
        (['P1', 'P1', 'P2', 'P2'], ['a', 'b', 'c', 'd'], 'category'),
        (['P1', 'P1', 'P2', 'P2'], [1, 2, 3, 4], 'int64'),
        # Mixed kinds in one object column.
        (['P1', 'P2', 'P3', 'P3', 'P4'], [1, 'a', 2.5, 2.5, 2**40], object),
        (['P1', 'P2', 'P3', 'P3'], [-(2**62), 2**63 + 5, 1, 2], object),
    ],
)
def test_kinds_of_result(monkeypatch, values):
    parcels, column_values, dtype = values
    linked = _linked(parcels, {'v': (dtype, column_values)})
    _compare(monkeypatch, linked, _spine(SPINE_IDS), ['v'])


def test_integer_parcel_keys(monkeypatch):
    linked = _linked(
        [3, 1, 1, 2, 2], {'v': ('str', ['a', 'b', 'b', 'c', 'd'])}, 'int64'
    )
    _compare(monkeypatch, linked, _spine([1, 2, 3, 4]), ['v'])


def test_no_linked_rows(monkeypatch):
    linked = _linked([], {'v': ('str', [])})
    _compare(monkeypatch, linked, _spine(SPINE_IDS), ['v'])


def test_centroids_in_the_spine_when_no_group_column(monkeypatch):
    spine = _spine(['P1', 'P2', 'P3'])
    linked = pd.DataFrame(
        {
            'lat': [0.5, 0.5, 0.5, 0.5, 0.5, 5.0],
            'long': [0.5, 1.2, 1.8, 2.2, 2.8, 5.0],
            'v': pd.Series(['a', 'b', 'b', 'c', 'd', 'e'], dtype='str'),
        }
    )
    _compare(monkeypatch, linked, spine, ['v'])
    # No centroid in any spine row: nothing to group.
    _compare(monkeypatch, linked.iloc[[5]], spine, ['v'])


def _random_linked(n_rows: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.choice([1, 1, 1, 1, 1, 2, 3, 4], size=n_rows)
    parcels = np.repeat(np.arange(len(sizes)), sizes)[:n_rows]
    parcel_ids = pd.Series(parcels).map('P{:06d}'.format).to_numpy(dtype=object)
    parcel_ids[rng.random(n_rows) < 0.01] = None
    tracts = rng.choice(['t1', 't2', 't3', None], p=[0.6, 0.2, 0.1, 0.1], size=n_rows)
    numbers = np.where(rng.random(n_rows) < 0.1, np.nan, rng.integers(0, 3, n_rows))
    columns = {
        'tract_str': ('str', list(tracts)),
        'tract_cat': ('category', list(tracts)),
        'tract_obj': (object, list(tracts)),
        'number': ('float64', numbers),
    }
    return _linked(list(parcel_ids), columns, parcel_dtype=object)


def test_randomized_rows_match_the_previous_code(monkeypatch):
    linked = _random_linked(4000)
    spine_ids = sorted(set(linked['parcel_id'].dropna()))[::2] + ['P9999999']
    columns = ['tract_str', 'tract_cat', 'tract_obj', 'number']
    _compare(monkeypatch, linked, _spine(spine_ids), columns)


# A parcel geospine's columns when link_geographic_ids runs: value
# `_source` sidecars at positions 8 to 10 (the Fort Bend layout, where
# land_value_source and improvement_value_source were seen to swap)
# and a source-native admin4_id the step overwrites in place.
GEOSPINE_COLUMNS = [
    'parcel_id_local',
    'admin4_id',
    'use_group',
    'improvement_value',
    'land_value',
    'year_built',
    'n_dwellings',
    'geometry_source',
    'land_value_source',
    'year_built_source',
    'improvement_value_source',
    'lat',
    'long',
]


def _geospine() -> gpd.GeoDataFrame:
    spine = _spine(['P1', 'P2', 'P3'])
    for column in GEOSPINE_COLUMNS:
        if column not in spine.columns:
            spine[column] = pd.Series(['x', 'y', 'z'], index=spine.index)
    # The ids the linked rows hold, so the step has nothing to warn of.
    spine['admin4_id'] = pd.Series(['a1', 'a2', 'a3'], index=spine.index)
    return spine[[*GEOSPINE_COLUMNS, 'geometry']]


def test_link_geographic_ids_keeps_the_column_order(monkeypatch):
    """The whole step writes the columns the previous code wrote, in order."""
    linked = _linked(
        ['P1', 'P2', 'P2', 'P3'],
        {
            'admin4_id': ('str', ['a1', 'a2', 'a2', 'a3']),
            'census_tract_id': ('str', ['t1', 't2', 't2', 't3']),
            'census_block_id': ('str', ['b1', 'b2', 'b2', 'b3']),
        },
    )
    monkeypatch.setattr(
        spine_module, 'get_entities', lambda *args, **kwargs: linked.copy()
    )
    monkeypatch.setattr(
        'openplaces.io.harmonizer.diagnostics.save_geographic_id_inheritance_conflicts',
        lambda state: None,
    )
    links = [
        {'output_column': 'census_tract_id', 'recipe_id': 'tracts'},
        {'output_column': 'admin4_id', 'admin_level': 4},
        {'output_column': 'census_block_id', 'recipe_id': 'blocks'},
    ]
    inherit_from = {'recipe_id': 'US_building-spine-2026', 'group_by': 'parcel_id'}

    def run():
        state = SimpleNamespace(
            spine=_geospine(), admin_id='US-ZZ-AAA', metadata={}, verbose=False
        )
        return spine_module.link_geographic_ids(
            state, links=links, inherit_from=inherit_from
        ).spine

    new = run()
    monkeypatch.setattr(
        spine_module, '_inherit_geographic_ids', _old_inherit_geographic_ids
    )
    old = run()

    expected = [*GEOSPINE_COLUMNS, 'geometry', 'census_tract_id', 'census_block_id']
    assert list(old.columns) == expected
    assert list(new.columns) == expected
    pd.testing.assert_frame_equal(new, old, check_exact=True)
