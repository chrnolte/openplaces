"""An outline two units both delivered is cut at the unit line, each
piece keeping its own unit's values. Every geometry, id and value below
is fabricated."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.delivery.slicing import SLICED_FROM, slice_at_unit_lines

# Two units meeting at longitude -71.0 (about 830 m by 1,100 m each).
UNITS = gpd.GeoDataFrame(
    geometry=[box(-71.01, 42.0, -71.0, 42.01), box(-71.0, 42.0, -70.99, 42.01)],
    index=pd.Index(['US-XX-AAA', 'US-XX-BBB'], name='admin3_id'),
    crs='EPSG:4326',
)


def _pooled(outlines):
    rows = pd.DataFrame(
        [{k: v for k, v in o.items() if k != 'geometry'} for o in outlines]
    ).set_index('footprint_id')
    return gpd.GeoDataFrame(
        rows, geometry=[o['geometry'] for o in outlines], crs='EPSG:4326'
    )


def test_a_straddling_outline_is_cut_and_each_piece_keeps_its_units_value():
    # A building about 40 m wide, two thirds in AAA, drawn by both units.
    house = box(-71.0003, 42.005, -70.99985, 42.0052)
    pooled = _pooled(
        [
            {
                'footprint_id': 'X1',
                'admin3_id': 'US-XX-AAA',
                'structure_value': 300.0,
                'area_m2': 1.0,
                'geometry': house,
            },
            {
                'footprint_id': 'X1',
                'admin3_id': 'US-XX-BBB',
                'structure_value': 120.0,
                'area_m2': 1.0,
                'geometry': house,
            },
            {
                'footprint_id': 'Y1',
                'admin3_id': 'US-XX-AAA',
                'structure_value': 50.0,
                'area_m2': 1.0,
                'geometry': box(-71.005, 42.005, -71.0049, 42.0051),
            },
        ]
    )
    out, provenance = slice_at_unit_lines(pooled, 'admin3_id', UNITS)
    assert len(out) == 3
    pieces = out[out[SLICED_FROM].notna()]
    assert sorted(pieces['admin3_id']) == ['US-XX-AAA', 'US-XX-BBB']
    assert set(pieces[SLICED_FROM]) == {'X1'}
    by_unit = pieces.set_index('admin3_id')
    assert by_unit.loc['US-XX-AAA', 'structure_value'] == 300.0
    assert by_unit.loc['US-XX-BBB', 'structure_value'] == 120.0
    # Each piece lies in its own unit and carries its own area.
    for unit, row in by_unit.iterrows():
        assert UNITS.geometry[unit].buffer(1e-9).contains(row.geometry)
    assert by_unit.loc['US-XX-AAA', 'area_m2'] > by_unit.loc['US-XX-BBB', 'area_m2']
    # Ids are new, distinct, and not the source's.
    assert pieces.index.is_unique and 'X1' not in pieces.index
    assert set(provenance['piece_id']) == set(pieces.index)
    assert set(provenance['source_id']) == {'X1'}
    assert out.loc['Y1', 'structure_value'] == 50.0


def test_a_sliver_is_not_cut_and_the_minority_copy_is_left_out():
    # Crosses the line by about 1 m of a 40 m building.
    house = box(-71.0005, 42.005, -70.99999, 42.0052)
    pooled = _pooled(
        [
            {
                'footprint_id': 'A1',
                'admin3_id': 'US-XX-AAA',
                'structure_value': 300.0,
                'geometry': house,
            },
            {
                'footprint_id': 'B1',
                'admin3_id': 'US-XX-BBB',
                'structure_value': 9.0,
                'geometry': house,
            },
        ]
    )
    out, provenance = slice_at_unit_lines(pooled, 'admin3_id', UNITS)
    assert list(out.index) == ['A1']
    assert out.loc['A1', 'structure_value'] == 300.0
    assert provenance.empty


def test_an_outline_only_one_unit_delivered_is_left_whole():
    house = box(-71.0003, 42.005, -70.99985, 42.0052)
    pooled = _pooled(
        [
            {
                'footprint_id': 'A1',
                'admin3_id': 'US-XX-AAA',
                'structure_value': 300.0,
                'geometry': house,
            }
        ]
    )
    out, provenance = slice_at_unit_lines(pooled, 'admin3_id', UNITS)
    assert list(out.index) == ['A1']
    assert out.geometry.iloc[0].equals(house)
    assert provenance.empty
