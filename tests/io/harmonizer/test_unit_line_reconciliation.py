"""A building crossing a unit line is reconciled to one envelope and each
unit keeps its piece. Every geometry and id below is fabricated."""

import geopandas as gpd
import pandas as pd
import pytest
import shapely.affinity
from shapely.geometry import box

import openplaces.io.harmonizer.neighbors as neighbors
import openplaces.io.harmonizer.spine as spine_module
from openplaces.core.constants import ENVELOPE_ID_COLUMN
from openplaces.recipe.dependencies import DepEdge

# Two units meeting at longitude -71.0.
UNITS = gpd.GeoDataFrame(
    geometry=[box(-71.01, 42.0, -71.0, 42.01), box(-71.0, 42.0, -70.99, 42.01)],
    index=pd.Index(['US-XX-AAA', 'US-XX-BBB'], name='admin3_id'),
    crs='EPSG:4326',
)


def _polygons(admin_id):
    own = UNITS.geometry[str(admin_id)]
    return own, UNITS[UNITS.index != str(admin_id)]


def _spine(rows):
    return gpd.GeoDataFrame(
        geometry=[g for _, g in rows],
        index=pd.Index([i for i, _ in rows], name='footprint_id'),
        crs='EPSG:4326',
    )


@pytest.fixture
def two_units(monkeypatch):
    monkeypatch.setattr(neighbors, 'unit_polygons', _polygons)


def test_a_straddling_envelope_is_cut_and_each_unit_keeps_its_piece(two_units):
    house = box(-71.0003, 42.005, -70.99985, 42.0052)  # two thirds in AAA
    inner = box(-71.005, 42.005, -71.0049, 42.0051)
    a = spine_module._cut_at_unit_lines(
        _spine([('H', house), ('I', inner)]), 'US-XX-AAA'
    )
    b = spine_module._cut_at_unit_lines(_spine([('H', house)]), 'US-XX-BBB')
    # The interior outline keeps its id and has no envelope.
    assert 'I' in a.index and pd.isna(a.loc['I', ENVELOPE_ID_COLUMN])
    piece_a = a[a[ENVELOPE_ID_COLUMN].notna()]
    piece_b = b[b[ENVELOPE_ID_COLUMN].notna()]
    assert len(piece_a) == 1 and len(piece_b) == 1
    # Both units name the same envelope, and the pieces tile it.
    assert piece_a[ENVELOPE_ID_COLUMN].iloc[0] == piece_b[ENVELOPE_ID_COLUMN].iloc[0]
    assert piece_a.index[0] != piece_b.index[0] != 'H'
    union = piece_a.geometry.iloc[0].union(piece_b.geometry.iloc[0])
    assert abs(union.area - house.area) < 1e-12
    assert piece_a.geometry.iloc[0].area > piece_b.geometry.iloc[0].area


def test_a_sliver_across_the_line_stays_whole_on_the_majority_side(two_units):
    house = box(-71.0005, 42.005, -70.99999, 42.0052)  # about 1 m over
    a = spine_module._cut_at_unit_lines(_spine([('H', house)]), 'US-XX-AAA')
    b = spine_module._cut_at_unit_lines(_spine([('H', house)]), 'US-XX-BBB')
    assert list(a.index) == ['H'] and a.geometry.iloc[0].equals(house)
    assert b.empty


def test_two_outlines_sharing_an_id_are_cut_and_dropped_one_by_one(two_units):
    # The merge may hold two sources' outlines under one id until the
    # spine makes ids unique; each is judged on its own geometry.
    left = box(-71.0003, 42.005, -70.99985, 42.0052)
    right = box(-71.00025, 42.0052, -70.9998, 42.0054)
    sliver = box(-71.0005, 42.006, -70.99999, 42.0062)
    rows = [
        ('D', left),
        ('D', right),
        ('S', sliver),
        ('S', box(-71.0, 42.006, -70.999, 42.0062)),
    ]
    a = spine_module._cut_at_unit_lines(_spine(rows), 'US-XX-AAA')
    pieces = a[a[ENVELOPE_ID_COLUMN].notna()]
    assert len(pieces) == 2 and pieces.index.is_unique and 'D' not in a.index
    # Of the two outlines named S, only the one wholly in BBB leaves.
    assert list(a.index[a.index == 'S']) == ['S']
    assert a.loc[['S']].geometry.iloc[0].equals(sliver)


def test_neighbor_inputs_are_the_shared_spine_sources(monkeypatch):
    recipe = {'pipeline': [{'step': 'resolve_spine', 'neighbor_reads': True}]}
    edges = {
        'US-XX-AAA': ['obm', 'massgis'],
        'US-XX-BBB': ['obm', 'massgis', 'only-in-b'],
    }

    def fake_dependencies(recipe, admin_id=None):
        return [
            DepEdge('r', up, 'auto_discover', step='resolve_spine')
            for up in edges[admin_id]
        ] + [DepEdge('r', 'nsi', 'recipe_id', step='link_to_reference')]

    monkeypatch.setattr(neighbors, 'touching_units', lambda admin: ('US-XX-BBB',))
    monkeypatch.setattr(neighbors, 'get_recipe_dependencies', fake_dependencies)
    assert neighbors.neighbor_inputs(recipe, 'US-XX-AAA') == [
        ('massgis', 'US-XX-BBB'),
        ('obm', 'US-XX-BBB'),
    ]
    monkeypatch.setenv(neighbors.NEIGHBOR_READS_ENV, '0')
    assert neighbors.neighbor_inputs(recipe, 'US-XX-AAA') == []


def test_each_neighbor_is_scanned_with_its_own_patches(monkeypatch):
    # AAA's patch names a closed source list; the harmonizer hands the
    # patched recipe over. BBB runs the national list, so of AAA's
    # sources it reads only the shared one, whichever recipe is passed.
    national = {
        'recipe_id': 'r',
        'pipeline': [{'step': 'resolve_spine', 'neighbor_reads': True}],
    }
    patched = {**national, 'patched_for': 'US-XX-AAA'}
    own_lists = {'US-XX-AAA': ['closed', 'shared'], 'US-XX-BBB': ['obm', 'shared']}

    def fake_dependencies(recipe, admin_id=None):
        # As get_recipe_dependencies does, the unit's own patch applies;
        # a recipe already patched keeps the patch it carries.
        unit = recipe.get('patched_for') or admin_id
        return [
            DepEdge('r', up, 'recipe_id', step='resolve_spine')
            for up in own_lists[unit]
        ]

    monkeypatch.setattr(neighbors, 'touching_units', lambda admin: ('US-XX-BBB',))
    monkeypatch.setattr(neighbors, 'get_recipe_dependencies', fake_dependencies)
    monkeypatch.setattr(neighbors, 'get_recipe_by_id', lambda rid: national)
    expected = [('shared', 'US-XX-BBB')]
    assert neighbors.neighbor_inputs(patched, 'US-XX-AAA') == expected
    assert neighbors.neighbor_inputs(national, 'US-XX-AAA') == expected


def test_a_recipe_without_neighbor_reads_reads_none(monkeypatch):
    recipe = {'pipeline': [{'step': 'resolve_spine'}]}
    assert neighbors.neighbor_inputs(recipe, 'US-XX-AAA') == []


def test_neighbor_rows_join_their_source_once(monkeypatch, two_units):
    house = box(-71.0003, 42.005, -70.99985, 42.0052)
    far = box(-70.995, 42.005, -70.9949, 42.0051)  # wholly in BBB
    own = _spine([('H', house)])
    neighbor = _spine([('H', house), ('F', far)])
    monkeypatch.setattr(
        neighbors, 'neighbor_inputs', lambda recipe, admin: [('obm', 'US-XX-BBB')]
    )
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: neighbor)

    class State:
        recipe = {}
        admin_id = 'US-XX-AAA'
        verbose = False

    sources = {'obm': own}
    spine_module._add_neighbor_rows(sources, State())
    assert list(sources['obm'].index) == ['H']


def test_neighbor_copies_and_id_clashes_keep_ids_unique(monkeypatch, two_units):
    # A county-split layer: the neighbor's copy of the straddling house
    # differs by float noise and is kept once. A different outline the
    # neighbor files under the same id stays, with a longer code.
    house = box(-71.0003, 42.005, -70.99985, 42.0052)
    noisy = shapely.affinity.translate(house, xoff=2e-13)
    other = box(-71.0001, 42.0058, -70.9998, 42.0059)
    own = _spine([('H', house), ('O', box(-71.0002, 42.0058, -71.0001, 42.0059))])
    neighbor = _spine([('H', noisy), ('O', other)])
    monkeypatch.setattr(
        neighbors, 'neighbor_inputs', lambda recipe, admin: [('obm', 'US-XX-BBB')]
    )
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: neighbor)

    class State:
        recipe = {}
        admin_id = 'US-XX-AAA'
        verbose = False

    sources = {'obm': own}
    spine_module._add_neighbor_rows(sources, State())
    merged = sources['obm']
    assert merged.index.is_unique and len(merged) == 3
    assert list(merged.index[:2]) == ['H', 'O']
    assert merged.geometry.iloc[2].equals(other)


def test_elongated_duplicates_are_judged_row_by_row():
    # Two candidates share an id; only the one beside the spine's
    # trailer is its duplicate.
    trailer = box(0, 0, 20, 4)
    spine = gpd.GeoDataFrame(geometry=[trailer], index=['T'], crs='EPSG:6933')
    to_add = gpd.GeoDataFrame(
        geometry=[box(0, 1, 20, 5), box(0, 500, 20, 504)],
        index=['D', 'D'],
        crs='EPSG:6933',
    )
    kept = spine_module.drop_elongated_duplicates(spine, to_add)
    assert len(kept) == 1 and kept.geometry.iloc[0].bounds[1] == 500


def test_a_row_almost_wholly_in_a_neighbor_never_enters_the_merge(two_units):
    # Such a row is the neighbor's building; inside this unit's merge it
    # would block this unit's own outlines nearby and then be dropped.
    neighbors_building = box(-71.00002, 42.005, -70.9995, 42.0052)
    inner = box(-71.0003, 42.005, -71.0001, 42.0051)
    straddler = box(-71.0003, 42.006, -70.99985, 42.0062)
    rows = _spine([('N', neighbors_building), ('I', inner), ('S', straddler)])
    kept, gone = spine_module._without_neighbor_outlines(rows, 'US-XX-AAA')
    assert list(kept.index) == ['I', 'S']
    assert len(gone) == 1


def test_another_sources_copy_of_a_left_out_building_is_dropped_too(two_units):
    # The neighbor keeps its building whole; a lower-ranked copy of it
    # here would otherwise be cut into a second, overlapping piece. A
    # distinct building nearby stays.
    neighbors_building = box(-71.00002, 42.005, -70.9995, 42.0052)
    copy = box(-71.00004, 42.00501, -70.99955, 42.00519)
    nearby = box(-71.0003, 42.005, -71.0001, 42.0051)
    left_out = _spine([('N', neighbors_building)]).geometry
    other_source = _spine([('C', copy), ('B', nearby)])
    kept = spine_module._drop_copies_of(other_source, left_out)
    assert list(kept.index) == ['B']
