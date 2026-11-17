"""`consolidate_condo_cluster_footprints` keeps a footprint whole where
its lots are condominium units.

The rule (plans/buildings-from-parcel-pieces.md, section 7.2): a
footprint over 9 or more lots, most classed Condominium by the parcel
curate's keyword ruleset, or over 33 or more lots. The tests read the
shipped ruleset. All geometry and labels are fabricated: lots are
squares of about 11 m near the equator, inside a unit drawn as the unit
square.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_mod
from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.recipe import get_recipe_by_id

FOOTPRINT_GEOSPINE = 'US_footprint-geospine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
UNIT = 'US-NC-BR'
SIDE = 0.0001


@pytest.fixture(autouse=True)
def unit_square(monkeypatch):
    unit = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)], index=pd.Index([UNIT]), crs='epsg:4326'
    )
    monkeypatch.setattr(spine_mod, 'get_admin', lambda *a, **k: unit)


@pytest.fixture(autouse=True)
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def _row_of_lots(prefix, n, y0, use):
    """n lots side by side and one footprint drawn exactly over them."""
    lots = gpd.GeoDataFrame(
        {'use_group': [use] * n},
        geometry=[box(i * SIDE, y0, (i + 1) * SIDE, y0 + SIDE) for i in range(n)],
        index=pd.Index([f'{prefix}{i}' for i in range(n)], name='parcel_id'),
        crs='epsg:4326',
    )
    outline = box(0, y0, n * SIDE, y0 + SIDE)
    return lots, outline


def _state(groups):
    """One footprint per group, the overlay linking it to its first lot
    only (as the min_fraction_of_largest trim leaves a large building)."""
    lots, outlines, ids, pairs = [], [], [], []
    for k, (n, use) in enumerate(groups):
        group_lots, outline = _row_of_lots(f'g{k}_', n, k * 10 * SIDE, use)
        lots.append(group_lots)
        outlines.append(outline)
        ids.append(f'f{k}')
        pairs.append((f'f{k}', group_lots.index[0]))
    ref = pd.concat(lots)
    spine = gpd.GeoDataFrame(
        {'geometry_source': ['obm'] * len(ids)},
        geometry=outlines,
        index=pd.Index(ids, name='footprint_id'),
        crs='epsg:4326',
    )
    crosswalk = pd.DataFrame(
        {'link': ['spatial_overlay'] * len(pairs), 'area_intersection_m2': 120.0},
        index=pd.MultiIndex.from_tuples(pairs, names=['footprint_id', 'parcel_id']),
    )
    return HarmonizeState(
        recipe=get_recipe_by_id(FOOTPRINT_GEOSPINE),
        admin_id=AdminId(UNIT),
        verbose=False,
        timer=None,
        spine=spine,
        references={PARCEL: ref},
        crosswalks={PARCEL: crosswalk},
        reference_types={PARCEL: 'parcel'},
    )


def _kept(state):
    source = state.spine['geometry_source'].astype(str)
    return sorted(state.spine.index[source.str.startswith('condo_cluster.')])


def test_lots_count_where_half_their_area_lies_under_the_footprint():
    lots = gpd.GeoSeries(
        [
            box(0, 0, SIDE, SIDE),
            box(SIDE, 0, 2 * SIDE, SIDE),
            box(2 * SIDE, 0, 3 * SIDE, SIDE),
        ],
        index=['a', 'b', 'c'],
        crs='epsg:4326',
    )
    # Over a, b and a quarter of c.
    footprint = gpd.GeoSeries(
        [box(0, 0, 2.25 * SIDE, SIDE)], index=['f'], crs='epsg:4326'
    )
    pairs = links_mod.lots_under_footprints(footprint, lots, 0.5)
    assert sorted(pairs['lot']) == ['a', 'b']
    assert pairs['share'].round(6).tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    ('n', 'use', 'whole'),
    [
        (9, 'Residential Condominium', True),
        (8, 'Residential Condominium', False),
        (9, 'Townhome Residential', False),
        (32, 'Single Family Residential', False),
        (33, 'Warehouse', True),
    ],
)
def test_the_rule_keeps_condominiums_and_very_large_groups_whole(n, use, whole):
    state = links_mod.consolidate_condo_cluster_footprints(_state([(n, use)]))
    assert _kept(state) == (['f0'] if whole else [])


def test_a_kept_footprint_gains_links_and_loses_none():
    state = _state([(9, 'Condominium'), (3, 'Townhome')])
    before = state.crosswalks[PARCEL].copy()
    state = links_mod.consolidate_condo_cluster_footprints(state)
    after = state.crosswalks[PARCEL]
    assert before.index.isin(after.index).all()
    added = after.loc[~after.index.isin(before.index)]
    assert set(added.index.get_level_values('footprint_id')) == {'f0'}
    assert len(added) == 8
    assert (added['link'] == links_mod.CLUSTER_LINK).all()
    # The true intersection: each lot lies wholly under the footprint.
    assert added['area_intersection_m2'].between(100, 140).all()
    assert state.spine.loc['f0', 'geometry_source'] == 'condo_cluster.obm'
    assert state.spine.loc['f1', 'geometry_source'] == 'obm'


@pytest.mark.parametrize(('code', 'whole'), [('004', True), ('001', False)])
def test_a_use_code_is_read_through_the_recipes_crosswalk(code, whole):
    # A layer with codes only (Florida's DOR use code, 004 condominium,
    # 001 single family): the crosswalk beside the recipe gives the text.
    florida = 'US-FL_parcel-floridagio-2026'
    state = _state([(9, 'unused')])
    ref = state.references.pop(PARCEL).drop(columns='use_group')
    ref['use_group_code'] = code
    state.references[florida] = ref
    state.crosswalks[florida] = state.crosswalks.pop(PARCEL)
    state.reference_types = {florida: 'parcel'}
    state = links_mod.consolidate_condo_cluster_footprints(state)
    assert _kept(state) == (['f0'] if whole else [])
    # The step reads the crosswalk; it writes no use column on the layer.
    assert 'use_group' not in state.references[florida].columns


def _with_displaced_lots():
    """A kept condo row (f1, y from 10 to 11 sides) and lots drawn off it.

    - x: condo, 30% under f1 and under nothing else;
    - z: condo, 20% under f1 and 40% under a small footprint f9;
    - w: single family, 30% under f1.
    """
    state = _state([(3, 'Townhome'), (9, 'Condominium')])
    y0 = 10 * SIDE
    extra = gpd.GeoDataFrame(
        {'use_group': ['Condominium', 'Condominium', 'Single Family']},
        geometry=[
            box(9 * SIDE - 0.3 * SIDE, y0, 9.7 * SIDE, y0 + SIDE),
            box(0, y0 - 0.8 * SIDE, SIDE, y0 + 0.2 * SIDE),
            box(2 * SIDE, y0 - 0.7 * SIDE, 3 * SIDE, y0 + 0.3 * SIDE),
        ],
        index=pd.Index(['x', 'z', 'w'], name='parcel_id'),
        crs='epsg:4326',
    )
    state.references[PARCEL] = pd.concat([state.references[PARCEL], extra])
    small = gpd.GeoDataFrame(
        {'geometry_source': ['obm']},
        geometry=[box(0, y0 - 0.8 * SIDE, SIDE, y0 - 0.4 * SIDE)],
        index=pd.Index(['f9'], name='footprint_id'),
        crs='epsg:4326',
    )
    state.spine = pd.concat([state.spine, small])
    return state


@pytest.mark.parametrize('on', [False, True])
def test_a_displaced_condo_lot_links_to_the_building_covering_most_of_it(on):
    state = links_mod.consolidate_condo_cluster_footprints(
        _with_displaced_lots(), condo_lots_to_largest_overlap=on
    )
    assert _kept(state) == ['f1']
    links = state.crosswalks[PARCEL]
    cluster = links[links['link'] == links_mod.CLUSTER_LINK]
    lots = set(cluster.index.get_level_values('parcel_id'))
    row = {f'g1_{i}' for i in range(1, 9)}
    # Only x: z lies mostly under f9, w is not a condo unit.
    assert lots == (row | {'x'} if on else row)
    if on:
        assert cluster.loc[('f1', 'x'), 'area_intersection_m2'] > 0


def test_a_trimmed_pair_in_the_sidecar_takes_the_cluster_label():
    # The sidecar holds every raw overlay pair; one the overlay's trim
    # dropped is there with a null label, which the curates read as no
    # link. The step labels it, keeps a labeled pair's label and
    # appends the pairs not yet there.
    from openplaces.geo.link import get_entity_link_path
    from openplaces.io import to_parquet
    from openplaces.io.harmonizer.links._shared import _LINK_METADATA_KEY
    from openplaces.recipe import get_recipe_id

    state = _state([(9, 'Condominium')])
    path = get_entity_link_path(get_recipe_id(state.recipe), PARCEL, state.admin_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = pd.DataFrame(
        {
            'footprint_id': ['f0', 'f0'],
            'parcel_id': ['g0_0', 'g0_3'],
            'area_intersection_m2': [120.0, 120.0],
            'link': ['spatial_overlay', None],
        }
    )
    to_parquet(sidecar, path, file_metadata={_LINK_METADATA_KEY: '{}'})
    links_mod.consolidate_condo_cluster_footprints(state)
    after = pd.read_parquet(path).set_index('parcel_id')['link']
    assert len(after) == 9
    assert after['g0_0'] == 'spatial_overlay'
    assert (after.drop('g0_0') == links_mod.CLUSTER_LINK).all()


def test_a_stray_lot_does_not_count(monkeypatch):
    state = _state([(9, 'Condominium')])
    ref = state.references[PARCEL]
    ref = ref.set_geometry(ref.geometry.translate(xoff=2.0))
    state.references[PARCEL] = ref
    state.spine = state.spine.set_geometry(state.spine.geometry.translate(xoff=2.0))
    state = links_mod.consolidate_condo_cluster_footprints(state)
    assert _kept(state) == []
