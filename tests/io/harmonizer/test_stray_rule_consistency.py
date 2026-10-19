"""The footprint and building geospines apply one stray rule, once.

Regression for the statewide WI run of 2026-10-18: the footprint
geospine of US-WI-WI turned 466 parcels lying outside the county's
extent into outlines (`infer_spine_additions` and
`consolidate_condo_cluster_footprints` read the parcel reference, which
no stray test had touched), and the building geospine, reading those
outlines as a source, dropped them by the stray rule. `load_geospine`
then refused the pair ("466 rows there are new"). Now the parcel steps
apply the rule (`spine.in_unit_extent`) the parcel geospine applies, and
`resolve_spine` does not test a spine product's rows a second time.

All geometry is fabricated: a unit drawn as the unit square, parcels
inside it and parcels two degrees east of it.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_mod
from openplaces.config import cfg
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import links as links_mod
from openplaces.recipe import get_recipe_by_id
from tests.links_patching import patch_links

FOOTPRINT_GEOSPINE = 'US_footprint-geospine-2026'
BUILDING_GEOSPINE = 'US_building-geospine-2026'
PARCEL = 'US-NC_parcel-nconemap-2025'
UNIT = 'US-NC-BR'
STRAY_X = 2.0


@pytest.fixture(autouse=True)
def unit_square(monkeypatch):
    """The unit is the unit square, whatever the admin layer says."""
    unit = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1)], index=pd.Index([UNIT]), crs='epsg:4326'
    )
    monkeypatch.setattr(spine_mod, 'get_admin', lambda *a, **k: unit)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    dirs = dict(cfg.config['directories'])
    dirs['data_root'] = tmp_path
    for name in ('core', 'external', 'raw', 'cache', 'out', 'share'):
        dirs[name] = tmp_path / 'data' / name
    dirs['heap'] = tmp_path / 'data/cache/_heap'
    dirs['logs'] = tmp_path / 'data/cache/_logs'
    monkeypatch.setitem(cfg.config, 'directories', dirs)
    return tmp_path


def test_rule_keeps_the_margin_and_drops_what_lies_beyond_it():
    rows = gpd.GeoDataFrame(
        geometry=[
            box(0.4, 0.4, 0.5, 0.5),
            box(1.005, 0.4, 1.01, 0.5),
            box(STRAY_X, 0.4, STRAY_X + 0.1, 0.5),
        ],
        crs='epsg:4326',
    )
    assert in_unit(rows).tolist() == [True, True, False]
    unplaced = gpd.GeoDataFrame(geometry=rows.geometry.to_numpy())
    assert in_unit(unplaced).tolist() == [True, True, True]


def in_unit(rows):
    return spine_mod.in_unit_extent(rows, UNIT)


def _inference_state():
    """20 linked parcels in the unit, one unlinked built parcel in, one out."""
    side = 0.01
    geoms = [box(i * side, 0, (i + 1) * side, side) for i in range(21)]
    geoms.append(box(STRAY_X, 0, STRAY_X + side, side))
    ids = [f'p{i}' for i in range(22)]
    ref = gpd.GeoDataFrame(
        {'improvement_value_per_ha': [100_000.0] * 20 + [150_000.0, 150_000.0]},
        geometry=geoms,
        index=pd.Index(ids, name='parcel_id'),
        crs='epsg:4326',
    )
    crosswalk = pd.DataFrame(
        {'link': ['spatial_overlay'] * 20},
        index=pd.MultiIndex.from_arrays(
            [[f'f{i}' for i in range(20)], ids[:20]],
            names=['footprint_id', 'parcel_id'],
        ),
    )
    spine = gpd.GeoDataFrame(
        geometry=geoms[:20],
        index=pd.Index([f'f{i}' for i in range(20)], name='footprint_id'),
        crs='epsg:4326',
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


def test_a_stray_parcel_is_not_inferred_as_an_outline():
    state = links_mod.infer_spine_additions(_inference_state(), entity_type='parcel')
    inferred = state.metadata[f'inferred_from_{PARCEL}']
    assert set(inferred['parcel_id']) == {'p20'}


def _condo_parcels(x0):
    return gpd.GeoDataFrame(
        {
            'geo_id': ['U1', 'U2'],
            'land_value': [0.0, 0.0],
            'improvement_value': [300_000.0, 250_000.0],
            'geometry': [
                box(x0, 0, x0 + 0.0001, 0.0001),
                box(x0 + 0.0001, 0, x0 + 0.0002, 0.0001),
            ],
        },
        crs='epsg:4326',
    )


@pytest.mark.parametrize(('x0', 'n_clusters'), [(0.5, 1), (STRAY_X, 0)])
def test_a_stray_condo_cluster_is_not_consolidated(
    data_root, monkeypatch, x0, n_clusters
):
    patch_links(monkeypatch, 'get_entities', lambda *a, **k: _condo_parcels(x0))
    state = HarmonizeState(
        recipe=get_recipe_by_id(FOOTPRINT_GEOSPINE),
        admin_id=AdminId(UNIT),
        verbose=False,
        timer=None,
        spine=gpd.GeoDataFrame(
            geometry=[box(0.1, 0.1, 0.1001, 0.1001)],
            index=pd.Index(['f_other'], name='footprint_id'),
            crs='epsg:4326',
        ),
    )
    state = links_mod._link_spatial_overlay(
        state,
        PARCEL,
        'parcel',
        {'min_fraction_of_largest': 0.1667, 'area_intersection_m2_min': 10},
        save_link=False,
    )
    state = links_mod.consolidate_condo_cluster_footprints(state)
    source = state.spine.get('geometry_source', pd.Series(dtype=object))
    assert int(source.astype(str).str.startswith('condo_cluster.').sum()) == n_clusters


def test_building_geospine_keeps_every_row_of_the_footprint_geospine(monkeypatch):
    # A footprint geospine row outside the unit's extent: built before
    # the parcel steps applied the rule, or by any later step that adds
    # rows. The building geospine is built from the footprint geospine
    # and must carry exactly its rows, or load_geospine refuses the pair.
    footprints = gpd.GeoDataFrame(
        {'geometry_source': ['obm', 'parcel.nconemap']},
        geometry=_one_in_one_out(),
        index=pd.Index(['in', 'out'], name='footprint_id'),
        crs='epsg:4326',
    )
    assert in_unit(footprints).tolist() == [True, False]
    monkeypatch.setattr(spine_mod, 'get_entities', lambda *a, **k: footprints)
    state = HarmonizeState(
        recipe=get_recipe_by_id(BUILDING_GEOSPINE),
        admin_id=AdminId(UNIT),
        verbose=False,
        timer=None,
    )
    state = spine_mod.resolve_spine(
        state,
        sources=[
            {'recipe_id': FOOTPRINT_GEOSPINE, 'label': 'footprint'},
            {'auto_discover': False},
        ],
        keep_columns=['geometry_source'],
    )
    assert sorted(state.spine.index) == ['in', 'out']


def _one_in_one_out():
    return [
        box(0.4, 0.4, 0.4001, 0.4001),
        box(STRAY_X, 0.4, STRAY_X + 0.001, 0.401),
    ]


def test_an_ingest_source_still_loses_its_strays(monkeypatch):
    outlines = gpd.GeoDataFrame(
        geometry=_one_in_one_out(),
        index=pd.Index(['in', 'out'], name='footprint_id'),
        crs='epsg:4326',
    )
    monkeypatch.setattr(spine_mod, 'get_entities', lambda *a, **k: outlines)
    state = HarmonizeState(
        recipe=get_recipe_by_id(FOOTPRINT_GEOSPINE),
        admin_id=AdminId(UNIT),
        verbose=False,
        timer=None,
    )
    with pytest.warns(UserWarning, match='carried strays'):
        state = spine_mod.resolve_spine(
            state,
            sources=[
                {'recipe_id': 'footprint-obm-2025', 'label': 'obm'},
                {'auto_discover': False},
            ],
        )
    assert np.array_equal(state.spine.index.to_numpy(), np.array(['in']))
