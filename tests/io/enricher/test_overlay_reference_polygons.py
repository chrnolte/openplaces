"""The open space overlay enrich step, on synthetic geometry.

Two protected sites meet at a town line; lots lie inside, across the
edge, and outside. The step promises: the share is the covered part
of the lot, the dominant polygon is the one covering most of it, a
withheld label stays empty while the codes are written, and a unit
the reference does not cover gets every column, all missing.
"""

import geopandas as gpd
import pandas as pd
import pytest
import shapely

from openplaces.core.schema import AdminId
from openplaces.io.enricher import EnrichState
from openplaces.io.enricher import overlay as overlay_module
from openplaces.io.enricher.overlay import (
    overlay_reference_polygons,
    overlay_shares,
)
from openplaces.recipe import (
    apply_recipe_patches,
    get_recipe_by_id,
    get_recipe_dependencies,
)

# Massachusetts State Plane (meters), so the scene lies in the state.
CRS = 'EPSG:26986'
X0, Y0 = 230_000, 905_000
REFERENCE = 'US-MA_rules-openspace-massgis-2026'
ENRICH = 'US-MA_parcel_rules-openspace-massgis-2026'


def _box(x0, y0, x1, y1):
    return shapely.box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


def _sites():
    """Two polygons of one site split by a town line at x = 100, and a
    privately owned restricted polygon to the south."""
    return gpd.GeoDataFrame(
        {
            'protected_open_space_site': ['Fells', 'Fells', 'Farm CR'],
            'protected_open_space_level': ['P', 'P', 'P'],
            'protected_open_space_owner_type': ['S', 'S', 'P'],
        },
        geometry=[_box(0, 0, 100, 100), _box(100, 0, 200, 100), _box(0, -100, 100, 0)],
        index=pd.Index(['176-1', '178-1', '176-9'], name='protected_open_space_id'),
        crs=CRS,
    )


def _lots():
    return gpd.GeoDataFrame(
        geometry=[
            _box(10, 10, 40, 40),  # inside the first town's polygon
            _box(150, 10, 180, 40),  # inside the second town's polygon
            _box(80, 10, 140, 40),  # across the town line, mostly east
            _box(150, 120, 180, 150),  # outside
            _box(10, -40, 40, -10),  # on the private polygon
            _box(180, 80, 220, 120),  # a quarter inside
        ],
        index=pd.Index(['a', 'b', 'c', 'd', 'e', 'f'], name='parcel_id'),
        crs=CRS,
    )


def test_share_and_dominant_polygon():
    share, dominant, dominant_share = overlay_shares(_lots().geometry, _sites())

    assert share['a'] == pytest.approx(1)
    assert share['c'] == pytest.approx(1)
    assert share['d'] == 0
    assert share['f'] == pytest.approx(0.25)
    assert dominant['a'] == '176-1'
    assert dominant['b'] == '178-1'
    # 20 m west and 40 m east of the line: the eastern polygon covers most
    assert dominant['c'] == '178-1'
    assert dominant_share['c'] == pytest.approx(40 / 60)
    assert pd.isna(dominant['d'])


def test_a_large_site_cut_into_pieces_counts_once():
    sites = _sites()
    # A polygon with many vertices is subdivided before the overlay;
    # the pieces of one polygon must add up, not compete.
    ring = shapely.Point(X0 + 100, Y0 + 300).buffer(80, quad_segs=256)
    sites.loc['176-2'] = ['Big', 'P', 'M', ring]
    lot = gpd.GeoSeries([_box(60, 260, 140, 340)], index=['g'], crs=CRS)
    share, dominant, _ = overlay_shares(lot, sites, max_vertices=32)

    assert dominant['g'] == '176-2'
    assert share['g'] == pytest.approx(1, abs=1e-6)


def _state(admin_id='US-MA-MED'):
    lots = _lots().to_crs('EPSG:4326')
    return EnrichState(
        recipe=get_recipe_by_id(ENRICH),
        entity_recipe=get_recipe_by_id('US_parcel-spine-2026'),
        admin_id=AdminId(admin_id),
        verbose=False,
        timer=None,
        spine=lots,
        evidence=pd.DataFrame(index=lots.index),
    )


def _run(state):
    step = next(
        s
        for s in get_recipe_by_id(ENRICH)['pipeline']
        if s['step'] == 'overlay_reference_polygons'
    )
    params = {k: v for k, v in step.items() if k != 'step'}
    return overlay_reference_polygons(state, **params)


def test_the_recipe_writes_codes_and_withholds_private_site_names(monkeypatch):
    calls = []

    def _load(recipe_id, admin_id):
        calls.append((recipe_id, admin_id))
        return _sites()

    monkeypatch.setattr(overlay_module, '_load_reference', _load)
    evidence = _run(_state()).evidence

    # A town's run reads the state-wide file
    assert calls == [(REFERENCE, 'US-MA')]
    assert evidence.loc['a', 'protected_open_space_site_massgis'] == 'Fells'
    assert evidence.loc['b', 'protected_open_space_site_massgis'] == 'Fells'
    assert evidence.loc['c', 'protected_open_space_id_massgis'] == '178-1'
    assert evidence.loc['a', 'protected_open_space_level_massgis'] == 'P'
    assert evidence.loc['d', 'protected_open_space_share_massgis'] == 0
    assert pd.isna(evidence.loc['d', 'protected_open_space_level_massgis'])
    # Private for-profit polygon: codes written, site name withheld
    assert evidence.loc['e', 'protected_open_space_owner_type_massgis'] == 'P'
    assert evidence.loc['e', 'protected_open_space_share_massgis'] == pytest.approx(
        1, rel=1e-3
    )
    assert pd.isna(evidence.loc['e', 'protected_open_space_site_massgis'])


def test_a_unit_outside_the_layer_gets_every_column_missing(monkeypatch):
    def _load(recipe_id, admin_id):
        raise AssertionError('no reference is read outside its scope')

    monkeypatch.setattr(overlay_module, '_load_reference', _load)
    evidence = _run(_state('US-NC-CUR')).evidence

    patch = get_recipe_by_id('US-MA_parcel-openplaces-2026')
    merged = patch['pipeline_patch'][0]['values'][0]['columns']
    assert set(merged) <= set(evidence.columns)
    assert evidence.isna().all().all()


def test_the_state_patch_merges_the_evidence_into_massachusetts_parcels():
    national = get_recipe_by_id('US_parcel-openplaces-2026')
    patched, applied = apply_recipe_patches(national, AdminId('US-MA-MED'))
    specs = next(s for s in patched['pipeline'] if s['step'] == 'merge_enrichments')[
        'recipes'
    ]

    assert 'US-MA_parcel-openplaces-2026' in applied
    assert ENRICH in [s['recipe_id'] for s in specs]
    unpatched, _ = apply_recipe_patches(national, AdminId('US-NC-CUR'))
    other = next(s for s in unpatched['pipeline'] if s['step'] == 'merge_enrichments')
    assert ENRICH not in [s['recipe_id'] for s in other['recipes']]
    # The viewer's two columns are shared
    shared = national['share']['columns']
    assert {'protected_open_space_share', 'protected_open_space_level'} <= set(shared)


def test_the_dag_sees_the_layer_behind_the_enrichment():
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(ENRICH)}

    assert {REFERENCE, 'US_parcel-spine-2026'} <= upstream


def test_every_merged_column_is_registered():
    from openplaces.core.attribute_registry import load_registry

    registry = load_registry().index
    patch = get_recipe_by_id('US-MA_parcel-openplaces-2026')
    canonical = patch['pipeline_patch'][0]['values'][0]['columns'].values()
    ingest = get_recipe_by_id(REFERENCE)['columns']

    assert set(canonical) <= set(registry)
    assert set(ingest) <= set(registry)
