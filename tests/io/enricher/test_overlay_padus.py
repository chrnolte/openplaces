"""The PAD-US parcel overlay, on fabricated geometry.

Fee polygons of different owners and two overlapping easement records
lie over a row of fabricated lots. The step promises: the easement
share counts land under two records once, the public-land share sees
only fee polygons of public owners, a flag is set at the recipe's
threshold and not below it, and a unit with no reference gets every
column, the flags included, all missing.
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
from openplaces.recipe import get_recipe_by_id, get_recipe_dependencies

# NAD83 / Conus Albers (meters); the scene is fabricated.
CRS = 'EPSG:5070'
X0, Y0 = 1_500_000, 1_500_000
FEE = 'US_rules-protectedarea-usgs-padus41_fee'
EASEMENT = 'US_rules-protectedarea-usgs-padus41_easement'
ENRICH = 'US_parcel_rules-protectedarea-usgs-padus41'


def _box(x0, y0, x1, y1):
    return shapely.box(X0 + x0, Y0 + y0, X0 + x1, Y0 + y1)


def _fee():
    """A federal tract, a land trust preserve, and a town park."""
    return gpd.GeoDataFrame(
        {
            'protected_area_category': ['Fee', 'Fee', 'Fee'],
            'protected_area_owner_type': ['FED', 'NGO', 'LOC'],
            'protected_area_manager_type': ['FED', 'NGO', 'LOC'],
            'protected_area_gap_status': ['2', '2', '4'],
            'protected_area_public_access': ['OA', 'RA', 'OA'],
            'protected_area_year_established': [1990.0, None, 2004.0],
        },
        geometry=[_box(0, 0, 100, 100), _box(100, 0, 200, 100), _box(200, 0, 300, 100)],
        index=pd.Index([11, 12, 13], name='protected_area_id'),
        crs=CRS,
    )


def _easements():
    """One easement recorded twice (two sources, same land), and a
    third that covers a corner of a lot."""
    return gpd.GeoDataFrame(
        {
            'protected_area_category': ['Easement'] * 3,
            'conservation_easement_holder_type': ['NGO', 'STAT', 'NGO'],
            'protected_area_gap_status': ['3', '3', '4'],
            'protected_area_year_established': [2011.0, 2011.0, None],
        },
        geometry=[
            _box(0, 200, 100, 300),
            _box(0, 200, 100, 300),
            _box(280, 200, 300, 220),
        ],
        index=pd.Index([21, 22, 23], name='protected_area_id'),
        crs=CRS,
    )


def _lots():
    return gpd.GeoDataFrame(
        geometry=[
            _box(10, 10, 90, 90),  # on the federal tract
            _box(110, 10, 190, 90),  # on the land trust preserve
            _box(150, 10, 250, 90),  # half preserve, half town park
            _box(290, 10, 390, 90),  # a tenth on the town park
            _box(0, 200, 100, 300),  # under the twice-recorded easement
            _box(200, 200, 300, 300),  # a 4% corner under an easement
            _box(500, 500, 600, 600),  # outside everything
        ],
        index=pd.Index(list('abcdefg'), name='parcel_id'),
        crs=CRS,
    )


def test_overlapping_records_count_once_when_dissolved():
    lots = _lots().geometry
    dissolved, dominant, _ = overlay_shares(lots, _easements(), dissolve=True)

    # A lot half under the twice-recorded easement: the sum counts that
    # half twice, the dissolved share once.
    half = gpd.GeoSeries([_box(50, 200, 150, 300)], index=['h'], crs=CRS)
    summed_half = overlay_shares(half, _easements())[0]['h']
    dissolved_half = overlay_shares(half, _easements(), dissolve=True)[0]['h']
    assert summed_half == pytest.approx(1)
    assert dissolved_half == pytest.approx(0.5)
    assert dissolved['e'] == pytest.approx(1)
    # One record only: dissolving changes nothing
    assert dissolved['f'] == pytest.approx(0.04)
    # A tie between the two records goes to the smaller id
    assert dominant['e'] == 21


def _state(admin_id='US-NC-TYR'):
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
    for step in get_recipe_by_id(ENRICH)['pipeline']:
        params = {k: v for k, v in step.items() if k != 'step'}
        state = overlay_reference_polygons(state, **params)
    return state.evidence


def _fake_load(recipe_id, admin_id):
    assert admin_id == 'US'
    return {FEE: _fee(), EASEMENT: _easements()}[recipe_id]


def _threshold(flag):
    (step,) = [
        s for s in get_recipe_by_id(ENRICH)['pipeline'] if s.get('flag_column') == flag
    ]
    return step['min_share']


def test_the_recipe_flags_easements_and_public_land(monkeypatch):
    monkeypatch.setattr(overlay_module, '_load_reference', _fake_load)
    ev = _run(_state())

    # Fee: any owner for the protected share, public owners for the flag
    assert ev.loc['b', 'protected_area_share_padus'] == pytest.approx(1)
    assert ev.loc['b', 'public_land_share_padus'] == 0
    assert not ev.loc['b', 'has_public_land_padus']
    assert ev.loc['a', 'has_public_land_padus']
    assert ev.loc['a', 'protected_area_owner_type_padus'] == 'FED'
    assert ev.loc['a', 'protected_area_year_established_padus'] == 1990
    # Half NGO, half town: protected all over, public on half
    assert ev.loc['c', 'protected_area_share_padus'] == pytest.approx(1)
    assert ev.loc['c', 'public_land_share_padus'] == pytest.approx(0.5)
    # A tenth on the park: below any threshold the recipe may state
    assert ev.loc['d', 'public_land_share_padus'] == pytest.approx(0.1)
    assert _threshold('has_public_land') > 0.1
    assert not ev.loc['d', 'has_public_land_padus']

    # Easements: land under two records counts once
    assert ev.loc['e', 'conservation_easement_share_padus'] == pytest.approx(1)
    assert ev.loc['e', 'has_conservation_easement_padus']
    assert ev.loc['e', 'conservation_easement_year_established_padus'] == 2011
    assert ev.loc['e', 'conservation_easement_holder_type_padus'] == 'NGO'
    # Ids are written as text, the registry's type for an id
    assert ev.loc['e', 'conservation_easement_id_padus'] == '21'
    # A 4% corner is a boundary sliver, not an encumbered parcel
    assert ev.loc['f', 'conservation_easement_share_padus'] == pytest.approx(0.04)
    assert not ev.loc['f', 'has_conservation_easement_padus']
    # Outside everything: zero shares, false flags, no attributes
    assert ev.loc['g', 'conservation_easement_share_padus'] == 0
    assert not ev.loc['g', 'has_conservation_easement_padus']
    assert pd.isna(ev.loc['g', 'protected_area_id_padus'])


def test_a_unit_without_the_reference_gets_every_column_missing(monkeypatch):
    monkeypatch.setattr(overlay_module, '_load_reference', lambda *a: None)
    ev = _run(_state())

    assert 'has_conservation_easement_padus' in ev.columns
    assert 'has_public_land_padus' in ev.columns
    assert ev.isna().all().all()


def test_a_flag_needs_a_threshold():
    with pytest.raises(ValueError, match='min_share'):
        overlay_reference_polygons(
            _state(),
            reference_recipe_id=FEE,
            share_column='public_land_share',
            flag_column='has_public_land',
        )


def test_the_dag_sees_both_layers_behind_the_enrichment():
    upstream = {e.upstream_recipe_id for e in get_recipe_dependencies(ENRICH)}

    assert {FEE, EASEMENT, 'US_parcel-spine-2026'} <= upstream


def test_every_column_is_registered_and_suffix_free_in_the_registry():
    from openplaces.core.attribute_registry import load_registry

    registry = set(load_registry().index)
    for recipe_id in (FEE, EASEMENT):
        assert set(get_recipe_by_id(recipe_id)['columns']) <= registry
    for step in get_recipe_by_id(ENRICH)['pipeline']:
        names = [step['share_column'], step.get('flag_column'), step.get('id_column')]
        names += list((step.get('columns') or {}).values())
        assert {n for n in names if n} <= registry
