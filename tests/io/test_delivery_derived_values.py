"""Values openplaces derived from a restricted source stay with it.

A footprint's structure value is apportioned from its parcel's
improvement value, and a missing land value is estimated from the rates
of the county's recorded ones. Where the county's roll restricts
redistribution, both are derived from it and are withheld wherever it
is (maintainer's decision, 2026-10-23), while an open county's estimates
ship. A city looked up from its ZIP code is tied to a place and ships,
and a classification computed from the roll's withheld use class goes
with it (maintainer, 2026-10-24). Sources and values are fabricated.
"""

from __future__ import annotations

import pandas as pd
import pytest

import openplaces.io.delivery as delivery_module
from openplaces.io.delivery import export_delivery, share_derived_from
from openplaces.io.delivery.redaction import (
    WITHHELD,
    find_restricted,
    units_supplied,
)
from openplaces.recipe import get_recipe_by_id
from tests.io.test_delivery import _county, _recipe

RESTRICTED_UNIT = 'US-NC-AL'
OPEN_UNIT = 'US-NC-BB'
KNOWN = {'closedroll', 'openroll', 'nsi', 'overture'}


def _roll(**overrides):
    """A county roll reaching footprints inside the curated parcels."""
    return {
        'source_id': 'closedroll',
        'recipe_id': f'{RESTRICTED_UNIT}_property-closedroll-2026',
        'license': 'fabricated: no redistribution',
        'terms_url': 'https://example.org/terms',
        'reasons': ['redistribution restricted'],
        'entity_type': 'property',
        'admin_id': RESTRICTED_UNIT,
        'attributes': [
            'structure_value',
            'land_value',
            'improvement_value_parcel',
            'year_built_parcel',
        ],
        'layer_source': False,
        'via_entities': ['parcel'],
        **overrides,
    }


def _frame():
    """Rows a to d in the restricted county, e and f in an open one."""
    units = [RESTRICTED_UNIT] * 4 + [OPEN_UNIT] * 2
    return pd.DataFrame(
        {
            'admin3_id': units,
            'structure_value': [100.0, 0.0, 30.0, 50.0, 10.0, 20.0],
            'structure_value_source': [
                'parcel.closedroll',
                'parcel',
                'parcel.city__is_residential+imputed',
                'nsi',
                'parcel',
                'parcel.city__is_residential+imputed',
            ],
            'structure_value_per_area': [1.0, 0.0, 0.3, 0.5, 0.1, 0.2],
            'land_value': [5.0, 6.0, 7.0, None, 8.0, 9.0],
            'land_value_source': [
                'parcel.closedroll',
                'parcel.closedroll',
                'parcel.address_street_city+imputed',
                None,
                'parcel.openroll',
                'parcel.city__is_residential+imputed',
            ],
            'year_built': [1990, 1991, 1992, 1993, 1994, 1995],
            'year_built_source': ['parcel', 'parcel', 'nsi', 'nsi', 'parcel', 'parcel'],
            'n_dwellings': [1, 1, 1, 1, 1, 1],
            'n_dwellings_source': ['overture'] * 6,
        },
        index=pd.Index(list('abcdef'), name='footprint_id'),
    )


def _cells(found, column):
    return found['cells'].get(column, pd.Series(False, index=list('abcdef'))).tolist()


# Which cells are derived from the roll.


def test_derived_cells_in_the_restricted_county_are_withheld():
    found = find_restricted(_frame(), [_roll()], 'admin3_id', KNOWN)

    # Its own value, its apportioned zero and an estimate through the
    # parcels; NSI's value is not the roll's.
    assert _cells(found, 'structure_value') == [True, True, True, False, False, False]
    assert _cells(found, 'land_value')[:3] == [True, True, True]
    # The roll's year, copied onto the footprint under the layer's name.
    assert _cells(found, 'year_built') == [True, True, False, False, False, False]
    # Overture's count is not derived from the roll.
    assert not any(_cells(found, 'n_dwellings'))


def test_an_open_countys_estimates_are_kept():
    found = find_restricted(_frame(), [_roll()], 'admin3_id', KNOWN)
    for column in ('structure_value', 'land_value', 'year_built'):
        assert _cells(found, column)[4:] == [False, False], column


def test_a_ratio_goes_with_its_value():
    found = find_restricted(_frame(), [_roll()], 'admin3_id', KNOWN)
    assert _cells(found, 'structure_value_per_area') == _cells(found, 'structure_value')


def test_a_ratio_follows_a_value_the_other_pass_withheld():
    """The evidence pass sees only the redacted sidecar of a value."""
    frame = pd.DataFrame(
        {
            'admin3_id': [RESTRICTED_UNIT, RESTRICTED_UNIT],
            'improvement_value_parcel_source': [WITHHELD, 'nsi'],
            'improvement_value_parcel_per_area': [2.0, 3.0],
        },
        index=pd.Index(['a', 'b'], name='footprint_id'),
    )
    found = find_restricted(frame, [_roll()], 'admin3_id', KNOWN, supplied={})
    assert found['cells']['improvement_value_parcel_per_area'].tolist() == [
        True,
        False,
    ]


def test_a_source_whose_token_is_on_no_cell_derives_nothing():
    """A statewide registry reaches the value columns through the parcels.

    The walk says only that it could; nothing in the bundle names it, so
    the layer's own values and estimates are not taken for its.
    """
    registry = _roll(source_id='deedregistry', admin_id='US-NC')
    found = find_restricted(_frame(), [registry], 'admin3_id', KNOWN)
    assert found['counts'] == {}


def test_a_source_read_directly_derives_nothing_through_a_layer():
    direct = _roll(via_entities=[])
    found = find_restricted(_frame(), [direct], 'admin3_id', KNOWN)
    # Only the cells naming it.
    assert _cells(found, 'structure_value') == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    assert not any(_cells(found, 'year_built'))


def test_supplied_units_are_read_from_the_columns_the_source_reaches():
    assert units_supplied(_frame(), [_roll()], 'admin3_id') == {
        'closedroll': {RESTRICTED_UNIT}
    }


def test_a_peer_group_name_no_longer_passes_for_a_source():
    """A layer roll's estimate named its tier, which exonerated it."""
    layer = _roll(
        layer_source=True,
        entity_type='parcel',
        attributes=['land_value'],
        via_entities=[],
    )
    frame = pd.DataFrame(
        {
            'admin3_id': [RESTRICTED_UNIT, RESTRICTED_UNIT, OPEN_UNIT],
            'land_value_imputed': [1.0, 2.0, 3.0],
            'land_value_imputed_source': [
                'city__is_residential+imputed',
                'openroll',
                'city__is_residential+imputed',
            ],
        },
        index=pd.Index(['a', 'b', 'c'], name='parcel_id'),
    )
    found = find_restricted(frame, [layer], 'admin3_id', KNOWN)
    # The estimate goes; a value another declared source supplied stays,
    # and so does the open county's estimate.
    assert found['cells']['land_value_imputed'].tolist() == [True, False, False]


def test_each_withheld_cell_is_counted_once_per_source():
    """A roll and its values table find the same cells."""
    table = _roll(recipe_id=f'{RESTRICTED_UNIT}_property-closedroll-2026_values')
    once = find_restricted(_frame(), [_roll()], 'admin3_id', KNOWN)['counts']
    twice = find_restricted(_frame(), [_roll(), table], 'admin3_id', KNOWN)['counts']
    assert once == twice


def test_a_withheld_base_no_longer_clears_its_parsed_parts():
    """The evidence pass reads `address_number` by `address_source`.

    That sidecar reads 'withheld' once the canonical pass emptied the
    address, and 'withheld' used to pass for another source.
    """
    layer = _roll(
        layer_source=True, entity_type='parcel', attributes=['address'], via_entities=[]
    )
    frame = pd.DataFrame(
        {
            'admin3_id': [RESTRICTED_UNIT, OPEN_UNIT],
            'address_source': [WITHHELD, WITHHELD],
            'address_number': ['1', '2'],
        },
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    found = find_restricted(frame, [layer], 'admin3_id', KNOWN)
    assert found['cells']['address_number'].tolist() == [True, False]


def test_a_column_named_for_another_source_is_not_the_layers():
    layer = _roll(
        layer_source=True,
        entity_type='parcel',
        attributes=['n_dwellings'],
        via_entities=[],
    )
    frame = pd.DataFrame(
        {
            'admin3_id': [RESTRICTED_UNIT],
            'n_dwellings': [2],
            'n_dwellings_source': [WITHHELD],
            'n_dwellings_overture': [3],
        },
        index=pd.Index(['a'], name='parcel_id'),
    )
    found = find_restricted(frame, [layer], 'admin3_id', KNOWN)
    assert 'n_dwellings_overture' not in found['cells']


# Estimates tied to a place, and classifications of the roll's codes.


def test_a_city_looked_up_from_its_zip_code_stays():
    frame = _frame()
    frame['city'] = ['Town'] * 6
    frame['city_source'] = ['zipcodes+imputed', 'parcel+imputed'] + [None] * 4
    found = find_restricted(frame, [_roll()], 'admin3_id', KNOWN)
    # The ZIP lookup stays; an estimate through the parcels goes.
    assert _cells(found, 'city')[:2] == [False, True]


def test_a_city_from_zip_codes_stays_beside_a_layer_roll():
    layer = _roll(layer_source=True, entity_type='parcel', via_entities=[])
    frame = pd.DataFrame(
        {
            'admin3_id': [RESTRICTED_UNIT, RESTRICTED_UNIT],
            'city': ['Town', 'Town'],
            'city_source': ['zipcodes+imputed', 'zipcodes+imputed'],
            'group_parcel': ['Residential', 'Commercial'],
            'group_parcel_source': ['group_statistic+imputed', 'parcel'],
        },
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    found = find_restricted(frame, [layer], 'admin3_id', KNOWN)
    assert 'city' not in found['cells']
    # A group statistic of the roll's values goes.
    assert found['cells']['group_parcel'].tolist() == [True, True]


DERIVED = {'occupancy_type_parcel': ['use_group_combined_parcel', 'group_parcel']}


def test_a_declared_derivation_goes_with_its_inputs():
    roll = _roll(attributes=[*_roll()['attributes'], 'use_group_combined_parcel'])
    frame = _frame()
    frame['use_group_combined_parcel'] = ['RES'] * 4 + ['RES', None]
    frame['occupancy_type_parcel'] = ['Single-Family'] * 6
    found = find_restricted(frame, [roll], 'admin3_id', KNOWN, derived_from=DERIVED)
    # The roll's use class is withheld in its county, so is its class.
    assert _cells(found, 'occupancy_type_parcel') == [True] * 4 + [False] * 2


def test_a_declared_derivation_follows_the_other_pass():
    frame = pd.DataFrame(
        {
            'admin3_id': [RESTRICTED_UNIT, RESTRICTED_UNIT],
            'group_parcel_source': [WITHHELD, 'nsi'],
            'occupancy_type_parcel': ['Single-Family', 'Multi-Family'],
        },
        index=pd.Index(['a', 'b'], name='footprint_id'),
    )
    found = find_restricted(
        frame, [_roll()], 'admin3_id', KNOWN, supplied={}, derived_from=DERIVED
    )
    assert found['cells']['occupancy_type_parcel'].tolist() == [True, False]


def test_a_derivation_needs_its_inputs_listed():
    recipe = _recipe()
    recipe['share']['derived_from'] = {'occupancy_type_parcel': []}
    with pytest.raises(ValueError, match='computed from'):
        share_derived_from(recipe)


def test_the_footprint_recipe_declares_its_roll_classification():
    derived = share_derived_from(get_recipe_by_id('US_footprint-openplaces-2026'))
    assert set(derived['occupancy_type_parcel']) == {
        'use_group_combined_parcel',
        'group_parcel',
    }
    assert 'fallback' not in get_recipe_by_id('US_footprint-openplaces-2026')['share']


# The bundle: public and team.


def _export_recipe():
    recipe = _recipe()
    recipe['share']['columns'] = [*recipe['share']['columns'], 'structure_value']
    recipe['share']['derived_from'] = DERIVED
    return recipe


def _write_counties():
    for ids, unit in ((['a', 'b', 'c'], RESTRICTED_UNIT), (['d', 'e'], OPEN_UNIT)):
        closed = unit == RESTRICTED_UNIT
        _county(
            ids,
            admin3_id=unit,
            extra={
                'structure_value': [100.0, 0.0, 30.0] if closed else [10.0, 20.0],
                'structure_value_source': (
                    [
                        'parcel.closedroll',
                        'parcel',
                        'parcel.city__is_residential+imputed',
                    ]
                    if closed
                    else ['parcel.openroll', 'parcel.city__is_residential+imputed']
                ),
                'structure_value_per_area': [1.0, 0.0, 0.3] if closed else [0.1, 0.2],
                'structure_value_building_nsi': (
                    [90.0, 80.0, 70.0] if closed else [60.0, 50.0]
                ),
                'land_value': [5.0, None, 7.0] if closed else [8.0, 9.0],
                'land_value_source': (
                    ['parcel.closedroll', None, 'parcel.city__is_residential+imputed']
                    if closed
                    else ['parcel.openroll', 'parcel.city__is_residential+imputed']
                ),
                'use_group_combined_parcel': ['RES'] * len(ids),
                'occupancy_type_parcel': ['Single-Family'] * len(ids),
                'city': ['Town'] * len(ids),
                'city_source': ['zipcodes+imputed'] * len(ids),
            },
        )


@pytest.fixture
def bundles(mock_data_root, monkeypatch):
    """The two counties shipped as a public and as a team bundle."""
    _write_counties()
    roll = {
        **_roll(attributes=[*_roll()['attributes'], 'use_group_combined_parcel']),
        'team_sharing_permitted': True,
    }
    monkeypatch.setattr(
        delivery_module, 'restricted_inputs', lambda *_args, **_kw: [roll]
    )
    public = {
        'region_id': 'test-region',
        'admin_level': 2,
        'admin_ids': [RESTRICTED_UNIT, OPEN_UNIT],
        'admin_id': None,
        'audience': 'public',
    }
    team = {
        **public,
        'region_id': 'test-region.team',
        'audience': 'team',
        'base_region_id': 'test-region',
    }
    monkeypatch.setattr(delivery_module, 'delivery_regions', lambda _: [public, team])
    recipe = _export_recipe()
    return {
        'public': export_delivery(recipe, region='test-region', allow_stale=True),
        'team': export_delivery(recipe, region='test-region.team', allow_stale=True),
    }


def _read(paths, role):
    return pd.read_parquet(paths[role])


def test_public_bundles_withhold_the_restricted_countys_estimates(bundles):
    evidence = _read(bundles['public'], 'evidence')
    # The land estimate in the restricted county goes, the open
    # county's stays.
    assert pd.isna(evidence.loc['c', 'land_value'])
    assert evidence.loc['c', 'land_value_source'] == WITHHELD
    assert evidence.loc['e', 'land_value'] == 9.0


def test_public_bundles_leave_withheld_structure_values_empty(bundles):
    canonical = _read(bundles['public'], 'canonical')
    point = _read(bundles['public'], 'point')

    # No other lane stands in: NSI's value is in the evidence file only.
    restricted = canonical.loc[['a', 'b', 'c']]
    assert restricted['structure_value'].isna().all()
    assert (restricted['structure_value_source'] == WITHHELD).all()
    assert point.loc[['a', 'b', 'c'], 'structure_value_per_area'].isna().all()
    # The open county keeps its roll's value and its own estimate.
    assert canonical.loc['d', 'structure_value'] == 10.0
    assert canonical.loc['d', 'structure_value_source'] == 'parcel.openroll'
    assert canonical.loc['e', 'structure_value'] == 20.0


def test_public_bundles_withhold_the_rolls_classification_not_the_city(bundles):
    evidence = _read(bundles['public'], 'evidence')
    assert evidence.loc[['a', 'b', 'c'], 'occupancy_type_parcel'].isna().all()
    assert evidence.loc[['d', 'e'], 'occupancy_type_parcel'].notna().all()
    assert (evidence['city'] == 'Town').all()


def test_team_bundles_keep_the_estimates_and_the_roll(bundles):
    canonical = _read(bundles['team'], 'canonical')
    evidence = _read(bundles['team'], 'evidence')
    assert evidence.loc['c', 'land_value'] == 7.0
    assert canonical.loc['a', 'structure_value'] == 100.0
    assert canonical.loc['a', 'structure_value_source'] == 'parcel.closedroll'
    assert evidence['occupancy_type_parcel'].notna().all()
