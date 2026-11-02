"""Tests for the structure value zero reasons (fabricated rows)."""

import pandas as pd

from openplaces.io.curator.zero_value_reasons import structure_value_zero_reasons

PARAMS = {
    'exempt_classes': ('Church', 'Average School'),
    'exempt_use_groups': ('totally exempt',),
}


def _row(**kwargs):
    row = {
        'parcel_id': 'P0',
        'structure_value': 0.0,
        'structure_value_source': 'parcel.example',
        'occupancy_type': 'Single-Family',
        'priority_on_parcel': 'primary',
        'area_m2': 150.0,
        'land_use_class_parcel': 'Single-Family',
        'use_group_combined_parcel': 'residential',
        'n_manufactured_homes_per_parcel': 0,
        'manufactured_home_community': False,
        'building_id_nsi': None,
        'year_built_parcel': None,
        'living_area_sqft_parcel': None,
        'year_built_source': None,
    }
    row.update(kwargs)
    return row


def test_each_reason_is_named_and_only_zeros_get_one():
    rows = [
        _row(parcel_id='A', structure_value=250000.0),
        _row(parcel_id='A', occupancy_type='Secondary', priority_on_parcel='secondary'),
        _row(parcel_id='B', occupancy_type='Manufactured Home', area_m2=90.0),
        _row(parcel_id='C', land_use_class_parcel='Manufactured Home Park'),
        _row(parcel_id='D', occupancy_type='Church'),
        _row(parcel_id='E', use_group_combined_parcel='residential | Totally Exempt'),
        _row(parcel_id='A', priority_on_parcel='secondary'),
        _row(
            parcel_id='F', occupancy_type='Agricultural', priority_on_parcel='secondary'
        ),
        _row(parcel_id='G', land_use_class_parcel='Vacant', building_id_nsi='N1'),
        _row(parcel_id='H', land_use_class_parcel='Vacant', area_m2=30.0),
        _row(parcel_id='I'),
        _row(parcel_id='J', structure_value=None),
    ]
    reasons = structure_value_zero_reasons(pd.DataFrame(rows), **PARAMS)
    assert reasons.tolist() == [
        pd.NA,
        'secondary',
        'personal_property',
        'personal_property',
        'exempt',
        'exempt',
        'included_elsewhere',
        'secondary',
        'not_on_roll',
        'no_value',
        'no_value',
        pd.NA,
    ]


def test_a_house_on_a_vacant_lot_needs_a_second_source():
    rows = [
        _row(land_use_class_parcel='Vacant'),
        _row(land_use_class_parcel='Vacant', year_built_parcel=1970),
    ]
    reasons = structure_value_zero_reasons(pd.DataFrame(rows), **PARAMS)
    assert reasons.tolist() == ['no_value', 'not_on_roll']


def test_a_dwelling_beside_a_manufactured_home_is_personal_property():
    rows = [_row(n_manufactured_homes_per_parcel=1)]
    reasons = structure_value_zero_reasons(pd.DataFrame(rows), **PARAMS)
    assert reasons.tolist() == ['personal_property']


def test_a_duplicate_index_and_missing_columns_are_tolerated():
    frame = pd.DataFrame(
        [_row(use_group_combined_parcel='totally exempt'), _row()], index=[7, 7]
    )
    assert structure_value_zero_reasons(frame, **PARAMS).tolist() == [
        'exempt',
        'no_value',
    ]
    bare = pd.DataFrame({'structure_value': [0.0, 1.0]})
    bare['structure_value_source'] = 'parcel.example'
    bare['occupancy_type'] = 'Retail'
    assert structure_value_zero_reasons(bare).tolist() == ['no_value', pd.NA]
