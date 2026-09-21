"""A roll row keyed on a split unit is re-keyed onto the unit's lot, and
an account on several lots has its land divided by area.

Every id and value below is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer.links import _move_units_to_lots


def _spine():
    return pd.DataFrame(
        {'parcel_id_local': ['A', 'B1', 'C', 'D'], 'area_ha': [1.0, 0.5, 3.0, 1.0]}
    )


def _roll():
    return pd.DataFrame(
        {
            'parcel_id_local': ['A1', 'A2', 'B1', 'CD', 'ZZ'],
            'land_value': [10.0, 10.0, 50.0, 400.0, 7.0],
            'improvement_value': [90.0, 80.0, 60.0, 1000.0, 0.0],
            'use_group': ['Condo', 'Condo', 'SF', 'Farm', 'SF'],
        }
    )


def _pairs():
    return pd.DataFrame(
        {
            'unit_key': ['A1', 'A2', 'CD', 'CD', 'B1'],
            'lot_key': ['A', 'A', 'C', 'D', 'X'],
        }
    )


def _moved():
    spine = _spine()
    return _move_units_to_lots(
        _roll(), 'parcel_id_local', spine['parcel_id_local'], _pairs(), spine
    )


def test_units_take_their_lots_key_and_nothing_else_moves():
    out, n_moved, n_divided = _moved()
    assert sorted(out['parcel_id_local']) == ['A', 'A', 'B1', 'C', 'D', 'ZZ']
    assert (n_moved, n_divided) == (3, 1)
    # B1 is on the spine under its own key: the pair naming it is ignored.
    assert out.loc[out['parcel_id_local'] == 'B1', 'land_value'].tolist() == [50.0]


def test_land_is_divided_by_lot_area_and_conserved():
    out, _, _ = _moved()
    land = out.set_index('parcel_id_local')['land_value']
    assert land['C'] == 300.0 and land['D'] == 100.0
    assert out['land_value'].sum() == _roll()['land_value'].sum()


def test_an_improvement_goes_whole_to_the_largest_lot():
    out, _, _ = _moved()
    improvement = out.set_index('parcel_id_local')['improvement_value']
    assert improvement['C'] == 1000.0
    assert pd.isna(improvement['D'])
    assert out['improvement_value'].sum() == _roll()['improvement_value'].sum()
    # A column that is not additive is simply carried to both lots.
    assert out.set_index('parcel_id_local').loc[['C', 'D'], 'use_group'].tolist() == [
        'Farm',
        'Farm',
    ]


def test_no_pair_means_no_change():
    spine = _spine()
    roll = _roll()
    out, n_moved, n_divided = _move_units_to_lots(
        roll,
        'parcel_id_local',
        spine['parcel_id_local'],
        pd.DataFrame({'unit_key': [], 'lot_key': []}, dtype='string'),
        spine,
    )
    assert out is roll and (n_moved, n_divided) == (0, 0)
