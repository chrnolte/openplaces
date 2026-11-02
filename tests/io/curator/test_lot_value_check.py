"""Standing lot-value-sharing check (io.curator.lot_value_check).

Every id and value below is fabricated.
"""

import pandas as pd

from openplaces.io.curator.lot_value_check import lot_value_sharing_counts


def _buildings(rows):
    columns = [
        'parcel_id',
        'occupancy_type',
        'area_m2',
        'structure_value',
        'structure_value_source',
        'improvement_value_parcel',
    ]
    return pd.DataFrame(rows, columns=columns)


ROLL = 'parcel.roll'


def test_a_complex_valued_on_one_of_three_buildings_is_unshared():
    frame = _buildings(
        [
            ('L1', 'Multi-Family', 400.0, 900_000.0, ROLL, 900_000.0),
            ('L1', 'Multi-Family', 400.0, None, None, 900_000.0),
            ('L1', 'Multi-Family', 400.0, None, None, 900_000.0),
            # Shared well: both large buildings carry the roll's value.
            ('L2', 'Multi-Family', 300.0, 400_000.0, ROLL, 800_000.0),
            ('L2', 'Multi-Family', 300.0, 400_000.0, ROLL, 800_000.0),
        ]
    )
    counts = lot_value_sharing_counts(frame)
    assert counts['n_lots_multi'] == 2
    assert counts['n_lots_unshared'] == 1
    assert counts['unshared_unvalued_m2'] == 800.0
    assert counts['n_unvalued_beside_valued'] == 2


def test_sheds_mobile_homes_and_non_roll_values_are_counted_as_intended():
    frame = _buildings(
        [
            ('L1', 'Single-Family', 150.0, 200_000.0, ROLL, 200_000.0),
            ('L1', 'Secondary', 30.0, None, None, 200_000.0),
            ('L1', 'Manufactured Home', 70.0, None, None, 200_000.0),
            # A value from NSI is not the roll's: still unvalued.
            ('L1', 'Single-Family', 120.0, 150_000.0, 'building.nsi', 200_000.0),
            # A lot without a roll value has nothing to double count.
            ('L3', 'Single-Family', 150.0, None, None, None),
            ('L3', 'Single-Family', 150.0, None, None, None),
        ]
    )
    counts = lot_value_sharing_counts(frame)
    assert counts['n_unvalued_beside_valued'] == 1
    assert counts['n_lots_unvalued_beside_valued'] == 1
    assert counts['n_lots_unshared'] == 0


def test_missing_columns_give_zero_counts_rather_than_an_error():
    frame = pd.DataFrame({'parcel_id': ['L1', 'L1']})
    counts = lot_value_sharing_counts(frame)
    assert set(counts.values()) == {0}
