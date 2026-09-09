"""A negative reference value is named as such, not as over-allocation.

The conservation check compares an allocated total, from which masked
secondary entities are removed, against an unmasked source total. A
negative share that lands on a masked entity leaves one side and stays
in the other, so the totals disagree in the direction that reads as
duplication. Duplin County NC carried -984.6M of negative improvement
value into curate and the resulting "over-allocated" error sent a
reader hunting for a duplicated link for an hour. The precondition is
now checked and named.
"""

import pandas as pd
import pytest

from openplaces.io.harmonizer.apportion import (
    NegativeReferenceValueError,
    apportion_reference_values,
)

SID = 'footprint_id'


def _inputs(improvement):
    pairs = pd.DataFrame(
        {
            SID: ['fp-1', 'fp-2', 'fp-3'],
            'parcel_id': ['p-a', 'p-a', 'p-b'],
            'area_intersection_m2': [60.0, 40.0, 100.0],
        }
    )
    ref_values = pd.DataFrame(
        {'improvement_value': improvement},
        index=pd.Index(['p-a', 'p-b'], name='parcel_id'),
    )
    return pairs, ref_values


def test_a_negative_reference_value_is_refused_by_name():
    pairs, ref_values = _inputs([-289_900.0, 125_000.0])

    with pytest.raises(NegativeReferenceValueError) as excinfo:
        apportion_reference_values(pairs, ref_values, spine_id_col=SID)

    message = str(excinfo.value)
    assert "'improvement_value'" in message
    assert '1 of 2' in message
    assert 'null_out_of_range' in message


def test_non_negative_references_apportion_and_conserve():
    pairs, ref_values = _inputs([100_000.0, 125_000.0])

    result = apportion_reference_values(pairs, ref_values, spine_id_col=SID)

    assert result['improvement_value'].sum() == pytest.approx(225_000.0)
    assert result.loc['fp-1', 'improvement_value'] == pytest.approx(60_000.0)
    assert result.loc['fp-2', 'improvement_value'] == pytest.approx(40_000.0)


def test_zero_is_a_value_not_a_violation():
    """A parcel with no buildings holds zero improvement, legitimately."""
    pairs, ref_values = _inputs([0.0, 125_000.0])

    result = apportion_reference_values(pairs, ref_values, spine_id_col=SID)

    assert result.loc['fp-1', 'improvement_value'] == 0.0
