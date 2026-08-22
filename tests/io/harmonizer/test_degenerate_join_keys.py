"""A placeholder shared by thousands of rows is not a join key.

`link_by_id` joined on whatever `parcel_id_local` held. Where a source
writes '0' for "parcel number unknown", or repeats one code across
thousands of unrelated parcels, every mode went wrong silently:
'attributes' picked one arbitrary row for all of them, 'count' reported the
whole group's size on each, and 'aggregate' **summed** their value columns
and wrote that sum back onto every member.

Measured before this guard: a quarter-acre Brazoria County lot carrying a
$7.5 billion `total_value`, 24,241 rows there sharing the key '0', and 152
eastern-NC footprints holding 62% of the entire delivered parcel value.
Healthy files in the same regions peak at 5 rows per key (Kleberg) and 49
(Pender, a real multi-unit building), which is what the threshold has to
leave alone.
"""

import warnings

import pandas as pd
import pytest

from openplaces.io.harmonizer.links import (
    DEGENERATE_KEY_MIN_ROWS,
    _neutralize_degenerate_keys,
    _placeholder_key_mask,
)


def _ref(keys, values=None):
    frame = pd.DataFrame({'parcel_id_local': keys})
    if values is not None:
        frame['total_value'] = values
    return frame


class TestPlaceholderKeys:
    @pytest.mark.parametrize(
        'value', ['0', '00', '000000', '000-00-0000', '0-0-0', '', '   ']
    )
    def test_all_zero_and_blank_codes_are_placeholders(self, value):
        assert bool(_placeholder_key_mask(pd.Series([value])).iloc[0])

    @pytest.mark.parametrize(
        'value', ['0123', '167144', '42047533000000', '10|504|70|37']
    )
    def test_a_real_code_is_not(self, value):
        assert not bool(_placeholder_key_mask(pd.Series([value])).iloc[0])

    def test_a_placeholder_is_neutralized_however_rare(self):
        """Two rows sharing '0' are still two different parcels."""
        ref = _ref(['0', '0', 'A1'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = _neutralize_degenerate_keys(ref, 'parcel_id_local')
        assert out['parcel_id_local'].isna().sum() == 2
        assert out.loc[2, 'parcel_id_local'] == 'A1'


class TestOverusedKeys:
    def test_a_key_on_thousands_of_rows_is_neutralized(self):
        keys = ['167144'] * 2_443 + [f'p{i}' for i in range(50_000)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = _neutralize_degenerate_keys(_ref(keys), 'parcel_id_local')
        assert out['parcel_id_local'].isna().sum() == 2_443
        assert out['parcel_id_local'].notna().sum() == 50_000

    def test_a_genuine_multi_unit_building_is_left_alone(self):
        """Pender's most-shared real key covers 49 rows -- a condominium,
        not a placeholder. The floor exists to keep those joined."""
        keys = ['42047533000000'] * 49 + [f'p{i}' for i in range(55_000)]
        out = _neutralize_degenerate_keys(_ref(keys), 'parcel_id_local')
        assert out['parcel_id_local'].notna().all()

    def test_the_floor_holds_for_a_small_reference(self):
        """A share alone would neutralize ordinary duplicates in a small
        file; the absolute floor prevents that."""
        keys = ['dup'] * 20 + [f'p{i}' for i in range(200)]
        out = _neutralize_degenerate_keys(_ref(keys), 'parcel_id_local')
        assert out['parcel_id_local'].notna().all()
        assert 20 < DEGENERATE_KEY_MIN_ROWS

    def test_it_warns_rather_than_dropping_silently(self):
        keys = ['0'] * 200 + [f'p{i}' for i in range(10)]
        with pytest.warns(UserWarning, match='degenerate'):
            _neutralize_degenerate_keys(_ref(keys), 'parcel_id_local', 'a-recipe')

    def test_a_clean_reference_is_returned_untouched(self):
        ref = _ref([f'p{i}' for i in range(100)])
        out = _neutralize_degenerate_keys(ref, 'parcel_id_local')
        assert out is ref


def test_the_value_inflation_this_prevents():
    """The whole point, end to end: summing value columns over a degenerate
    key both inflates the total and broadcasts it back to every member."""
    keys = ['0'] * 300 + ['real']
    values = [1_000_000.0] * 300 + [250_000.0]
    ref = _ref(keys, values)

    naive = (
        ref.dropna(subset=['parcel_id_local'])
        .groupby('parcel_id_local')['total_value']
        .sum()
    )
    assert naive['0'] == 300_000_000.0  # what used to happen

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        guarded = _neutralize_degenerate_keys(ref, 'parcel_id_local')
    kept = (
        guarded.dropna(subset=['parcel_id_local'])
        .groupby('parcel_id_local')['total_value']
        .sum()
    )
    assert set(kept.index) == {'real'}
    assert kept['real'] == 250_000.0
