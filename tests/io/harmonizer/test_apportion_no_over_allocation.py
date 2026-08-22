"""A parcel's value is split among its buildings, never repeated on each.

`land_value`, `land_value_imputed` and `total_value` are whole-*parcel*
figures. The original reading of "whole value on the principal entity" gave
the full amount to every footprint marked `primary`, and a parcel may carry
several principal structures -- an apartment complex, a school, a refinery.
Each building then looked correct on its own while the region's assessed
value came out 1.23x (NC Brunswick) to 1.69x (TX Kleberg) too high, because
multi-building parcels are disproportionately the high-value commercial
tail.

Two guarantees are pinned here: the value is divided among the qualifying
entities, and `_assert_not_over_allocated` fails the run if any future
change ever hands out more than a reference holds. The gate lives in the
one function both the harmonize and the curate stage apportion through, so
it covers every recipe and every dataset.
"""

import numpy as np
import pandas as pd
import pytest

from openplaces.io.harmonizer.apportion import (
    ValueOverAllocationError,
    apportion_reference_values,
)


def _pairs(rows):
    return pd.DataFrame(
        rows, columns=['footprint_id', 'parcel_id', 'area_intersection_m2']
    )


def _apportion(pairs, ref_values, priority=None, **kwargs):
    return apportion_reference_values(
        pairs,
        ref_values,
        spine_id_col='footprint_id',
        priority=priority,
        **kwargs,
    )


class TestWholeValueIsSplitNotRepeated:
    def test_two_primary_buildings_split_the_parcel_value(self):
        pairs = _pairs([('f1', 'p1', 300.0), ('f2', 'p1', 100.0)])
        ref = pd.DataFrame({'land_value': [400_000.0]}, index=['p1'])
        ref.index.name = 'parcel_id'
        priority = pd.Series({'f1': 'primary', 'f2': 'primary'})

        out = _apportion(pairs, ref, priority=priority)

        # 3:1 overlap area -> 3:1 split, and the parcel's total is preserved
        assert out.loc['f1', 'land_value'] == pytest.approx(300_000.0)
        assert out.loc['f2', 'land_value'] == pytest.approx(100_000.0)
        assert out['land_value'].sum() == pytest.approx(400_000.0)

    def test_a_lone_primary_building_still_gets_the_whole_value(self):
        """The overwhelmingly common case must be unchanged by the split."""
        pairs = _pairs([('f1', 'p1', 250.0), ('f2', 'p1', 40.0)])
        ref = pd.DataFrame({'land_value': [400_000.0]}, index=['p1'])
        priority = pd.Series({'f1': 'primary', 'f2': 'secondary'})

        out = _apportion(pairs, ref, priority=priority)

        assert out.loc['f1', 'land_value'] == pytest.approx(400_000.0)
        assert pd.isna(out.loc['f2', 'land_value'])

    @pytest.mark.parametrize(
        'column', ['land_value', 'land_value_imputed', 'total_value']
    )
    def test_every_whole_value_column_is_split(self, column):
        pairs = _pairs([('f1', 'p1', 100.0), ('f2', 'p1', 100.0)])
        ref = pd.DataFrame({column: [900_000.0]}, index=['p1'])
        priority = pd.Series({'f1': 'primary', 'f2': 'primary'})

        out = _apportion(pairs, ref, priority=priority)

        assert out[column].sum() == pytest.approx(900_000.0)
        assert out.loc['f1', column] == pytest.approx(450_000.0)

    def test_zero_area_primaries_divide_equally(self):
        """A degenerate or point-derived geometry must not blank the value."""
        pairs = _pairs([('f1', 'p1', 0.0), ('f2', 'p1', 0.0)])
        ref = pd.DataFrame({'total_value': [100_000.0]}, index=['p1'])
        priority = pd.Series({'f1': 'primary', 'f2': 'primary'})

        out = _apportion(pairs, ref, priority=priority)

        assert out['total_value'].sum() == pytest.approx(100_000.0)
        assert out.loc['f1', 'total_value'] == pytest.approx(50_000.0)

    def test_each_parcel_is_split_independently(self):
        pairs = _pairs(
            [
                ('f1', 'p1', 100.0),
                ('f2', 'p1', 100.0),
                ('f3', 'p2', 500.0),
            ]
        )
        ref = pd.DataFrame({'land_value': [200_000.0, 750_000.0]}, index=['p1', 'p2'])
        priority = pd.Series({'f1': 'primary', 'f2': 'primary', 'f3': 'primary'})

        out = _apportion(pairs, ref, priority=priority)

        assert out.loc['f1', 'land_value'] == pytest.approx(100_000.0)
        assert out.loc['f3', 'land_value'] == pytest.approx(750_000.0)
        assert out['land_value'].sum() == pytest.approx(950_000.0)

    def test_a_suppressed_building_does_not_take_a_share(self):
        """Dwelling-linked suppression removes an entity from the split
        entirely; its share goes to the remaining principals, not nowhere."""
        pairs = _pairs([('f1', 'p1', 100.0), ('f2', 'p1', 100.0)])
        ref = pd.DataFrame({'land_value': [300_000.0]}, index=['p1'])
        priority = pd.Series({'f1': 'primary', 'f2': 'primary'})

        out = _apportion(pairs, ref, priority=priority, dwelling_linked_ids={'f1'})

        assert out.loc['f1', 'land_value'] == pytest.approx(300_000.0)
        assert pd.isna(out.loc['f2', 'land_value'])
        assert out['land_value'].sum() == pytest.approx(300_000.0)

    def test_no_priority_falls_back_to_sole_entity_rule(self):
        """Without a priority series, only a lone entity takes the value --
        the pre-existing safe branch, which must keep working."""
        pairs = _pairs([('f1', 'p1', 100.0), ('f2', 'p1', 100.0), ('f3', 'p2', 100.0)])
        ref = pd.DataFrame({'land_value': [500_000.0, 90_000.0]}, index=['p1', 'p2'])

        out = _apportion(pairs, ref)

        assert pd.isna(out.loc['f1', 'land_value'])
        assert out.loc['f3', 'land_value'] == pytest.approx(90_000.0)


class TestTheGate:
    def test_over_allocation_raises_naming_the_reference(self, monkeypatch):
        """Simulate the old duplicating behavior and confirm the gate stops
        it: the invariant has to be enforced, not merely implemented."""
        from openplaces.io.harmonizer import apportion as mod

        pairs = _pairs([('f1', 'p1', 100.0), ('f2', 'p1', 100.0)])
        ref = pd.DataFrame({'land_value': [400_000.0]}, index=['p1'])

        result = pd.DataFrame(
            {'land_value': [400_000.0, 400_000.0]},
            index=pd.Index(['f1', 'f2'], name='footprint_id'),
        )
        dominant_ref = pd.Series(['p1', 'p1'], index=['f1', 'f2'], name='parcel_id')

        with pytest.raises(ValueOverAllocationError, match="'land_value'"):
            mod._assert_not_over_allocated(
                result,
                pairs=pairs,
                ref_values=ref,
                value_cols=['land_value'],
                spine_id_col='footprint_id',
                ref_id_col='parcel_id',
                dominant_ref=dominant_ref,
            )

    def test_under_allocation_is_allowed(self):
        """A parcel whose buildings are all secondary keeps its value
        unassigned; that is correct, not a violation."""
        pairs = _pairs([('f1', 'p1', 100.0)])
        ref = pd.DataFrame({'land_value': [400_000.0]}, index=['p1'])
        priority = pd.Series({'f1': 'secondary'})

        out = _apportion(pairs, ref, priority=priority)

        assert pd.isna(out.loc['f1', 'land_value'])

    def test_rounding_noise_does_not_trip_the_gate(self):
        """Many small shares of an odd amount must not accumulate into a
        false positive."""
        n = 97
        pairs = _pairs([(f'f{i}', 'p1', 1.0) for i in range(n)])
        ref = pd.DataFrame({'total_value': [1_000_000.01]}, index=['p1'])
        priority = pd.Series({f'f{i}': 'primary' for i in range(n)})

        out = _apportion(pairs, ref, priority=priority)

        assert out['total_value'].sum() == pytest.approx(1_000_000.01)

    def test_proportional_columns_stay_within_their_source(self):
        pairs = _pairs([('f1', 'p1', 300.0), ('f2', 'p1', 100.0)])
        ref = pd.DataFrame({'improvement_value': [800_000.0]}, index=['p1'])
        priority = pd.Series({'f1': 'primary', 'f2': 'primary'})

        out = _apportion(pairs, ref, priority=priority)

        assert out['improvement_value'].sum() <= 800_000.0 + 0.01
        assert out.loc['f1', 'improvement_value'] == pytest.approx(600_000.0)


def test_the_shipped_failure_shape_is_now_correct():
    """The exact configuration measured in Kleberg and Brunswick: one
    high-value parcel carrying several principal structures alongside many
    ordinary single-building parcels. The regional sum must equal the
    parcels' own total, not a multiple of it."""
    rows = [(f'big_f{i}', 'big_p', 100.0) for i in range(4)]
    rows += [(f'f{i}', f'p{i}', 100.0) for i in range(20)]
    pairs = _pairs(rows)

    values = {'big_p': 10_000_000.0}
    values.update({f'p{i}': 200_000.0 for i in range(20)})
    ref = pd.DataFrame({'total_value': pd.Series(values)})
    priority = pd.Series(
        {sid: 'primary' for sid, _, _ in [(r[0], r[1], r[2]) for r in rows]}
    )

    out = _apportion(pairs, ref, priority=priority)

    expected = 10_000_000.0 + 20 * 200_000.0
    assert out['total_value'].sum() == pytest.approx(expected)
    # the four structures on the big parcel share it, one quarter each
    assert out.loc['big_f0', 'total_value'] == pytest.approx(2_500_000.0)
    assert not np.isclose(out['total_value'].sum(), expected + 3 * 10_000_000.0)
