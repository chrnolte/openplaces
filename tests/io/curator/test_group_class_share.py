"""Neighborhood context as a groupby, not a geometric operation.

`flag_manufactured_home_communities` counts manufactured homes per *parcel*,
so it only ever sees the landlord-owned park where many homes share one
parcel. In a subdivided community each home has its own lot, every parcel
holds exactly one home, and the flag can never fire -- measured on one Harris
County block, 36 of 48 footprints were already classified Manufactured Home
while the flag was False for all of them. `derive_group_class_share` supplies
the missing context by grouping on an id the row already carries.

The step performs no geometric operation by design (see its Method note);
these tests pin that it needs nothing but an id column.

The step is landed as **evidence only** -- no recipe votes on it. A block
share computed from the assessor's own text correlates +0.577 with the
keyword rule that reads the same column, and the points it uniquely moves
are 0.294 precise against a 0.425 base rate, so wiring it into the occupancy
vote made the inventory worse. Kept because it is the mechanism
`notebooks/05_curate/mmh_separability.py` measures with, and because a
future *independent* neighborhood source would need exactly this shape.
"""

import pandas as pd
import pytest

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_group_class_share


def _state(frame):
    state = CurateState.__new__(CurateState)
    state.curated = frame
    state.verbose = False
    return state


def _share(frame, **kwargs):
    kwargs.setdefault('group_column', 'census_block_id')
    kwargs.setdefault('output', 'mh_share')
    kwargs.setdefault('evidence_columns', ['group_nsi'])
    kwargs.setdefault('match_values', ['Manufactured Home'])
    return derive_group_class_share(_state(frame), **kwargs).curated


class TestGroupClassShare:
    def test_a_subdivided_community_is_visible_where_a_parcel_count_is_not(self):
        """The reported failure, reduced: one home per parcel, so no parcel
        ever holds enough to trip a per-parcel count, while the block is
        overwhelmingly manufactured homes."""
        n = 10
        frame = pd.DataFrame(
            {
                'census_block_id': ['b1'] * n,
                'parcel_id': [f'p{i}' for i in range(n)],  # one home per parcel
                'group_nsi': ['Manufactured Home'] * 8 + ['Single Family'] * 2,
            }
        )
        out = _share(frame)

        # every parcel holds exactly one home: a per-parcel count sees nothing
        assert out.groupby('parcel_id').size().max() == 1
        # the misclassified row still sees a strongly manufactured-home block
        assert out.loc[9, 'mh_share'] == pytest.approx(8 / 9)

    def test_the_row_is_excluded_from_its_own_share(self):
        """Otherwise the signal partly restates the value it should correct."""
        frame = pd.DataFrame(
            {
                'census_block_id': ['b1'] * 4,
                'group_nsi': ['Manufactured Home'] * 3 + ['Single Family'],
            }
        )
        out = _share(frame)

        # a matching row sees 2 of the other 3; the non-matching row sees 3 of 3
        assert out.loc[0, 'mh_share'] == pytest.approx(2 / 3)
        assert out.loc[3, 'mh_share'] == pytest.approx(1.0)

    def test_self_inclusion_is_available_but_not_the_default(self):
        frame = pd.DataFrame(
            {
                'census_block_id': ['b1'] * 4,
                'group_nsi': ['Manufactured Home'] * 3 + ['Single Family'],
            }
        )
        out = _share(frame, exclude_self=False)
        assert out.loc[0, 'mh_share'] == pytest.approx(3 / 4)

    def test_groups_are_independent(self):
        frame = pd.DataFrame(
            {
                'census_block_id': ['b1', 'b1', 'b1', 'b2', 'b2', 'b2'],
                'group_nsi': ['Manufactured Home'] * 3 + ['Single Family'] * 3,
            }
        )
        out = _share(frame)
        assert out.loc[0, 'mh_share'] == pytest.approx(1.0)
        assert out.loc[3, 'mh_share'] == pytest.approx(0.0)

    def test_any_evidence_column_can_carry_the_class(self):
        frame = pd.DataFrame(
            {
                'census_block_id': ['b1'] * 3,
                'group_nsi': ['Single Family', None, 'Single Family'],
                'group_fema': [None, 'Manufactured Home', 'Manufactured Home'],
            }
        )
        out = _share(frame, evidence_columns=['group_nsi', 'group_fema'])
        # rows 1 and 2 match via FEMA; row 0 sees both of them
        assert out.loc[0, 'mh_share'] == pytest.approx(1.0)

    def test_a_group_too_small_to_mean_anything_gets_no_share(self):
        frame = pd.DataFrame(
            {
                'census_block_id': ['solo', 'b1', 'b1'],
                'group_nsi': ['Manufactured Home'] * 3,
            }
        )
        out = _share(frame)
        assert pd.isna(out.loc[0, 'mh_share'])
        assert out.loc[1, 'mh_share'] == pytest.approx(1.0)

    def test_rows_without_a_group_id_get_no_share(self):
        frame = pd.DataFrame(
            {
                'census_block_id': [None, 'b1', 'b1', 'b1'],
                'group_nsi': ['Manufactured Home'] * 4,
            }
        )
        out = _share(frame)
        assert pd.isna(out.loc[0, 'mh_share'])

    def test_an_absent_group_column_is_a_no_op(self):
        frame = pd.DataFrame({'group_nsi': ['Manufactured Home'] * 3})
        out = _share(frame, group_column='census_block_id')
        assert 'mh_share' not in out.columns

    def test_absent_evidence_columns_are_a_no_op(self):
        frame = pd.DataFrame({'census_block_id': ['b1'] * 3})
        out = _share(frame, evidence_columns=['group_nsi', 'group_fema'])
        assert 'mh_share' not in out.columns

    def test_the_raw_count_can_be_emitted_too(self):
        frame = pd.DataFrame(
            {
                'census_block_id': ['b1'] * 4,
                'group_nsi': ['Manufactured Home'] * 3 + ['Single Family'],
            }
        )
        out = _share(frame, count_output='n_mh_per_block')
        assert out.loc[3, 'n_mh_per_block'] == 3
        assert out.loc[0, 'n_mh_per_block'] == 2  # itself excluded

    def test_it_needs_no_geometry(self):
        """The step is a groupby on an id, deliberately not a spatial
        operation -- see its Method note. A frame with no geometry at all
        must work."""
        frame = pd.DataFrame(
            {'census_block_id': ['b1'] * 3, 'group_nsi': ['Manufactured Home'] * 3}
        )
        out = _share(frame)
        assert out['mh_share'].notna().all()
