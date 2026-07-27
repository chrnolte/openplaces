"""Tests for the harmonize-stage `reconcile_addresses` wrapper.

Mirrors a representative subset of `tests/io/curator/test_reconcile_addresses.py`
(the curate-stage wrapper's tests) against `HarmonizeState`, to confirm the
two thin wrappers around the shared
`openplaces.io.harmonizer.addresses.reconcile_addresses_df` core behave
identically -- plus a harmonize-specific `state.spine is None` no-op case
with no curate-stage equivalent.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.addresses import reconcile_addresses


def make_state(spine) -> HarmonizeState:
    return HarmonizeState(
        recipe={},
        admin_id='US-MA-MI',
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_reconcile_addresses_harmonize_step():
    spine = pd.DataFrame(
        {
            'address_parcel': [
                '123 MAIN ST, APT 4B',
                '456 BROADWAY AVE',
                None,
                '789 ELM ST',
            ],
            'address_street_overture': ['Main St', None, 'Oak Rd', 'Pine St'],
            # 999 doesn't match 789 in the last row
            'address_number_overture': ['123', None, '100', '999'],
        }
    )
    state = reconcile_addresses(
        make_state(spine),
        sources={
            'parcel': {'address_full': 'address_parcel'},
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
            },
        },
    )
    res = state.spine

    # Row 0: both agree -> reconciled, parcel base keeps its unit
    assert res.loc[0, 'address'] == '123 Main St Apt 4B'
    assert res.loc[0, 'address_source'] == 'reconciled'

    # Row 1: only parcel present
    assert res.loc[1, 'address'] == '456 Broadway Ave'
    assert res.loc[1, 'address_source'] == 'parcel'

    # Row 2: only the dwelling point present
    assert res.loc[2, 'address'] == '100 Oak Rd'
    assert res.loc[2, 'address_source'] == 'dwelling_overture'

    # Row 3: house numbers disagree -> higher-priority parcel wins alone,
    # and the disagreement is summarized in address_conflict
    assert res.loc[3, 'address'] == '789 Elm St'
    assert res.loc[3, 'address_source'] == 'parcel'
    assert (
        res.loc[3, 'address_conflict']
        == 'parcel: 789 ELM ST | dwelling_overture: 999 PINE ST'
    )
    assert res.loc[[0, 1, 2], 'address_conflict'].isna().all()


def test_reconcile_addresses_writes_street_output_col():
    spine = pd.DataFrame(
        {'address_parcel': ['9-11 PEARSON AVE UNIT 1', '7 PEARSON AVE']}
    )
    state = reconcile_addresses(
        make_state(spine),
        sources={'parcel': {'address_full': 'address_parcel'}},
    )
    res = state.spine
    # Case-formatted, like `address`, not the internal matching-only
    # uppercase representation.
    assert res.loc[0, 'address_street'] == 'Pearson Ave'
    assert res.loc[1, 'address_street'] == 'Pearson Ave'


def test_reconcile_addresses_complete_from_admin():
    spine = pd.DataFrame(
        {
            'address_street_overture': ['David Dr'],
            'address_number_overture': ['502'],
            'city_overture': ['EMERALD ISLE'],
        }
    )
    state = reconcile_addresses(
        make_state(spine),  # admin_id US-MA-MI -> state MA
        sources={
            'dwelling_overture': {
                'address_street': 'address_street_overture',
                'address_number': 'address_number_overture',
                'city': 'city_overture',
            },
        },
        complete_from_admin={'state': 2},
    )
    # US-MA-MI's level-2 code is 'MA'; only rows with another non-street
    # component (here, the parsed city) get completed.
    assert state.spine.loc[0, 'address'] == '502 David Dr, Emerald Isle, MA'


def test_reconcile_addresses_no_op_when_spine_is_none():
    state = reconcile_addresses(
        make_state(None),
        sources={'parcel': {'address_full': 'address_parcel'}},
    )
    assert state.spine is None


def test_reconcile_addresses_missing_source_columns_is_a_no_op():
    spine = pd.DataFrame({'other_column': [1, 2]})
    state = reconcile_addresses(
        make_state(spine),
        sources={'parcel': {'address_full': 'address_parcel'}},
    )
    assert 'address' not in state.spine.columns
