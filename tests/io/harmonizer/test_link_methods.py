"""Recording which rule linked a row.

Every key here is fabricated.
"""

import pandas as pd
import pytest

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import link_methods as lm

PARCELS = pd.DataFrame(
    {
        'parcel_id_local': ['p1', 'p2', None],
        'address_id_local': ['a1', 'a2', 'a3'],
    }
)


def _state(spine):
    state = HarmonizeState.__new__(HarmonizeState)
    state.spine = spine
    state.admin_id = 'XX-YY-ZZ'
    state.metadata = {}
    state.timer = None
    state.verbose = False
    return state


@pytest.fixture(autouse=True)
def _reference(monkeypatch):
    monkeypatch.setattr(lm, 'get_entities', lambda *args, **kwargs: PARCELS.copy())


def _both_passes(spine):
    state = lm.record_link_method(
        _state(spine),
        label='parcel_id_local',
        spine_key='parcel_id_local',
        recipe_id='ref',
    )
    return lm.record_link_method(
        state,
        label='address_id_local',
        spine_key='address_id_local',
        recipe_id='ref',
    ).spine


def test_the_first_rule_to_reach_a_row_names_it_and_a_later_one_never_renames():
    spine = pd.DataFrame(
        {
            # Number and address both match; number only; address only
            # (number unknown to the reference); address only (no
            # number); neither.
            'parcel_id_local': ['p1', 'p2', 'p9', None, 'p9'],
            'address_id_local': ['a1', None, 'a2', 'a3', 'a9'],
        }
    )
    out = _both_passes(spine)['parcel_link_method']
    assert out.tolist()[:4] == [
        'parcel_id_local',
        'parcel_id_local',
        'address_id_local',
        'address_id_local',
    ]
    assert pd.isna(out[4])


def test_a_missing_key_never_matches_a_missing_reference_key():
    spine = pd.DataFrame({'parcel_id_local': [None, pd.NA]})
    out = lm.record_link_method(
        _state(spine),
        label='parcel_id_local',
        spine_key='parcel_id_local',
        recipe_id='ref',
    ).spine
    assert out['parcel_link_method'].isna().all()


def test_a_spine_without_the_key_is_left_alone_and_a_pass_must_name_its_source():
    spine = pd.DataFrame({'other': ['x']})
    out = lm.record_link_method(
        _state(spine), label='k', spine_key='parcel_id_local', recipe_id='ref'
    ).spine
    assert 'parcel_link_method' not in out.columns
    with pytest.raises(ValueError):
        lm.record_link_method(
            _state(pd.DataFrame({'parcel_id_local': ['p1']})),
            label='k',
            spine_key='parcel_id_local',
        )
