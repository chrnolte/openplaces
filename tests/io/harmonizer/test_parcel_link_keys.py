"""The key a record finds its parcel by. Every key here is fabricated."""

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import link_methods as lm
from openplaces.io.harmonizer import parcel_link_keys as plk

PAIRS = pd.DataFrame(
    {
        # u1 and u2 stand on lot L1; u3 was seen on two lots.
        'unit_key': ['u1', 'u2', 'u3', 'u3'],
        'lot_key': ['L1', 'L1', 'L2', 'L3'],
    },
    dtype='string',
)


def _state(spine):
    state = HarmonizeState.__new__(HarmonizeState)
    state.spine = spine
    state.admin_id = 'XX-YY-ZZ'
    state.metadata = {}
    state.timer = None
    state.verbose = False
    return state


def test_a_unit_joins_on_its_lot_and_a_unit_on_two_lots_is_not_guessed(monkeypatch):
    monkeypatch.setattr(plk, 'load_unit_lot_pairs', lambda admin_id: PAIRS)
    spine = pd.DataFrame({'parcel_id_local': ['u1', 'p7', 'u3', None]})
    out = plk.derive_parcel_link_key(_state(spine)).spine
    assert out['lot_id_local'].tolist()[0] == 'L1'
    assert out['lot_id_local'][1:].isna().all()
    assert out['parcel_link_key'].tolist()[:3] == ['L1', 'p7', 'u3']
    assert pd.isna(out['parcel_link_key'][3])
    # The number the record states is never rewritten.
    assert out['parcel_id_local'].tolist()[:3] == ['u1', 'p7', 'u3']


def test_without_any_split_layer_every_row_keeps_its_own_number(monkeypatch):
    empty = pd.DataFrame({'unit_key': [], 'lot_key': []}, dtype='string')
    monkeypatch.setattr(plk, 'load_unit_lot_pairs', lambda admin_id: empty)
    spine = pd.DataFrame({'parcel_id_local': ['p1', 'p2']})
    out = plk.derive_parcel_link_key(_state(spine)).spine
    assert out['parcel_link_key'].tolist() == ['p1', 'p2']
    assert out['lot_id_local'].isna().all()


def test_a_sale_that_reached_its_parcel_through_a_lot_says_so(monkeypatch):
    parcels = pd.DataFrame({'parcel_id_local': ['L1', 'p7']})
    monkeypatch.setattr(lm, 'get_entities', lambda *a, **k: parcels.copy())
    spine = pd.DataFrame(
        {
            'parcel_link_key': ['L1', 'p7', 'p9'],
            'lot_id_local': ['L1', None, None],
        }
    )
    out = lm.record_link_method(
        _state(spine),
        label='parcel_id_local',
        spine_key='parcel_link_key',
        ref_key='parcel_id_local',
        recipe_id='ref',
        via_column='lot_id_local',
        via_label='stacked_units',
    ).spine['parcel_link_method']
    assert out.tolist()[:2] == ['stacked_units', 'parcel_id_local']
    assert pd.isna(out[2])
