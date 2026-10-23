"""Maryland's placeholder accounts leave the curated parcels, nothing else.

The state patch (US-MD_parcel-openplaces-2026) extends the national
`exclude_by_value` list. Its all-digit values have to stay strings: an
unquoted YAML number would never match the string `parcel_id_assessor`.
Fabricated rows throughout; the placeholder values are SDAT's sentinels.
"""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.filters import exclude_by_value
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id

PATCH = 'US-MD_parcel-openplaces-2026'
PLACEHOLDERS = ['17019999999', '17019999998', 'GCE', 'PATH']


def _exclude_step(admin_id):
    national = get_recipe_by_id('US_parcel-openplaces-2026')
    patched, applied = apply_recipe_patches(national, admin_id)
    steps = [s for s in patched['pipeline'] if s['step'] == 'exclude_by_value']
    assert len(steps) == 1
    return steps[0], applied


def test_the_state_patch_reaches_every_md_county():
    for admin_id in ('US-MD-PG', 'US-MD-MO'):
        step, applied = _exclude_step(admin_id)
        assert PATCH in applied
        assert step['column'] == 'parcel_id_assessor'
        assert all(isinstance(v, str) for v in step['values'])
        assert set(PLACEHOLDERS) <= set(step['values'])
        # The national values stay.
        assert {'WATER', 'ROW'} <= set(step['values'])


def test_the_patch_stays_out_of_other_states():
    step, applied = _exclude_step('US-VA-FX')
    assert PATCH not in applied
    assert not set(PLACEHOLDERS) & set(step['values'])


def test_placeholders_drop_and_real_accounts_stay():
    step, _ = _exclude_step('US-MD-PG')
    frame = pd.DataFrame(
        {
            'parcel_id_assessor': pd.Series(
                ['17019999999', '17019999998', '17020000001', 'GCE', '17020000002'],
                dtype='string',
            ),
            'land_value': [None, None, 1000.0, None, 2000.0],
        }
    )
    state = CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-MD-PG'),
        verbose=False,
        timer=None,
        curated=frame,
    )
    out = exclude_by_value(state, column=step['column'], values=step['values'])
    assert out.curated['parcel_id_assessor'].tolist() == ['17020000001', '17020000002']
