"""Fairfax County's assessed values reach team bundles only.

`US-VA-FA_property-fairfaxcounty-2026_values` is the only Fairfax table
carrying land, improvement and total values. Its terms grant internal
or personal reproduction only, and the maintainer cleared it for the
research team on 2026-10-20. Checked on the real recipes with
fabricated values: the parcel curate's restricted-input walk for
US-VA-FA must name the table with its value columns and the team
clearance, the public washington-dc bundle must empty those values, and
the team twin must keep them.
"""

import pandas as pd

from openplaces.io.delivery import delivery_regions
from openplaces.io.delivery.redaction import withhold
from openplaces.io.delivery.terms import restricted_inputs
from openplaces.recipe import get_recipe_by_id

VALUES = 'US-VA-FA_property-fairfaxcounty-2026_values'
ROLL = 'US-VA-FA_property-fairfaxcounty-2026'
PARCEL = 'US_parcel-openplaces-2026'
UNIT = 'US-VA-FA'
VALUE_COLUMNS = ['land_value', 'improvement_value', 'total_value']


def test_the_values_table_supplements_the_roll_on_its_parcel_key():
    recipe = get_recipe_by_id(VALUES)
    assert recipe['supplements'] == ROLL
    for column in VALUE_COLUMNS:
        assert column in recipe['columns']
    # Joined on parcel_id_local like the roll, with the roll's conv.
    assert 'supplements_key' not in recipe
    roll_conv = get_recipe_by_id(ROLL)['parcel_id_local']['instruction']
    assert recipe['parcel_id_local']['instruction'] == roll_conv


def test_the_values_table_is_restricted_and_cleared_for_the_team():
    source = get_recipe_by_id(VALUES)['entity'].source
    assert source.source_id == 'fairfaxcounty'
    assert source.redistribution_restricted is True
    assert source.team_sharing_permitted is True


def _entries():
    return {
        e['recipe_id']: e for e in restricted_inputs(get_recipe_by_id(PARCEL), [UNIT])
    }


def test_the_parcel_curate_names_the_values_table_with_its_columns():
    entry = _entries()[VALUES]
    assert entry['team_sharing_permitted'] is True
    for column in VALUE_COLUMNS:
        assert column in entry['attributes']


def _bundle(audience):
    """Fabricated Fairfax rows as the parcel bundle redacts them."""
    entries = list(_entries().values())
    if audience == 'team':
        entries = [e for e in entries if not e.get('team_sharing_permitted')]
    frame = pd.DataFrame(
        {
            'admin3_id': [UNIT, UNIT],
            'land_value': [100.0, 200.0],
            'land_value_source': ['fairfaxcounty', 'fairfaxcounty'],
            'improvement_value': [300.0, 400.0],
            'improvement_value_source': ['fairfaxcounty', 'fairfaxcounty'],
            'total_value': [400.0, 600.0],
            'total_value_source': ['fairfaxcounty', 'fairfaxcounty'],
        },
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    redacted, _ = withhold(frame, entries, 'admin3_id')
    return redacted


def test_the_public_bundle_withholds_fairfax_values():
    redacted = _bundle('public')
    for column in VALUE_COLUMNS:
        assert redacted[column].isna().all()
        assert (redacted[f'{column}_source'] == 'withheld').all()


def test_the_team_bundle_keeps_fairfax_values():
    redacted = _bundle('team')
    for column in VALUE_COLUMNS:
        assert redacted[column].notna().all()


def test_washington_dc_parcels_ship_a_team_bundle():
    regions = {r['region_id']: r for r in delivery_regions(PARCEL)}
    assert regions['washington-dc.team']['audience'] == 'team'
    assert regions['washington-dc']['audience'] == 'public'
