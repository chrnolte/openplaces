"""Arlington County's assessed values stay out of every bundle.

`US-VA-AR_property-arlington-2026_assessment` supplies Arlington's land,
improvement and total values. Its terms are unread, so it records
`redistribution_restricted: true` and no team clearance. Once the curated
parcels carry its land value, the footprints take their structure value
from it (`parcel.arlington`). Checked on the real recipes with fabricated
values: the footprint curate's restricted-input walk for US-VA-AR must
name the table, reaching footprints through the parcels, and both the
public washington-dc footprint bundle and its team twin must empty the
structure value it supplied, while NSI's value stays.
"""

import pandas as pd
import pytest

from openplaces.io.delivery import delivery_regions, share_derived_from
from openplaces.io.delivery.redaction import WITHHELD, withhold
from openplaces.io.delivery.terms import restricted_inputs
from openplaces.recipe import get_recipe_by_id

ASSESSMENT = 'US-VA-AR_property-arlington-2026_assessment'
FOOTPRINT = 'US_footprint-openplaces-2026'
UNIT = 'US-VA-AR'


def _entries():
    return {
        e['recipe_id']: e
        for e in restricted_inputs(get_recipe_by_id(FOOTPRINT), [UNIT])
    }


def test_the_assessment_table_is_restricted_and_not_cleared_for_the_team():
    source = get_recipe_by_id(ASSESSMENT)['entity'].source
    assert source.source_id == 'arlington'
    assert source.redistribution_restricted is True
    assert not source.team_sharing_permitted


def test_the_footprint_curate_reaches_the_table_through_the_parcels():
    entry = _entries()[ASSESSMENT]
    assert not entry.get('team_sharing_permitted')
    assert 'parcel' in entry['via_entities']


def _bundle(audience):
    """Fabricated Arlington footprints as a washington-dc bundle redacts them."""
    entries = list(_entries().values())
    if audience == 'team':
        entries = [e for e in entries if not e.get('team_sharing_permitted')]
    frame = pd.DataFrame(
        {
            'admin3_id': [UNIT, UNIT, UNIT],
            'structure_value': [300.0, 0.0, 250.0],
            'structure_value_source': ['parcel.arlington', 'parcel', 'nsi'],
            'structure_value_per_area': [3.0, 0.0, 2.5],
            'improvement_value_parcel': [300.0, 0.0, None],
            'improvement_value_parcel_source': [
                'parcel.arlington',
                'parcel.arlington',
                None,
            ],
        },
        index=pd.Index(['a', 'b', 'c'], name='footprint_id'),
    )
    redacted, _ = withhold(
        frame,
        entries,
        'admin3_id',
        derived_from=share_derived_from(get_recipe_by_id(FOOTPRINT)),
    )
    return redacted


@pytest.mark.parametrize('audience', ['public', 'team'])
def test_washington_dc_bundles_withhold_arlingtons_structure_value(audience):
    redacted = _bundle(audience)
    rolls = redacted.loc[['a', 'b']]
    assert rolls['structure_value'].isna().all()
    assert (rolls['structure_value_source'] == WITHHELD).all()
    assert rolls['structure_value_per_area'].isna().all()
    assert rolls['improvement_value_parcel'].isna().all()
    # NSI's value is not the roll's.
    assert redacted.loc['c', 'structure_value'] == 250.0


def test_washington_dc_footprints_ship_a_team_bundle():
    regions = {r['region_id']: r for r in delivery_regions(FOOTPRINT)}
    assert regions['washington-dc.team']['audience'] == 'team'
    assert regions['washington-dc']['audience'] == 'public'
