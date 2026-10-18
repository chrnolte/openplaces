"""The Fathom-free placeslab predictions reach team bundles only.

`US_parcel-placeslab-fmv2026open` is restricted and cleared for team
sharing; `US_parcel-placeslab-fmv2026`, which holds the families with
Fathom and TNC inputs, is restricted and not cleared. Checked on the
real recipes: a curate step merging the open evidence must carry the
team clearance down the dependency walk, and the merged column must
still be withheld wherever the bundle is not a team bundle (the team
side of `export_delivery` is pinned in test_delivery_restriction_gate).
"""

import copy

import pandas as pd

from openplaces.io.delivery import delivery_regions
from openplaces.io.delivery.redaction import find_restricted, withhold
from openplaces.io.delivery.terms import restricted_inputs
from openplaces.recipe import get_recipe_by_id

OPEN = 'US_parcel-placeslab-fmv2026open'
RESTRICTED = 'US_parcel-placeslab-fmv2026'
COLUMN = 'land_value_per_ha_log_region_nb_bare'


def _source(recipe_id):
    return get_recipe_by_id(recipe_id)['entity'].source


def test_the_open_source_is_restricted_and_cleared_for_the_team():
    source = _source(OPEN)
    assert source.redistribution_restricted is True
    assert source.team_sharing_permitted is True


def test_the_fathom_families_are_not_cleared_for_the_team():
    source = _source(RESTRICTED)
    assert source.redistribution_restricted is True
    assert not source.team_sharing_permitted


def _curate_merging_open_evidence():
    recipe = copy.deepcopy(get_recipe_by_id('US_parcel-openplaces-2026'))
    recipe['pipeline'] = [
        {
            'step': 'merge_enrichments',
            'recipes': [
                {
                    'recipe_id': 'US_parcel_parcel-placeslab-fmv2026open',
                    'columns': {f'{COLUMN}_2010_fmv2026open': COLUMN},
                }
            ],
        }
    ]
    return recipe


def test_merged_open_columns_are_withheld_unless_the_bundle_is_a_team_bundle():
    entries = [
        entry
        for entry in restricted_inputs(_curate_merging_open_evidence())
        if entry['recipe_id'] == OPEN
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry['reasons'] == ['redistribution restricted']
    assert entry['team_sharing_permitted'] is True
    # Reached through named columns, so only those are its values.
    assert entry['layer_source'] is False
    assert entry['attributes'] == [COLUMN]

    # Fabricated values in two Massachusetts towns.
    frame = pd.DataFrame(
        {'admin3_id': ['US-MA-NTC', 'US-MA-BOS'], COLUMN: [1.0, 2.0]},
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    found = find_restricted(frame, [entry], 'admin3_id')
    assert found['cells'][COLUMN].tolist() == [True, True]


# The Nantucket patch (US-MA-NTC_parcel-openplaces-2026) merges the
# open evidence into the national parcel curate for that unit only.
UNIT = 'US-MA-NTC'
PARCEL = 'US_parcel-openplaces-2026'
AOA = 'area_of_applicability_region_nb_bare'
FAMILY = 'land_value_per_ha_log_region_nb'


def _bundle(audience):
    """Fabricated Nantucket rows as the parcel bundle redacts them."""
    entries = restricted_inputs(get_recipe_by_id(PARCEL), [UNIT])
    if audience == 'team':
        entries = [e for e in entries if not e.get('team_sharing_permitted')]
    frame = pd.DataFrame(
        {
            'admin3_id': [UNIT, UNIT],
            COLUMN: [1.0, 2.0],
            AOA: [0.5, 1.5],
            FAMILY: [3.0, 4.0],
            'land_use_class': ['Vacant', 'Vacant'],
            'land_use_class_source': ['parcel', 'parcel'],
        },
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    redacted, _ = withhold(frame, entries, 'admin3_id')
    return redacted


def test_a_patch_merging_open_evidence_is_read_by_its_columns():
    entries = {
        e['recipe_id']: e for e in restricted_inputs(get_recipe_by_id(PARCEL), [UNIT])
    }
    entry = entries[OPEN]
    # Read from the patched pipeline: a layer entry mapping nothing let
    # these columns through and emptied land_use_class instead.
    assert entry['layer_source'] is False
    assert COLUMN in entry['attributes']
    assert AOA in entry['attributes']
    assert len(entry['attributes']) == 14
    assert not entries[RESTRICTED]['team_sharing_permitted']


def test_the_public_bundle_withholds_both_placeslab_sources():
    redacted = _bundle('public')
    assert redacted[COLUMN].isna().all()
    assert redacted[AOA].isna().all()
    assert redacted[FAMILY].isna().all()
    assert redacted['land_use_class'].notna().all()


def test_the_team_bundle_keeps_only_the_open_source():
    redacted = _bundle('team')
    assert redacted[COLUMN].notna().all()
    assert redacted[AOA].notna().all()
    assert redacted[FAMILY].isna().all()
    assert redacted['land_use_class'].notna().all()


def test_nantucket_parcels_ship_a_team_bundle():
    regions = {r['region_id']: r for r in delivery_regions(PARCEL)}
    assert regions['nantucket-ma.team']['audience'] == 'team'
    assert regions['nantucket-ma']['audience'] == 'public'
