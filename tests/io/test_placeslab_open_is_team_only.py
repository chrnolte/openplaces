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

from openplaces.io.delivery.redaction import find_restricted
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
