"""Regressions for the parcel land-use vote, run against the shipped recipe.

These load `US_parcel-openplaces-2026`'s own `derive_indicators` and
`resolve_by_vote` blocks rather than hand-copied fixtures, so a future edit to
the recipe that reintroduces one of these bugs fails here.

Each case covers a decision that used to be resolved by rule-list position
rather than by evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_indicators
from openplaces.io.curator.reconcilers import resolve_by_vote
from openplaces.recipe import get_recipe_by_id

RECIPE_ID = 'US_parcel-openplaces-2026'


@pytest.fixture(scope='module')
def recipe():
    return get_recipe_by_id(RECIPE_ID)


def _step(recipe, name):
    return next(s for s in recipe['pipeline'] if s['step'] == name)


def _classify(recipe, rows: list[dict]) -> pd.Series:
    """Run the recipe's real derive + vote blocks over synthetic parcels."""
    # Columns every decision may reference; absent ones would silently cast no
    # vote, which would make a test pass for the wrong reason.
    defaults = {
        'use_group_combined': None,
        'group_parcel': None,
        'group_footprint_fema': None,
        'n_small_elongated_footprints_per_parcel': 0,
        'n_footprints_per_parcel': 0,
        'n_primary_footprints_per_parcel': 0,
        'max_parcels_per_footprint': 0,
        'max_footprint_area_m2': 100.0,
        'footprint_area_m2_in_parcel': 100.0,
        'footprint_area_m2_primary': 100.0,
        'area_ha': 1.0,
        'land_value': 50_000.0,
        'improvement_value': 150_000.0,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])

    state = CurateState(
        recipe=recipe,
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=frame,
    )
    derive = _step(recipe, 'derive_indicators')
    state = derive_indicators(state, derive['indicators'])

    vote = _step(recipe, 'resolve_by_vote')
    state = resolve_by_vote(
        state,
        target=vote['target'],
        decisions=vote['decisions'],
        preserve_base=vote.get('preserve_base', True),
        default_source=vote.get('default_source', 'vote'),
        flag_column=vote.get('flag_column'),
        flag_class=vote.get('flag_class'),
    )
    return state.curated['land_use_class'].astype(object)


def test_manufactured_home_without_park_scale_is_not_a_park(recipe):
    # Keyword + both group sources say manufactured home, but there is no
    # park-scale footprint evidence at all. This used to tie Manufactured
    # Home Park against Manufactured Home at score 3 and resolve to the park
    # purely because that decision is listed first.
    result = _classify(
        recipe,
        [
            {
                'use_group_combined': 'MANUFACTURED HOME',
                'group_parcel': 'Manufactured Home',
                'group_footprint_fema': 'Manufactured Home',
                'n_small_elongated_footprints_per_parcel': 1,
            }
        ],
    )
    assert result.iloc[0] == 'Manufactured Home'


def test_park_scale_footprint_count_still_yields_a_park(recipe):
    # The mirror case: with genuine park-scale evidence the park decision
    # must still win, so the fix gates on evidence rather than disabling it.
    result = _classify(
        recipe,
        [
            {
                'use_group_combined': 'MOBILE HOME PARK',
                'group_parcel': 'Manufactured Home',
                'group_footprint_fema': 'Manufactured Home',
                'n_small_elongated_footprints_per_parcel': 12,
            }
        ],
    )
    assert result.iloc[0] == 'Manufactured Home Park'


def test_bank_parcel_resolves_to_office_on_evidence_not_order(recipe):
    # 'BANK' matched Office's keyword while group_parcel='Bank' also sat in
    # Commercial's in_set, so both scored 1 and Office won by position.
    # Office now claims both signals and Commercial claims neither.
    result = _classify(
        recipe,
        [{'use_group_combined': 'BANK BRANCH', 'group_parcel': 'Bank'}],
    )
    assert result.iloc[0] == 'Office'


def test_hotel_parcel_still_resolves_to_commercial(recipe):
    # Proves the Office fix narrowed Commercial rather than breaking it.
    result = _classify(
        recipe,
        [{'use_group_combined': 'COMMERCIAL LODGING', 'group_parcel': 'Hotel'}],
    )
    assert result.iloc[0] == 'Commercial'


def test_retail_parcel_resolves_to_retail_not_commercial(recipe):
    result = _classify(
        recipe,
        [{'use_group_combined': 'RETAIL STORE', 'group_parcel': 'Retail'}],
    )
    assert result.iloc[0] == 'Retail'


def test_wholesale_resolves_to_industrial_matching_the_class_map(recipe):
    # group_parcel='Wholesale' used to score toward Commercial via the rule
    # vote while the land-use class map independently mapped the same raw
    # value to Industrial -- two paths contradicting each other.
    result = _classify(recipe, [{'group_parcel': 'Wholesale'}])
    assert result.iloc[0] == 'Industrial'


def test_wholesale_agrees_with_the_land_use_class_map(recipe):
    """The rule vote and the crosswalk fallback must not disagree."""
    from openplaces.io.transform import get_crosswalk

    reconcile = _step(recipe, 'reconcile_land_use')
    crosswalk = get_crosswalk({'recipe_id': reconcile['class_map_id']})
    assert crosswalk['Wholesale'] == 'Industrial'
    assert _classify(recipe, [{'group_parcel': 'Wholesale'}]).iloc[0] == 'Industrial'


def test_keyword_class_assigns_exactly_one_land_use_class(recipe):
    # The structural half of the park fix: the park pattern precedes the bare
    # manufactured-home pattern, so first-match-wins makes the ambiguous
    # double match that caused the tie impossible to express.
    state = CurateState(
        recipe=recipe,
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=pd.DataFrame(
            {
                'use_group_combined': [
                    'MOBILE HOME PARK',
                    'MANUFACTURED HOME',
                    'VACANT LAND',
                    'SOMETHING UNMAPPED',
                ]
            }
        ),
    )
    derive = _step(recipe, 'derive_indicators')
    spec = next(i for i in derive['indicators'] if i['output'] == 'keyword_class')
    out = derive_indicators(state, [spec]).curated['keyword_class']

    assert out.iloc[0] == 'Manufactured Home Park'
    assert out.iloc[1] == 'Manufactured Home'
    assert out.iloc[2] == 'Vacant'
    assert pd.isna(out.iloc[3])


def test_service_station_is_not_swept_into_rv_park(recipe):
    # The old RV pattern was an unanchored 'RV', so any label containing
    # those two letters -- 'SERVICE' among them -- matched. The ruleset
    # anchors it to a word boundary.
    state = CurateState(
        recipe=recipe,
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=pd.DataFrame({'use_group_combined': ['SERVICE STATION', 'RV PARK']}),
    )
    derive = _step(recipe, 'derive_indicators')
    spec = next(i for i in derive['indicators'] if i['output'] == 'keyword_class')
    out = derive_indicators(state, [spec]).curated['keyword_class']

    # Matches nothing at all now, rather than matching the RV pattern.
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == 'RV Park'
