"""Vermont transfer tax returns: the US-VT curate patch grades them."""

from types import SimpleNamespace

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator.inferers import derive_indicators
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _grading_step(admin):
    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-openplaces-2026'), AdminId(admin)
    )
    return [s for s in recipe['pipeline'] if s['step'] == 'derive_indicators'][0]


def _grade(rows):
    state = SimpleNamespace(curated=pd.DataFrame(rows), verbose=False, recipe={})
    indicators = _grading_step('US-VT-RUT')['indicators']
    return derive_indicators(state, indicators).curated


def test_vermont_replaces_the_national_grading_step_only_in_vermont():
    vt = [i['output'] for i in _grading_step('US-VT-RUT')['indicators']]
    assert vt[0] == '_exemption_grade'
    assert 'sale_record_kind' in vt
    fl = [i['output'] for i in _grading_step('US-FL-LA')['indicators']]
    assert '_exemption_grade' not in fl


def test_vermont_codes_are_not_read_as_florida_codes():
    # '05' is a family transfer in Vermont and a qualified multi-parcel
    # sale in Florida's undecoded codes.
    out = _grade(
        {
            'sale_qualification_code': ['05'],
            'sale_party_relationship': ['02. Parent/Child'],
            'conveyance_type': ['Fee Simple'],
            'sale_record_kind': ['deed'],
        }
    )
    assert out['sale_arms_length_confidence'].iloc[0] == 0.0
    assert out['sale_qualification_group'].iloc[0] == 'related_parties_family'


def test_vermont_grades_are_the_lowest_of_three_answers():
    out = _grade(
        {
            'sale_qualification_code': ['00', '00', '00', '99', '00', '27'],
            'sale_party_relationship': [None, '05. Other', None, None, None, None],
            'conveyance_type': [
                'Fee Simple',
                'Fee Simple',
                'Timeshare',
                'Fee Simple',
                'Other',
                'Fee Simple',
            ],
            'sale_record_kind': ['deed'] * 6,
        }
    )
    confidence = out['sale_arms_length_confidence'].tolist()
    assert confidence[:5] == [1.0, 0.7, 0.0, 0.7, 0.7]
    # An exemption code the patch does not list stays missing.
    assert pd.isna(confidence[5])
    group = out['sale_qualification_group'].tolist()
    assert group[:3] == ['qualified', 'related_parties_unspecified', 'disqualified']
    assert pd.isna(group[3]) and pd.isna(group[4])
