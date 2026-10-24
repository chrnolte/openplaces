"""Massachusetts LA3 sales: the US-MA curate patch grades the NAL code."""

from types import SimpleNamespace

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator.inferers import derive_indicators
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _grading_steps(admin):
    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-openplaces-2026'), AdminId(admin)
    )
    return [s for s in recipe['pipeline'] if s['step'] == 'derive_indicators']


def _grade(rows):
    state = SimpleNamespace(curated=pd.DataFrame(rows), verbose=False, recipe={})
    # The national step, then the state's, as the pipeline runs them.
    for step in _grading_steps('US-MA-SOM')[:2]:
        state = derive_indicators(state, step['indicators'])
    return state.curated


def test_massachusetts_grades_after_the_national_step_only_in_massachusetts():
    ma = _grading_steps('US-MA-SOM')
    assert [i['output'] for i in ma[0]['indicators']] == ['sale_record_kind']
    assert ma[1]['indicators'][0]['output'] == '_nal_letter'
    fl = [i['output'] for s in _grading_steps('US-FL-LA') for i in s['indicators']]
    assert '_nal_letter' not in fl


def test_a_blank_code_is_a_valid_sale_and_an_unknown_code_is_ungraded():
    out = _grade(
        {
            'source': ['mador'] * 6,
            'sale_record_kind': ['deed'] * 6,
            'sale_qualification_code': [None, 'A', 'v', 'N7', 'X', 'G,R'],
        }
    )
    confidence = out['sale_arms_length_confidence'].tolist()
    assert confidence[:4] == [1.0, 0.0, 0.7, 0.0]
    assert pd.isna(confidence[4]) and pd.isna(confidence[5])
    group = out['sale_qualification_group'].tolist()
    assert group[:4] == [
        'qualified',
        'related_parties_family',
        'qualified_multi_parcel',
        'disqualified',
    ]
    assert pd.isna(group[4]) and pd.isna(group[5])


def test_a_massgis_last_sale_without_a_code_is_not_read_as_valid():
    out = _grade(
        {
            'source': ['massgis', 'mador'],
            'sale_record_kind': ['assessor_last_sale', 'deed'],
            'sale_qualification_code': [None, None],
        }
    )
    assert pd.isna(out['sale_arms_length_confidence'].iloc[0])
    assert pd.isna(out['sale_qualification_group'].iloc[0])
    assert out['sale_arms_length_confidence'].iloc[1] == 1.0


def test_codes_describing_the_buyer_or_exposure_get_no_group():
    out = _grade(
        {
            'source': ['mador'] * 3,
            'sale_record_kind': ['deed'] * 3,
            'sale_qualification_code': ['R', 'T', 'U'],
        }
    )
    assert out['sale_arms_length_confidence'].tolist() == [0.7, 0.7, 0.7]
    assert out['sale_qualification_group'].isna().all()
