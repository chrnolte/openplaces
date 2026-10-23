"""Sarpy County NE: the county curate patch grades the qualified flag."""

from types import SimpleNamespace

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.curator.inferers import derive_indicators
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _grade(admin, rows):
    recipe, _ = apply_recipe_patches(
        get_recipe_by_id('US_transaction-openplaces-2026'), AdminId(admin)
    )
    state = SimpleNamespace(curated=pd.DataFrame(rows), verbose=False, recipe={})
    for step in recipe['pipeline']:
        if step['step'] == 'derive_indicators':
            state.curated = derive_indicators(state, step['indicators']).curated
    return state.curated


def test_sarpy_grades_the_qualified_flag_and_leaves_an_empty_one_missing():
    out = _grade('US-NE-SA', {'sale_qualification_code': ['Yes', 'No', None]})
    confidence = out['sale_arms_length_confidence'].tolist()
    assert confidence[:2] == [1.0, 0.0]
    assert pd.isna(confidence[2])
    group = out['sale_qualification_group'].tolist()
    assert group[:2] == ['qualified', 'disqualified']
    assert pd.isna(group[2])


def test_the_sarpy_flag_is_not_graded_outside_the_county():
    out = _grade('US-NE-DO', {'sale_qualification_code': ['Yes', 'No']})
    if 'sale_arms_length_confidence' in out:
        assert out['sale_arms_length_confidence'].isna().all()
