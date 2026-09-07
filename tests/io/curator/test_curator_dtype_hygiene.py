"""Dtype and error-message hygiene across three curate modules.

Three unrelated defects with one shape: a value arrives in a dtype the step
did not expect, and the step fails (or silently corrupts) instead of
handling it. Each test names the shipping column or recipe that reaches it.
"""

import pandas as pd
import pytest

from openplaces.io.curator import CurateState
from openplaces.io.curator.formatters import cast_categoricals
from openplaces.io.curator.indicators import evaluate_indicator
from openplaces.io.curator.occupancy import coerce_to_class, match_ruleset


def _state(curated):
    return CurateState(
        recipe={'entity': {'entity_type': 'footprint'}},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=curated,
    )


# cast_categoricals


def test_confidence_columns_stay_numeric_and_comparable():
    """The shipping roof-shape confidence columns must not become Categorical.

    US_footprint-openplaces-2026 merges roof_shape_confidence_building_cheer
    onto roof_shape_confidence and then runs cast_categoricals. A bare
    startswith('roof_shape') match cast the float column to an unordered
    Categorical over its own values, and every later `> 0.6` comparison
    raised "Unordered Categoricals can only compare equality or not".
    """
    curated = pd.DataFrame(
        {
            'roof_shape': ['gable', 'hip'],
            'roof_shape_building_cheer': ['gable', 'flat'],
            'roof_shape_confidence': [0.91, 0.42],
            'roof_shape_confidence_building_cheer': [0.91, 0.42],
            'roof_shape_confidence_2': [0.88, 0.31],
        }
    )
    result = cast_categoricals(_state(curated)).curated

    assert result['roof_shape'].dtype == 'category'
    assert result['roof_shape_building_cheer'].dtype == 'category'
    for col in (
        'roof_shape_confidence',
        'roof_shape_confidence_building_cheer',
        'roof_shape_confidence_2',
    ):
        assert pd.api.types.is_numeric_dtype(result[col]), col
        assert (result[col] > 0.6).tolist() == [True, False]


def test_suffixed_and_unregistered_categoricals_are_still_cast():
    """The fix must not stop casting the columns the step exists for."""
    curated = pd.DataFrame(
        {
            'occupancy_type': ['Single-Family', 'Multi-Family'],
            'occupancy_type_dwelling_overture': ['Single-Family', None],
            # Not a registry name and not a provenance suffix; the
            # categorical-attribute prefix fallback still has to catch it.
            'occupancy_type_nsi_class': ['Single-Family', 'Multi-Family'],
            'occupancy_type_source': ['nsi', 'parcel+imputed'],
        }
    )
    result = cast_categoricals(_state(curated)).curated
    for col in curated.columns:
        assert result[col].dtype == 'category', col


# keyword matching on an int-typed code column


def test_keyword_indicator_matches_an_int_typed_code_column():
    """A numeric code column must match, not abort the county's curate."""
    curated = pd.DataFrame({'use_group_code': [92, 65, 100]})
    indicator = {'type': 'keyword', 'column': 'use_group_code', 'pattern': '^9'}
    matched = evaluate_indicator(curated, indicator)
    assert matched.tolist() == [True, False, False]


def test_keyword_indicator_drops_the_decimal_of_a_whole_float_code():
    curated = pd.DataFrame({'use_group_code': [92.0, 65.0, None]})
    indicator = {'type': 'keyword', 'column': 'use_group_code', 'pattern': '^92$'}
    assert evaluate_indicator(curated, indicator).tolist() == [True, False, False]


_RULES = [
    {
        'pattern': '92',
        'match_type': 'contains',
        'occupancy_type': 'Manufactured Home',
        'reviewed': True,
    }
]


def test_coerce_to_class_handles_an_int_typed_code_column():
    coerced = coerce_to_class(pd.Series([92, 65, None]), _RULES)
    assert coerced.tolist()[:2] == ['Manufactured Home', 65]
    assert pd.isna(coerced.iloc[2])


def test_match_ruleset_handles_an_int_typed_code_column():
    proposal, reviewed = match_ruleset(pd.Series([92, 65]), _RULES)
    assert proposal.tolist() == ['Manufactured Home', pd.NA]
    assert reviewed.tolist() == [True, False]


# a duplicate key in an overrides crosswalk


def test_duplicate_override_key_names_the_crosswalk_and_the_key(monkeypatch):
    """The raise must name the crosswalk and the repeated key.

    pandas' own InvalidIndexError from keys.map(corrections) names
    neither, leaving no way to find the offending row.
    """
    from openplaces.io import transform
    from openplaces.io.curator import imputers

    corrections = pd.Series(
        ['Alpha', 'Beta'],
        index=pd.Index([' shared ', 'shared'], name='group'),
        name='corrected',
    )
    monkeypatch.setattr(transform, 'get_crosswalk', lambda spec: corrections.copy())

    curated = pd.DataFrame({'group': ['shared', 'other'], 'value': ['Alpha', 'Beta']})
    with pytest.raises(ValueError, match=r"'a-crosswalk-recipe'.*'shared'"):
        imputers.impute_from_group_statistic(
            _state(curated),
            group_column='group',
            value_column='value',
            output='corrected',
            overrides='a-crosswalk-recipe',
        )
