"""Shared occupancy-classification helpers for the curate stage.

Vocabulary-neutral: all class names, evidence columns, thresholds, and the
raw-label -> class mapping come from the recipe ``occupancy`` config block and the
class-map ruleset CSV, never from this module. Used by the inferer, reconciler,
and diagnostics steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

# The ruleset reader and matcher live in core so a harmonize step can
# apply the same files; the curate stage keeps reaching them here.
from openplaces.core.rulesets import match_ruleset, read_ruleset  # noqa: F401

if TYPE_CHECKING:
    from openplaces.io.curator import CurateState


def get_occupancy_config(state: CurateState) -> dict:
    """Return the recipe's ``occupancy`` config block (or an empty dict)."""
    return state.recipe.get('occupancy') or {}


def load_ruleset(
    state: CurateState, ruleset: str, class_column: str | None = None
) -> list[dict]:
    """Load an ordered label-mapping ruleset CSV stored beside the curate recipe.

    Columns: ``pattern``, ``match_type`` (contains | regex; default contains),
    the output class column, and an optional ``reviewed`` (true | false;
    default false). Rows apply top to bottom, first match wins, so more
    specific terms must precede general ones. Matching is case-insensitive.

    Rules are returned with the output class under the key ``occupancy_type``
    whatever the CSV calls it, so every caller reads one shape.

    Parameters
    ----------
    ruleset : str
        Filename (or recipe-id stem) of the CSV beside the curate recipe; a
        ``.csv`` extension is added when missing.
    class_column : str, optional
        CSV column holding the output class. Defaults to whichever of
        ``occupancy_type`` or ``land_use_class`` the file actually has, so a
        ruleset naming a non-occupancy vocabulary need not mislabel its
        column.
    """
    from openplaces.path import recipe_path

    recipe = state.recipe
    filename = ruleset if ruleset.endswith('.csv') else f'{ruleset}.csv'
    csv_path = recipe_path(
        recipe['admin_id'],
        recipe.get('entity') or recipe.get('dataset'),
        filename=filename,
    )
    return read_ruleset(csv_path, class_column)


def coerce_to_class(series: pd.Series, rules: list[dict]) -> pd.Series:
    """Map raw occupancy labels to coarse classes via *rules*; unmatched pass through.

    First matching rule wins (case-insensitive contains/regex). Values matching no
    rule are returned unchanged (e.g. non-residential categories); missing values
    stay missing.
    """
    from openplaces.io.curator.indicators import as_matchable_text

    terms = as_matchable_text(series)
    result = series.astype(object).copy()
    unmatched = pd.Series(True, index=series.index)
    for rule in rules:
        mask = unmatched & terms.str.contains(
            rule['pattern'],
            case=False,
            na=False,
            regex=rule['match_type'] == 'regex',
        ).astype(bool)
        if mask.any():
            result.loc[mask] = rule['occupancy_type']
            unmatched.loc[mask] = False
    return result.where(series.notna())


def bucket_classes(coerced: pd.Series, config: dict) -> pd.Series:
    """Collapse already-coerced occupancy classes to residential granularity.

    Every class outside ``residential_classes`` plus the ``secondary_class`` is
    replaced with a single ``conflict_other_label`` (default ``Non-Residential``),
    so two differing non-residential categories (e.g. Retail vs Hotel) do not
    register as a disagreement while residential subtypes (Single-Family vs
    Multi-Family) do. Missing values stay missing.
    """
    keep = set(config.get('residential_classes', []))
    secondary = config.get('secondary_class')
    if secondary is not None:
        keep.add(secondary)
    other_label = config.get('conflict_other_label', 'Non-Residential')
    return coerced.where(coerced.isna() | coerced.isin(keep), other_label)


def bucket_to_residential(
    series: pd.Series, config: dict, rules: list[dict]
) -> pd.Series:
    """Coerce *series* to occupancy classes, collapsing non-residential to a bucket.

    Used to compare occupancy evidence sources at residential granularity; see
    :func:`bucket_classes` for the bucketing semantics.
    """
    return bucket_classes(coerce_to_class(series, rules), config)
