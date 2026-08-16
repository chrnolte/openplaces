"""Shared occupancy-classification helpers for the curate stage.

Vocabulary-neutral: all class names, evidence columns, thresholds, and the
raw-label -> class mapping come from the recipe ``occupancy`` config block and the
class-map ruleset CSV, never from this module. Used by the inferer, reconciler,
and diagnostics steps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

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
    if not csv_path.exists():
        raise FileNotFoundError(f'Classification ruleset not found: {csv_path}')

    table = pd.read_csv(csv_path)
    if class_column is None:
        class_column = next(
            (c for c in ('occupancy_type', 'land_use_class') if c in table.columns),
            'occupancy_type',
        )
    if class_column not in table.columns:
        raise KeyError(
            f'Ruleset {csv_path.name} has no {class_column!r} column; '
            f'found {list(table.columns)}.'
        )

    rules = []
    for _, row in table.iterrows():
        rules.append(
            {
                'pattern': str(row['pattern']),
                'match_type': str(row.get('match_type', 'contains')).strip().lower(),
                'occupancy_type': str(row[class_column]),
                'reviewed': str(row.get('reviewed', '')).strip().lower()
                in ('true', '1', 'yes'),
            }
        )
    return rules


def coerce_to_class(series: pd.Series, rules: list[dict]) -> pd.Series:
    """Map raw occupancy labels to coarse classes via *rules*; unmatched pass through.

    First matching rule wins (case-insensitive contains/regex). Values matching no
    rule are returned unchanged (e.g. non-residential categories); missing values
    stay missing.
    """
    terms = series.astype(object)
    result = terms.copy()
    unmatched = pd.Series(True, index=series.index)
    for rule in rules:
        mask = unmatched & terms.str.contains(
            rule['pattern'],
            case=False,
            na=False,
            regex=rule['match_type'] == 'regex',
        )
        if mask.any():
            result.loc[mask] = rule['occupancy_type']
            unmatched.loc[mask] = False
    return result.where(series.notna())


def match_ruleset(terms: pd.Series, rules: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Apply an ordered label ruleset, returning (matched class, reviewed).

    First match wins, mirroring :func:`coerce_to_class` -- except a term
    matching no rule yields a missing class rather than passing through
    unchanged, because this answers "which class did the label assert?", not
    "normalize this label".
    """
    proposal = pd.Series(pd.NA, index=terms.index, dtype=object)
    reviewed = pd.Series(False, index=terms.index)
    unmatched = pd.Series(True, index=terms.index)
    for rule in rules:
        mask = unmatched & terms.str.contains(
            rule['pattern'],
            case=False,
            na=False,
            regex=rule['match_type'] == 'regex',
        )
        if mask.any():
            proposal.loc[mask] = rule['occupancy_type']
            reviewed.loc[mask] = rule['reviewed']
            unmatched.loc[mask] = False
    return proposal, reviewed


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
