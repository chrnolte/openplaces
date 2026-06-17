"""Registered curation steps that derive canonical values from evidence."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register


def _load_occupancy_ruleset(state: CurateState, ruleset: str) -> list[dict]:
    """Load the ordered parcel keyword ruleset CSV beside the curate recipe.

    Columns: ``pattern``, ``match_type`` (contains | regex), ``occupancy_type``,
    ``reviewed`` (true | false). Rows are applied top to bottom (first match
    wins), so more specific terms must precede general ones.
    """
    from openplaces.path import recipe_path

    recipe = state.recipe
    csv_path = recipe_path(
        recipe['admin_id'],
        recipe.get('entity') or recipe.get('dataset'),
        filename=ruleset,
    )
    if not csv_path.exists():
        raise FileNotFoundError(f'Occupancy ruleset not found: {csv_path}')

    table = pd.read_csv(csv_path)
    rules = []
    for _, row in table.iterrows():
        rules.append(
            {
                'pattern': str(row['pattern']),
                'match_type': str(row.get('match_type', 'contains')).strip().lower(),
                'occupancy_type': str(row['occupancy_type']),
                'reviewed': str(row['reviewed']).strip().lower()
                in ('true', '1', 'yes'),
            }
        )
    return rules


@_register('resolve_occupancy')
def resolve_occupancy(
    state: CurateState,
    ruleset: str,
    parcel_column: str = 'purpose_group_combined_parcel',
) -> CurateState:
    """Resolve occupancy from parcel land-use terms, with priority over NSI.

    Applies an NSI-independent keyword ruleset to the parcel
    ``purpose_group_combined`` string to propose an occupancy class, then
    reconciles it against the NSI-derived ``occupancy_type``:

    - Agreement, or no parcel proposal → keep the NSI value.
    - Disagreement, rule marked ``reviewed`` → override with the parcel value.
    - Disagreement, rule not reviewed → keep the NSI value but flag the row in
      ``occupancy_type_conflict`` so the proposal can be inspected before it is
      trusted. The proposal is always retained in ``occupancy_type_parcel``.

    ``Secondary`` footprints (non-primary residential) are left untouched.

    Parameters
    ----------
    ruleset : str
        Filename of the keyword ruleset CSV stored beside the curate recipe.
    parcel_column : str, optional
        Parcel land-use column to match against (default
        ``purpose_group_combined_parcel``).
    """
    curated = state.curated
    if 'occupancy_type' not in curated or parcel_column not in curated:
        return state

    rules = _load_occupancy_ruleset(state, ruleset)
    if not rules:
        return state

    terms = curated[parcel_column].astype(object)
    proposal = pd.Series(pd.NA, index=curated.index, dtype=object)
    reviewed = pd.Series(False, index=curated.index)
    matched_pattern = pd.Series(pd.NA, index=curated.index, dtype=object)
    unmatched = pd.Series(True, index=curated.index)
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
            matched_pattern.loc[mask] = rule['pattern']
            unmatched.loc[mask] = False

    base = curated['occupancy_type'].astype(object).copy()
    differs = proposal.notna() & (proposal != base) & base.ne('Secondary')
    apply = differs & reviewed
    conflict = differs & ~reviewed

    base.loc[apply] = proposal.loc[apply]

    curated['occupancy_type'] = pd.Categorical(base)
    curated['occupancy_type_parcel'] = pd.Categorical(proposal)
    curated['occupancy_type_conflict'] = conflict
    state.curated = curated

    summary = pd.DataFrame(
        {
            parcel_column: terms[conflict],
            'occupancy_type_parcel': proposal[conflict],
            'occupancy_type_nsi': base[conflict],
            'matched_pattern': matched_pattern[conflict],
        }
    )
    summary = (
        summary.groupby(list(summary.columns), dropna=False)
        .size()
        .rename('count')
        .reset_index()
        .sort_values('count', ascending=False, ignore_index=True)
    )

    report_path = None
    if len(summary):
        from openplaces.path import reports_path

        report_path = reports_path(state.admin_id, filename='occupancy-conflicts.csv')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(report_path, index=False)

    if state.verbose:
        print(
            f'  resolve_occupancy: {int(apply.sum()):,} parcel overrides applied, '
            f'{int(conflict.sum()):,} unreviewed conflicts surfaced'
        )
        if report_path is not None:
            print('  Top 10 unreviewed conflicts:')
            top = summary.head(10).to_string(index=False)
            print('\n'.join('    ' + line for line in top.splitlines()))
            print(f'  Full conflict report: {report_path}')

    return state


@_register('refine_occupancy_height')
def refine_occupancy_height(
    state: CurateState,
    thresholds: dict | None = None,
) -> CurateState:
    """Derive ``occupancy_type_cheer`` by splitting Multi-Family by ``n_stories``.

    Writes a new ``occupancy_type_cheer`` column: a copy of ``occupancy_type``
    in which the generic ``Multi-Family`` class is refined into HAZUS low-,
    mid-, and high-rise bands wherever a story count is available. All other
    classes (Single-Family, Mobile Home, Secondary, non-residential categories)
    are carried over unchanged, and the base ``occupancy_type`` is left intact.

    The story count reflects merged NSI and BRAILS evidence, so this step must
    run after ``merge_enrichments``. Multi-Family rows with a missing
    ``n_stories`` stay labeled ``Multi-Family``.

    Parameters
    ----------
    thresholds : dict, optional
        ``low_max_stories`` (int, default 3) — highest story count still
        classed as Low-Rise Multi-Family.
        ``mid_max_stories`` (int, default 7) — highest story count still
        classed as Mid-Rise Multi-Family; taller buildings are High-Rise.
    """
    curated = state.curated
    if 'occupancy_type' not in curated or 'n_stories' not in curated:
        return state

    thresholds = thresholds or {}
    low_max = float(thresholds.get('low_max_stories', 3))
    mid_max = float(thresholds.get('mid_max_stories', 7))

    occupancy = curated['occupancy_type'].astype(object).copy()
    n_stories = curated['n_stories']

    multi_family = occupancy.eq('Multi-Family') & n_stories.notna()
    low = multi_family & (n_stories <= low_max)
    mid = multi_family & (n_stories > low_max) & (n_stories <= mid_max)
    high = multi_family & (n_stories > mid_max)

    occupancy.loc[low] = 'Low-Rise Multi-Family'
    occupancy.loc[mid] = 'Mid-Rise Multi-Family'
    occupancy.loc[high] = 'High-Rise Multi-Family'

    curated['occupancy_type_cheer'] = pd.Categorical(occupancy)
    state.curated = curated

    if state.verbose:
        counts = occupancy.value_counts(dropna=False)
        print(
            '  refine_occupancy_height: '
            + ', '.join(f'{k}={v:,d}' for k, v in counts.items())
        )

    return state
