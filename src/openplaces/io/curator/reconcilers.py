"""Curation steps that resolve conflicts between competing source columns.

Reconciliation picks (or blends) a single canonical value when two or more
columns disagree. It is distinct from imputation (filling missing values) and
inference (deriving a brand-new concept).
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register


def _source_token(col: str, default: str | None = None) -> str:
    """Return the provenance token for *col*: its source suffix, or *default*.

    An imputed-evidence column (e.g. ``land_value_imputed_parcel``) would
    otherwise have its ``_imputed`` marker swallowed by the registered
    ``_parcel`` suffix strip, losing the real/imputed distinction in the
    provenance sidecar. A column with no provenance suffix at all (e.g. a
    bare canonical name like ``structure_value``) falls back to *default*
    when given, else to *col* itself.
    """
    from openplaces.io.curator.formatters import _split_source

    if '_imputed' in col:
        return 'imputed'
    token = _split_source(col)[1]
    if token is not None:
        return token
    return col if default is None else default


@_register('reconcile_values')
def reconcile_values(
    state: CurateState,
    priority: dict[str, list[str]],
) -> CurateState:
    """Select each canonical value from competing source columns by priority.

    For every target feature, the first non-null value across the listed source
    columns wins (Lochhead et al. 2026, Step C). Source columns are given by
    their explicit, already-suffixed names so the selection is unambiguous.

    Parameters
    ----------
    priority : dict of {feature: [source_column, ...]}
        Each key is the canonical output column; the value is an ordered list of
        existing source columns to try. Missing columns are skipped.

        Example::

            priority:
              n_dwellings: [n_dwellings_overture, n_dwellings_parcel]
              year_built: [year_built_parcel]
              improvement_value: [improvement_value_parcel]
    """
    from openplaces.io.curator.provenance import record_source

    curated = state.curated
    for feature, source_cols in priority.items():
        cols = [c for c in source_cols if c in curated.columns]
        if not cols:
            continue
        sub = curated[cols]
        curated[feature] = sub.bfill(axis=1).iloc[:, 0]
        # Record which source column supplied each chosen value: the token is the
        # winning column's provenance suffix (parcel/nsi/overture).
        notnull = sub.notna()
        has_value = notnull.any(axis=1)
        winning = notnull.idxmax(axis=1)
        for col in cols:
            mask = has_value & winning.eq(col)
            if mask.any():
                record_source(curated, feature, mask, _source_token(col))
    state.curated = curated
    return state


def _resolve_admin_group(
    state: CurateState, curated, admin_level: int
) -> pd.Series | None:
    """Return a per-row admin-unit id `Series` at *admin_level*, or `None`.

    `None` means every row of *curated* belongs to the same admin unit --
    either because *admin_level* is at or above the level this curate call
    already processes one unit of (see :func:`impute_land_value`'s note that
    each call already processes exactly one admin unit, with no per-row
    admin id column required), or because a finer per-row group could not be
    resolved (no usable geometry, or no admin boundaries at that level) --
    callers should then treat the whole chunk (`state.admin_id`) as one
    group.

    When a finer level is requested and an ``admin{admin_level}_id`` column
    isn't already present, one is derived by spatially joining each row's
    centroid to :func:`~openplaces.io.readers.get_admin`'s boundaries for
    *admin_level* within `state.admin_id` -- the same fallback path
    :func:`~openplaces.io.enricher.parcels._resolve_calibrate_group_col`
    uses for town-level grouping. Never raises.
    """
    if admin_level <= state.admin_id.get_level():
        return None

    existing_col = f'admin{admin_level}_id'
    if existing_col in curated.columns:
        return curated[existing_col]

    geom = getattr(curated, 'geometry', None)
    if geom is None or geom.isna().all():
        return None

    try:
        from openplaces.io.readers import get_admin

        units = get_admin(state.admin_id, admin_level, geom=True)
    except Exception:
        return None
    if units.empty:
        return None

    import warnings

    import geopandas as gpd

    with warnings.catch_warnings():
        # Coarse "which admin unit is this footprint in" -- a geographic-CRS
        # centroid is imprecise by at most a few meters, irrelevant next to
        # an admin unit's own size, so this is safe to silence rather than
        # reproject first.
        warnings.filterwarnings('ignore', 'Geometry is in a geographic CRS')
        centroids = geom.centroid
    points = gpd.GeoDataFrame(
        geometry=centroids, index=curated.index, crs=curated.crs
    ).to_crs(units.crs)
    joined = gpd.sjoin(points, units[['geometry']], how='left', predicate='within')
    id_col = existing_col if existing_col in joined.columns else 'index_right'
    return joined[id_col].reindex(curated.index)


@_register('select_value_source_by_admin_unit')
def select_value_source_by_admin_unit(
    state: CurateState,
    output: str,
    parcel_column: str,
    other_column: str,
    admin_level: int = 4,
    coverage_threshold: float = 0.5,
    priority_column: str = 'priority_on_parcel',
    exclude_priority: str = 'secondary',
    min_group_size: int = 5,
) -> CurateState:
    """Pick *parcel_column* or *other_column* per admin unit, not per row.

    Blending two value sources row by row within one place is what causes
    errors here: whether *parcel_column* (a parcel-apportioned assessor
    value) is usable at all is a property of the local assessor data, not
    something that varies building by building within a well-covered area.
    This step instead decides, once per admin unit, which source that whole
    unit uses -- an admin unit with good parcel coverage keeps
    *parcel_column* everywhere in it (even for the few rows individually
    missing one -- no per-row top-off from *other_column*); an admin unit
    with poor coverage uses *other_column* everywhere in it (discarding
    *parcel_column* even where individually present).

    Coverage is computed only over rows where *priority_column* is not
    *exclude_priority* (default excludes ``'secondary'``): a
    ``'secondary'``-priority footprint structurally never receives an
    apportioned value from
    :func:`~openplaces.io.curator.evidence.apportion_curated_values`
    regardless of local data quality, so including it in the denominator
    would make every admin unit look artificially low-coverage. The
    resulting per-unit decision still applies to every row in the unit,
    including ``'secondary'``-priority ones.

    Admin units are grouped at *admin_level* (default 4, town/MCD) via
    :func:`_resolve_admin_group`, falling back to the whole processing
    chunk (`state.admin_id`) as one group where a per-row group can't be
    resolved. A group with fewer than *min_group_size* eligible rows falls
    back to its enclosing chunk's coverage instead, so a handful of rows
    near a chunk boundary don't get an unstable, small-sample estimate.

    Parameters
    ----------
    output : str
        Canonical column to write.
    parcel_column, other_column : str
        Competing source columns (e.g. ``'structure_value'``,
        ``'structure_value_building_nsi'``). Skipped (returns *state*
        unchanged) if either is absent from `state.curated`.
    admin_level : int, optional
        Admin level to group by (default 4).
    coverage_threshold : float, optional
        A group switches to *other_column* when the fraction of its eligible
        rows with a real, positive *parcel_column* falls below this (default
        0.5 -- a majority of buildings must be missing a value to switch).
    priority_column : str, optional
        Column marking each row's role in the parcel apportionment (default
        ``'priority_on_parcel'``); rows are all treated as eligible when
        absent.
    exclude_priority : str, optional
        *priority_column* value excluded from the coverage denominator
        (default ``'secondary'``).
    min_group_size : int, optional
        Minimum eligible-row count for a group's own coverage to be trusted
        (default 5); smaller groups fall back to the chunk-wide coverage.
    """
    curated = state.curated
    if parcel_column not in curated.columns or other_column not in curated.columns:
        if state.verbose:
            print(
                '  select_value_source_by_admin_unit: '
                f'{parcel_column!r}/{other_column!r} missing; skipping.'
            )
        return state

    if priority_column in curated.columns:
        eligible = curated[priority_column].astype(object) != exclude_priority
    else:
        eligible = pd.Series(True, index=curated.index)

    parcel_value = pd.to_numeric(curated[parcel_column], errors='coerce')
    has_value = eligible & parcel_value.notna() & (parcel_value > 0)

    chunk_key = str(state.admin_id)
    chunk_keys = pd.Series(chunk_key, index=curated.index)
    group = _resolve_admin_group(state, curated, admin_level)
    group_keys = chunk_keys if group is None else group.fillna(chunk_key)

    def _coverage(keys: pd.Series) -> tuple[pd.Series, pd.Series]:
        size = eligible.groupby(keys).transform('sum')
        covered = has_value.groupby(keys).transform('sum')
        return (covered / size).where(size > 0), size

    coverage, group_size = _coverage(group_keys)
    chunk_coverage, _ = _coverage(chunk_keys)
    too_small = group_size.fillna(0) < min_group_size
    coverage = coverage.where(~too_small, chunk_coverage)

    # No eligible evidence anywhere for a unit (coverage still unknown after
    # the chunk-wide fallback) is itself a sign the parcel source has
    # nothing usable here -- default to the other source rather than the
    # parcel one.
    use_other = coverage.fillna(0.0) < coverage_threshold

    curated[output] = curated[other_column].where(use_other, curated[parcel_column])

    from openplaces.io.curator.provenance import record_source

    record_source(
        curated, output, ~use_other, _source_token(parcel_column, default='parcel')
    )
    record_source(
        curated, output, use_other, _source_token(other_column, default='nsi')
    )
    state.curated = curated

    if state.verbose:
        n_units = group_keys.nunique()
        switched = group_keys[use_other].nunique()
        print(
            f'  select_value_source_by_admin_unit: {switched:,}/{n_units:,} '
            f'admin unit(s) switched to {other_column!r} '
            f'({int(use_other.sum()):,}/{len(curated):,} row(s)).'
        )
    return state


@_register('suppress_where')
def suppress_where(
    state: CurateState,
    column: str,
    condition_column: str,
    condition_value: object = True,
) -> CurateState:
    """Null *column* wherever *condition_column* equals *condition_value*.

    A generic evidence-validity gate: some upstream determination (e.g. a
    land-use classification) can invalidate an otherwise-present value
    without itself being a competing source to reconcile against. Distinct
    from ``reconcile_values`` (picks among several present sources) and
    imputation (fills a *missing* value) — this only removes a value that
    should not have been trusted in the first place.

    Parameters
    ----------
    column : str
        Column to null out.
    condition_column : str
        Column whose value triggers the suppression.
    condition_value : optional
        Value that triggers suppression (default ``True``, for a boolean
        flag column).
    """
    import numpy as np

    curated = state.curated
    if column not in curated.columns or condition_column not in curated.columns:
        return state

    condition = (curated[condition_column].astype(object) == condition_value).fillna(
        False
    )
    mask = condition & curated[column].notna()
    if mask.any():
        curated.loc[mask, column] = np.nan
    state.curated = curated

    if state.verbose:
        print(
            f'  suppress_where: {int(mask.sum()):,} {column!r} value(s) suppressed '
            f'where {condition_column!r} == {condition_value!r}.'
        )
    return state


def _summarize_conflicts(
    present: list[tuple[str, pd.Series]],
    index: pd.Index,
) -> pd.Series:
    """Summarize disagreeing evidence values per row as a compact string.

    *present* is a list of (label, values) pairs, each values Series aligned
    to *index*. Returns an object Series that is missing except where at
    least two present values disagree; there, sources are grouped by unique
    value — groups ordered by first-appearing label, labels within a group
    joined with '/' — e.g. 'nsi/parcel: Single Family | fema: Manufactured
    Home', so agreements and disagreements are both visible at a glance.
    """
    from itertools import combinations

    conflict = pd.Series(pd.NA, index=index, dtype=object)
    if len(present) < 2:
        return conflict

    differ = pd.Series(False, index=index)
    for (_, class_a), (_, class_b) in combinations(present, 2):
        both = class_a.notna() & class_b.notna()
        differ = differ | (both & class_a.ne(class_b))
    if not differ.any():
        return conflict

    labels = [label for label, _ in present]
    stacked = pd.concat(
        {label: values.astype(object) for label, values in present}, axis=1
    )

    def _row_summary(row) -> str:
        groups: dict[str, list[str]] = {}
        for label in labels:
            value = row[label]
            if pd.notna(value):
                groups.setdefault(str(value), []).append(label)
        return ' | '.join(f'{"/".join(who)}: {value}' for value, who in groups.items())

    conflict.loc[differ] = stacked.loc[differ].apply(_row_summary, axis=1)
    return conflict


@_register('resolve_occupancy')
def resolve_occupancy(
    state: CurateState,
    ruleset: str,
    parcel_column: str = 'use_group_combined_parcel',
) -> CurateState:
    """Apply parcel-side corrections over the base occupancy and flag conflicts.

    The base ``occupancy_type`` (from ``impute_occupancy_type``) follows the
    recipe's evidence priority. This step applies the high-confidence reviewed
    keyword override, records the parcel-proposed class, sets a review flag, and
    summarizes evidence disagreements. All thresholds, columns, and class labels
    come from the recipe ``occupancy`` block; the keyword rules come from
    *ruleset*.

    The single class correction here is a ``reviewed`` keyword rule whose class
    differs from the base. Value-share and dwelling-count class assignment now
    happen in the generic ``resolve_by_vote`` step, which weighs them against
    one another rather than letting the last correction win.

    ``occupancy_type_parcel`` is the keyword proposal, else the class coerced from
    the parcel evidence column (the evidence entry whose ``label`` is ``parcel``).
    A review-flag column marks footprints whose improvement value is a small
    nonzero share of total value
    (``0 < improvement/(improvement+land) < review_max_ratio``).
    ``occupancy_type_conflict`` is a categorical summary of every present
    occupancy evidence (NSI, FEMA, parcel, and any other source in
    ``occupancy.evidence``) for rows where two or more disagree (else null),
    with sources grouped by unique value — e.g.
    ``"nsi/parcel: Single Family | fema: Manufactured Home"`` (see
    :func:`_summarize_conflicts`). To keep the column low-cardinality, every
    non-residential class is collapsed into a single bucket label
    (``occupancy.conflict_other_label``, default ``Non-Residential``), so only
    residential — or residential-vs-non-residential — disagreements are
    surfaced.

    Parameters
    ----------
    ruleset : str
        Filename of the keyword ruleset CSV stored beside the curate recipe.
    parcel_column : str, optional
        Parcel land-use column the keyword rules match against.
    """
    from openplaces.io.curator.occupancy import (
        bucket_to_residential,
        coerce_to_class,
        get_occupancy_config,
        load_ruleset,
    )

    curated = state.curated
    if 'occupancy_type' not in curated or parcel_column not in curated:
        return state

    config = get_occupancy_config(state)
    class_rules = load_ruleset(state, config['class_map'])
    evidence = config.get('evidence', [])
    columns = config.get('columns', {})
    rule_cfg = config.get('rules', {})

    # Keyword proposal from the parcel land-use string.
    terms = curated[parcel_column].astype(object)
    proposal = pd.Series(pd.NA, index=curated.index, dtype=object)
    reviewed = pd.Series(False, index=curated.index)
    unmatched = pd.Series(True, index=curated.index)
    for rule in load_ruleset(state, ruleset):
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

    # occupancy_type_parcel: keyword proposal, else the parcel-evidence class.
    # Select the parcel evidence by its label so inserting other sources (e.g.
    # FEMA) ahead of it does not shift a positional index onto the wrong column.
    parcel_ev = next((ev for ev in evidence if ev.get('label') == 'parcel'), None)
    if parcel_ev is None and len(evidence) > 1:
        parcel_ev = evidence[1]
    parcel_col = parcel_ev['column'] if parcel_ev else None
    parcel_class = proposal.copy()
    if parcel_col and parcel_col in curated.columns:
        parcel_class = parcel_class.fillna(
            coerce_to_class(curated[parcel_col], class_rules)
        )

    from openplaces.io.curator.provenance import record_source

    base = curated['occupancy_type'].astype(object).copy()
    secondary = config.get('secondary_class')

    # Correction 1: reviewed keyword override.
    apply_kw = proposal.notna() & reviewed & (proposal != base)
    if secondary is not None:
        apply_kw &= base.ne(secondary)
    base.loc[apply_kw] = proposal.loc[apply_kw]
    if apply_kw.any():
        record_source(curated, 'occupancy_type', apply_kw, 'keyword')

    # Value-share review flag: a small nonzero improvement share of total value
    # marks ambiguous footprints for manual inspection. The class assignment for
    # zero/low value and for high dwelling counts now happens in resolve_by_vote,
    # which weighs those signals against one another.
    mh_rule = rule_cfg.get('manufactured_home_value', {})
    imp_col = columns.get('improvement_value')
    land_col = columns.get('land_value')
    review = pd.Series(False, index=curated.index)
    ratio_max = mh_rule.get('review_max_ratio')
    if (
        ratio_max is not None
        and imp_col
        and imp_col in curated.columns
        and land_col
        and land_col in curated.columns
    ):
        improvement = pd.to_numeric(curated[imp_col], errors='coerce')
        land = pd.to_numeric(curated[land_col], errors='coerce')
        total = improvement + land
        ratio = improvement.where(total > 0) / total.where(total > 0)
        review = (improvement > 0) & (ratio < float(ratio_max))

    # Conflict summary across every present occupancy evidence (NSI, FEMA, parcel,
    # and any future source), coerced and compared at residential granularity.
    # Each non-residential class is collapsed into one bucket label so the column
    # stays low-cardinality — residential disagreements (incl. residential vs
    # non-residential) are surfaced, while two differing non-residential categories
    # (e.g. Retail vs Hotel) are not.
    present = [
        (
            ev.get('label', ev['column']),
            bucket_to_residential(curated[ev['column']], config, class_rules),
        )
        for ev in evidence
        if ev['column'] in curated.columns
    ]
    conflict = _summarize_conflicts(present, curated.index)

    review_col = config.get('review_column', 'occupancy_type_review')
    curated['occupancy_type'] = pd.Categorical(base)
    curated['occupancy_type_parcel'] = pd.Categorical(parcel_class)
    curated['occupancy_type_conflict'] = pd.Categorical(conflict)
    curated[review_col] = review.to_numpy()
    state.curated = curated

    has_conflict = conflict.notna()
    report_path = None
    if has_conflict.any():
        from openplaces.path import reports_path

        summary = (
            pd.DataFrame(
                {
                    parcel_column: terms[has_conflict],
                    'occupancy_type_conflict': conflict[has_conflict],
                }
            )
            .groupby(['occupancy_type_conflict', parcel_column], dropna=False)
            .size()
            .rename('count')
            .reset_index()
            .sort_values('count', ascending=False, ignore_index=True)
        )
        report_path = reports_path(state.admin_id, filename='occupancy-conflicts.csv')
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(report_path, index=False)

    if state.verbose:
        print(
            f'  resolve_occupancy: {int(apply_kw.sum()):,} keyword overrides, '
            f'{int(has_conflict.sum()):,} evidence conflicts, '
            f'{int(review.sum()):,} review-flagged'
            + (f' (report: {report_path})' if report_path is not None else '')
        )

    return state


@_register('reconcile_land_use')
def reconcile_land_use(
    state: CurateState,
    columns: list[dict],
    output: str = 'land_use_class',
    tiebreaker: str = 'group_parcel',
    class_map_id: str | None = None,
    conflict_column: str = 'land_use_class_conflict',
    report: str | None = None,
) -> CurateState:
    """Fill missing land-use classes by vote across group-vocabulary evidence.

    Each listed column casts one vote per row with its (non-null) value; the
    value with the most votes wins. On a tie, the *tiebreaker* column's value
    wins when it is among the tied values; a residual tie (tiebreaker absent)
    falls to the earliest listed column voting for a tied value. The winning
    group is mapped through the *class_map_id* crosswalk to the coarse
    land-use class and fills only rows where *output* is missing — classes
    already assigned by the rule-based vote (``classify_parcel_land_use``)
    stay on top. The ``{output}_source`` sidecar records the winning value's
    contributing labels joined with '/' (e.g. ``nsi/parcel``).

    Also writes *conflict_column* (see :func:`_summarize_conflicts`), a
    grouped summary like ``"nsi/parcel: Single Family | fema: Manufactured
    Home"`` for rows where the present values disagree, and saves its most
    frequent combinations (count-sorted) to the reports directory.

    Parameters
    ----------
    state : CurateState
        The curation state with the target GeoDataFrame in state.curated.
    columns : list of dict
        Voting columns in priority order, each ``{column, label}``. All are
        expected to share one vocabulary (normalize upstream, e.g. via
        ``remap_column``); missing columns are skipped.
    output : str, optional
        Land-use class column to fill (default ``land_use_class``).
    tiebreaker : str, optional
        Column whose value breaks ties when present among the tied values
        (default ``group_parcel``).
    class_map_id : str, optional
        Recipe id of the group -> class crosswalk CSV applied to the winning
        value. Winning groups missing from the map leave the row unfilled.
        When omitted, the winning group is written as-is.
    conflict_column : str, optional
        Output column for the grouped disagreement summary.
    report : str, optional
        Filename for the conflict-combination counts CSV written to the
        reports directory (skipped when omitted or no conflicts exist).
    """
    from openplaces.io.curator.provenance import record_source
    from openplaces.io.transform import get_crosswalk

    curated = state.curated

    present = [
        (spec.get('label', spec['column']), curated[spec['column']].astype(object))
        for spec in columns
        if spec['column'] in curated.columns
    ]
    if not present:
        if state.verbose:
            print('  reconcile_land_use: no evidence columns present; skipping.')
        return state
    labels = [label for label, _ in present]
    tiebreaker_label = next(
        (
            spec.get('label', spec['column'])
            for spec in columns
            if spec['column'] == tiebreaker and spec['column'] in curated.columns
        ),
        None,
    )

    conflict = _summarize_conflicts(present, curated.index)
    curated[conflict_column] = pd.Categorical(conflict)

    stacked = pd.concat(dict(present), axis=1)

    def _vote(row) -> tuple:
        votes = [(label, row[label]) for label in labels if pd.notna(row[label])]
        if not votes:
            return (pd.NA, pd.NA)
        counts: dict[str, int] = {}
        for _, value in votes:
            counts[value] = counts.get(value, 0) + 1
        max_votes = max(counts.values())
        # dict preserves first-vote order, so ties fall to the earliest
        # listed column unless the tiebreaker claims one of the tied values.
        tied = [value for value, n in counts.items() if n == max_votes]
        winner = tied[0]
        if len(tied) > 1 and tiebreaker_label is not None:
            tiebreaker_value = row[tiebreaker_label]
            if pd.notna(tiebreaker_value) and tiebreaker_value in tied:
                winner = tiebreaker_value
        token = '/'.join(label for label, value in votes if value == winner)
        return (winner, token)

    mask_any = stacked.notna().any(axis=1)
    winner = pd.Series(pd.NA, index=curated.index, dtype=object)
    token = pd.Series(pd.NA, index=curated.index, dtype=object)
    if mask_any.any():
        voted = stacked.loc[mask_any].apply(_vote, axis=1)
        winner.loc[mask_any] = voted.str[0]
        token.loc[mask_any] = voted.str[1]

    if class_map_id:
        filled = winner.map(get_crosswalk({'recipe_id': class_map_id}))
    else:
        filled = winner
    if output in curated.columns:
        to_fill = curated[output].isna() & filled.notna()
    else:
        to_fill = filled.notna()

    values = (
        curated[output].astype(object)
        if output in curated.columns
        else pd.Series(pd.NA, index=curated.index, dtype=object)
    )
    values.loc[to_fill] = filled.loc[to_fill]
    curated[output] = pd.Categorical(values)
    for fill_token in token.loc[to_fill].dropna().unique():
        record_source(curated, output, to_fill & token.eq(fill_token), fill_token)

    report_path = None
    if report and conflict.notna().any():
        import warnings

        try:
            from openplaces.path import reports_path

            summary = (
                conflict.dropna()
                .value_counts()
                .rename('count')
                .rename_axis(conflict_column)
                .reset_index()
            )
            report_path = reports_path(state.admin_id, filename=report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(report_path, index=False)
        except Exception as exception:
            warnings.warn(f'reconcile_land_use: conflict report failed: {exception}')
            report_path = None

    if state.verbose:
        print(
            f'  reconcile_land_use: {int(to_fill.sum()):,} filled, '
            f'{int(conflict.notna().sum()):,} conflicts'
            + (f' (report: {report_path})' if report_path is not None else '')
        )
    state.curated = curated
    return state


@_register('resolve_by_vote')
def resolve_by_vote(
    state: CurateState,
    target: str,
    decisions: list[dict],
) -> CurateState:
    """Override *target* by tallying weighted votes from independent indicators.

    Each decision proposes one class and lists indicators (predicates over
    existing columns). For every row, a decision's score is the sum of the
    weights of its matched indicators; the decision is eligible where that score
    reaches its ``min_score``. Among the eligible decisions the highest score
    wins (ties broken by recipe order), and the winning class overwrites
    *target*. Rows with no eligible decision keep their existing value.

    This is the generic, vocabulary-neutral reconciliation seam: it holds no
    class names or thresholds of its own, so the same step resolves any
    categorical column. New evidence (e.g. a model probability) joins a decision
    as one more weighted indicator without code changes — a
    ``numeric_at_least`` over ``p_manufactured_home`` is all it takes.

    Parameters
    ----------
    target : str
        Categorical column to override (e.g. ``occupancy_type``). Created as
        an all-missing column first when not already present, so this step can
        also populate a brand-new derived classification, not just correct an
        existing one.
    decisions : list of dict
        Ordered candidate classes. Each is
        ``{class, min_score, indicators, require, source}``, where
        ``indicators`` is a list of indicator specs (see
        :func:`~openplaces.io.curator.indicators.evaluate_indicator`);
        ``min_score`` defaults to 1 and each indicator's ``weight`` defaults to
        1. ``require`` is an optional list of indicator specs (same vocabulary)
        that must *all* hold, on top of reaching ``min_score`` — a hard
        precondition rather than one more weighted vote, for evidence that
        should veto a decision outright regardless of how strongly the other
        indicators favor it (e.g. a minimum footprint size). The optional
        ``source`` is the provenance token recorded in ``{target}_source`` for
        rows this decision wins (default ``'vote'``), so the single reason
        column distinguishes one decision's outcome from another.
    """
    from openplaces.io.curator.indicators import evaluate_indicator
    from openplaces.io.curator.provenance import record_source

    curated = state.curated
    if not decisions:
        return state
    if target not in curated.columns:
        curated[target] = pd.Series(pd.NA, index=curated.index, dtype=object)

    base = curated[target].astype(object).copy()
    winner = pd.Series(pd.NA, index=curated.index, dtype=object)
    token = pd.Series(pd.NA, index=curated.index, dtype=object)
    best_score = pd.Series(-1.0, index=curated.index)
    for decision in decisions:
        score = pd.Series(0.0, index=curated.index)
        for indicator in decision.get('indicators', []):
            weight = float(indicator.get('weight', 1.0))
            matched = evaluate_indicator(curated, indicator).astype(float)
            score = score + matched * weight
        eligible = score >= float(decision.get('min_score', 1))
        for req in decision.get('require', []):
            eligible = eligible & evaluate_indicator(curated, req)
        # Strict > keeps the earlier decision on ties (recipe order).
        take = eligible & (score > best_score)
        winner.loc[take] = decision['class']
        token.loc[take] = decision.get('source', 'vote')
        best_score.loc[take] = score.loc[take]

    assign = winner.notna()
    base.loc[assign] = winner.loc[assign]
    curated[target] = pd.Categorical(base)
    for tok in token[assign].dropna().unique():
        record_source(curated, target, assign & token.eq(tok), tok)
    state.curated = curated

    if state.verbose:
        counts = winner[assign].value_counts()
        summary = ', '.join(f'{k}={v:,d}' for k, v in counts.items()) or 'none'
        print(f'  resolve_by_vote: {target} overridden -> {summary}')

    return state


# Recipe role keys accepted by reconcile_addresses: address_full is a
# one-line string to parse; the rest are the component keys of
# openplaces.geo.address.ADDRESS_COMPONENTS, used verbatim.
def reconcile_addresses(state: CurateState, **kwargs) -> CurateState:
    """Curate-stage wrapper: reconcile addresses on ``state.curated``.

    Thin wrapper around the shared, state-agnostic
    :func:`~openplaces.io.harmonizer.addresses.reconcile_addresses_df` (see
    that function's docstring for parameters and behavior) -- the harmonize
    stage now runs the same reconciliation earlier, against ``state.spine``,
    for entities this step's config was moved there for (see
    ``US_footprint-spine-2026.yaml``/``US_parcel-spine-2026.yaml``); this
    curate-stage registration remains available for any recipe that still
    wants to reconcile addresses at curate time.
    """
    from openplaces.io.harmonizer.addresses import reconcile_addresses_df

    state.curated = reconcile_addresses_df(
        state.curated, state.admin_id, state.verbose, **kwargs
    )
    return state
