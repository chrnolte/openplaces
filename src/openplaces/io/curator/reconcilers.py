"""Curation steps that resolve conflicts between competing source columns.

Reconciliation picks (or blends) a single canonical value when two or more
columns disagree. It is distinct from imputation (filling missing values) and
inference (deriving a brand-new concept).
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register


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
    from openplaces.io.curator.formatters import _split_source
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
            token = _split_source(col)[1] or col
            mask = has_value & winning.eq(col)
            if mask.any():
                record_source(curated, feature, mask, token)
    state.curated = curated
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
_ADDRESS_ROLES = (
    'address_full',
    'address_number',
    'address_street',
    'unit_number',
    'city',
    'state',
    'postal_code',
)


@_register('reconcile_addresses')
def reconcile_addresses(
    state: CurateState,
    sources: dict[str, dict[str, str]],
    output_col: str = 'address',
    similarity_threshold: float = 80,
    conflict_column: str = 'address_conflict',
    complete_from_admin: dict[str, int] | None = None,
) -> CurateState:
    """Reconcile street addresses from any number of source inputs.

    Each key of *sources* is a provenance token recorded in the output's
    source sidecar; its value maps component roles to evidence columns.
    Roles: address_full (a one-line address string, parsed via
    openplaces.geo.address.parse_address), address_number, address_street,
    unit_number, city, state, postal_code.

    Declaration order is priority. The base address comes from the
    highest-priority source with a usable address on each row (non-empty
    street; sources that declare address_number must also have the number).
    A lower-priority source agrees with the base when house numbers match
    and streets match per openplaces.geo.address.match_streets — notation
    equivalences from address_equivalences.csv are applied, then rapidfuzz
    similarity against *similarity_threshold* (0-100);
    agreeing sources fill the base's missing components (every component
    outside MATCH_COMPONENTS) and mark the row 'reconciled'. Disagreeing
    sources
    are excluded from selection but summarized in *conflict_column* (null
    when the sources agree or only one is present; see
    :func:`_summarize_conflicts`). Missing columns are skipped, like in
    reconcile_values.

    Parameters
    ----------
    sources : dict of {token: {role: column}}
        Ordered mapping of provenance token to role-column spec.

        Example::

            sources:
              parcel:
                address_full: address_parcel
              dwelling_overture:
                address_street: address_street_dwelling_overture
                address_number: address_number_dwelling_overture
    output_col : str
        Canonical output column (default 'address').
    similarity_threshold : float
        Minimum street similarity (0-100) for two sources to agree.
    conflict_column : str
        Output column for the grouped disagreement summary (default
        'address_conflict').
    complete_from_admin : dict of {component: admin_level}, optional
        Fill components that no source provided from the run's admin id,
        e.g. ``{state: 2}`` completes a missing state with the admin unit's
        level-2 code (validated against ISO 3166-2 for the unit's country).
        Only rows that already carry another non-street component are
        completed, so street-only addresses stay untouched.
    """
    from openplaces.core.schema import AdminId
    from openplaces.geo.address import (
        ADDRESS_COMPONENTS,
        MATCH_COMPONENTS,
        get_admin2_codes,
        harmonize_address_case,
        match_streets,
        normalize_address_components,
        parse_address,
    )
    from openplaces.io.curator.provenance import record_source

    curated = state.curated
    components = list(ADDRESS_COMPONENTS)
    fillable = [c for c in components if c not in MATCH_COMPONENTS]
    admin_str = str(state.admin_id) if state.admin_id else ''
    admin_levels = AdminId(admin_str).levels if admin_str else ()
    admin1_id = admin_levels[0] if admin_levels else None

    def normalize_component(value: str, component: str) -> str:
        kwargs = {'address_street': '', 'admin1_id': admin1_id}
        kwargs[component] = value
        return normalize_address_components(**kwargs)[component]

    # Build one normalized component frame per source (over unique values)
    frames: dict[str, pd.DataFrame] = {}
    needs_number: dict[str, bool] = {}
    for name, spec in sources.items():
        unknown = set(spec) - set(_ADDRESS_ROLES)
        if unknown:
            raise ValueError(
                f'reconcile_addresses: unknown role(s) {sorted(unknown)} for '
                f'source {name!r}; valid roles: {sorted(_ADDRESS_ROLES)}'
            )
        present = {role: col for role, col in spec.items() if col in curated.columns}
        if not present:
            continue
        frame = pd.DataFrame('', index=curated.index, columns=components)
        full_col = present.get('address_full')
        if full_col is not None:
            keys = curated[full_col].map(
                lambda v: str(v).strip() if pd.notna(v) else ''
            )
            empty = dict.fromkeys(components, '')
            lookup = {'': empty}
            for value in keys.unique():
                if not value:
                    continue
                parsed = parse_address(value, admin1_id=admin1_id).components
                lookup[value] = normalize_address_components(
                    address_street=parsed['address_street'] or '',
                    address_number=parsed['address_number'],
                    unit_number=parsed['unit_number'],
                    postal_code=parsed['postal_code'],
                    city=parsed['city'],
                    state=parsed['state'],
                    admin1_id=admin1_id,
                )
            for comp in components:
                frame[comp] = keys.map(lambda k: lookup[k][comp])
        for role, col in present.items():
            if role == 'address_full':
                continue
            comp = role
            values = curated[col].map(lambda v: str(v) if pd.notna(v) else '')
            normed = {v: normalize_component(v, comp) for v in values.unique()}
            frame[comp] = values.map(normed)
        frames[name] = frame
        needs_number[name] = 'address_number' in present

    if not frames:
        if state.verbose:
            print('  reconcile_addresses: no address source columns found.')
        return state

    # Base = highest-priority source with a usable address on each row
    usable: dict[str, pd.Series] = {}
    for name, frame in frames.items():
        mask = frame['address_street'].str.len() > 0
        if needs_number[name]:
            mask &= frame['address_number'].str.len() > 0
        usable[name] = mask

    base = pd.Series('', index=curated.index, dtype=object)
    for name in frames:
        take = base.eq('') & usable[name]
        base.loc[take] = name

    merged = pd.DataFrame('', index=curated.index, columns=components)
    for name, frame in frames.items():
        take = base.eq(name)
        merged.loc[take] = frame.loc[take]

    # Agreeing lower-priority sources fill the base's missing components;
    # sources that fail the agreement check are tracked for conflict flagging
    corroborated = pd.Series(False, index=curated.index)
    disagreeing = pd.Series(False, index=curated.index)
    for name, frame in frames.items():
        others = usable[name] & base.ne('') & base.ne(name)
        candidate = others & (
            merged['address_number'].ne('')
            & frame['address_number'].ne('')
            & merged['address_number'].eq(frame['address_number'])
        )
        agree = pd.Series(False, index=curated.index)
        if candidate.any():
            pairs = pd.DataFrame(
                {
                    'a': merged.loc[candidate, 'address_street'],
                    'b': frame.loc[candidate, 'address_street'],
                }
            )
            uniq = pairs.drop_duplicates()
            matched = {
                (a, b): match_streets(a, b, similarity_threshold, admin1_id)
                for a, b in zip(uniq['a'], uniq['b'])
            }
            hits = pd.Series(
                [matched[(a, b)] for a, b in zip(pairs['a'], pairs['b'])],
                index=pairs.index,
            )
            agree_index = hits.index[hits]
            agree.loc[agree_index] = True
            corroborated |= agree
            for comp in fillable:
                fill = merged.loc[agree_index, comp].eq('') & frame.loc[
                    agree_index, comp
                ].ne('')
                fill_index = fill.index[fill]
                merged.loc[fill_index, comp] = frame.loc[fill_index, comp]
        disagreeing |= others & ~agree

    # Recipe-configured completion from the run's admin unit (e.g. state: 2
    # fills a missing state with the level-2 code, since most spines carry no
    # state evidence). Level-2 codes are validated against ISO 3166-2 for the
    # unit's country; only rows that already carry another non-street
    # component are completed, so street-only addresses stay untouched.
    for comp, level in (complete_from_admin or {}).items():
        level = int(level)
        if comp not in fillable or len(admin_levels) < level:
            continue
        code = admin_levels[level - 1]
        if level == 2 and admin1_id and code not in get_admin2_codes(admin1_id):
            continue
        others = [c for c in fillable if c != comp]
        fill = merged[comp].eq('') & merged[others].ne('').any(axis=1)
        merged.loc[fill, comp] = code

    # Summarize the disagreeing evidence per row (exact-equal values are
    # never summarized, and rows the fuzzy check accepted are masked out)
    compare = [
        (
            name,
            frame[MATCH_COMPONENTS[0]]
            .str.cat(frame[list(MATCH_COMPONENTS[1:])], sep=' ')
            .str.strip()
            .where(usable[name]),
        )
        for name, frame in frames.items()
    ]
    conflict = _summarize_conflicts(compare, curated.index).where(disagreeing)
    curated[conflict_column] = conflict

    # Format over unique component tuples, then assign and record provenance
    has = base.ne('')
    subset = merged.loc[has]
    keys = list(zip(*(subset[comp] for comp in components)))
    formats = {
        key: harmonize_address_case(
            **{comp: value or None for comp, value in zip(components, key)},
            admin1_id=admin1_id,
        )
        for key in set(keys)
    }
    formatted = pd.Series([formats[k] for k in keys], index=subset.index)

    if output_col not in curated.columns:
        curated[output_col] = pd.Series(pd.NA, index=curated.index, dtype=object)
    curated.loc[formatted.index, output_col] = formatted

    tokens = base.copy()
    tokens[corroborated] = 'reconciled'
    for token in sorted(set(tokens[has])):
        record_source(curated, output_col, has & tokens.eq(token), token)

    state.curated = curated

    if state.verbose:
        counts = tokens[has].value_counts()
        summary = ', '.join(f'{k}={v:,d}' for k, v in counts.items()) or 'none'
        n_conflicts = int(conflict.notna().sum())
        print(
            f'  reconcile_addresses: {output_col} populated -> {summary}, '
            f'conflicts={n_conflicts:,d}'
        )
        if n_conflicts:
            print('    sample conflicts:')
            for sample in conflict.dropna().head(5):
                print(f'      {sample}')
            print(
                '    tip: pairs that denote the same address in different '
                'notations\n    (e.g. HIGHWAY ~ HWY) can be added to '
                'src/openplaces/geo/address_equivalences.csv (kind=match).'
            )

    return state
