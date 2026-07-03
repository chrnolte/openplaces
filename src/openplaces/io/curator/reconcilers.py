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


@_register('resolve_occupancy')
def resolve_occupancy(
    state: CurateState,
    ruleset: str,
    parcel_column: str = 'use_group_combined_parcel',
) -> CurateState:
    """Apply parcel-side corrections over the base occupancy and flag conflicts.

    The base ``occupancy_type`` (from ``infer_occupancy_type``) follows the
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
    ``occupancy_type_conflict`` is a categorical ``"{label}: {class} | ..."``
    summary listing every present occupancy evidence (NSI, FEMA, parcel, and any
    other source in ``occupancy.evidence``) for rows where two or more disagree
    (else null). To keep the column low-cardinality, every non-residential class is
    collapsed into a single bucket label (``occupancy.conflict_other_label``,
    default ``Non-Residential``), so only residential — or
    residential-vs-non-residential — disagreements are surfaced.

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
    # (e.g. Retail vs Hotel) are not. The string lists all present sources in
    # recipe order so agreements and disagreements are both visible.
    from itertools import combinations

    conflict = pd.Series(pd.NA, index=curated.index, dtype=object)
    present = [
        (
            ev.get('label', ev['column']),
            bucket_to_residential(curated[ev['column']], config, class_rules),
        )
        for ev in evidence
        if ev['column'] in curated.columns
    ]
    if len(present) >= 2:
        differ = pd.Series(False, index=curated.index)
        for (_, class_a), (_, class_b) in combinations(present, 2):
            both = class_a.notna() & class_b.notna()
            differ = differ | (both & class_a.ne(class_b))
        if differ.any():
            parts = []
            for label, classes in present:
                mask = classes.notna()
                part = pd.Series(pd.NA, index=curated.index, dtype=object)
                part.loc[mask] = f'{label}: ' + classes[mask].astype(str)
                parts.append(part)
            # Join each row's present "label: class" parts, skipping absent sources.
            stacked = pd.concat(parts, axis=1)
            joined = stacked.apply(
                lambda row: ' | '.join(v for v in row if isinstance(v, str)), axis=1
            )
            conflict.loc[differ] = joined.loc[differ]

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
