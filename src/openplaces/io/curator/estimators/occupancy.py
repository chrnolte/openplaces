"""Estimate a missing occupancy type from the evidence classes on
the row, through the shared vote vocabulary."""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator import CurateState, _register


def _vote_evidence_class(
    curated,
    evidence: list[dict],
    config: dict,
    rules: list[dict],
) -> tuple[pd.Series, pd.Series]:
    """Weighted consensus vote across the coerced occupancy evidence columns.

    Each present evidence entry casts its ``weight`` (default 1.0) for its
    residential-bucketed class (see
    :func:`~openplaces.io.curator.occupancy.bucket_classes` — the same
    granularity as the ``occupancy_type_conflict`` summary, so agreeing
    non-residential sources pool their votes). The heaviest bucket wins; ties
    fall to the bucket of the earliest listed evidence, preserving the recipe
    ordering as precedence when there is no majority. Returns
    ``(classes, tokens)``: the concrete class is the first-listed winning
    voter's coerced class (a pooled non-residential win still yields a
    specific class), and the token joins the winning voters' labels with '/'.
    """
    from openplaces.io.curator.indicators import vote_dynamic_values
    from openplaces.io.curator.occupancy import bucket_classes, coerce_to_class

    values: dict[str, pd.Series] = {}
    buckets: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for ev in evidence:
        col = ev['column']
        if col not in curated.columns:
            continue
        coerced = coerce_to_class(curated[col], rules)
        label = ev.get('label', col)
        values[label] = coerced
        buckets[label] = bucket_classes(coerced, config)
        weights[label] = float(ev.get('weight', 1.0))
    if not values:
        empty = pd.Series(pd.NA, index=curated.index, dtype=object)
        return empty, empty.copy()

    return vote_dynamic_values(values, weights, buckets=buckets)


@_register('impute_occupancy_type', phase='infer')
def impute_occupancy_type(state: CurateState) -> CurateState:
    """Impute ``occupancy_type`` from ordered evidence, then dwellings.

    Vocabulary, evidence columns, and thresholds all come from the recipe
    ``occupancy`` config block; this step holds no source- or class-specific
    names.

    Sets the base class from ``occupancy.evidence`` and nothing else. Default
    (``evidence_mode: cascade``): walk the entries in priority order, coerce
    each column to a class via the class-map ruleset, and take the first
    non-null (the recipe ordering sets precedence, e.g. a structure source
    before an area source). With ``evidence_mode: vote``: a weighted consensus
    vote across all present evidence, so agreeing lower-priority sources can
    outvote a lone higher-priority one (see :func:`_vote_evidence_class`);
    per-entry ``weight`` (default 1.0) tunes each source's say.

    Three one-shot rules that used to follow it here have moved out to
    ``resolve_by_vote``, where they compete on evidence instead of claiming
    rows first and unopposed:

    - the footprint-geometry manufactured-home signal, which wrote the class
      from shape alone ahead of — and unreachable by — the vote that owns the
      final call, so it bypassed that vote's minimum-area precondition
      entirely. Shape now enters as weighted indicators built from the
      ``aspect_ratio`` that ``derive_indicators`` produces.
    - the ``n_dwellings`` single-family gap-fill, which could only ever fill a
      null and so could never contest a class the evidence vote had already
      assigned — the structural reason single-family was the pipeline's
      least-corroborated class.
    - the secondary-class demotion and its habitable-park-home exception, now
      a ``Secondary`` decision plus a paired community/size indicator on the
      manufactured-home decision. The habitable threshold is expressed as a
      ``cohort_threshold`` indicator, measured against a source classification
      rather than against this step's own output.
    """
    from openplaces.io.curator.occupancy import (
        coerce_to_class,
        get_occupancy_config,
        load_ruleset,
    )
    from openplaces.io.curator.provenance import record_source

    curated = state.curated
    config = get_occupancy_config(state)
    rules = load_ruleset(state, config['class_map'])

    result = pd.Series(pd.NA, index=curated.index, dtype=object)

    # Base class from the evidence columns: weighted consensus vote, or the
    # default first-non-null cascade (recipe order = precedence).
    evidence_list = config.get('evidence', [])
    if config.get('evidence_mode', 'cascade') == 'vote':
        classes, tokens = _vote_evidence_class(curated, evidence_list, config, rules)
        fill = classes.notna()
        result.loc[fill] = classes.loc[fill]
        for token in tokens.loc[fill].dropna().unique():
            record_source(curated, 'occupancy_type', fill & tokens.eq(token), token)
    else:
        for evidence in evidence_list:
            col = evidence['column']
            if col not in curated.columns:
                continue
            coerced = coerce_to_class(curated[col], rules)
            fill = result.isna() & coerced.notna()
            result.loc[fill] = coerced.loc[fill]
            if fill.any():
                record_source(
                    curated, 'occupancy_type', fill, evidence.get('label', col)
                )

    curated['occupancy_type'] = pd.Categorical(result)
    state.curated = curated

    if state.save_statistics:
        from openplaces.io.curator.diagnostics import (
            save_geometry_statistics,
            save_group_dwelling_candidates,
            save_occupancy_evidence_comparison,
        )

        save_geometry_statistics(state)
        save_group_dwelling_candidates(state)
        save_occupancy_evidence_comparison(state)

    if state.verbose:
        counts = curated['occupancy_type'].value_counts(dropna=False)
        print(
            '  impute_occupancy_type: '
            + ', '.join(f'{k}={v:,d}' for k, v in counts.items())
        )
    return state
