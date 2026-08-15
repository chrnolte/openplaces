"""Registered curation steps that derive new canonical values from evidence."""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.curator import CurateState, _register


def _habitable_threshold(curated, result, mh_label, config) -> float:
    """Minimum footprint area (m2) for a park home to count as habitable.

    Realizes "average manufactured-home footage x habitable_fraction": the
    average is taken over the footprints already classed as *mh_label*
    (falling back to the config ``manufactured_home_avg_m2`` when too few
    samples), and the result is floored at ``habitable_floor_m2`` so a
    degenerate average cannot drop it to zero.
    """
    fraction = float(config.get('habitable_fraction', 0.5))
    floor = float(config.get('habitable_floor_m2', 25.0))
    areas = pd.to_numeric(
        curated.loc[result.eq(mh_label), 'area_m2'], errors='coerce'
    ).dropna()
    avg = (
        float(areas.mean())
        if len(areas) >= 3
        else float(config.get('manufactured_home_avg_m2', 90.0))
    )
    return max(floor, fraction * avg)


@_register('derive_metrics')
def derive_metrics(state: CurateState) -> CurateState:
    """Compute polygon area and per-area value ratios.

    Adds a canonical area column (entity-type-aware: ``area_ha`` for parcels,
    ``area_m2`` for others) and, for the canonical ``value`` column plus
    every ``improvement_value*`` / ``structure_value*`` evidence column, a
    matching ``{column}_per_area`` ratio.

    For parcels, ``area_ha`` is computed once during harmonize spine
    assembly (``derive_geometry_attributes``) and carried through here
    unchanged -- not recomputed. For other entities, ``area_m2`` is
    computed here, left missing on synthetic reference-derived rows
    (``geometry_source`` like ``'parcel.spine'``, whose geometry is the
    reference boundary rather than a real outline); their ``_per_area``
    ratios inherit the missing denominator either way.
    """
    from openplaces.core.schema import is_synthetic_geometry
    from openplaces.geo.polygon import get_areas
    from openplaces.io.curator.provenance import SOURCE_SUFFIX

    curated = state.curated

    entity = state.recipe.get('entity')
    entity_type = (
        entity.get('entity_type')
        if isinstance(entity, dict)
        else getattr(entity, 'entity_type', None)
    )
    if entity_type and str(entity_type) == 'parcel':
        area_col = 'area_ha'
    else:
        area_col = 'area_m2'
        area_mask = ~is_synthetic_geometry(curated, entity)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            curated[area_col] = get_areas(curated, unit='m2', mask=area_mask)

    for col in list(curated.columns):
        if col.endswith('_per_area') or col.endswith(SOURCE_SUFFIX):
            continue
        if (
            col == 'value'
            or col.startswith('improvement_value')
            or col.startswith('structure_value')
        ):
            curated[f'{col}_per_area'] = curated[col] / curated[area_col]

    state.curated = curated
    return state


def _match_ruleset(terms: pd.Series, rules: list[dict]) -> tuple[pd.Series, pd.Series]:
    """Apply an ordered label ruleset, returning (matched class, reviewed).

    First match wins, mirroring
    :func:`~openplaces.io.curator.occupancy.coerce_to_class` -- except a term
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


def _derive_ruleset_class(state: CurateState, spec: dict) -> pd.Series | None:
    """Classify a label column through an ordered ruleset CSV."""
    from openplaces.io.curator.occupancy import load_ruleset

    curated = state.curated
    column = spec['column']
    if column not in curated.columns:
        return None
    rules = load_ruleset(state, spec['ruleset'], spec.get('class_column'))
    proposal, reviewed = _match_ruleset(curated[column].astype(object), rules)
    if spec.get('reviewed_only'):
        # Null out matches whose winning rule is unreviewed rather than
        # pre-filtering the ruleset: pre-filtering would let a term that
        # legitimately matched an unreviewed rule fall through to a later
        # reviewed one, silently changing which class it asserts.
        proposal = proposal.where(reviewed)
    return proposal


def _derive_pooled_vote(state: CurateState, spec: dict) -> pd.Series | None:
    """Pool several same-vocabulary columns into one value by weighted vote."""
    from openplaces.io.curator.indicators import vote_dynamic_values

    curated = state.curated
    values: dict[str, pd.Series] = {}
    weights: dict[str, float] = {}
    for entry in spec['columns']:
        column = entry['column']
        if column not in curated.columns:
            continue
        label = entry.get('label', column)
        values[label] = curated[column].astype(object)
        weights[label] = float(entry.get('weight', 1.0))
    if not values:
        return None

    tiebreaker = spec.get('tiebreaker')
    tiebreak_series = None
    if tiebreaker is not None and tiebreaker in curated.columns:
        tiebreak_series = curated[tiebreaker].astype(object)
    winner, _ = vote_dynamic_values(values, weights, tiebreaker=tiebreak_series)
    return winner


def _derive_ratio(state: CurateState, spec: dict) -> pd.Series | None:
    """Express one column as a share of a column sum or of the entity's area."""
    curated = state.curated
    numerator = spec['numerator']
    if numerator not in curated.columns:
        return None
    value = pd.to_numeric(curated[numerator], errors='coerce')

    denominator = spec['denominator']
    if denominator == 'own_area':
        from openplaces.geo.polygon import resolve_area

        # resolve_area returns a plain array; index it to align with value.
        total = pd.Series(resolve_area(curated, unit='m2'), index=curated.index)
    else:
        columns = [denominator] if isinstance(denominator, str) else denominator
        if any(c not in curated.columns for c in columns):
            return None
        total = sum(pd.to_numeric(curated[c], errors='coerce') for c in columns)
    # Guard a zero or negative total the same way the value_share
    # indicators do, so the ratio is missing rather than infinite.
    return value.where(total > 0) / total.where(total > 0)


def _derive_shape_metric(state: CurateState, spec: dict) -> pd.Series | None:
    """Measure a minimum-bounding-rectangle dimension of each geometry."""
    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims

    curated = state.curated
    if 'geometry' not in curated.columns or curated.empty:
        return None

    # get_oriented_dims measures raw coordinates, so the geometry must be
    # metric first -- on lon/lat degrees the x axis is compressed by
    # cos(latitude) and every dimension (aspect ratio included) is skewed.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        geometry = curated.geometry.to_crs(local_metric_crs(curated))
    dims = geometry.map(get_oriented_dims)
    length = dims.map(lambda d: d[1])
    width = dims.map(lambda d: d[2])

    metric = spec.get('metric', 'aspect_ratio')
    if metric == 'aspect_ratio':
        return length / width.clip(lower=1e-6)
    if metric == 'length':
        return length
    if metric == 'width':
        return width
    if metric == 'orientation':
        return dims.map(lambda d: d[0])
    raise ValueError(
        f'Unknown shape_metric {metric!r}; expected aspect_ratio, length, '
        f'width, or orientation.'
    )


def _derive_group_statistic(state: CurateState, spec: dict) -> pd.Series | None:
    """Score a value against the distribution of its own cohort."""
    import numpy as np

    curated = state.curated
    group_column = spec['group_column']
    value_column = spec['value_column']
    if group_column not in curated.columns or value_column not in curated.columns:
        return None

    value = pd.to_numeric(curated[value_column], errors='coerce')
    transform = spec.get('transform', 'log1p')
    if transform == 'log1p':
        value = np.log1p(value)
    elif transform is not None:
        raise ValueError(f"Unknown transform {transform!r}; expected 'log1p' or None.")

    grouped = value.groupby(curated[group_column], observed=True)
    statistic = spec.get('statistic', 'zscore')
    if statistic == 'zscore':
        mean = grouped.transform('mean')
        # A zero-variance cohort scores missing rather than dividing by 0.
        std = grouped.transform('std', ddof=0).replace(0, np.nan)
        return (value - mean) / std
    if statistic == 'percentile':
        return grouped.rank(pct=True)
    raise ValueError(
        f"Unknown statistic {statistic!r}; expected 'zscore' or 'percentile'."
    )


_INDICATOR_DERIVATIONS = {
    'ruleset_class': _derive_ruleset_class,
    'pooled_vote': _derive_pooled_vote,
    'ratio': _derive_ratio,
    'shape_metric': _derive_shape_metric,
    'group_statistic': _derive_group_statistic,
}


@_register('derive_indicators')
def derive_indicators(state: CurateState, indicators: list[dict]) -> CurateState:
    """Compute the named precursor columns the voting steps score against.

    The derivation half of the curate stage's two-layer classification: this
    step produces indicator *columns*, and
    :func:`~openplaces.io.curator.reconcilers.resolve_by_vote` turns them into
    a class. Each spec carries an ``output`` column name and a ``type`` from a
    small closed vocabulary, mirroring how
    :func:`~openplaces.io.curator.indicators.evaluate_indicator` dispatches its
    own predicate types -- so both layers read the same way.

    Deriving a value once and naming it is what keeps a threshold from being
    restated per rule: recipes reference ``aspect_ratio`` or ``keyword_class``
    by name, and the cutoff applied to it lives in exactly one place, the vote.
    Values are deliberately *not* thresholded here -- an indicator column holds
    a measurement or a label, never a pre-baked boolean, so the vote layer
    stays the single home of every cutoff.

    Supported ``type`` values:

    - ``ruleset_class``: classify ``column`` through an ordered ruleset CSV
      (``ruleset``), first match wins; unmatched rows are missing. With
      ``reviewed_only`` true, only rows whose winning rule is marked reviewed
      keep their class -- the high-confidence subset.
    - ``pooled_vote``: pool several same-vocabulary ``columns``
      (``{column, label, weight}``) into one value by weighted vote, with an
      optional ``tiebreaker`` column.
    - ``ratio``: ``numerator`` over the sum of ``denominator`` columns, or over
      the entity's own area when ``denominator`` is ``own_area``. A
      non-positive total yields a missing ratio.
    - ``shape_metric``: a minimum-bounding-rectangle ``metric`` of each
      geometry -- ``aspect_ratio`` (default), ``length``, ``width``, or
      ``orientation``. Measured on a locally-projected metric copy.
    - ``group_statistic``: score ``value_column`` against its ``group_column``
      cohort via ``statistic`` (``zscore`` default, or ``percentile``), after
      an optional ``transform`` (``log1p`` default, or None).

    Parameters
    ----------
    indicators : list of dict
        Ordered specs, each ``{output, type, ...}`` with the type-specific keys
        above. A spec whose input columns are absent is skipped, leaving the
        output column unwritten so downstream indicators simply cast no vote.
    """
    curated = state.curated
    written = []
    for spec in indicators:
        kind = spec['type']
        if kind not in _INDICATOR_DERIVATIONS:
            raise ValueError(
                f'Unknown indicator derivation type: {kind!r}. Expected one of '
                f'{", ".join(sorted(_INDICATOR_DERIVATIONS))}.'
            )
        derived = _INDICATOR_DERIVATIONS[kind](state, spec)
        if derived is None:
            continue
        curated[spec['output']] = derived
        written.append(spec['output'])

    state.curated = curated
    if state.verbose:
        summary = ', '.join(written) or 'none'
        print(f'  derive_indicators: wrote {len(written)} column(s) -> {summary}')
    return state


@_register('derive_stories_from_height')
def derive_stories_from_height(
    state: CurateState,
    column: str = 'n_stories_footprint_fema',
    height_column: str = 'height_footprint_fema',
    floor_height_m: float = 3.05,
) -> CurateState:
    """Derive a story count from a measured building height.

    Approximates the story count as ``height / floor_height_m``, rounded and
    floored at one story. A missing or non-positive height yields a missing
    story count rather than a fabricated minimum.

    Parameters
    ----------
    state : CurateState
        The curation state with the target GeoDataFrame in state.curated.
    column : str, optional
        Output column name for the derived story count.
    height_column : str, optional
        Source column holding measured building height (metres). No-op if
        absent from ``state.curated``.
    floor_height_m : float, optional
        Assumed height per story, in metres.
    """
    curated = state.curated
    if height_column not in curated.columns:
        return state

    height = pd.to_numeric(curated[height_column], errors='coerce')
    stories = (height / floor_height_m).round().clip(lower=1)
    curated[column] = stories.where(height > 0)

    state.curated = curated
    return state


@_register('classify_parcel_land_use')
def classify_parcel_land_use(
    state: CurateState,
    rules: list[dict],
    output: str = 'land_use_class',
    flag_column: str | None = None,
    flag_class: str | None = None,
    score_columns: dict[str, str] | None = None,
    review_column: str | None = None,
    review_margin: float = 1.0,
) -> CurateState:
    """Classify each parcel's land-use class by weighted indicator voting.

    .. deprecated::
        Kept only so recipes not yet migrated keep working. This is now a thin
        wrapper over :func:`~openplaces.io.curator.reconcilers.resolve_by_vote`
        with ``preserve_base=False`` and ``default_source='rule'``, which is
        exactly what it always did — the parcel-curation "rules" and the
        occupancy "decisions" were the same weighted vote written twice. Point
        recipes at ``resolve_by_vote`` directly; this wrapper is removed once
        none reference it.

    Parameters
    ----------
    rules : list of dict
        Ordered candidate classes, each ``{class, min_score, indicators, weight}``;
        ``indicators`` is a list of specs (see
        :func:`~openplaces.io.curator.indicators.evaluate_indicator`). For each
        parcel a rule's score sums its matched indicator weights; among the rules
        reaching ``min_score`` (default 1) the highest score wins, ties broken by
        order. Parcels matching no rule are left null.
    output : str, optional
        Output class column (default ``land_use_class``).
    flag_column, flag_class : str, optional
        When both are given, write a boolean ``flag_column`` set where ``output``
        equals ``flag_class`` (e.g. ``manufactured_home_community``).
    score_columns : dict of {class: column}, optional
        For named classes, also write that rule's raw weighted score — not
        gated by its own ``min_score`` — to the given column. This is an
        ordered/graded signal (e.g. a vacancy likelihood) usable by later
        steps even for parcels where that class did not win the vote.
    review_column : str, optional
        Boolean column flagging parcels where the winning class's score beat
        the runner-up's (among rules that individually reached their own
        ``min_score``) by less than *review_margin* — the "unresolved"
        confusion cases worth a second look (e.g. Manufactured Home Park vs.
        RV Park scoring close or tied). Left unset (``None``) by default.
    review_margin : float, optional
        Score margin below which ``review_column`` is set (default 1.0).
    """
    from openplaces.io.curator.reconcilers import resolve_by_vote

    return resolve_by_vote(
        state,
        target=output,
        decisions=rules,
        preserve_base=False,
        default_source='rule',
        score_columns=score_columns,
        review_column=review_column,
        review_margin=review_margin,
        flag_column=flag_column,
        flag_class=flag_class,
    )


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


@_register('impute_occupancy_type')
def impute_occupancy_type(state: CurateState) -> CurateState:
    """Impute ``occupancy_type`` from ordered evidence, then dwellings.

    Vocabulary, evidence columns, and thresholds all come from the recipe
    ``occupancy`` config block; this step holds no source- or class-specific
    names.

    1. Base class from ``occupancy.evidence``. Default (``evidence_mode:
       cascade``): walk the entries in priority order, coerce each column to a
       class via the class-map ruleset, and take the first non-null (the recipe
       ordering sets precedence, e.g. a structure source before an area
       source). With ``evidence_mode: vote``: a weighted consensus vote across
       all present evidence, so agreeing lower-priority sources can outvote a
       lone higher-priority one (see :func:`_vote_evidence_class`); per-entry
       ``weight`` (default 1.0) tunes each source's say.
    2. Non-primary ``residential_classes`` footprints become ``secondary_class``.

    Two one-shot rules that used to sit inside this step have moved out to
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

    Multi-Family is later refined into height bands by ``refine_occupancy_height``.
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
    rule_cfg = config.get('rules', {})

    result = pd.Series(pd.NA, index=curated.index, dtype=object)

    # 1. Base class from the evidence columns: weighted consensus vote, or the
    #    default first-non-null cascade (recipe order = precedence).
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

    # 2. Non-primary residential footprints become the secondary class — as do
    #    explicit-secondary footprints with no occupancy evidence (an accessory
    #    structure on a parcel whose primary building is elsewhere). A non-primary
    #    footprint with a known non-residential class keeps that class. Exception:
    #    habitable-size homes on a manufactured-home-community parcel are the
    #    community's dwellings (Manufactured Home), not accessory structures;
    #    only sub-threshold footprints there (sheds) stay secondary. This flag
    #    is set by the parcel curation lane (classify_parcel_land_use) and
    #    joined in by link_curated_entity, under the same name
    #    flag_manufactured_home_communities later refines from the final
    #    footprint occupancy -- read here before that refinement runs, so this
    #    step still sees the parcel-lane's one-pass value. Absent either,
    #    behaviour is unchanged.
    residential = list(config.get('residential_classes', []))
    secondary = config.get('secondary_class')
    mh_label = rule_cfg.get('manufactured_home_geometry', {}).get('class')
    if residential and secondary and 'priority_on_parcel' in curated.columns:
        priority = curated['priority_on_parcel'].astype(object)
        non_primary = (result.isin(residential) & ~priority.eq('primary')) | (
            result.isna() & priority.eq('secondary')
        )
        in_park = (
            curated['manufactured_home_community'].astype('boolean').fillna(False)
            if 'manufactured_home_community' in curated.columns
            else pd.Series(False, index=curated.index)
        )
        to_mh = pd.Series(False, index=curated.index)
        if mh_label and bool(in_park.any()) and 'area_m2' in curated.columns:
            threshold = _habitable_threshold(curated, result, mh_label, config)
            habitable = (
                pd.to_numeric(curated['area_m2'], errors='coerce') >= threshold
            ).fillna(False)
            to_mh = non_primary & in_park & habitable
            if to_mh.any():
                result.loc[to_mh] = mh_label
                record_source(curated, 'occupancy_type', to_mh, 'park')
        sec_mask = non_primary & ~to_mh
        result.loc[sec_mask] = secondary
        record_source(curated, 'occupancy_type', sec_mask, 'secondary')

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


@_register('flag_manufactured_home_communities')
def flag_manufactured_home_communities(
    state: CurateState,
    min_homes: int = 3,
    output: str = 'manufactured_home_community',
    count_column: str = 'n_manufactured_homes_per_parcel',
) -> CurateState:
    """Flag parcels with more than *min_homes* manufactured-home footprints.

    Recomputed from the FINAL footprint occupancy (after imagery, vote, and height
    refinement), so it reflects the richest manufactured-home evidence — a
    correction the one-pass parcel lane cannot see, since it runs before footprint
    curation. Written as footprint columns: a per-parcel count and a boolean flag,
    under the same name (*output*, default ``manufactured_home_community``) the
    parcel curation lane's own ``classify_parcel_land_use`` flag uses -- this
    step's value is the intentional final word, overwriting whatever
    ``link_curated_entity`` relayed from the parcel lane earlier in this
    recipe (already consumed by then, see ``impute_occupancy_type``). A
    future second parcel pass can write this correction back to the parcel
    dataset.

    Parameters
    ----------
    min_homes : int, optional
        A parcel is a community when it carries strictly more than this many
        manufactured-home footprints (default 3, i.e. 4+).
    output : str, optional
        Boolean community-flag column (default ``manufactured_home_community``).
    count_column : str, optional
        Per-parcel manufactured-home count column
        (default ``n_manufactured_homes_per_parcel``).
    """
    from openplaces.io.curator.occupancy import get_occupancy_config

    curated = state.curated
    # Prefer the globally-unique parcel_id over locally-scoped fallbacks (see
    # link_curated_entity for why); grouping by a non-unique key would
    # silently pool unrelated parcels' counts together.
    parcel_col = next(
        (
            c
            for c in (
                'parcel_id',
                'parcel_id_local',
                'parcel_id_tax',
                'parcel_id_assessor',
            )
            if c in curated.columns
        ),
        None,
    )
    if parcel_col is None or 'occupancy_type' not in curated.columns:
        return state

    config = get_occupancy_config(state)
    mh_label = (
        config.get('rules', {})
        .get('manufactured_home_geometry', {})
        .get('class', 'Manufactured Home')
    )
    is_mh = curated['occupancy_type'].astype(object).eq(mh_label).astype(int)
    counts = is_mh.groupby(curated[parcel_col]).transform('sum')
    curated[count_column] = counts.fillna(0).astype('int64')
    curated[output] = (curated[count_column] > min_homes).to_numpy()
    state.curated = curated

    if state.verbose:
        n_comm = int(curated.loc[curated[output], parcel_col].nunique())
        print(
            f'  flag_manufactured_home_communities: {n_comm:,} community parcels '
            f'(> {min_homes} manufactured-home footprints).'
        )
    return state


def _score_manufactured_home_candidates(
    work,
    assessor_labels,
    *,
    mh_label,
    sf_label,
    aspect_min,
    area_max,
    plausible_aspect_min,
    plausible_area_max_m2,
    model_type,
    min_training_samples,
    verbose,
):
    """Score candidate footprints for manufactured-home probability.

    Runs the heavy morphology, neighborhood, and model work on the candidate
    subset only. Geometry is reprojected to a local metre-based CRS (centered on
    the data) so distances — ``dwithin`` queries and nearest neighbor — are
    correct regardless of the stored CRS. Returns a dict with the
    ``p_manufactured_home`` Series indexed like *work*.
    """
    import numpy as np
    import pandas as pd
    from shapely.strtree import STRtree
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims

    orig_index = work.index
    work = work.reset_index(drop=True)
    assessor_labels = pd.Series(
        np.asarray(assessor_labels, dtype=object), index=work.index
    )
    n_rows = len(work)

    # Metric geometry for every distance/shape computation.
    geom = work.geometry.to_crs(local_metric_crs(work))
    area = pd.to_numeric(work['area_m2'], errors='coerce').to_numpy()
    perimeter = geom.length.values
    perimeter_sq = np.clip(perimeter**2, a_min=1e-6, a_max=None)
    compactness = 4 * np.pi * area / perimeter_sq

    dims = geom.map(get_oriented_dims)
    angle = np.array([x[0] for x in dims])
    length = np.array([x[1] for x in dims])
    width = np.array([x[2] for x in dims])
    width_clip = np.clip(width, a_min=1e-6, a_max=None)
    aspect_ratio = length / width_clip
    rectangularity = area / np.clip(length * width, a_min=1e-6, a_max=None)
    n_vertices = geom.map(
        lambda g: len(g.exterior.coords) - 1 if hasattr(g, 'exterior') else 0
    ).values

    # Prefer the globally-unique parcel_id over locally-scoped fallbacks, same
    # rationale as flag_manufactured_home_communities above. An explicit
    # preference tuple, not work.columns iteration order, so this doesn't
    # depend on incidental column placement.
    parcel_col = next(
        (
            c
            for c in (
                'parcel_id',
                'parcel_id_local',
                'parcel_id_tax',
                'parcel_id_assessor',
            )
            if c in work.columns
        ),
        None,
    )
    if parcel_col is not None and work[parcel_col].notna().any():
        n_structures_on_parcel = (
            work[parcel_col].map(work[parcel_col].value_counts()).values
        )
    else:
        n_structures_on_parcel = np.ones(n_rows)

    tree = STRtree(geom.values)
    idx_query, idx_tree = tree.query(geom.values, predicate='dwithin', distance=100.0)
    df_pairs = pd.DataFrame({'query_idx': idx_query, 'tree_idx': idx_tree})
    df_pairs_no_self = df_pairs[df_pairs['query_idx'] != df_pairs['tree_idx']]

    # Per-row neighbor statistics over the 100m pairs (positional, empty-safe).
    is_elongated = (aspect_ratio >= aspect_min) & (area <= area_max)
    q = df_pairs_no_self['query_idx'].to_numpy()
    t = df_pairs_no_self['tree_idx'].to_numpy()
    if len(q):
        index = range(n_rows)
        density = np.bincount(q, minlength=n_rows)[:n_rows]
        size_std = (
            pd.Series(area[t]).groupby(q).std(ddof=0).reindex(index, fill_value=0.0)
        ).to_numpy()
        orientation_std = (
            pd.Series(angle[t]).groupby(q).std(ddof=0).reindex(index, fill_value=0.0)
        ).to_numpy()
        share_elongated = (
            pd.Series(is_elongated[t].astype(float))
            .groupby(q)
            .mean()
            .reindex(index, fill_value=0.0)
        ).to_numpy()
    else:
        density = np.zeros(n_rows, dtype=int)
        size_std = np.zeros(n_rows)
        orientation_std = np.zeros(n_rows)
        share_elongated = np.zeros(n_rows)

    idx_q2, idx_t2 = tree.query(geom.values, predicate='dwithin', distance=200.0)
    df_pairs2 = pd.DataFrame({'query_idx': idx_q2, 'tree_idx': idx_t2})
    df_pairs2 = df_pairs2[df_pairs2['query_idx'] != df_pairs2['tree_idx']]
    if len(df_pairs2) > 0:
        left = geom.iloc[df_pairs2['query_idx'].to_numpy()].reset_index(drop=True)
        right = geom.iloc[df_pairs2['tree_idx'].to_numpy()].reset_index(drop=True)
        df_pairs2['distance'] = left.distance(right).to_numpy()
        nn_dist = (
            df_pairs2.groupby('query_idx')['distance']
            .min()
            .reindex(range(n_rows), fill_value=200.0)
            .values
        )
    else:
        nn_dist = np.full(n_rows, 200.0)

    features = pd.DataFrame(
        {
            'area': area,
            'perimeter': perimeter,
            'compactness': compactness,
            'length': length,
            'width': width,
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'n_vertices': n_vertices,
            'n_structures_on_parcel': n_structures_on_parcel,
            'local_density': density,
            'size_std': size_std,
            'orientation_std': orientation_std,
            'share_elongated': share_elongated,
            'nn_dist': nn_dist,
        }
    )

    # Tier 3: imagery predictions, if present.
    cv_prob = pd.Series(np.nan, index=work.index)
    cv_cols = [
        'p_manufactured_home_cv',
        'manufactured_home_cv',
        'occupancy_brails',
        'occupancy_type_brails',
    ]
    cv_col = next((c for c in cv_cols if c in work.columns), None)
    if cv_col is not None:
        if pd.api.types.is_numeric_dtype(work[cv_col]):
            cv_prob = pd.to_numeric(work[cv_col], errors='coerce')
        else:
            terms = work[cv_col].astype(str).str.upper()
            cv_prob.loc[terms.str.contains('MANUFACTURED|MOBILE', na=False)] = 1.0
            cv_prob.loc[terms.str.contains('SINGLE FAMILY|SINGLE-FAMILY', na=False)] = (
                0.0
            )

    # Tier 2: local footprint-morphology model.
    train_mask = assessor_labels.notna()
    n_mfg = int((assessor_labels == mh_label).sum())
    n_sf = int((assessor_labels == sf_label).sum())

    xgb_available = False
    if model_type == 'xgboost':
        try:
            from xgboost import XGBClassifier

            xgb_available = True
        except ImportError:
            model_type = 'calibrated_logistic'

    X = features.fillna(0.0)
    p_mfg_morph = pd.Series(0.0, index=work.index)
    model_trained = False

    if n_mfg >= min_training_samples and n_sf >= min_training_samples:
        X_train = X[train_mask.to_numpy()]
        y_train = (assessor_labels[train_mask] == mh_label).astype(int)

        if model_type == 'calibrated_logistic':
            model = LogisticRegression(
                solver='liblinear', max_iter=1000, random_state=42
            )
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            base_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=3)
        elif model_type == 'svm':
            model = SVC(probability=True, random_state=42)
        elif model_type == 'xgboost' and xgb_available:
            from xgboost import XGBClassifier

            model = XGBClassifier(random_state=42, eval_metric='logloss')
        else:
            model = LogisticRegression(
                solver='liblinear', max_iter=1000, random_state=42
            )

        try:
            model.fit(X_train, y_train)
            if isinstance(model, LogisticRegression):
                coef = model.coef_[0]
                intercept = model.intercept_[0]
                z = X.mul(coef, axis=1).sum(axis=1) + intercept
                probs = 1.0 / (1.0 + np.exp(-z.to_numpy()))
                p_mfg_morph = pd.Series(probs, index=work.index)
            elif hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                p_mfg_morph = pd.Series(probs[:, 1], index=work.index)
            else:
                p_mfg_morph = pd.Series(model.predict(X), index=work.index, dtype=float)
            model_trained = True
        except Exception as e:
            if verbose:
                print(
                    f'    Failed to fit model {model_type}: {e}. '
                    f'Falling back to rule-based morphology.'
                )

    if not model_trained:
        # Fallback rule-based morphology classifier.
        aspect = X['aspect_ratio']
        area_val = X['area']
        aspect_score = np.clip((aspect - 1.5) / 1.0, 0, 1) * 0.5
        area_score = np.clip((plausible_area_max_m2 - area_val) / 100.0, 0, 1) * 0.5
        p_mfg_morph = aspect_score + area_score
        model_type = 'rule_based_fallback'

    # Integrate all tiers into a probability and provenance (vectorized).
    plausible_dim = (features['area'] <= plausible_area_max_m2) & (
        features['aspect_ratio'] >= plausible_aspect_min
    )

    is_candidate = (assessor_labels == mh_label) | (p_mfg_morph >= 0.5)
    idx_q, idx_t = tree.query(geom.values, predicate='dwithin', distance=100.0)
    df_p = pd.DataFrame({'query_idx': idx_q, 'tree_idx': idx_t})
    df_p = df_p[df_p['query_idx'] != df_p['tree_idx']]
    if len(df_p) > 0:
        cand_vals = is_candidate.to_numpy().astype(int)
        nearby_cands = (
            pd.Series(cand_vals[df_p['tree_idx'].to_numpy()])
            .groupby(df_p['query_idx'].to_numpy())
            .sum()
            .reindex(range(n_rows), fill_value=0)
            .to_numpy()
        )
    else:
        nearby_cands = np.zeros(n_rows, dtype=int)
    cluster_support = (nearby_cands >= 2) | (
        features['n_structures_on_parcel'].to_numpy() >= 3
    )

    assessor_mh = (assessor_labels == mh_label).to_numpy()
    assessor_sf = (assessor_labels == sf_label).to_numpy()
    morph = p_mfg_morph.to_numpy()
    cv = cv_prob.to_numpy(dtype=float)
    plausible = plausible_dim.to_numpy()

    rule_sf = assessor_sf & ~((morph > 0.75) & plausible)
    decided = assessor_mh | rule_sf
    cv_valid = ~decided & ~np.isnan(cv) & (cv >= 0.5) & plausible
    morph_cluster = ~decided & ~cv_valid & (morph >= 0.5) & cluster_support

    conditions = [assessor_mh, rule_sf, cv_valid, morph_cluster]
    p_mfg_out = pd.Series(
        np.select(
            conditions, [np.ones(n_rows), np.zeros(n_rows), cv, morph], default=morph
        ),
        index=work.index,
    )

    return {
        'p_manufactured_home': pd.Series(p_mfg_out.to_numpy(), index=orig_index),
    }


@_register('classify_manufactured_homes')
def classify_manufactured_homes(
    state: CurateState,
    ruleset: str | None = None,
    model_type: str = 'calibrated_logistic',
    min_training_samples: int = 10,
    plausible_aspect_min: float = 1.8,
    plausible_area_max_m2: float = 250.0,
    update_occupancy: bool = False,
) -> CurateState:
    """Estimate manufactured vs single-family probability from a 3-tier model.

    Emits the ``p_manufactured_home`` evidence column. Class names and the
    geometry thresholds come from the recipe ``occupancy`` block; assessor labels
    come from the shared keyword *ruleset*, so this step stays vocabulary-neutral.
    By default it does not assign ``occupancy_type`` — the generic
    ``resolve_by_vote`` step weighs ``p_manufactured_home`` against the other
    indicators and makes the canonical call.

    Parameters
    ----------
    state : CurateState
        The curation state with the target GeoDataFrame in state.curated.
    ruleset : str, optional
        Filename of the keyword ruleset CSV (beside the curate recipe) used to
        derive Tier 1 assessor labels from the parcel use column. When omitted,
        Tier 1 is inactive and the morphology model relies on its fallback.
    model_type : str, optional
        Type of footprint morphology classifier:
        'calibrated_logistic' (default), 'random_forest', 'gradient_boosting',
        'svm', or 'xgboost' (if installed).
    min_training_samples : int, optional
        Minimum number of assessor-labeled structures for each class to train
        the local morphology model. If not met, falls back to a rule-based
        scoring model.
    plausible_aspect_min, plausible_area_max_m2 : float, optional
        Relaxed geometry envelope (aspect ratio at least, area at most) used to
        gate imagery/morphology overrides of assessor labels.
    update_occupancy : bool, optional
        If True, also writes the manufactured/single-family call straight into
        ``occupancy_type``. Default False: leave that to ``resolve_by_vote``.
    """
    import pandas as pd

    from openplaces.io.curator.occupancy import (
        coerce_to_class,
        get_occupancy_config,
        load_ruleset,
    )

    curated = state.curated
    if curated.empty:
        return state

    # Vocabulary and thresholds from the recipe occupancy block.
    config = get_occupancy_config(state)
    rule_cfg = config.get('rules', {})
    geom_rule = rule_cfg.get('manufactured_home_geometry', {})
    mh_label = geom_rule.get('class', 'Manufactured Home')
    sf_label = rule_cfg.get('single_family_dwellings', {}).get('class', 'Single-Family')
    aspect_min = float(geom_rule.get('aspect_min', 2.5))
    area_max = float(geom_rule.get('area_max_m2', 185.0))

    # --- Tier 1 assessor labels (cheap keyword coercion; computed for all rows
    # so the candidate gate and the morphology model can both use them) ---
    assessor_labels = pd.Series(pd.NA, index=curated.index, dtype=object)
    parcel_use_cols = [
        'use_subgroup_parcel',
        'use_group_combined_parcel',
        'use_group_parcel',
        'use_subgroup',
        'use_group_combined',
        'use_group',
    ]
    use_col = next((c for c in parcel_use_cols if c in curated.columns), None)
    if use_col is not None and ruleset is not None:
        coerced = coerce_to_class(curated[use_col], load_ruleset(state, ruleset))
        assessor_labels = coerced.where(coerced.isin([mh_label, sf_label]))

    if 'area_m2' not in curated.columns:
        from openplaces.core.schema import is_synthetic_geometry
        from openplaces.geo.polygon import get_areas

        area_mask = ~is_synthetic_geometry(curated, state.recipe.get('entity'))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            curated['area_m2'] = get_areas(curated, unit='m2', mask=area_mask)

    # --- Candidate gate ---
    # Manufactured vs single-family discrimination only applies to small
    # residential footprints; structures that are large, or positively classified
    # as non-residential, are never manufactured homes. Restrict the heavy
    # morphology / neighborhood / model / clustering work to candidates (plus any
    # assessor-labeled row, so the model keeps both-class training data). Each
    # clause filters: `is_small` drops large structures, `not_nonresidential`
    # drops known commercial/industrial, `has_label` adds back labeled rows.
    residential_classes = set(config.get('residential_classes', []))
    # Secondary footprints (non-primary structures, common for park homes the
    # priority rule demoted) must stay eligible so they get a real probability.
    secondary_class = config.get('secondary_class')
    keep_classes = residential_classes | (
        {secondary_class} if secondary_class else set()
    )
    area_vals = pd.to_numeric(curated['area_m2'], errors='coerce')
    is_small = area_vals <= plausible_area_max_m2
    if 'occupancy_type' in curated.columns and residential_classes:
        occ = curated['occupancy_type'].astype(object)
        # Unknown occupancy is kept (cannot be ruled out); only a known class
        # outside the residential/secondary set is excluded.
        not_nonresidential = occ.isna() | occ.isin(keep_classes)
    else:
        group_col = next(
            (
                c
                for c in ('use_group_combined_parcel', 'use_group', 'purpose_group')
                if c in curated.columns
            ),
            None,
        )
        if group_col is not None:
            grp = curated[group_col].astype(object)
            not_nonresidential = grp.isna() | grp.str.startswith('Residential').fillna(
                False
            )
        else:
            not_nonresidential = pd.Series(True, index=curated.index)
    has_label = assessor_labels.notna()
    candidate = (is_small & not_nonresidential) | has_label

    if state.verbose:
        print(
            f'  classify_manufactured_homes: {int(candidate.sum()):,} candidates of '
            f'{len(curated):,} structures (small={int(is_small.sum()):,}, '
            f'not-nonresidential={int(not_nonresidential.sum()):,}, '
            f'assessor-labeled={int(has_label.sum()):,}); model_type={model_type}'
        )

    # Full-length output defaults to "not a manufactured home"; candidates are
    # scored on a reprojected metric geometry inside the helper.
    p_mfg_out = pd.Series(0.0, index=curated.index)
    if candidate.any():
        scored = _score_manufactured_home_candidates(
            curated.loc[candidate],
            assessor_labels.loc[candidate],
            mh_label=mh_label,
            sf_label=sf_label,
            aspect_min=aspect_min,
            area_max=area_max,
            plausible_aspect_min=plausible_aspect_min,
            plausible_area_max_m2=plausible_area_max_m2,
            model_type=model_type,
            min_training_samples=min_training_samples,
            verbose=state.verbose,
        )
        idx = scored['p_manufactured_home'].index
        p_mfg_out.loc[idx] = scored['p_manufactured_home']

    curated['p_manufactured_home'] = p_mfg_out.astype(float)
    p_sf_out = 1.0 - p_mfg_out

    # --- Optional: write the call straight into occupancy_type ---
    # Off by default: resolve_by_vote owns the canonical occupancy decision and
    # weighs p_manufactured_home against the other indicators.
    if update_occupancy:
        mask_mfg = p_mfg_out >= 0.5
        mask_sf = (p_sf_out >= 0.5) & (curated.get('occupancy_type') == mh_label)

        occ = (
            curated['occupancy_type'].astype(object).copy()
            if 'occupancy_type' in curated.columns
            else pd.Series(pd.NA, index=curated.index, dtype=object)
        )
        occ.loc[mask_mfg] = mh_label
        occ.loc[mask_sf] = sf_label
        curated['occupancy_type'] = pd.Categorical(occ)
        from openplaces.io.curator.provenance import record_source

        record_source(curated, 'occupancy_type', mask_mfg | mask_sf, 'classifier')

    state.curated = curated
    if state.verbose:
        mfg_cnt = int((p_mfg_out >= 0.5).sum())
        sf_cnt = int((p_sf_out >= 0.5).sum())
        print(
            f'  classify_manufactured_homes: classified {mfg_cnt:,} manufactured '
            f'homes and {sf_cnt:,} single family homes.'
        )

    return state
