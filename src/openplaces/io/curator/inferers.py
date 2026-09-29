"""Registered curation steps that derive new canonical values from evidence."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('derive_metrics', phase='infer')
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
            col == 'total_value'
            or col.startswith('improvement_value')
            or col.startswith('structure_value')
        ):
            values = curated[col]
            if not pd.api.types.is_numeric_dtype(values):
                # A value column can arrive str-typed from an ingest whose
                # source shipped numbers as text (seen: TX txgio
                # improvement_value), and one such column must not crash
                # the whole county's curate. Coerce and warn -- the ingest
                # recipe still owes the registry-declared cast.
                warnings.warn(
                    f'derive_metrics: {col!r} is {values.dtype}, not numeric; '
                    'coercing for the _per_area ratio. Fix the ingest '
                    "recipe's dtype (attribute registry declares float)."
                )
                values = pd.to_numeric(values, errors='coerce')
            curated[f'{col}_per_area'] = values / curated[area_col]

    state.curated = curated
    return state


def _derive_ruleset_class(state: CurateState, spec: dict) -> pd.Series | None:
    """Classify a label column through an ordered ruleset CSV."""
    from openplaces.io.curator.occupancy import load_ruleset, match_ruleset

    curated = state.curated
    column = spec['column']
    if column not in curated.columns:
        return None
    rules = load_ruleset(state, spec['ruleset'], spec.get('class_column'))
    proposal, reviewed = match_ruleset(curated[column].astype(object), rules)
    if spec.get('reviewed_only'):
        # Null out matches whose winning rule is unreviewed rather than
        # pre-filtering the ruleset: pre-filtering would let a term that
        # legitimately matched an unreviewed rule fall through to a later
        # reviewed one, silently changing which class it asserts.
        proposal = proposal.where(reviewed)
    where_null = spec.get('where_null')
    if where_null is not None and where_null in curated.columns:
        # A second reading of the same assessor text must not reach a
        # decision's min_score on its own: the structure description
        # speaks only where the land use named no class.
        proposal = proposal.where(curated[where_null].isna())
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


def _derive_cohort_threshold(state: CurateState, spec: dict) -> pd.Series | None:
    """Express each value relative to a threshold set by a reference cohort.

    For "is this structure big enough to be a dwelling rather than a shed",
    where the answer depends on how big dwellings actually are around here. The
    threshold is a *fraction* of the reference cohort's mean, floored so a
    degenerate cohort cannot drive it to zero, and replaced by a fallback when
    the cohort is too small to average meaningfully.

    Returns the value/threshold ratio, not a boolean -- the vote applies the
    cutoff (typically ``numeric_at_least`` with ``min: 1.0``), keeping this
    layer's contract that indicator columns hold measurements, never
    pre-baked decisions. The parameters here define the reference statistic,
    not the class cutoff. A non-positive threshold yields a missing ratio,
    mirroring the ``ratio`` type's zero-total guard.
    """
    curated = state.curated
    value_column = spec['value_column']
    cohort_column = spec.get('cohort_column')
    if value_column not in curated.columns:
        return None

    values = pd.to_numeric(curated[value_column], errors='coerce')
    cohort = values
    if cohort_column and cohort_column in curated.columns:
        in_cohort = curated[cohort_column].astype(object).eq(spec['cohort_value'])
        cohort = values.where(in_cohort)
    sample = cohort.dropna()

    fraction = float(spec.get('fraction', 0.5))
    floor = float(spec.get('floor', 0.0))
    fallback = float(spec.get('fallback', 0.0))
    min_samples = int(spec.get('min_samples', 3))
    mean = float(sample.mean()) if len(sample) >= min_samples else fallback
    threshold = max(floor, fraction * mean)
    if threshold <= 0:
        return pd.Series(pd.NA, index=curated.index, dtype='Float64')
    return values / threshold


def _derive_value_map(state: CurateState, spec: dict) -> pd.Series | None:
    """Map one column's values through a recipe-supplied table.

    The per-source half of a source-neutral indicator: a source's own
    vocabulary (Florida's twelve sale qualification labels) is graded
    into a value every consumer reads the same way (an arm's-length
    confidence from 0 to 1). The mapping lives in the recipe, since it
    is a statement about that source; a value the mapping does not name
    takes ``default``, or stays missing.

    ``column`` may be a list, for a grade that depends on more than one
    source field (Wisconsin's RETR: whether the conveyance was a sale,
    and how the parties say they are related). The mapping is then keyed
    on the values joined by ``|``, and a row missing any of them has no
    key and stays missing: half an answer is not graded.
    """
    curated = state.curated
    column = spec['column']
    mapping = spec['mapping']
    if isinstance(column, list):
        if any(c not in curated.columns for c in column):
            return None
        parts = [curated[c].astype('string') for c in column]
        joined = parts[0]
        for part in parts[1:]:
            joined = joined + '|' + part
        values = joined.astype(object).where(joined.notna())
        mapped = values.map(mapping)
        default = spec.get('default')
        if default is not None:
            known = values.isin(list(mapping)) | values.isna()
            mapped = mapped.where(known, default)
        return mapped
    if column not in curated.columns:
        return None
    values = curated[column].astype(object)
    mapped = values.map(mapping)
    default = spec.get('default')
    if default is not None:
        mapped = mapped.where(values.isin(list(mapping)), default)
    return mapped


def _derive_minimum(state: CurateState, spec: dict) -> pd.Series | None:
    """Row-wise minimum of several graded ``columns``.

    For a grade that holds only if every component does: a transfer is
    at arm's length only if it is a sale *and* the parties are
    unrelated, so its confidence is the lower of the two grades. A row
    missing any component stays missing, because half an answer is not a
    grade. Skipped when any column is absent.
    """
    curated = state.curated
    columns = spec['columns']
    if any(c not in curated.columns for c in columns):
        return None
    values = curated[columns].apply(pd.to_numeric, errors='coerce')
    return values.min(axis=1).where(values.notna().all(axis=1))


def _derive_group_share(state: CurateState, spec: dict) -> pd.Series | None:
    """Share of a row's group whose ``column`` meets ``predicate``.

    The context a row cannot supply about itself, as a value: the share
    of single-family sales in a county that carry a price says whether
    the county discloses prices at all, which no single sale can say. A
    plain groupby over an id column the row already carries, never a
    spatial operation. ``predicate`` is ``positive`` (default: not
    missing and greater than zero), ``notnull``, or ``above`` a
    ``threshold``; ``restrict`` limits
    both numerator and denominator to rows where a column equals a value
    (the single-family sales), and rows outside it get the group's share
    all the same, so a consumer can read it on any row.
    """
    curated = state.curated
    group_column = spec.get('group_column')
    column = spec['column']
    if column not in curated.columns:
        return None
    if group_column is not None and group_column not in curated.columns:
        return None
    values = pd.to_numeric(curated[column], errors='coerce')
    predicate = spec.get('predicate', 'positive')
    if predicate == 'positive':
        met = values.notna() & (values > 0)
    elif predicate == 'notnull':
        met = values.notna()
    elif predicate == 'above':
        if 'threshold_column' in spec:
            threshold = pd.to_numeric(
                curated[spec['threshold_column']], errors='coerce'
            )
        else:
            threshold = float(spec['threshold'])
        met = values.notna() & (values > threshold)
    else:
        raise ValueError(f'Unknown group_share predicate: {predicate!r}')
    restrict = spec.get('restrict')
    in_scope = pd.Series(True, index=curated.index)
    if restrict:
        if restrict['column'] not in curated.columns:
            return None
        in_scope = curated[restrict['column']].astype(object) == restrict['equals']
    # No group column: the whole table is one group, which is the county
    # when the curate stage runs one admin unit at a time.
    groups = (
        curated[group_column]
        if group_column is not None
        else pd.Series('all', index=curated.index)
    )
    counted = met[in_scope].groupby(groups[in_scope]).sum()
    total = in_scope.groupby(groups).sum()
    share = counted.reindex(total.index).fillna(0) / total.where(total > 0)
    return groups.map(share).astype(float)


def _derive_nominal_floor(state: CurateState, spec: dict) -> pd.Series | None:
    """The price below which a source's values are a recording convention.

    A recorder's nominal consideration is not a price: Florida stamps
    100 dollars on a non-market transfer and North Carolina records 0.
    Which value a source uses is the recorder's choice, so it is read
    off the data rather than assumed. A nominal value is a mass point:
    a single price carrying at least ``min_share`` of all rows (default
    5%) and at least ``isolation`` times (default 20) the rows of every
    other price within ten percent of it. The floor is the largest mass
    point at or below ``cap`` (default 5,000), or 0 when there is none,
    written on every row of the group (the whole table without
    ``group_column``), so a consumer can read what "disclosed" meant
    for this source.

    The share is what tells a convention from a cluster, and it is set
    high on purpose. Measured 2026-09-12 across every ingested source
    with a price: Florida's SDF sales (1.9M rows) and last sales (10.7M
    parcels) carry 11% to 88% of their rows at 0 and 4% to 28% at 100,
    both 0% arm's length; Currituck, Nash and Wilson NC and Wisconsin's
    RETR carry 13% to 46% at 0 and nothing else. Real cheap sales also
    cluster on round numbers, but thinly: Polk County's 700 dollar
    spike is 0.6% of rows and 69% arm's length, Wilson's 1,000 dollar
    spike 0.5%. A 0.5% share would have called those conventions and
    hidden real sales; at 5% only the recorder's own value qualifies.
    """
    curated = state.curated
    column = spec['column']
    if column not in curated.columns:
        return None
    values = pd.to_numeric(curated[column], errors='coerce')
    group_column = spec.get('group_column')
    if group_column is not None and group_column not in curated.columns:
        return None
    groups = (
        curated[group_column]
        if group_column is not None
        else pd.Series('all', index=curated.index)
    )
    cap = float(spec.get('cap', 5000))
    min_share = float(spec.get('min_share', 0.05))
    isolation = float(spec.get('isolation', 20))

    def floor_of(group_values: pd.Series) -> float:
        low = group_values[(group_values >= 0) & (group_values <= cap)]
        if low.empty:
            return 0.0
        counts = low.value_counts()
        floor = 0.0
        for value, count in counts.items():
            if count < min_share * len(group_values):
                break  # value_counts is sorted descending
            near = low[(low > value * 0.9 - 1) & (low < value * 1.1 + 1)]
            neighbors = len(near) - count
            if neighbors * isolation <= count:
                floor = max(floor, float(value))
        return floor

    floors = values.groupby(groups).apply(floor_of)
    return groups.map(floors).astype(float)


_INDICATOR_DERIVATIONS = {
    'ruleset_class': _derive_ruleset_class,
    'value_map': _derive_value_map,
    'minimum': _derive_minimum,
    'group_share': _derive_group_share,
    'nominal_floor': _derive_nominal_floor,
    'pooled_vote': _derive_pooled_vote,
    'ratio': _derive_ratio,
    'shape_metric': _derive_shape_metric,
    'group_statistic': _derive_group_statistic,
    'cohort_threshold': _derive_cohort_threshold,
}


@_register('derive_indicators', phase='infer')
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

    - ``value_map``: map ``column`` through a recipe-supplied ``mapping``
      (value to value), with an optional ``default`` for values the
      mapping does not name. The per-source half of a source-neutral
      indicator: a source's vocabulary graded into a value a consumer can
      threshold (a sale's arm's-length confidence from the source's
      qualification labels).
      ``column`` may be a list: the mapping is then keyed on the values
      joined by ``|``, a row missing any of them stays missing, and
      ``default`` applies only to a complete key the mapping lacks.
    - ``minimum``: the row-wise minimum of several graded ``columns``,
      missing where any is; for a grade that holds only if each of its
      components does.
    - ``group_share``: the share of the row's ``group_column`` cohort (the
      whole table when omitted, which is the admin unit being curated)
      whose ``column`` meets ``predicate`` (``positive``, ``notnull``, or
      ``above`` a ``threshold``, or a per-row ``threshold_column``),
      optionally counting only rows where ``restrict: {column, equals}``
      holds; every row of the group receives the share.
    - ``nominal_floor``: the largest mass point of ``column`` at or below
      ``cap`` (a value carrying ``min_share`` of the rows and ``isolation``
      times its neighbors), per ``group_column`` or for the whole table:
      the recorder's nominal consideration, read off the data.
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
    - ``cohort_threshold``: each ``value_column`` entry as a ratio to a
      threshold derived from a reference cohort's own mean -- ``fraction`` of
      it, at least ``floor``, using ``fallback`` when fewer than
      ``min_samples`` rows match ``cohort_column``/``cohort_value``. For
      "large enough to be a dwelling here", where what counts as large
      depends on the local building stock; the vote applies the cutoff
      (``numeric_at_least`` over the ratio, ``min: 1.0``).

    Parameters
    ----------
    indicators : list of dict
        Ordered specs, each ``{output, type, ...}`` with the type-specific keys
        above. A spec whose input columns are absent is skipped, leaving the
        output column unwritten so downstream indicators simply cast no vote.
        A spec with ``fill_only: true`` writes only where ``output`` is
        still missing, so that several sources, each grading its own
        vocabulary, can contribute to one source-neutral column.
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
        if spec.get('fill_only') and spec['output'] in curated.columns:
            # A second source's contribution to one output: a unit
            # holding both kinds of record keeps what the first wrote.
            existing = curated[spec['output']]
            derived = existing.where(existing.notna(), derived)
        curated[spec['output']] = derived
        written.append(spec['output'])

    state.curated = curated
    if state.verbose:
        summary = ', '.join(written) or 'none'
        print(f'  derive_indicators: wrote {len(written)} column(s) -> {summary}')
    return state


# The steps below lived in this module until 2026-09-29 and are imported
# from here by the tests and the docs; each now lives with its concern.
from openplaces.io.curator.estimators.manufactured_homes import (  # noqa: E402
    classify_manufactured_homes,
    flag_manufactured_home_communities,
)
from openplaces.io.curator.estimators.occupancy import (  # noqa: E402
    impute_occupancy_type,
)
from openplaces.io.curator.estimators.stories import (  # noqa: E402
    derive_stories_from_height,
)
from openplaces.io.curator.evidence import derive_admin_attribute  # noqa: E402
from openplaces.io.curator.group_context import (  # noqa: E402
    derive_group_class_share,
    derive_group_count,
    derive_group_rank,
)

__all__ = [
    'derive_metrics',
    'derive_indicators',
    'derive_stories_from_height',
    'impute_occupancy_type',
    'classify_manufactured_homes',
    'flag_manufactured_home_communities',
    'derive_group_class_share',
    'derive_group_count',
    'derive_group_rank',
    'derive_admin_attribute',
]
