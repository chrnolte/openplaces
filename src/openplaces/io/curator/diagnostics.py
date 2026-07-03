"""Curation diagnostics: geometry-indicator distributions and use-group
separability, written to the cache to interrogate occupancy and geometry
decisions.

Enabled by ``save_statistics`` on the curate entrypoint (or a recipe-level
``save_statistics: true``). These functions never modify the curated output;
they only write CSV summaries to the cache, and a failure is warned about rather
than raised so it can never break a curation run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Tail-weighted quantile grid (>=10 points, finer in the tails) for the
# geometry-indicator distributions.
_QUANTILES = [
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
]

# Geometry indicators interrogated for use-group separability.
_INDICATORS = [
    'area_m2',
    'aspect_ratio',
    'rectangularity',
    'compactness',
    'convexity',
    'perimeter_m',
    'n_vertices',
    'n_stories',
    'volume_m3',
]


def _count_vertices(geom) -> int:
    """Return the number of exterior vertices of a (multi)polygon."""
    if geom is None or geom.is_empty:
        return 0
    if getattr(geom, 'exterior', None) is not None:
        return len(geom.exterior.coords)
    if hasattr(geom, 'geoms'):
        return sum(len(g.exterior.coords) for g in geom.geoms if g.exterior is not None)
    return 0


def compute_geometry_indicators(curated) -> pd.DataFrame:
    """Return per-footprint geometry indicators used to interrogate occupancy.

    Geometry is projected to a local UTM CRS so areas, lengths, and the
    oriented-bounding-box dimensions are in metres. Indicators: area (m2),
    oriented-bbox aspect ratio (length/width), rectangularity
    (area / oriented-bbox area), Polsby-Popper compactness (4*pi*A / P^2),
    convexity (area / convex-hull area), perimeter (m), exterior vertex count,
    inferred n_stories, and a volume proxy (area * n_stories).
    """
    from openplaces.io.harmonizer.spine import get_oriented_dims

    geom = curated.geometry
    proj = geom.to_crs(geom.estimate_utm_crs())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        area = proj.area.astype(float)
        perim = proj.length.astype(float)
        hull_area = proj.convex_hull.area.astype(float).clip(lower=1e-9)
        dims = proj.map(get_oriented_dims)

    length = dims.map(lambda d: d[1]).astype(float)
    width = dims.map(lambda d: d[2]).astype(float).clip(lower=1e-9)
    bbox_area = (length * width).clip(lower=1e-9)

    n_stories = (
        pd.to_numeric(curated['n_stories'], errors='coerce')
        if 'n_stories' in curated.columns
        else pd.Series(np.nan, index=curated.index)
    )

    out = pd.DataFrame(
        {
            'area_m2': area,
            'aspect_ratio': length / width,
            'rectangularity': area / bbox_area,
            'compactness': 4 * np.pi * area / perim.pow(2).clip(lower=1e-9),
            'convexity': area / hull_area,
            'perimeter_m': perim,
            'n_vertices': geom.map(_count_vertices).astype(float),
            'n_stories': n_stories.astype(float),
        },
        index=curated.index,
    )
    out['volume_m3'] = out['area_m2'] * out['n_stories']
    return out


def _quantile_table(indicators: pd.DataFrame) -> pd.DataFrame:
    """Tail-weighted quantiles of each indicator, grouped by use_group."""
    rows = []
    for use_group, group in indicators.groupby('use_group', dropna=False):
        n = len(group)
        n_stories_cov = round(float(group['n_stories'].notna().mean()), 3)
        for indicator in _INDICATORS:
            series = group[indicator].dropna()
            if series.empty:
                continue
            for q, value in series.quantile(_QUANTILES).items():
                rows.append(
                    {
                        'use_group': use_group,
                        'indicator': indicator,
                        'quantile': q,
                        'value': value,
                        'n': n,
                        'n_stories_coverage': n_stories_cov,
                    }
                )
    return pd.DataFrame(rows)


def _auc_one_vs_rest(x: pd.Series, positive: pd.Series) -> float:
    """Rank-based one-vs-rest AUC of indicator *x* separating the positive class.

    0.5 means no separation; values far from 0.5 (either direction) mean the
    indicator distinguishes the class. Equivalent to the normalised
    Mann-Whitney U statistic.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = x.rank()
    sum_ranks_pos = ranks[positive].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _eta_squared(x: pd.Series, groups: pd.Series) -> float:
    """ANOVA variance-explained (eta^2) of *groups* over indicator *x*."""
    grand = x.mean()
    ss_total = float(((x - grand) ** 2).sum())
    if ss_total == 0:
        return float('nan')
    ss_between = 0.0
    for _, group in x.groupby(groups):
        ss_between += len(group) * (group.mean() - grand) ** 2
    return float(ss_between / ss_total)


def _separability_table(indicators: pd.DataFrame) -> pd.DataFrame:
    """Per-indicator use-group separability: one-vs-rest AUC and eta^2."""
    rows = []
    groups = indicators['use_group']
    classes = list(pd.unique(groups.dropna()))
    for indicator in _INDICATORS:
        x = indicators[indicator]
        mask = x.notna() & groups.notna()
        if mask.sum() < 2 or groups[mask].nunique() < 2:
            continue
        eta2 = round(_eta_squared(x[mask], groups[mask]), 4)
        for cls in classes:
            in_cls = groups[mask] == cls
            if in_cls.sum() == 0 or (~in_cls).sum() == 0:
                continue
            rows.append(
                {
                    'indicator': indicator,
                    'use_group': cls,
                    'n': int(in_cls.sum()),
                    'auc_one_vs_rest': round(_auc_one_vs_rest(x[mask], in_cls), 4),
                    'eta_squared': eta2,
                }
            )
    return pd.DataFrame(rows)


def save_geometry_statistics(state) -> None:
    """Write geometry-indicator quantiles and use-group separability to cache.

    Groups footprints by their parcel ``use_group`` evidence and saves, per use
    group, tail-weighted quantiles of each geometry indicator (including the
    manufactured-home criterion inputs aspect ratio and area, and inferred
    n_stories), plus a separability summary (one-vs-rest AUC and eta^2). No-op
    unless ``state.save_statistics`` is set; never raises.
    """
    from openplaces.path import cache_path

    curated = state.curated
    if curated is None or 'geometry' not in curated or not len(curated):
        return
    try:
        indicators = compute_geometry_indicators(curated)
    except Exception as exc:  # diagnostics must never break a curation run
        warnings.warn(f'save_geometry_statistics: skipped ({exc}).')
        return

    use_col = (
        'use_group_parcel'
        if 'use_group_parcel' in curated.columns
        else next((c for c in curated.columns if c.startswith('use_group')), None)
    )
    use = (
        curated[use_col].astype('string')
        if use_col is not None
        else pd.Series('all', index=curated.index, dtype='string')
    ).fillna('n/a')
    indicators = indicators.assign(use_group=use.to_numpy())

    entity = state.recipe.get('entity')
    quant_path = cache_path(
        state.admin_id, entity, filename='geometry-indicator-quantiles.csv'
    )
    sep_path = cache_path(
        state.admin_id, entity, filename='use_group-geometry-separability.csv'
    )
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    _quantile_table(indicators).to_csv(quant_path, index=False)
    _separability_table(indicators).to_csv(sep_path, index=False)


_GROUP_KEY = 'use_group_combined_parcel'


def _group_dwelling_candidates(
    curated, rules: list[dict], multi_family_class: str
) -> pd.DataFrame:
    """Per parcel use-group: footprint count, mean Overture dwellings, group, flag.

    Returns one row per ``use_group_combined_parcel`` value with the footprint
    count, the mean ``n_dwellings_overture`` per footprint (missing treated as
    0), the modal ``group_parcel``, and ``flag_multifamily_low_dwellings`` (True
    when the group coerces to *multi_family_class* but the mean dwelling count is
    closer to 1 than 2, i.e. < 1.5). Flagged rows are sorted first. The class-map
    *rules* and the multi-family class come from the recipe ``occupancy`` config.
    """
    from openplaces.io.curator.occupancy import coerce_to_class

    dwellings = (
        pd.to_numeric(curated['n_dwellings_overture'], errors='coerce')
        if 'n_dwellings_overture' in curated.columns
        else pd.Series(np.nan, index=curated.index)
    ).fillna(0.0)
    df = pd.DataFrame(
        {
            _GROUP_KEY: curated[_GROUP_KEY],
            'group_parcel': curated['group_parcel'],
            'n_dwellings_overture': dwellings.to_numpy(),
        }
    )
    candidates = (
        df.groupby(_GROUP_KEY, observed=True)
        .agg(
            n_footprints=('group_parcel', 'size'),
            mean_n_dwellings_overture=('n_dwellings_overture', 'mean'),
            group_parcel=(
                'group_parcel',
                lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA,
            ),
        )
        .reset_index()
    )
    occ = coerce_to_class(candidates['group_parcel'], rules)
    candidates['flag_multifamily_low_dwellings'] = (occ == multi_family_class) & (
        candidates['mean_n_dwellings_overture'] < 1.5
    )
    return candidates.sort_values(
        ['flag_multifamily_low_dwellings', 'n_footprints'],
        ascending=[False, False],
        ignore_index=True,
    )


def save_group_dwelling_candidates(state) -> None:
    """Flag parcel use-groups whose dwelling evidence contradicts the occupancy.

    Surfaces candidates for manual group corrections. Writes the per use-group
    candidate table (see :func:`_group_dwelling_candidates`) plus a
    use_group -> NSI group co-occurrence audit (counts and within-use
    fractions). No-op unless save_statistics is set; never raises.
    """
    from openplaces.path import cache_path

    curated = state.curated
    if curated is None or _GROUP_KEY not in curated or 'group_parcel' not in curated:
        return

    try:
        from openplaces.io.curator.occupancy import get_occupancy_config, load_ruleset

        config = get_occupancy_config(state)
        rules = (
            load_ruleset(state, config['class_map']) if config.get('class_map') else []
        )
        mf_rule = config.get('rules', {}).get('multi_family_dwellings', {})
        multi_family_class = mf_rule.get('class', 'Multi-Family')
        candidates = _group_dwelling_candidates(curated, rules, multi_family_class)
        linkage = None
        if 'group_building_nsi' in curated.columns:
            linkage = (
                curated[[_GROUP_KEY, 'group_building_nsi']]
                .dropna()
                .value_counts()
                .rename('count')
                .reset_index()
            )
            totals = linkage.groupby(_GROUP_KEY)['count'].transform('sum')
            linkage['fraction'] = (linkage['count'] / totals).round(3)
    except Exception as exc:  # diagnostics must never break a curation run
        warnings.warn(f'save_group_dwelling_candidates: skipped ({exc}).')
        return

    entity = state.recipe.get('entity')
    out_path = cache_path(
        state.admin_id, entity, filename='parcel-group-dwelling-candidates.csv'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_path, index=False)
    if linkage is not None:
        linkage.to_csv(
            cache_path(state.admin_id, entity, filename='parcel-group-linkage.csv'),
            index=False,
        )


# Occupancy evidence sources compared by the comparison diagnostic, in the
# provisional cascade order; the report informs the final rank choice.
_EVIDENCE_SOURCES = {
    'nsi': 'occupancy_type_building_nsi',
    'fema': 'occupancy_type_footprint_fema',
    'parcel': 'group_parcel',
    'overture': 'occupancy_type_dwelling_overture',
}


def save_occupancy_evidence_comparison(state) -> None:
    """Compare the NSI / FEMA / parcel occupancy evidences for the rank decision.

    Coerces each available evidence column to its occupancy class and writes three
    cache CSVs: pairwise agreement rates, the most common conflict class-pairs, and
    a sample of disagreement cases (id + each source's class) to inspect. This is
    the low-cost basis for choosing the evidence cascade order empirically. No-op
    unless ``state.save_statistics`` is set; never raises.
    """
    from itertools import combinations

    from openplaces.io.curator.occupancy import (
        bucket_to_residential,
        get_occupancy_config,
        load_ruleset,
    )
    from openplaces.path import cache_path

    curated = state.curated
    if curated is None or not len(curated):
        return
    try:
        config = get_occupancy_config(state)
        rules = load_ruleset(state, config['class_map'])
        # Bucket non-residential exactly like occupancy_type_conflict so the
        # report and the column agree on what counts as a disagreement.
        coerced = {
            label: bucket_to_residential(curated[col], config, rules).astype('string')
            for label, col in _EVIDENCE_SOURCES.items()
            if col in curated.columns
        }
        labels = list(coerced)
        if len(labels) < 2:
            return
        frame = pd.DataFrame(coerced)

        agreement_rows = []
        conflict_frames = []
        any_conflict = pd.Series(False, index=frame.index)
        for a, b in combinations(labels, 2):
            both = frame[[a, b]].dropna()
            n = len(both)
            differ = both[a] != both[b]
            agree = int((~differ).sum())
            agreement_rows.append(
                {
                    'source_a': a,
                    'source_b': b,
                    'n_both_present': n,
                    'n_agree': agree,
                    'n_conflict': n - agree,
                    'agreement_rate': round(agree / n, 4) if n else float('nan'),
                }
            )
            any_conflict.loc[both.index] |= differ
            diff = both[differ]
            if len(diff):
                pair = (
                    diff.groupby([a, b], dropna=False)
                    .size()
                    .rename('count')
                    .reset_index()
                    .rename(columns={a: 'class_a', b: 'class_b'})
                )
                pair.insert(0, 'source_b', b)
                pair.insert(0, 'source_a', a)
                conflict_frames.append(pair)
    except Exception as exc:  # diagnostics must never break a curation run
        warnings.warn(f'save_occupancy_evidence_comparison: skipped ({exc}).')
        return

    entity = state.recipe.get('entity')
    agree_path = cache_path(
        state.admin_id, entity, filename='occupancy-evidence-agreement.csv'
    )
    agree_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(agreement_rows).to_csv(agree_path, index=False)

    if conflict_frames:
        conflicts = pd.concat(conflict_frames, ignore_index=True).sort_values(
            'count', ascending=False, ignore_index=True
        )
        conflicts.to_csv(
            cache_path(
                state.admin_id, entity, filename='occupancy-evidence-conflicts.csv'
            ),
            index=False,
        )

    cases = frame[any_conflict].head(1000).reset_index()
    if len(cases):
        cases.to_csv(
            cache_path(
                state.admin_id, entity, filename='occupancy-evidence-conflict-cases.csv'
            ),
            index=False,
        )
