"""The manufactured-home scores are what they were before S3.

plans/slow-steps-same-output-faster.md, S3: the scorer reuses its
100 m neighbor pairs, measures within 200 m only the pairs whose
envelopes can hold a row's nearest neighbor, and counts exterior
vertices without a Python call per row. The previous implementation is
kept below and both run on the same outlines; probabilities, nearest
distances and vertex counts are compared exactly. Every outline, id
and label is fabricated (docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.strtree import STRtree

from openplaces.io.curator.estimators.manufactured_homes import (
    _exterior_vertex_counts,
    _nearest_distances,
    _score_manufactured_home_candidates,
)


def _old_vertex_counts(geoms) -> np.ndarray:
    """The vertex count before S3."""
    return (
        gpd.GeoSeries(geoms)
        .map(lambda g: len(g.exterior.coords) - 1 if hasattr(g, 'exterior') else 0)
        .values
    )


def _old_nearest_distances(geom: gpd.GeoSeries, tree, n_rows: int) -> np.ndarray:
    """The nearest-neighbor block before S3."""
    idx_q2, idx_t2 = tree.query(geom.values, predicate='dwithin', distance=200.0)
    df_pairs2 = pd.DataFrame({'query_idx': idx_q2, 'tree_idx': idx_t2})
    df_pairs2 = df_pairs2[df_pairs2['query_idx'] != df_pairs2['tree_idx']]
    if len(df_pairs2) > 0:
        left = geom.iloc[df_pairs2['query_idx'].to_numpy()].reset_index(drop=True)
        right = geom.iloc[df_pairs2['tree_idx'].to_numpy()].reset_index(drop=True)
        df_pairs2['distance'] = left.distance(right).to_numpy()
        return (
            df_pairs2.groupby('query_idx')['distance']
            .min()
            .reindex(range(n_rows), fill_value=200.0)
            .values
        )
    return np.full(n_rows, 200.0)


def _old_score(
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
    """`_score_manufactured_home_candidates` as it stood before S3."""
    from sklearn.linear_model import LogisticRegression

    from openplaces.geo.polygon import local_metric_crs
    from openplaces.io.harmonizer.spine import get_oriented_dims

    orig_index = work.index
    work = work.reset_index(drop=True)
    assessor_labels = pd.Series(
        np.asarray(assessor_labels, dtype=object), index=work.index
    )
    n_rows = len(work)

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

    cv_prob = pd.Series(np.nan, index=work.index)
    train_mask = assessor_labels.notna()
    n_mfg = int((assessor_labels == mh_label).sum())
    n_sf = int((assessor_labels == sf_label).sum())

    X = features.fillna(0.0)
    p_mfg_morph = pd.Series(0.0, index=work.index)
    model_trained = False

    if n_mfg >= min_training_samples and n_sf >= min_training_samples:
        X_train = X[train_mask.to_numpy()]
        y_train = (assessor_labels[train_mask] == mh_label).astype(int)
        model = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        coef = model.coef_[0]
        intercept = model.intercept_[0]
        z = X.mul(coef, axis=1).sum(axis=1) + intercept
        probs = 1.0 / (1.0 + np.exp(-z.to_numpy()))
        p_mfg_morph = pd.Series(probs, index=work.index)
        model_trained = True

    if not model_trained:
        aspect = X['aspect_ratio']
        area_val = X['area']
        aspect_score = np.clip((aspect - 1.5) / 1.0, 0, 1) * 0.5
        area_score = (
            np.clip((plausible_area_max_m2 - area_val) / plausible_area_max_m2, 0, 1)
            * 0.5
        )
        p_mfg_morph = aspect_score + area_score

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


def _ring(x, y, size):
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]


def _special_outlines():
    """Holes, multipolygons, ties, duplicates and isolated outlines (m)."""
    with_hole = Polygon(_ring(0, 0, 20), [_ring(5, 5, 5), _ring(12, 12, 3)])
    multi = MultiPolygon([box(40, 0, 50, 8), box(52, 0, 60, 8)])
    multi_with_hole = MultiPolygon(
        [Polygon(_ring(70, 0, 15), [_ring(74, 4, 4)]), box(90, 0, 95, 5)]
    )
    # A row of equal squares 10 m apart: every inner square has two
    # nearest neighbors at the same distance.
    row = [box(x, 100, x + 10, 110) for x in range(0, 200, 20)]
    duplicate = [box(300, 0, 316, 6), box(300, 0, 316, 6)]
    # Exactly 200 m and just over 200 m from the duplicates.
    edge = [box(516, 0, 520, 4), box(300, 206.5, 304, 210)]
    isolated = [box(5000, 5000, 5014, 5004), box(-4000, 3000, -3990, 3012)]
    triangle = Polygon([(600, 0), (620, 0), (610, 9)])
    rotated = affinity.rotate(box(700, 0, 718, 5), 33, origin='center')
    return [
        with_hole,
        multi,
        multi_with_hole,
        *row,
        *duplicate,
        *edge,
        *isolated,
        triangle,
        rotated,
    ]


def _random_outlines(n_rows: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    side = np.sqrt(n_rows) * 40.0
    outlines = []
    for _ in range(n_rows):
        x, y = rng.uniform(0, side, 2)
        length = rng.uniform(6, 30)
        width = rng.uniform(3, 14)
        outline = affinity.rotate(
            box(x, y, x + length, y + width), rng.uniform(0, 180), origin='center'
        )
        kind = rng.random()
        if kind < 0.05:
            outline = outline.difference(outline.centroid.buffer(1.0))
        elif kind < 0.08:
            outline = MultiPolygon([outline, box(x + 40, y, x + 44, y + 3)])
        outlines.append(outline)
    return outlines


@pytest.mark.parametrize('with_missing', [False, True])
def test_vertex_counts_match_the_previous_code(with_missing):
    outlines = [*_special_outlines(), Polygon(), *_random_outlines(300, seed=1)]
    if with_missing:
        outlines.append(None)
    geoms = gpd.GeoSeries(outlines).values
    new = _exterior_vertex_counts(geoms)
    old = _old_vertex_counts(outlines)
    np.testing.assert_array_equal(new, old)
    assert new.dtype == old.dtype


@pytest.mark.parametrize(
    'outlines',
    [
        _special_outlines(),
        [box(0, 0, 5, 5)],
        [box(0, 0, 5, 5), box(1000, 0, 1005, 5)],
        [box(0, 0, 5, 5), box(0, 0, 5, 5)],
        [*_special_outlines(), Polygon(), None],
        _random_outlines(3000, seed=2),
    ],
)
def test_nearest_distances_match_the_previous_code(outlines):
    geom = gpd.GeoSeries(outlines)
    tree = STRtree(geom.values)
    new = _nearest_distances(tree, geom.values, len(geom))
    old = _old_nearest_distances(geom, tree, len(geom))
    np.testing.assert_array_equal(new, old)
    assert new.dtype == old.dtype


def _work(outlines, seed: int, labels: bool) -> tuple[gpd.GeoDataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n_rows = len(outlines)
    # A projected CRS in meters; the scorer reprojects to its own.
    work = gpd.GeoDataFrame(
        {
            'parcel_id': rng.integers(0, max(n_rows // 2, 1), n_rows).astype(str),
        },
        geometry=[
            affinity.translate(g, 500000, 3900000) if g is not None else None
            for g in outlines
        ],
        crs=32617,
        index=pd.Index([f'b{i:05d}' for i in range(n_rows)], name='building_id'),
    )
    work['area_m2'] = work.geometry.area
    pool = np.array(['Manufactured Home', 'Single-Family', None], dtype=object)
    if labels:
        assessor = pool[rng.integers(0, 3, n_rows)]
    else:
        assessor = np.full(n_rows, None, dtype=object)
    return work, pd.Series(assessor, index=work.index)


@pytest.mark.parametrize(
    ('outlines', 'labels'),
    [
        (_special_outlines(), False),
        (_special_outlines() + _random_outlines(200, seed=4), True),
        (_random_outlines(3000, seed=5), True),
        (_random_outlines(3000, seed=6), False),
    ],
)
def test_scores_match_the_previous_code(outlines, labels):
    work, assessor = _work(outlines, seed=9, labels=labels)
    options = {
        'mh_label': 'Manufactured Home',
        'sf_label': 'Single-Family',
        'aspect_min': 2.5,
        'area_max': 185.0,
        'plausible_aspect_min': 1.8,
        'plausible_area_max_m2': 250.0,
        'model_type': 'calibrated_logistic',
        'min_training_samples': 10,
        'verbose': False,
    }
    new = _score_manufactured_home_candidates(work, assessor, **options)
    old = _old_score(work, assessor, **options)
    pd.testing.assert_series_equal(
        new['p_manufactured_home'],
        old['p_manufactured_home'],
        check_dtype=True,
        check_exact=True,
    )
