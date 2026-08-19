"""Score a curated classification against hand-labeled ground-truth points.

Vocabulary-neutral and geography-neutral by construction: no class names, no
admin units, and no source paths appear here. Callers supply the labelled
points, the curated entities, and the class list, so the same code validates
building occupancy in North Carolina, land use elsewhere, or any future
labelled set (a state manufactured-housing registry, building permits) without
being edited.

Two things this module insists on that a naive accuracy check gets wrong:

- **Identity beats proximity when linking.** The nearest footprint to a survey
  pin is very often a shed or the neighbor's house, so an address match is
  tried first and distance is only the fallback. Which route matched is
  recorded, because a distance-linked row is weaker evidence than an
  address-linked one and the difference should stay visible downstream.
  Identity alone does not finish the job, though: a house and its garage
  share one address, so several entities routinely match the same point.
  Callers rank those ties with `prefer_column`; without it the pick falls to
  row order, which is arbitrary.
- **Precision and recall are reported separately, per class.** A rule that
  labels almost everything one class scores excellent recall for it, and an
  aggregate agreement figure hides the whole problem: it is entirely possible
  for overall agreement to rise while two of three classes get worse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per-row validation outcomes. A misclassified row is, strictly, both an
# omission for its true class and a commission for the predicted one; as a
# single per-row label, 'commission' means "we asserted a class and it was
# wrong" and 'omission' means "we asserted nothing where a label exists".
RESULT_CORRECT = 'correct'
RESULT_OMISSION = 'omission'
RESULT_COMMISSION = 'commission'


def normalize_house_number(value) -> str | None:
    """Normalize a house number so it compares across dtypes.

    A number that round-tripped through a float column arrives as ``5114.0``
    while the entity side stores ``'5114'``. Comparing the raw strings matches
    nothing at all, and does so silently -- every row falls through to the
    distance fallback and the linkage still looks like it worked.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text or None


def link_points_to_entities(
    points,
    entities,
    *,
    number_column: str = 'address_number',
    street_column: str = 'address_street',
    lon_column: str = 'lon',
    lat_column: str = 'lat',
    max_distance_m: float = 15.0,
    street_threshold: float = 80.0,
    admin1_id: str | None = None,
    suffix: str = '_inv',
    prefer_column: str | None = None,
    prefer_values: tuple = (),
) -> pd.DataFrame:
    """Link labelled *points* to *entities*, by address first then distance.

    Parameters
    ----------
    points : pandas.DataFrame
        Labelled points, carrying *lon_column*/*lat_column* and optionally the
        address columns.
    entities : geopandas.GeoDataFrame
        Curated entities to link against; every column is carried through.
    number_column, street_column : str, optional
        Address component columns, expected on both sides. Address matching is
        skipped when either is absent from *entities*.
    max_distance_m : float, optional
        Fallback radius for rows with no address match (default 15).
    street_threshold : float, optional
        Fuzzy street-similarity cutoff, 0-100 (default 80).
    admin1_id : str, optional
        Region hint for street canonicalization (e.g. ``'US-NC'``).
    suffix : str, optional
        Appended to every entity column (default ``'_inv'``). Both sides
        routinely share column names -- the class being validated most of all
        -- and overwriting the ground-truth side would make every comparison
        trivially agree with itself.
    prefer_column : str, optional
        Entity column used to rank several entities matching one point's
        address. Entities whose value is in *prefer_values* win; remaining
        ties keep row order. Without this, an address shared by a house and
        its outbuildings resolves to whichever happens to come first.
    prefer_values : tuple, optional
        Values of *prefer_column* that mark an entity as preferred. Named by
        the caller, since this module knows no vocabulary of its own.

    Returns
    -------
    pandas.DataFrame
        One row per linked point: the point's own columns, every entity column
        suffixed, and ``matched_by`` recording ``'address'`` or ``'distance'``.
        Points that matched nothing are absent.
    """
    import geopandas as gpd

    from openplaces.geo.address import match_streets

    if points.empty or entities is None or len(entities) == 0:
        return pd.DataFrame()

    entities = entities.reset_index(drop=True)
    has_address = {number_column, street_column} <= set(entities.columns)

    matched: dict[int, int] = {}
    if has_address and {number_column, street_column} <= set(points.columns):
        # An exact house number narrows the candidates to a handful, so the
        # fuzzy street comparison stays cheap.
        by_number: dict[str, list[int]] = {}
        for position, number in enumerate(entities[number_column]):
            key = normalize_house_number(number)
            if key:
                by_number.setdefault(key, []).append(position)

        preferred = None
        if prefer_column and prefer_column in entities.columns:
            wanted = set(prefer_values)
            preferred = entities[prefer_column].isin(wanted).to_numpy()

        for index, row in points.iterrows():
            key = normalize_house_number(row.get(number_column))
            street = row.get(street_column)
            if key is None or pd.isna(street):
                continue
            candidates: list[int] = []
            for position in by_number.get(key, ()):
                if match_streets(
                    street,
                    entities[street_column].iloc[position],
                    threshold=street_threshold,
                    admin1_id=admin1_id,
                ):
                    candidates.append(position)
                    if preferred is None:
                        # Nothing to rank by, so the rest of the
                        # bucket cannot change the answer.
                        break
            if not candidates:
                continue
            if len(candidates) > 1:
                # Stable sort: preferred entities move ahead, and
                # anything still tied keeps row order.
                candidates.sort(key=lambda position: not preferred[position])
            matched[index] = candidates[0]

    nearest: dict[int, int] = {}
    remaining = points.drop(index=list(matched))
    if not remaining.empty:
        metric_crs = entities.estimate_utm_crs()
        located = gpd.GeoDataFrame(
            remaining,
            geometry=gpd.points_from_xy(remaining[lon_column], remaining[lat_column]),
            crs='EPSG:4326',
        ).to_crs(metric_crs)
        joined = gpd.sjoin_nearest(
            located[['geometry']],
            entities[['geometry']].to_crs(metric_crs),
            how='left',
            distance_col='dist_m',
            lsuffix='pt',
            rsuffix='ent',
        )
        joined = joined.sort_values('dist_m').groupby(level=0).first()
        joined = joined[joined['dist_m'] <= max_distance_m]
        nearest = joined['index_ent'].dropna().astype(int).to_dict()

    pairs = {**matched, **nearest}
    if not pairs:
        return pd.DataFrame()

    point_index = list(pairs)
    linked = points.loc[point_index].reset_index(drop=True)
    picked = entities.iloc[[pairs[i] for i in point_index]].reset_index(drop=True)

    linked['matched_by'] = [
        'address' if i in matched else 'distance' for i in point_index
    ]
    for column in picked.columns:
        linked[f'{column}{suffix}'] = picked[column].to_numpy()
    return linked


def classify_validation_result(truth: pd.Series, predicted: pd.Series) -> pd.Series:
    """Label each row ``correct``, ``omission`` or ``commission``.

    ``omission`` is reserved for rows where a label exists but nothing was
    predicted -- the classifier declined to answer. ``commission`` is an
    answer that was wrong. Keeping them apart matters because they have
    different fixes: an omission means evidence was missing or a threshold was
    too strict, a commission means the evidence present was misread.
    """
    truth_values = truth.astype(object)
    predicted_values = predicted.astype(object)
    result = pd.Series(RESULT_COMMISSION, index=truth.index, dtype=object)
    result[predicted_values.isna()] = RESULT_OMISSION
    result[predicted_values.notna() & (predicted_values == truth_values)] = (
        RESULT_CORRECT
    )
    return result


def summarize_sources(
    sources: dict[str, pd.Series], separator: str = ' | '
) -> pd.Series:
    """Render each row's per-source values as one readable string.

    Produces e.g. ``'truth: Multi-Family | nsi: Single-Family | fema: -'``, so
    a reviewer can see the whole evidence picture for a disputed row without
    scanning a dozen columns. Missing values are rendered rather than dropped:
    a source that said nothing is itself informative.
    """
    if not sources:
        return pd.Series(dtype=object)
    frame = pd.DataFrame(
        {label: series.astype(object) for label, series in sources.items()}
    )
    return frame.apply(
        lambda row: separator.join(
            f'{label}: {"-" if pd.isna(row[label]) else row[label]}'
            for label in frame.columns
        ),
        axis=1,
    )


def compare_classifications_paired(
    truth: pd.Series,
    baseline: pd.Series,
    proposed: pd.Series,
    classes: list[str],
    n_draws: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired bootstrap of the per-class F1 change from *baseline* to *proposed*.

    Comparing two independent point estimates cannot resolve a change
    smaller than the sample's own spread -- on a survey of a thousand-odd
    labelled points that spread is several F1 points, far wider than the
    change a careful edit produces. Resampling the *same* point indices for
    both predictions cancels the variation the two runs share, because most
    points are classified identically either way, so the interval reflects
    only the rows that actually moved.

    Parameters
    ----------
    truth : pandas.Series
        Hand-assigned class per point.
    baseline : pandas.Series
        The accepted run's prediction for the same points, already aligned
        to *truth* (same index, same order).
    proposed : pandas.Series
        The candidate run's prediction for those points.
    classes : list of str
        Classes to score, in report order. An ``ALL`` row is appended.
    n_draws : int, optional
        Bootstrap draws (default 400).
    seed : int, optional
        Seed for the resampler, so a gate decision is reproducible.
    alpha : float, optional
        Two-sided interval width (default 0.05, i.e. a 95% interval).

    Returns
    -------
    pandas.DataFrame
        One row per class plus ``ALL``, with ``f1_base``, ``f1_new``,
        ``d_f1`` (the point estimate of new minus base), ``d_low``/``d_high``
        (the paired interval), and ``p_worse`` (share of draws in which the
        class lost F1). A class the proposed run never predicts and the
        baseline never predicted scores NaN, as in
        :func:`score_classification`.
    """
    truth = truth.reset_index(drop=True)
    baseline = baseline.reset_index(drop=True)
    proposed = proposed.reset_index(drop=True)
    if not (len(truth) == len(baseline) == len(proposed)):
        raise ValueError(
            f'paired comparison needs aligned inputs; got {len(truth)} truth, '
            f'{len(baseline)} baseline and {len(proposed)} proposed rows.'
        )

    def _f1(idx, predicted):
        table = score_classification(
            truth.iloc[idx].reset_index(drop=True),
            predicted.iloc[idx].reset_index(drop=True),
            classes,
        )
        return table.set_index('class')['f1']

    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {}
    for _ in range(n_draws):
        idx = rng.integers(0, len(truth), len(truth))
        base_f1 = _f1(idx, baseline)
        new_f1 = _f1(idx, proposed)
        for label in base_f1.index:
            deltas.setdefault(label, []).append(new_f1[label] - base_f1[label])

    whole = np.arange(len(truth))
    base_point = _f1(whole, baseline)
    new_point = _f1(whole, proposed)

    rows = []
    for label in base_point.index:
        draws = np.array(deltas.get(label, []), dtype=float)
        finite = draws[np.isfinite(draws)]
        low, high = (
            np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            if len(finite)
            else (np.nan, np.nan)
        )
        rows.append(
            {
                'class': label,
                'f1_base': base_point[label],
                'f1_new': new_point[label],
                'd_f1': new_point[label] - base_point[label],
                'd_low': low,
                'd_high': high,
                'p_worse': float((finite < 0).mean()) if len(finite) else np.nan,
                'n_draws': int(len(finite)),
            }
        )
    return pd.DataFrame(rows)


def score_classification(
    truth: pd.Series, predicted: pd.Series, classes: list[str]
) -> pd.DataFrame:
    """Per-class precision, recall and F1, plus an overall agreement row.

    Recall answers "of the real Xs, how many did we find"; precision answers
    "of those we called X, how many really are". Reporting only one invites
    the failure this whole module exists to prevent.
    """
    truth = truth.astype(object)
    predicted = predicted.astype(object)
    present = predicted.notna()

    records = []
    for cls in classes:
        is_class = truth == cls
        scored = is_class & present
        called = present & (predicted == cls)
        n_correct = int((predicted[scored] == cls).sum())
        n_called = int(called.sum())
        recall = n_correct / scored.sum() if scored.sum() else None
        precision = int((truth[called] == cls).sum()) / n_called if n_called else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall
            else None
        )
        records.append(
            {
                'class': cls,
                'n_truth': int(is_class.sum()),
                'n_scored': int(scored.sum()),
                'n_correct': n_correct,
                'n_predicted': n_called,
                'recall': round(recall, 4) if recall is not None else None,
                'precision': round(precision, 4) if precision is not None else None,
                'f1': round(f1, 4) if f1 is not None else None,
            }
        )

    agreement = (
        (predicted[present] == truth[present]).sum() / present.sum()
        if present.sum()
        else None
    )
    records.append(
        {
            'class': 'ALL',
            'n_truth': int(len(truth)),
            'n_scored': int(present.sum()),
            'n_correct': int((predicted[present] == truth[present]).sum()),
            'n_predicted': int(present.sum()),
            'recall': round(agreement, 4) if agreement is not None else None,
            'precision': round(agreement, 4) if agreement is not None else None,
            'f1': round(agreement, 4) if agreement is not None else None,
        }
    )
    return pd.DataFrame(records)
