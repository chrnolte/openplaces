"""Manufactured homes: the calibrated morphology classifier that
scores a footprint's probability of being one, and the per-parcel
community flag read off the final classes."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState, _register


@_register('flag_manufactured_home_communities', phase='infer')
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
        # Fallback rule-based morphology classifier. The area term grades
        # over the full plausible range: an earlier /100 denominator
        # saturated it at 0.5 for everything under
        # plausible_area_max_m2 - 100 (~150 m2), so a shed, a single-wide
        # and a bungalow all scored alike and p >= 0.5 fired on 17-58% of
        # all footprints per county. Graded continuously, a high p needs
        # genuine elongation AND a genuinely small area together.
        aspect = X['aspect_ratio']
        area_val = X['area']
        aspect_score = np.clip((aspect - 1.5) / 1.0, 0, 1) * 0.5
        area_score = (
            np.clip((plausible_area_max_m2 - area_val) / plausible_area_max_m2, 0, 1)
            * 0.5
        )
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


@_register('classify_manufactured_homes', phase='infer')
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
        # Crosswalked-vocabulary variants first: this ruleset is matched
        # with coerce_to_class, which honours no `reviewed` flag, so an
        # unreviewed rule such as `\bMH\b` would otherwise fire on a raw
        # land-use code that merely contains those letters.
        'use_group_combined_labeled_parcel',
        'use_group_combined_labeled',
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
