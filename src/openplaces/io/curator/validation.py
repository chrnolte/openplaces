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

from pathlib import Path

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


def class_from_ruleset(
    recipe,
    terms: pd.Series,
    ruleset: str,
    *,
    reviewed_only: bool = False,
) -> pd.Series | None:
    """Reconstruct a class column from raw evidence through a recipe ruleset.

    Applies the same ordered ruleset a curate vote used, so a
    reconstructed column scores identically to one the recipe's
    formatting stage dropped from published output.

    Parameters
    ----------
    recipe : str or dict
        Curate recipe (id or dict) whose sidecar rulesets to read.
    terms : pandas.Series
        Raw label text the class column is derived from.
    ruleset : str
        Filename of the ruleset CSV beside the curate recipe.
    reviewed_only : bool, optional
        Keep only matches whose winning rule is marked reviewed,
        mirroring the recipe's own flag. Nulling unreviewed matches
        after the fact, rather than dropping those rules up front, is
        deliberate: pre-filtering would let a term fall through to a
        later reviewed rule and assert a class the vote never saw.

    Returns
    -------
    pandas.Series or None
        The reconstructed class column, or None when the ruleset cannot
        be located, leaving the caller to omit that source.
    """
    from types import SimpleNamespace

    from openplaces.io.curator.occupancy import load_ruleset, match_ruleset
    from openplaces.recipe import get_recipe_by_id

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    try:
        state = SimpleNamespace(recipe=recipe)
        rules = load_ruleset(state, ruleset)
    except (FileNotFoundError, KeyError):
        return None
    proposal, reviewed = match_ruleset(terms.astype(object), rules)
    if reviewed_only:
        proposal = proposal.where(reviewed)
    return proposal


def reference_confidence_tier(
    frame: pd.DataFrame,
    *,
    count_column: str = 'n_permits_with_occupancy_type',
    mode_pct_column: str = 'occupancy_type_mode_pct',
    mode_column: str = 'occupancy_type_mode',
    matched_via_column: str = 'matched_via',
    id_match_value: str = 'parcel_id_local',
) -> pd.Series:
    """Confidence tier for a reference-label table, high to low.

    Two things separate a strong claim from a weak one: how the
    reference reached the entity (an id join beats an address or point
    match) and whether its records agree (a unanimous mode over at
    least two label-bearing records beats a single uncorroborated one).

    Parameters
    ----------
    frame : pandas.DataFrame
        Entity-keyed reference labels carrying the four columns named by
        the keyword arguments (the permit pair-table schema by default).

    Returns
    -------
    pandas.Series
        One of `1_id_strong`, `2_id_weak`, `3_addr_strong`,
        `4_addr_weak`, or `none` where no record names a label.
    """
    n_labels = pd.to_numeric(frame[count_column], errors='coerce')
    unanimous = frame[mode_pct_column].ge(0.999) & n_labels.ge(2)
    by_id = frame[matched_via_column].eq(id_match_value)
    spoke = frame[mode_column].notna()
    tier = pd.Series('none', index=frame.index)
    tier[spoke & ~by_id & ~unanimous] = '4_addr_weak'
    tier[spoke & ~by_id & unanimous] = '3_addr_strong'
    tier[spoke & by_id & ~unanimous] = '2_id_weak'
    tier[spoke & by_id & unanimous] = '1_id_strong'
    return tier


class ValidationContext:
    """Everything a validation notebook needs, built from recipe data.

    A curate recipe declares its validation configuration in a
    `validation:` block, the same way delivery columns live in `share:`:
    the hand-labelled reference it is scored against, the class
    vocabulary and how the inventory's finer bands collapse onto it,
    linkage thresholds, and the evidence columns each vote input is
    scored from. Reference tables whose source is licence-restricted
    are declared in an untracked sidecar
    (`{recipe_id}_validation-references.yaml` beside the recipe) that
    is merged over the committed block when present, so the committed
    surface never names such a source.

    Notebooks build one context and read the same names the block
    declares; nothing geography- or source-specific lives in this
    class.
    """

    def __init__(self, recipe, references_state=None):
        """Build a context for one curate recipe.

        Parameters
        ----------
        recipe : str or dict
            Curate recipe id or dict carrying a `validation:` block.
        references_state : str, optional
            Which entry of the sidecar's `references:` mapping to
            activate (e.g. a state code). Without it the
            reference-table helpers raise when used.
        """
        from openplaces.recipe import get_recipe_by_id, get_recipe_id

        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        self.recipe = recipe
        self.recipe_id = get_recipe_id(recipe)
        config = dict(recipe.get('validation') or {})
        sidecar = self._load_references_sidecar()
        if sidecar:
            config = {**config, **sidecar}
        if not config:
            raise ValueError(
                f'{self.recipe_id} declares no validation: block and has '
                'no validation-references sidecar.'
            )
        self.config = config
        self.classes = tuple(config.get('classes') or ())
        self.collapse = dict(config.get('collapse') or {})
        self.single_dwelling_classes = tuple(
            config.get('single_dwelling_classes') or ()
        )
        link = dict(config.get('link') or {})
        self.max_distance_m = link.get('max_distance_m', 15)
        self.street_threshold = link.get('street_threshold', 80.0)
        self.prefer_column = link.get('prefer_column')
        self.prefer_values = tuple(link.get('prefer_values') or ())
        self.prediction_key = list(config.get('prediction_key') or [])
        self.class_map = config.get('class_map')
        self.keyword_ruleset = config.get('keyword_ruleset')
        self.source_columns = dict(config.get('source_columns') or {})
        self.derived_source_columns = dict(config.get('derived_source_columns') or {})
        self.dwelling_count_column = config.get('dwelling_count_column')
        self.inventory_suffix = config.get('inventory_suffix', '_inv')
        self.references_state = references_state
        self.reference = None
        if references_state is not None:
            table = dict(config.get('references') or {})
            if references_state not in table:
                raise KeyError(
                    f'No validation reference declared for '
                    f'{references_state!r}; the untracked sidecar '
                    f'{self.recipe_id}_validation-references.yaml '
                    f'declares: {sorted(table)}'
                )
            self.reference = dict(table[references_state])

    # Configuration resolution

    def _load_references_sidecar(self):
        import yaml

        from openplaces.path import recipe_path

        base = recipe_path(
            self.recipe.get('admin_id'),
            self.recipe.get('entity') or self.recipe.get('dataset'),
            # recipe_path prefixes the recipe id itself
            filename='validation-references',
        )
        path = Path(str(base))
        if path.suffix != '.yaml':
            path = path.with_suffix('.yaml')
        if not path.exists():
            return {}
        with open(path, encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    @property
    def ground_truth_path(self):
        from openplaces.core.schema import Entity
        from openplaces.path import external_dir

        spec = dict(self.config.get('ground_truth') or {})
        entity = Entity(spec['entity_type'], spec['source'], str(spec['version']))
        return external_dir(spec['admin_id'], entity=entity) / spec['filename']

    def _cache_path(self, filename=None, **kwargs):
        from openplaces.path import cache_path

        return cache_path(
            str(self.recipe.get('admin_id') or 'US'),
            entity=self.recipe.get('entity'),
            filename=filename,
            **kwargs,
        )

    @property
    def validation_dir(self):
        return self._cache_path(as_dir=True)

    @property
    def linked_path(self):
        return self._cache_path('validation-footprints')

    @property
    def baseline_path(self):
        return self._cache_path('occupancy-baseline', default_extension='csv')

    @property
    def baseline_predictions_path(self):
        return self._cache_path(
            'occupancy-baseline-predictions', default_extension='csv'
        )

    # Reference tables (entity-keyed label pairs), from the sidecar

    def _reference_dir(self):
        from openplaces.core.schema import Entity
        from openplaces.path import external_dir

        if not self.reference:
            raise ValueError(
                'This context was built without references_state; pass '
                "one, e.g. ValidationContext(recipe, 'NC')."
            )
        entity = Entity(
            self.reference['entity_type'],
            self.reference['source'],
            str(self.reference['version']),
        )
        return external_dir(
            self.reference['admin_id'], entity=entity
        ) / self.reference.get('subdir', 'validation')

    @property
    def reference_region(self):
        return self.reference['region'] if self.reference else None

    @property
    def reference_dir(self):
        return self._reference_dir()

    @property
    def reference_strong_tiers(self):
        return tuple(
            (self.reference or {}).get(
                'strong_tiers', ('1_id_strong', '2_id_weak', '3_addr_strong')
            )
        )

    def reference_admin_ids(self):
        """Admin units with a complete footprint+parcel pair on disk.

        An incomplete pair means a mid-write unit, not a unit without
        records, so it is skipped rather than read. The admin id
        pattern is anchored to the sidecar's declared admin scope and
        code width so files written under superseded id mints cannot
        double-count their units.
        """
        import re

        scope = self.reference['admin_id']
        width = int(self.reference.get('admin_code_width', 3))
        kinds = {}
        for path in sorted(
            self._reference_dir().glob(f'{scope}-*_occupancy_validation.parquet')
        ):
            match = re.match(
                rf'({re.escape(scope)}-\w{{{width}}})_'
                r'(footprint|parcel)_occupancy_validation',
                path.stem,
            )
            if match:
                kinds.setdefault(match.group(1), set()).add(match.group(2))
        return sorted(c for c, k in kinds.items() if k == {'footprint', 'parcel'})

    def load_reference(self, admin_id, kind='footprint'):
        """Load one unit's entity-level reference labels, or None."""
        path = self._reference_dir() / f'{admin_id}_{kind}_occupancy_validation.parquet'
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        frame.index.name = f'{kind}_id'
        return frame

    def reference_tier(self, frame):
        """Confidence tier of loaded reference labels (module helper)."""
        return reference_confidence_tier(frame)

    # Stands in for "no class" when two class columns are compared, so a
    # missing value on either side compares equal to itself and unequal
    # to every real class.
    _NO_CLASS = '<none>'

    def link_ground_truth(self, counties=None, *, verbose=False, save=True):
        """Link the hand-labelled points to curated entities, per unit.

        Address identity is tried before proximity (see
        :func:`link_points_to_entities`): the nearest footprint to a
        survey pin is very often a shed or the neighbour's house.

        Parameters
        ----------
        counties : tuple of str, optional
            Admin units to link. Defaults to :meth:`survey_admin_ids`.
        verbose : bool, optional
            Report per-unit linkage counts.
        save : bool, optional
            Write the linked frame to :attr:`linked_path` (parquet, plus
            a CSV sidecar for review). Default True.

        Returns
        -------
        geopandas.GeoDataFrame
            One row per linked point: the reference columns, every
            curated column suffixed with :attr:`inventory_suffix`,
            `matched_by`, and the derived comparison columns
            `predicted`, `is_single_dwelling`, `validation_result`,
            `occupancy_type_conflict_sources` and `sources_disagree`.
            The geometry is the matched entity, not the reference pin.
        """
        import geopandas as gpd

        import openplaces as op

        counties = tuple(counties) if counties else self.survey_admin_ids()
        admin1_id = str((self.config.get('ground_truth') or {}).get('admin_id') or '')
        frames = []
        crs = None
        for admin_id in counties:
            points = self.load_ground_truth((admin_id,))
            if points.empty:
                continue
            entities = op.get_entities(
                self.recipe_id, admin_id, geom=True, missing='ignore'
            )
            if entities is None or entities.empty:
                if verbose:
                    print(f'{admin_id}: no curated output on disk, skipped')
                continue
            crs = entities.crs
            # reset_index carries the entity id through as a column, so
            # a reference-table notebook can join on an id that is
            # stable across branches.
            linked = link_points_to_entities(
                points,
                entities.reset_index(),
                max_distance_m=self.max_distance_m,
                street_threshold=self.street_threshold,
                admin1_id=admin1_id or None,
                prefer_column=self.prefer_column,
                prefer_values=self.prefer_values,
            )
            if linked.empty:
                if verbose:
                    print(f'{admin_id}: {len(points)} points, none linked')
                continue
            if verbose:
                by_route = linked['matched_by'].value_counts()
                print(
                    f'{admin_id}: {len(linked)}/{len(points)} points linked '
                    f'(address {by_route.get("address", 0)}, '
                    f'distance {by_route.get("distance", 0)})'
                )
            frames.append(linked)

        if not frames:
            return gpd.GeoDataFrame()

        suffix = self.inventory_suffix
        linked = pd.concat(frames, ignore_index=True)
        linked['predicted'] = self.collapse_bands(linked[f'occupancy_type{suffix}'])
        linked['is_single_dwelling'] = linked['occupancy_type_canonical'].isin(
            self.single_dwelling_classes
        )
        linked['validation_result'] = classify_validation_result(
            linked['occupancy_type_canonical'], linked['predicted']
        )

        sources = self.source_values(linked)
        linked['occupancy_type_conflict_sources'] = summarize_sources(
            {'ground_truth': linked['occupancy_type_canonical'], **sources}
        )
        # Flag the rows worth reading by hand: any input that spoke and
        # was overruled. Both sides compare through a sentinel: a row
        # the vote declined to classify makes `ne` return pd.NA on a
        # nullable column, and reading a missing vote as a disagreement
        # is the intended answer, not a convenience.
        predicted = linked['predicted'].astype(object).fillna(self._NO_CLASS)
        disagree = pd.Series(False, index=linked.index)
        for label, values in sources.items():
            if label == 'final_vote':
                continue
            values = values.astype(object)
            disagree |= values.notna() & values.fillna(self._NO_CLASS).ne(predicted)
        linked['sources_disagree'] = disagree

        # link_points_to_entities returns a plain DataFrame, so the
        # matched geometry arrives as a CRS-less object column; restore
        # it from the entities it came from.
        linked = gpd.GeoDataFrame(
            linked.drop(columns=f'geometry{suffix}'),
            geometry=gpd.GeoSeries(linked[f'geometry{suffix}'], crs=crs),
        )
        if save:
            self.validation_dir.mkdir(parents=True, exist_ok=True)
            linked.to_parquet(self.linked_path)
            linked.drop(columns='geometry').to_csv(
                Path(str(self.linked_path)).with_suffix('.csv'), index=False
            )
            if verbose:
                print(f'wrote {len(linked)} linked points to {self.linked_path}')
        return linked

    # Vocabulary

    def collapse_bands(self, values):
        """Map the inventory's finer class bands onto the reference's."""
        return values.astype(object).replace(self.collapse)

    def class_from_ruleset(self, terms, ruleset=None, **kwargs):
        """Recipe-bound wrapper for the module-level class_from_ruleset."""
        return class_from_ruleset(
            self.recipe, terms, ruleset or self.class_map, **kwargs
        )

    # Survey ground truth

    def survey_admin_ids(self):
        """Admin units present in the ground-truth table, sorted.

        Discovered rather than hardcoded: the table's own admin column
        already reflects points that landed outside their declared
        source sheet.
        """
        path = self.ground_truth_path
        if not path.exists():
            spec = dict(self.config.get('ground_truth') or {})
            raise FileNotFoundError(
                f'Ground truth not found at {path}. ' + spec.get('regenerate_hint', '')
            )
        admin_ids = pd.read_csv(path, usecols=['admin_id'])['admin_id']
        return tuple(sorted(admin_ids.dropna().unique()))

    def load_ground_truth(self, counties=None):
        """Load the hand-labelled points, optionally restricted by unit."""
        points = pd.read_csv(self.ground_truth_path)
        points = points[points['admin_id'].notna()]
        if counties:
            points = points[points['admin_id'].isin(tuple(counties))]
        return points.reset_index(drop=True)

    # Sources and scoring

    def source_values(self, linked):
        """The vote and each input it arbitrates, on the reference vocabulary.

        Parameters
        ----------
        linked : pandas.DataFrame
            Linked frame carrying curated columns under
            `inventory_suffix`.

        Returns
        -------
        dict of str to pandas.Series
            Source label to comparable class values, bands collapsed.
        """
        values = {
            label: self.collapse_bands(linked[column])
            for label, column in self.source_columns.items()
            if column in linked.columns
        }
        for label, spec in self.derived_source_columns.items():
            column = spec['column'] + self.inventory_suffix
            if label in values or column not in linked.columns:
                continue
            derived = self.class_from_ruleset(
                linked[column],
                spec.get('ruleset', self.class_map),
                reviewed_only=spec.get('reviewed_only', False),
            )
            if derived is not None:
                values[label] = self.collapse_bands(derived)
        count_col = self.dwelling_count_column
        if count_col and count_col in linked.columns:
            dwellings = pd.to_numeric(linked[count_col], errors='coerce')
            values['overture'] = pd.Series(
                [
                    'Multi-Family' if n >= 2 else 'Single-Family'
                    for n in dwellings.fillna(0)
                ],
                index=linked.index,
                dtype=object,
            )
        return values

    def score_sources(self, linked):
        """Score the vote and each of its inputs against the hand labels."""
        tables = []
        for label, values in self.source_values(linked).items():
            table = score_classification(
                linked['occupancy_type_canonical'], values, list(self.classes)
            )
            table.insert(0, 'source', label)
            tables.append(table)
        return pd.concat(tables, ignore_index=True)

    # Baseline bookkeeping for the paired gate

    def save_baseline_predictions(self, linked, path=None):
        """Write the accepted run's per-point predictions."""
        path = Path(path or self.baseline_predictions_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [*self.prediction_key, 'occupancy_type_canonical', 'predicted']
        linked[columns].to_csv(path, index=False)
        return path

    def load_baseline_predictions(self, path=None):
        """Read the baseline predictions, failing with a how-to hint."""
        path = Path(path or self.baseline_predictions_path)
        if not path.exists():
            raise FileNotFoundError(
                f'No baseline predictions at {path}. Run the validation '
                'once with --write_baseline to record the accepted run '
                'before gating against it.'
            )
        return pd.read_csv(path)

    def align_to_baseline(self, linked, baseline):
        """Line the current run's predictions up with the baseline's.

        Returns the points both runs share; points only one run has are
        counted in the report rather than silently dropped, so the gate
        cannot quietly score a different set of buildings than the
        baseline did.
        """
        key = self.prediction_key
        current = linked[[*key, 'occupancy_type_canonical', 'predicted']].copy()
        merged = current.merge(
            baseline, on=key, how='inner', suffixes=('', '_base'), validate='1:1'
        )
        report = {
            'n_shared': len(merged),
            'n_baseline_only': len(baseline) - len(merged),
            'n_current_only': len(current) - len(merged),
        }
        report['n_truth_changed'] = int(
            merged['occupancy_type_canonical']
            .astype(object)
            .ne(merged['occupancy_type_canonical_base'].astype(object))
            .sum()
        )
        return (
            merged['occupancy_type_canonical'],
            merged['predicted_base'],
            merged['predicted'],
            report,
        )

    @staticmethod
    def check_baseline_coverage(table, baseline):
        """Fail loudly when a baseline row finds no counterpart in table.

        The gate merges on (source, class); a source missing from the
        scored table would silently shrink the comparison while the
        gate still reports a pass.
        """
        expected = set(map(tuple, baseline[['source', 'class']].to_numpy()))
        actual = set(map(tuple, table[['source', 'class']].to_numpy()))
        missing = sorted(expected - actual)
        if missing:
            raise SystemExit(
                f'FAIL: {len(missing)} baseline row(s) had no counterpart '
                f'to compare against, so the gate would have scored only '
                f'{len(actual)} of {len(expected)} rows: {missing}'
            )


def validation_context(recipe, references_state=None):
    """Build a :class:`ValidationContext`; see the class docstring."""
    return ValidationContext(recipe, references_state)
