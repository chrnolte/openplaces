"""A recipe's validation setup: references, ground truth,
baselines and the notebooks a region's validate job runs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openplaces.io.curator.validation.accuracy import (
    MIN_STRATUM_ROWS,
    NON_RESIDENTIAL_LABEL,
    OTHER_LABEL,
    score_classification,
)
from openplaces.io.curator.validation.linking import (
    classify_validation_result,
    link_points_to_entities,
    summarize_sources,
)
from openplaces.io.curator.validation.references import (
    class_from_ruleset,
    reference_confidence_tier,
)
from openplaces.io.curator.validation.reports import (
    write_confusion_report,
)


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
        occupancy = dict(recipe.get('occupancy') or {})
        self.secondary_class = occupancy.get('secondary_class')
        self.residential_classes = tuple(occupancy.get('residential_classes') or ())
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
            Source label to comparable class values, bands collapsed. A
            derived source declaring `keep_unmapped: true` keeps its raw
            evidence value wherever the ruleset maps it to no class, so
            a source that asserts a class outside the ruleset (NSI's
            `Agricultural`, say) scores as that assertion rather than as
            no prediction; a row without raw evidence stays missing.
        """
        values = {
            label: self.collapse_bands(linked[column])
            for label, column in self.source_columns.items()
            if column in linked.columns
        }
        for label, spec in self.derived_source_columns.items():
            column = spec['column'] + self.inventory_suffix
            if column not in linked.columns:
                continue
            keep_unmapped = spec.get('keep_unmapped', False)
            if label in values and not keep_unmapped:
                continue
            ruleset = spec.get('ruleset', self.class_map)
            derived = self.class_from_ruleset(
                linked[column],
                ruleset,
                reviewed_only=spec.get('reviewed_only', False),
            )
            if label in values:
                # A class column, where present, already holds this
                # ruleset's answer; only its unmapped rows are filled.
                derived = values[label]
            elif derived is None:
                continue
            derived = self.collapse_bands(derived).astype(object)
            if keep_unmapped:
                derived = self._with_unmapped_evidence(derived, linked[column], ruleset)
            values[label] = derived
        count_col = self.dwelling_count_column
        if count_col and count_col in linked.columns:
            dwellings = pd.to_numeric(linked[count_col], errors='coerce')
            # No dwelling record is not a count of zero. Filling it made
            # "Overture never saw this entity" indistinguishable from
            # "Overture saw one dwelling", so the source was credited and
            # penalized for a Single-Family it never asserted. Absent
            # evidence stays missing, and score_classification's
            # predicted.notna() gate excludes it from this source's score.
            implied = pd.Series(None, index=linked.index, dtype=object)
            observed = dwellings.notna()
            implied[observed & (dwellings >= 2)] = 'Multi-Family'
            implied[observed & (dwellings < 2)] = 'Single-Family'
            values['overture'] = implied
        return values

    def _with_unmapped_evidence(self, classes, raw, ruleset):
        """Fill rows no rule matched with the raw evidence value itself.

        Before this, a class map covering residential classes only
        turned every non-residential assertion (NSI's `Professional
        Technical Services` on a surveyed manufactured home) into a
        missing prediction, so the matrices could not tell "the source
        said nothing" from "the source said something non-residential".
        Rows a rule matched keep the ruleset's answer, even one nulled
        for being unreviewed, and blank raw values stay missing.
        """
        raw = raw.astype(object)
        present = raw.notna() & raw.astype(str).str.strip().ne('')
        matched = self.class_from_ruleset(raw, ruleset)
        unmatched = (
            matched.isna() if matched is not None else pd.Series(True, raw.index)
        )
        fill = classes.isna() & unmatched & present
        return classes.mask(fill, raw)

    def matrix_labels(self):
        """Confusion-matrix sentinel labels that follow this recipe.

        Returns
        -------
        dict
            `secondary` (the recipe's `occupancy: secondary_class`),
            `other`, which reads `Non-residential` when the recipe
            declares its residential classes, and `kept_classes`, the
            recipe's residential classes (collapsed as the validation
            block collapses them) that `validation: classes` does not
            score, such as `RV Dwelling` against a reference with no
            such class. Kept apart, a prediction of one is never
            reported as a non-residential assertion. Pass as keyword
            arguments to :func:`write_confusion_report` or
            :func:`confusion_matrix`.
        """
        residential = getattr(self, 'residential_classes', ())
        scored = set(getattr(self, 'classes', ()))
        collapse = dict(getattr(self, 'collapse', None) or {})
        kept = [
            label
            for label in dict.fromkeys(collapse.get(c, c) for c in residential)
            if label not in scored
        ]
        return {
            'secondary': getattr(self, 'secondary_class', None),
            'other': NON_RESIDENTIAL_LABEL if residential else OTHER_LABEL,
            'kept_classes': kept,
        }

    def entity_source_values(self, entities):
        """:meth:`source_values` for curated entities read straight from disk.

        Scoring against an entity-keyed reference (permits) starts from
        the curated table itself rather than a survey linkage, so its
        columns carry no inventory suffix. Suffixing them here lets both
        references score exactly the same sources, reconstructed the same
        way.

        Parameters
        ----------
        entities : pandas.DataFrame
            Curated entity columns, unsuffixed. `occupancy_type` becomes
            the vote.

        Returns
        -------
        dict of str to pandas.Series
            As :meth:`source_values`, indexed like *entities*.
        """
        frame = pd.DataFrame(entities).drop(columns='geometry', errors='ignore')
        frame = frame.add_suffix(self.inventory_suffix)
        if 'occupancy_type' in entities.columns:
            frame['predicted'] = self.collapse_bands(entities['occupancy_type'])
        return self.source_values(frame)

    def survey_strata(self, linked):
        """Strata the survey matrices are split by, where present.

        The county the point was surveyed in, and the source of the
        footprint geometry it matched (a building outline traced from
        imagery and a parcel-derived placeholder fail differently).
        """
        strata = {}
        if 'admin_id' in linked.columns:
            strata['county'] = linked['admin_id']
        geometry_source = f'geometry_source{self.inventory_suffix}'
        if geometry_source in linked.columns:
            strata['geometry_source'] = linked[geometry_source]
        return strata

    def score_sources(
        self,
        linked,
        out_dir=None,
        *,
        name=None,
        strata=None,
        min_rows=MIN_STRATUM_ROWS,
        notes=None,
    ):
        """Score the vote and each of its inputs against the hand labels.

        Parameters
        ----------
        linked : pandas.DataFrame
            Output of :meth:`link_ground_truth`.
        out_dir : str or pathlib.Path, optional
            When given, confusion matrices and producer's/consumer's
            accuracies for every scored source are written there through
            :func:`write_confusion_report`, pooled and per stratum. This
            is the default output of a validation run; the return value
            is unchanged either way.
        name : str, optional
            File stem, default `{recipe_id}_occupancy-survey`.
        strata : dict of str to pandas.Series, optional
            Default :meth:`survey_strata`.
        min_rows : int, optional
            Fewest points a written stratum may hold (default 10).
        notes : str, optional
            Recorded in the report's JSON sidecar.

        Returns
        -------
        pandas.DataFrame
            :func:`score_classification` per source, with a `source`
            column in front.
        """
        truth = linked['occupancy_type_canonical']
        sources = self.source_values(linked)
        tables = []
        for label, values in sources.items():
            table = score_classification(truth, values, list(self.classes))
            table.insert(0, 'source', label)
            tables.append(table)
        if out_dir is not None:
            spec = dict(self.config.get('ground_truth') or {})
            write_confusion_report(
                truth,
                sources,
                list(self.classes),
                out_dir,
                name or f'{self.recipe_id}_occupancy-survey',
                strata=self.survey_strata(linked) if strata is None else strata,
                reference=' '.join(
                    str(part)
                    for part in ('survey', spec.get('source'), spec.get('version'))
                    if part
                ),
                notes=notes,
                min_rows=min_rows,
                **self.matrix_labels(),
            )
        return pd.concat(tables, ignore_index=True)

    # Which notebooks score which delivered region

    def reference_regions(self):
        """Delivery region each declared reference scores, by reference key.

        `ground_truth` maps to the survey's `region`; every entry of the
        sidecar's `references:` to its own `region`. A reference without a
        region, or a sidecar that is absent, contributes nothing.
        """
        regions = {}
        survey_region = (self.config.get('ground_truth') or {}).get('region')
        if survey_region:
            regions['ground_truth'] = str(survey_region)
        for key, spec in (self.config.get('references') or {}).items():
            if (spec or {}).get('region'):
                regions[str(key)] = str(spec['region'])
        return regions

    def notebooks_for_region(self, region):
        """Validation notebooks that score one delivered region.

        Read from the `notebooks:` list of the `validation:` block, each
        entry naming a notebook (relative to the repository root) and
        the reference it scores against. A notebook whose reference is
        not available (the untracked sidecar is absent) is not listed.

        Parameters
        ----------
        region : str
            A delivery region id.

        Returns
        -------
        list of str
            Notebook paths, in declared order.
        """
        regions = self.reference_regions()
        return [
            str(entry['notebook'])
            for entry in self.config.get('notebooks') or []
            if regions.get(str(entry.get('reference'))) == str(region)
        ]

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


def validation_notebooks(recipe, region) -> list[str]:
    """Validation notebooks declared for one delivery region of a recipe.

    Parameters
    ----------
    recipe : str or dict
        Curate recipe id or dict.
    region : str
        Delivery region id.

    Returns
    -------
    list of str
        Notebook paths relative to the repository root; empty when the
        recipe declares no `validation:` block or nothing scores
        *region*.
    """
    try:
        context = ValidationContext(recipe)
    except ValueError:
        return []
    return context.notebooks_for_region(region)
