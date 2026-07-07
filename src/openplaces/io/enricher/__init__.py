"""
Recipe-driven entity enrichment.

Enrichment reads a harmonized entity spine plus one or more input recipes and
writes entity-keyed evidence tables beside the selected spine output.
"""

from __future__ import annotations

import json
import pkgutil as _pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module as _import_module

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io import read_parquet, save_parquet
from openplaces.io.aggregate import COVERAGE_ALL, read_partition_coverage
from openplaces.io.readers import get_admin_ids
from openplaces.recipe import (
    find_entity_recipe_id,
    get_output_path,
    get_recipe_by_id,
    get_save_admin_level,
)
from openplaces.timing import get_timer


@dataclass
class EnrichState:
    """Mutable container passed through every enrich pipeline step.

    Parameters
    ----------
    recipe
        Loaded enrichment recipe.
    entity_recipe
        Loaded concrete entity recipe being enriched.
    admin_id
        Current admin unit being processed.
    verbose
        Print per-step progress messages.
    timer
        Timing helper.
    spine
        Harmonized entity spine.
    evidence
        Entity-keyed evidence table being built.
    image_admin_ids
        When set, restrict image-based steps to these admin units
        (deeper than the process level) instead of all children.
    metadata
        Step-specific intermediate data.
    """

    recipe: dict
    entity_recipe: dict
    admin_id: AdminId
    verbose: bool
    timer: object | None
    spine: pd.DataFrame
    evidence: pd.DataFrame
    image_admin_ids: list[str] | None = None
    metadata: dict = field(default_factory=dict)


_STEP_REGISTRY: dict[str, Callable] = {}

# Footer-coverage sentinel: the evidence file covers all admin units.
# Defined in io.aggregate so lower-layer lifecycle checks share it.
_COVERAGE_ALL = COVERAGE_ALL


def _register(*names: str):
    """Decorator: register an enrich step under one or more names."""

    def decorator(fn: Callable) -> Callable:
        for name in names:
            _STEP_REGISTRY[name] = fn
        return fn

    return decorator


_steps_loaded = False


def _load_steps() -> None:
    """Import this stage's step submodules so their @_register runs.

    Deferred until a step is first dispatched; the ``ispkg`` skip keeps the
    heavy ``detectors`` subpackage out of the import.
    """
    global _steps_loaded
    if _steps_loaded:
        return
    for _m in _pkgutil.iter_modules(__path__):
        if not _m.ispkg:
            _import_module(f'{__name__}.{_m.name}')
    _steps_loaded = True


def _recipe_entity_type(recipe: dict) -> str:
    """Return the entity type targeted by an enrichment recipe."""
    entity = recipe.get('entity')
    if entity is None:
        raise ValueError("Enrich recipes require 'entity'.")
    return str(entity.entity_type)


def _resolve_entity_recipe(
    recipe: dict,
    entity_recipe_id: str | dict | None = None,
) -> dict:
    """Resolve the concrete entity recipe for an enrichment run."""
    if isinstance(entity_recipe_id, dict):
        return entity_recipe_id
    if isinstance(entity_recipe_id, str):
        return get_recipe_by_id(entity_recipe_id)

    recipe_id = find_entity_recipe_id(
        recipe.get('admin_id'),
        _recipe_entity_type(recipe),
        stage='harmonize',
        source_id='spine',
        silent=True,
    )
    if recipe_id is None:
        raise ValueError(
            f'No entity recipe found for enrichment recipe {recipe.get("dataset")}.'
        )
    return get_recipe_by_id(recipe_id)


class Enricher:
    """Run an enrichment recipe for one or more admin units."""

    def __init__(
        self,
        recipe: str | dict,
        admin_ids: str | list | None = None,
        entity_recipe_id: str | dict | None = None,
        verbose: bool = False,
    ):
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        if recipe is None:
            raise ValueError(
                '`recipe` must be a recipe dict or a valid recipe ID string.'
            )
        stage = recipe.get('stage', 'ingest')
        if stage != 'enrich':
            raise ValueError(
                f"Recipe stage is '{stage}', expected 'enrich'. "
                'Pass an enrichment recipe, not an ingest or harmonize recipe.'
            )

        self.recipe = recipe
        self.entity_recipe = _resolve_entity_recipe(recipe, entity_recipe_id)
        if self.entity_recipe.get('stage') != 'harmonize':
            raise ValueError(
                "Enrich 'entity_recipe_id' must reference a harmonization recipe."
            )
        self.verbose = verbose
        self._timer = None
        self.admin_ids = self._resolve_admin_ids(admin_ids)

    @property
    def _process_level(self) -> int:
        process_by = self.recipe.get('process_by') or {}
        if 'admin_level' in process_by:
            return process_by['admin_level']
        return get_save_admin_level(self.entity_recipe)

    def _resolve_admin_ids(self, admin_ids: str | list | None) -> list[str]:
        """Resolve requested admin IDs to process-level IDs.

        Admin IDs deeper than the process level (e.g. a town within a
        county-level recipe) are grouped under their process-level
        ancestor in `self.sub_admin_ids`, which restricts image input
        and coverage-aware skip checks to those units.
        """
        process_level = self._process_level
        recipe_admin_id = self.entity_recipe['admin_id']
        self.sub_admin_ids: dict[str, list[str]] = {}

        if admin_ids is None:
            return get_admin_ids(process_level, admin_id=recipe_admin_id)

        if isinstance(admin_ids, str | AdminId):
            admin_ids = [admin_ids]

        expanded: list[str] = []
        for aid in admin_ids:
            aid = AdminId(aid) if not isinstance(aid, AdminId) else aid
            if aid.get_level() < process_level:
                expanded += get_admin_ids(process_level, admin_id=aid)
            elif aid.get_level() == process_level:
                expanded.append(str(aid))
            else:
                parent = str(AdminId(*aid.levels[:process_level]))
                self.sub_admin_ids.setdefault(parent, []).append(str(aid))
                expanded.append(parent)

        expanded = list(dict.fromkeys(expanded))

        # A full process-level request supersedes sub-level subsets.
        full = {
            str(AdminId(aid))
            for aid in admin_ids
            if AdminId(aid).get_level() <= process_level
        }
        for parent in list(self.sub_admin_ids):
            if any(
                AdminId(full_id).is_parent_or_equal_of(AdminId(parent))
                for full_id in full
            ):
                del self.sub_admin_ids[parent]

        invalid = [
            aid
            for aid in expanded
            if not recipe_admin_id.is_parent_or_equal_of(AdminId(aid))
        ]
        if invalid:
            raise ValueError(
                f'Admin IDs are not children of entity_recipe admin_id '
                f'({recipe_admin_id}): {invalid}'
            )

        return expanded

    def enrich(
        self,
        reprocess: bool = False,
        cleanup: str | None = None,
        include_images: bool = False,
    ) -> None:
        """Run enrichment for all configured admin IDs.

        Parameters
        ----------
        reprocess : bool
            If True, re-run even when existing evidence covers the request.
        cleanup : str, optional
            ``'consumed'`` reclaims this recipe's direct inputs after each
            admin unit finishes. The image cache is deleted only when
            *include_images* opts in AND every enrich recipe sharing the
            ``image_recipe`` has complete evidence for the unit (standard
            consumer refcounting — a partial coverage footer blocks it).
        include_images : bool
            Opt-in for image-cache deletion (paid re-fetch); falls back to
            retention.cleanup.include_images.
        """
        if cleanup not in (None, 'consumed'):
            raise ValueError(f"Unknown cleanup mode: {cleanup!r} (use 'consumed').")
        for admin_id_str in self.admin_ids:
            admin_id = AdminId(admin_id_str)
            sub_admin_ids = self.sub_admin_ids.get(admin_id_str)
            out_path = get_output_path(
                self.recipe,
                admin_id,
                entity_recipe_id=self.entity_recipe,
            )
            if not reprocess and self._is_covered(out_path, sub_admin_ids):
                if self.verbose:
                    print(f'[skip] {admin_id}: enrichment output exists.')
                continue
            if self.verbose:
                print(f'[enrich] {admin_id}')
                if sub_admin_ids:
                    print(f'  restricted to: {", ".join(sub_admin_ids)}')
            self._timer = get_timer(
                'Enricher',
                admin_id=str(admin_id),
                verbose=self.verbose,
                overwrite=True,
            )
            self._enrich_one(admin_id, sub_admin_ids)
            self._timer.finish()
            if cleanup == 'consumed':
                from openplaces.io.cleanup import cleanup_consumed_inputs

                cleanup_consumed_inputs(
                    self.recipe,
                    admin_id,
                    include_images=include_images,
                    verbose=self.verbose,
                )

    @staticmethod
    def _read_coverage(out_path) -> set[str] | None:
        """Return the admin units covered by existing output.

        None means complete coverage: either the file records the
        `_COVERAGE_ALL` sentinel, or it predates coverage tracking.
        """
        coverage = read_partition_coverage(out_path)
        if not coverage or _COVERAGE_ALL in coverage:
            return None
        return coverage

    def _is_covered(self, out_path, sub_admin_ids: list[str] | None) -> bool:
        """Check whether existing output already covers the request."""
        if not out_path.exists():
            return False
        coverage = self._read_coverage(out_path)
        if coverage is None:
            return True
        return sub_admin_ids is not None and set(sub_admin_ids) <= coverage

    def _enrich_one(
        self,
        admin_id: AdminId,
        sub_admin_ids: list[str] | None = None,
    ) -> None:
        pipeline = self.recipe.get('pipeline')
        if not pipeline:
            warnings.warn(
                f"Recipe has no 'pipeline' section; nothing to do for {admin_id}."
            )
            return

        spine_path = get_output_path(self.entity_recipe, admin_id)
        spine = read_parquet(spine_path, geom=bool(self.recipe.get('spine_geom')))
        evidence = pd.DataFrame(index=spine.index)

        state = EnrichState(
            recipe=self.recipe,
            entity_recipe=self.entity_recipe,
            admin_id=admin_id,
            verbose=self.verbose,
            timer=self._timer,
            spine=spine,
            evidence=evidence,
            image_admin_ids=sub_admin_ids,
        )

        for step_cfg in pipeline:
            step_name = step_cfg.get('step')
            if not step_name:
                raise ValueError(
                    f"Pipeline entry is missing the 'step' key: {step_cfg!r}"
                )
            fn = _STEP_REGISTRY.get(step_name)
            if fn is None:
                _load_steps()
                fn = _STEP_REGISTRY.get(step_name)
            if fn is None:
                raise ValueError(
                    f"Unknown enrich step: '{step_name}'. "
                    f'Registered steps: {", ".join(sorted(_STEP_REGISTRY))}.'
                )
            params = {k: v for k, v in step_cfg.items() if k != 'step'}
            state = fn(state, **params)

        out_path = get_output_path(
            self.recipe,
            admin_id,
            entity_recipe_id=self.entity_recipe,
        )
        evidence = state.evidence
        coverage = [_COVERAGE_ALL]
        if sub_admin_ids is not None:
            coverage = self._merged_coverage(out_path, sub_admin_ids)
            if out_path.exists():
                evidence = self._merge_evidence(
                    read_parquet(out_path),
                    evidence,
                    state.metadata.get('attempted_keys'),
                )
        save_parquet(
            evidence,
            out_path,
            file_metadata={'openplaces:partitions': json.dumps(sorted(coverage))},
        )
        # Resume checkpoints are superseded by the saved evidence.
        for checkpoint in state.metadata.get('checkpoints', []):
            checkpoint.delete()

    def _merged_coverage(self, out_path, sub_admin_ids: list[str]) -> list[str]:
        """Union the run's admin units with existing footer coverage."""
        if not out_path.exists():
            return list(sub_admin_ids)
        existing = self._read_coverage(out_path)
        if existing is None:
            # Complete before this run; updating units keeps it complete.
            return [_COVERAGE_ALL]
        return sorted(existing | set(sub_admin_ids))

    @staticmethod
    def _merge_evidence(
        existing: pd.DataFrame,
        new: pd.DataFrame,
        attempted_keys: set | None,
    ) -> pd.DataFrame:
        """Update existing evidence with rows attempted in this run.

        Rows whose keys were attempted (images processed, even when the
        result is missing) take the new values; all other rows keep the
        existing evidence.
        """
        if attempted_keys is None:
            return new.combine_first(existing.reindex(new.index))
        merged = new.copy()
        existing = existing.reindex(merged.index)
        keep = ~merged.index.isin(attempted_keys)
        for column in merged.columns.intersection(existing.columns):
            merged.loc[keep, column] = existing.loc[keep, column]
        return merged


def enrich(
    recipe: str | dict,
    admin_ids: str | list | None = None,
    entity_recipe_id: str | dict | None = None,
    reprocess: bool = False,
    verbose: bool = False,
    cleanup: str | None = None,
    include_images: bool = False,
) -> None:
    """Instantiate and run enrichment for *recipe*.

    ``cleanup='consumed'`` reclaims direct inputs after each admin unit;
    image caches additionally require ``include_images`` (or the
    retention.cleanup.include_images config) and complete evidence from
    every enrich recipe sharing the ``image_recipe``.
    """
    Enricher(
        recipe,
        admin_ids=admin_ids,
        entity_recipe_id=entity_recipe_id,
        verbose=verbose,
    ).enrich(reprocess=reprocess, cleanup=cleanup, include_images=include_images)


__all__ = [
    'EnrichState',
    'Enricher',
    'enrich',
    '_STEP_REGISTRY',
    '_register',
]
