"""
Recipe-driven harmonization: pipeline architecture where each step is a
standalone function that reads from and writes to a shared ``HarmonizeState``.
The recipe's ``pipeline`` section declares which steps to run and with what
parameters, making the process composable and entity-type-agnostic.
"""

from __future__ import annotations

import pkgutil as _pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module as _import_module

import geopandas as gpd

from openplaces.core.schema import AdminId, SourceGeometryType
from openplaces.io import save_parquet
from openplaces.io.cleanup import discard_receipt, receipt_justifies_skip
from openplaces.io.readers import get_admin_ids
from openplaces.recipe import get_output_path, get_recipe_by_id
from openplaces.timing import get_timer


@dataclass
class HarmonizeState:
    """Mutable container passed through every pipeline step.

    Fields are populated progressively as steps run.

    Parameters
    ----------
    recipe : dict
        The loaded harmonization recipe driving this run.
    admin_id : AdminId or None
        Current admin unit being processed, or ``None`` for global runs.
    verbose : bool
        Print per-step progress messages.
    timer : object or None
        Timing helper (from :func:`~openplaces.timing.get_timer`).
    spine : GeoDataFrame or None
        The primary entity GeoDataFrame being built (e.g., footprints).
    references : dict[str, GeoDataFrame]
        Reference datasets keyed by resolved ``recipe_id``.
    crosswalks : dict[str, GeoDataFrame]
        Tabular spine ↔ reference join tables, keyed by resolved recipe_id.
    overlays : dict[str, GeoDataFrame]
        Geometry-bearing overlay results keyed by resolved recipe_id.
        Populated by ``link_to_reference`` for spatial_overlay joins and
        made available to subsequent steps (e.g., ``link_to_reference``
        for spatial_point that needs footprint-parcel geometries).
    reference_types : dict[str, str]
        Maps resolved recipe_id → entity_type (e.g.
        ``{'US-MA_parcel-mapc-2024': 'parcel'}``).  Lets steps look up all
        crosswalks of a given entity type without knowing the exact recipe_id.
    source_geometry_types : dict[str, SourceGeometryType]
        Maps resolved recipe_id → :class:`~openplaces.core.schema.SourceGeometryType`.
        Populated by ``link_to_reference`` when ``source_geometry_type`` is declared
        in the recipe step.  Used by ``classify_footprint_priority`` to identify which
        linked datasets are evidence of primary buildings.
    simplified_geometry : GeoSeries or None
        Set by ``simplify_geometries``; written as a sidecar by the save step.
    metadata : dict
        Arbitrary step-specific intermediate data (e.g. discovered admin
        sources, parcel-inferred footprint DataFrames).
    reprocess : bool
        True when the run was invoked with ``reprocess=True``; steps with
        persisted artifacts (e.g. ``link_to_reference`` with
        ``save_link``) must ignore and rewrite them.
    """

    recipe: dict
    admin_id: AdminId | None
    verbose: bool
    timer: object | None
    spine: gpd.GeoDataFrame | None = None
    references: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    crosswalks: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    overlays: dict[str, gpd.GeoDataFrame] = field(default_factory=dict)
    reference_types: dict[str, str] = field(default_factory=dict)
    source_geometry_types: dict[str, SourceGeometryType] = field(default_factory=dict)
    simplified_geometry: gpd.GeoSeries | None = None
    metadata: dict = field(default_factory=dict)
    save_statistics: bool = False
    reprocess: bool = False

    def get_crosswalks_by_type(self, entity_type: str) -> dict[str, gpd.GeoDataFrame]:
        """Return all crosswalks whose reference matches ``entity_type``."""
        return {
            rid: cw
            for rid, cw in self.crosswalks.items()
            if self.reference_types.get(rid) == entity_type
        }

    def get_references_by_type(self, entity_type: str) -> dict[str, gpd.GeoDataFrame]:
        """Return all reference GeoDataFrames matching ``entity_type``."""
        return {
            rid: ref
            for rid, ref in self.references.items()
            if self.reference_types.get(rid) == entity_type
        }


#: Maps step name strings (as used in recipe ``pipeline`` sections) to the
#: callable that implements that step.
_STEP_REGISTRY: dict[str, Callable] = {}


def _register(*names: str):
    """Decorator: register a step function under one or more names."""

    def decorator(fn: Callable) -> Callable:
        for name in names:
            _STEP_REGISTRY[name] = fn
        return fn

    return decorator


_steps_loaded = False


def _load_steps() -> None:
    """Import this stage's step submodules so their @_register runs.

    Deferred until a step is first dispatched so that merely importing the
    harmonizer package does not pull every step module.
    """
    global _steps_loaded
    if _steps_loaded:
        return
    for _m in _pkgutil.iter_modules(__path__):
        if not _m.ispkg:
            _import_module(f'{__name__}.{_m.name}')
    _steps_loaded = True


class Harmonizer:
    """Recipe-driven harmonization via a composable step pipeline.

    Reads the ``pipeline`` list from the recipe and executes each declared
    step in order, passing a shared :class:`HarmonizeState` between them.
    All configuration (thresholds, sources, enrichment) lives in the recipe —
    there are no entity-type-specific subclasses.

    For recipes with ``process_by.admin_level > 0``, harmonization runs once
    per admin unit.  For recipes with ``process_by.admin_level == 0``
    (e.g., global admin geometry), harmonization runs once globally.

    Parameters
    ----------
    recipe : str or dict
        Harmonization recipe ID string or pre-loaded dict.
    admin_ids : str, list, or None
        Admin IDs to harmonize.  IDs coarser than ``process_by.admin_level``
        are automatically expanded to matching children.  ``None`` processes
        all children of the recipe's ``admin_id`` at the process level.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        recipe: str | dict,
        admin_ids: str | list | None = None,
        verbose: bool = False,
        save_statistics: bool = False,
    ):
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        if recipe is None:
            raise ValueError(
                '`recipe` must be a recipe dict or a valid recipe ID string.'
            )
        stage = recipe.get('stage', 'ingest')
        if stage != 'harmonize':
            raise ValueError(
                f"Recipe stage is '{stage}', expected 'harmonize'. "
                'Pass a harmonization recipe, not an ingest recipe.'
            )
        self.recipe = recipe
        self.verbose = verbose
        self.save_statistics = save_statistics or bool(
            recipe.get('save_statistics', False)
        )
        self._timer = None

        process_level = self._process_level
        if process_level == 0:
            self.admin_ids: list[str] = []
            return

        recipe_admin_id = recipe['admin_id']
        if admin_ids is None:
            self.admin_ids = get_admin_ids(process_level, admin_id=recipe_admin_id)
        else:
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
                    raise ValueError(
                        f'admin_id {aid} is at level {aid.get_level()}, '
                        f'which is deeper than process_level {process_level}.'
                    )
            self.admin_ids = expanded

        invalid = [
            a
            for a in self.admin_ids
            if not recipe_admin_id.is_parent_or_equal_of(AdminId(a))
        ]
        if invalid:
            raise ValueError(
                f'Admin IDs are not children of recipe admin_id '
                f'({recipe_admin_id}): {invalid}'
            )

    @property
    def _process_level(self) -> int:
        process_by = self.recipe.get('process_by') or {}
        level = process_by.get('admin_level')
        if level is not None:
            return level
        admin_id = self.recipe.get('admin_id')
        if admin_id is None:
            return 0
        return admin_id.get_level()

    def harmonize(self, reprocess: bool = False) -> None:
        """Run harmonization for all configured admin IDs.

        Parameters
        ----------
        reprocess : bool
            If ``False`` (default), skip admin IDs whose output file already
            exists.
        """
        if self._process_level == 0:
            self._run_global(reprocess=reprocess)
        else:
            for admin_id_str in self.admin_ids:
                admin_id = AdminId(admin_id_str)
                out_path = get_output_path(self.recipe, admin_id)
                if not reprocess and (
                    out_path.exists() or receipt_justifies_skip(self.recipe, admin_id)
                ):
                    if self.verbose:
                        print(f'[skip] {admin_id}: output exists.')
                    continue
                if reprocess:
                    discard_receipt(out_path)
                if self.verbose:
                    print(f'[harmonize] {admin_id}')
                self._timer = get_timer(
                    'Harmonizer',
                    admin_id=str(admin_id),
                    verbose=self.verbose,
                    overwrite=True,
                )
                self._harmonize_one(admin_id, reprocess=reprocess)
                self._timer.finish()

    def _run_global(self, reprocess: bool = False) -> None:
        """Run a single global harmonization (process_level == 0)."""
        admin_level = self.recipe.get('admin_level', '')
        label = f'admin{admin_level}' if admin_level else 'global'
        out_path = get_output_path(self.recipe, admin_id=None)
        if not reprocess and (
            out_path.exists() or receipt_justifies_skip(self.recipe, None)
        ):
            if self.verbose:
                print(f'[skip] {label}: output exists.')
            return
        if reprocess:
            discard_receipt(out_path)
        if self.verbose:
            print(f'[harmonize] {label}')
        self._timer = get_timer(
            'Harmonizer',
            admin_id=label,
            verbose=self.verbose,
            overwrite=True,
        )
        self._harmonize_one(None, reprocess=reprocess)
        self._timer.finish()

    def _harmonize_one(self, admin_id: AdminId | None, reprocess: bool = False) -> None:
        """Build a :class:`HarmonizeState` and execute the recipe pipeline."""
        pipeline = self.recipe.get('pipeline')
        if not pipeline:
            warnings.warn(
                f"Recipe has no 'pipeline' section; nothing to do for {admin_id}."
            )
            return

        state = HarmonizeState(
            recipe=self.recipe,
            admin_id=admin_id,
            verbose=self.verbose,
            timer=self._timer,
            save_statistics=self.save_statistics,
            reprocess=reprocess,
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
                    f"Unknown pipeline step: '{step_name}'. "
                    f'Registered steps: {", ".join(sorted(_STEP_REGISTRY))}.'
                )
            params = {k: v for k, v in step_cfg.items() if k != 'step'}
            state = fn(state, **params)

        if state.spine is None:
            warnings.warn(f'Pipeline for {admin_id} produced no spine; nothing saved.')
            return

        # Restore the spine's original index name if resolve_spine renamed it to
        # avoid the spatial-overlay 'parcel_id' reference-level clash.
        restore_name = state.metadata.get('spine_index_name')
        if restore_name is not None and state.spine.index.name != restore_name:
            state.spine.index = state.spine.index.rename(restore_name)

        out_path = get_output_path(self.recipe, admin_id)
        save_parquet(
            state.spine,
            out_path,
            simplified_geometry=state.simplified_geometry,
        )


def harmonize(
    recipe: str | dict,
    admin_ids: str | list | None = None,
    reprocess: bool = False,
    verbose: bool = False,
    save_statistics: bool = False,
) -> None:
    """Instantiate and run harmonization for *recipe*.

    Convenience wrapper around ``Harmonizer(recipe, ...).harmonize()``.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string or loaded recipe dict.
    admin_ids : str, list, or None
        Admin IDs to process.
    reprocess : bool
        If True, re-run even if output already exists.
    verbose : bool
        Print progress messages.
    save_statistics : bool
        If True (or the recipe sets ``save_statistics: true``), write diagnostic
        tables (e.g. the parcel-NSI occupancy linkage) to the cache without
        changing the harmonized output.
    """
    Harmonizer(
        recipe,
        admin_ids=admin_ids,
        verbose=verbose,
        save_statistics=save_statistics,
    ).harmonize(reprocess=reprocess)


__all__ = [
    'Harmonizer',
    'HarmonizeState',
    'SourceGeometryType',
    'harmonize',
    '_STEP_REGISTRY',
    '_register',
]
