"""
Recipe-driven harmonization: pipeline architecture where each step is a
standalone function that reads from and writes to a shared ``HarmonizeState``.
The recipe's ``pipeline`` section declares which steps to run and with what
parameters, making the process composable and entity-type-agnostic.
"""

from __future__ import annotations

import json
import pkgutil as _pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module as _import_module
from pathlib import Path

import geopandas as gpd
import pandas as pd

from openplaces.core.attribute_registry import (
    PROVENANCE_SOURCE_SUFFIX as _PROVENANCE_SUFFIX,
)
from openplaces.core.schema import AdminId, SourceGeometryType
from openplaces.io import (
    coerce_mixed_object_columns,
    release_unused_memory,
    save_parquet,
)
from openplaces.io.cleanup import (
    cleanup_consumed_inputs,
    discard_receipt,
    receipt_justifies_skip,
)
from openplaces.io.readers import get_admin, get_admin_ids
from openplaces.recipe import get_output_path, get_recipe_by_id, saves_geometry
from openplaces.timing import get_timer

#: Footer key on a harmonized output's attribute parquet carrying the
#: pipeline metadata an attribute-only successor recipe needs to restore
#: (spine_index_name, spine_source_recipe_ids, spine_keep_columns) --
#: in-memory HarmonizeState.metadata does not survive the recipe split.
HARMONIZE_METADATA_KEY = 'openplaces:harmonize'


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
    step_index : int or None
        Position of the currently executing entry in the recipe's
        ``pipeline`` list, set by the dispatch loop before each step runs.
        Lets a step reason about what ran before it -- the link-sidecar
        fingerprint includes the configs of every *prior* geometry-phase
        step (see ``_STEP_PHASES``), so a changed spine threshold
        invalidates a persisted overlay downstream of it.
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
    step_index: int | None = None

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


def _attribute_checkpoint_path(recipe, admin_id):
    """Cache path of an attribute recipe's mid-pipeline checkpoint."""
    from openplaces.recipe import get_output_path

    out = get_output_path(recipe, admin_id)
    return out.with_name(out.stem + '_checkpoint.parquet')


def _checkpoint_chain(recipe, pipeline, upto, admin_id):
    """Validity fingerprint for a checkpoint after step *upto*.

    The ordered config hashes of every step up to and including the
    checkpointed one, plus the size/mtime of the entity_recipe output
    the pipeline loads (the geospine for the split spine recipes): a
    change to any covered step config or to the loaded input
    invalidates the checkpoint, a change to a later step does not.
    """
    import hashlib
    import json as _json

    from openplaces.io.harmonizer.links import _fingerprint_safe_step
    from openplaces.recipe import get_output_path, get_recipe_by_id

    hashes = []
    for step_cfg in pipeline[: upto + 1]:
        safe = _fingerprint_safe_step(step_cfg)
        hashes.append(
            hashlib.sha256(
                _json.dumps(safe, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
        )
    source = None
    entity_recipe_id = recipe.get('entity_recipe')
    if entity_recipe_id:
        try:
            upstream = get_recipe_by_id(str(entity_recipe_id))
            path = get_output_path(upstream, admin_id)
            stat = path.stat()
            source = {
                'path': path.name,
                'size': stat.st_size,
                'mtime': round(stat.st_mtime, 3),
            }
        except Exception:
            source = None
    return {'format': 1, 'steps': hashes, 'source': source}


def _load_attribute_checkpoint(recipe, admin_id, chain, verbose=False):
    """Return the checkpointed spine when *chain* still validates, else None."""
    import json as _json

    import geopandas as gpd
    import pyarrow.parquet as pq

    path = _attribute_checkpoint_path(recipe, admin_id)
    if not path.exists():
        return None
    try:
        meta = pq.read_schema(path).metadata or {}
        stored = _json.loads(meta[b'openplaces:checkpoint'])
    except Exception:
        return None
    if stored != chain:
        if verbose:
            print('  Checkpoint stale; running the full pipeline.')
        return None
    try:
        try:
            spine = gpd.read_parquet(path)
        except Exception:
            spine = pd.read_parquet(path)
    except Exception:
        return None
    if verbose:
        print(
            f'  Checkpoint: restored {len(spine):,d} rows after step '
            f'{len(chain["steps"])}; earlier steps skipped.'
        )
    return spine


def _save_attribute_checkpoint(recipe, admin_id, spine, chain, verbose=False):
    """Persist *spine* plus its validity *chain* in the parquet footer."""
    import json as _json

    import pyarrow.parquet as pq

    from openplaces.io import to_parquet

    path = _attribute_checkpoint_path(recipe, admin_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        to_parquet(coerce_mixed_object_columns(spine.copy()), path)
        table = pq.read_table(path)
        meta = dict(table.schema.metadata or {})
        meta[b'openplaces:checkpoint'] = _json.dumps(chain).encode()
        pq.write_table(table.replace_schema_metadata(meta), path)
        if verbose:
            print(f'  Checkpoint saved: {path.name}')
    except Exception as exc:
        warnings.warn(f'Could not save attribute checkpoint: {exc}')


def restrict_to_admin_by_name(df, recipe_id: str, admin_id: AdminId):
    """Fall back to a plain-text admin-name filter for an over-broad source.

    ``get_entities`` cannot restrict a non-spatial source to *admin_id* when
    the source has no matching admin-id column of its own (e.g. a statewide
    transaction table with only a free-text county name column, no
    ``admin3_id``) -- it returns the whole unfiltered table instead, which
    would otherwise duplicate that entire table into every child admin unit's
    output (or pool every admin unit's counts/aggregates together). Shared by
    :mod:`spine` (``union_spine_sources``) and :mod:`links` (``link_by_id``'s
    reference load), since both can hit the same over-broad source.

    When *df* carries an ``admin{level}_name`` column and its own recipe is
    scoped coarser than *admin_id*, filter locally by matching that column
    (case/whitespace-insensitively) against *admin_id*'s registered name. A
    no-op when no such column exists or the source is already scoped at or
    finer than *admin_id*.
    """
    if not isinstance(admin_id, AdminId):
        admin_id = AdminId(admin_id)
    level = admin_id.get_level()
    name_col = f'admin{level}_name'
    if name_col not in df.columns:
        return df
    source_admin_id = get_recipe_by_id(recipe_id)['admin_id']
    if not isinstance(source_admin_id, AdminId):
        source_admin_id = AdminId(source_admin_id)
    if source_admin_id.get_level() >= level:
        return df
    target = get_admin(admin_id, level)
    if target.empty:
        return df
    target_name = str(target['name'].iloc[0]).strip().casefold()
    match = df[name_col].astype('string').str.strip().str.casefold() == target_name
    return df[match]


def _ensure_object_source_column(df, column: str) -> None:
    """Ensure ``column`` exists on *df* and can hold string tokens.

    Creates it as an all-missing object column if absent; casts an existing
    Categorical or numeric column (e.g. an all-NaN float64 column a prior
    join left behind without ever populating it) to object first, since a
    numeric/categorical dtype can't hold an arbitrary source-id string.
    """
    if column not in df.columns:
        df[column] = pd.Series(pd.NA, index=df.index, dtype=object)
    elif isinstance(
        df[column].dtype, pd.CategoricalDtype
    ) or pd.api.types.is_numeric_dtype(df[column].dtype):
        df[column] = df[column].astype(object)


def _record_source(df, column: str, mask, token: str) -> None:
    """Set the ``{column}_source`` sidecar to *token* for the *mask* rows.

    Harmonize-stage peer of
    :func:`openplaces.io.curator.provenance.record_source` -- this package
    may not import from ``io.curator`` (a later pipeline stage), so this is
    a shared, independent copy for :mod:`spine` and :mod:`links` (mirroring
    how :mod:`addresses` already inlines its own private copy of the same
    idiom for the identical layering reason). *mask* may be a boolean
    Series aligned to *df*'s index.
    """
    side = f'{column}{_PROVENANCE_SUFFIX}'
    _ensure_object_source_column(df, side)
    df.loc[mask, side] = token


def _rename_right_index(
    gdf: pd.DataFrame,
    right_index_name: str | None,
    target: str,
) -> pd.DataFrame:
    """Rename the right-index column added by ``gpd.sjoin`` to *target*.

    geopandas names the right-index column ``'{name}_right'`` when there is a
    column-name conflict, ``'{name}'`` when there is no conflict (geopandas
    1.x), and ``'index_right'`` when the right index has no name.  This helper
    handles all three conventions. Shared by :mod:`links` and :mod:`spine`.
    """
    candidates = (
        [f'{right_index_name}_right', right_index_name, 'index_right']
        if right_index_name
        else ['index_right']
    )
    for col in candidates:
        if col in gdf.columns:
            return gdf.rename(columns={col: target})
    return gdf


#: Maps step name strings (as used in recipe ``pipeline`` sections) to the
#: callable that implements that step.
_STEP_REGISTRY: dict[str, Callable] = {}

#: Maps step name -> 'geometry' | 'attributes'. Geometry-phase steps mutate
#: spine rows/geometry or run spatial joins; their configs are part of the
#: link-sidecar fingerprint (a changed spine threshold must invalidate the
#: persisted overlay), and they are the steps a geospine recipe hosts under
#: the geometry/attribute recipe split. Attribute-phase steps only read or
#: annotate -- changing them never requires geometry rework.
_STEP_PHASES: dict[str, str] = {}


def _register(*names: str, phase: str = 'attributes'):
    """Decorator: register a step function under one or more names.

    Parameters
    ----------
    phase : str
        ``'geometry'`` for steps that mutate spine rows/geometry or run
        spatial joins (fingerprinted, geospine-hosted); ``'attributes'``
        (default) for steps that only read or annotate.
    """
    if phase not in ('geometry', 'attributes'):
        raise ValueError(f"phase must be 'geometry' or 'attributes', got {phase!r}")

    def decorator(fn: Callable) -> Callable:
        for name in names:
            _STEP_REGISTRY[name] = fn
            _STEP_PHASES[name] = phase
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


def _missing_link_sidecars(recipe, admin_id) -> list[Path]:
    """Link sidecars this recipe owes for `admin_id` but has not written.

    A geospine's contract is its spine *plus* one sidecar per
    `link_to_reference` join, and only the spine is checked before
    skipping. The two come apart whenever a run wrote the spine while an
    upstream ingest was failing: the spine exists, so the next run skips,
    and the attribute recipe fails with "missing or stale link sidecar"
    naming a geospine that looks finished. Treating a missing sidecar as
    "not done" closes that gap, so a repaired upstream is picked up
    without anyone having to reach for `reprocess`.

    A reference with no output of its own is *not* owed a sidecar. The
    linking step reads it with `missing='warn'` and writes nothing, so
    demanding one would re-run the geometry phase on every single pass
    for as long as that upstream stays unbuilt -- turning a warning into
    a permanent rebuild.
    """
    from openplaces.geo.link import get_entity_link_path
    from openplaces.io.harmonizer.links import _resolve_reference_recipe

    def _reference_is_built(reference_id) -> bool:
        try:
            reference = get_recipe_by_id(reference_id)
            return get_output_path(reference, admin_id).exists()
        except Exception:  # noqa: BLE001 - unresolvable is "not built"
            return False

    owed: list[Path] = []
    for step in recipe.get('pipeline') or []:
        if not isinstance(step, dict):
            continue
        if step.get('step') == 'link_to_reference':
            if step.get('save_link') is False:
                continue
        elif not step.get('save_link'):
            continue
        try:
            reference_id, _ = _resolve_reference_recipe(
                step.get('recipe_id'), step.get('entity_type'), admin_id
            )
        except Exception:  # noqa: BLE001 - an unresolvable reference is
            continue  # the linking step's error to raise, not ours
        if reference_id is None or not _reference_is_built(reference_id):
            continue
        path = get_entity_link_path(recipe['recipe_id'], reference_id, admin_id)
        if not path.exists():
            owed.append(path)
    return owed


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

    def harmonize(self, reprocess: bool = False, cleanup: str | None = None) -> None:
        """Run harmonization for all configured admin IDs.

        Parameters
        ----------
        reprocess : bool
            If ``False`` (default), skip admin IDs whose output file already
            exists.
        cleanup : str, optional
            ``'consumed'`` deletes this recipe's direct inputs after each
            admin unit finishes, iff every consumer in the recipe tree is
            complete (see
            :func:`~openplaces.io.cleanup.cleanup_consumed_inputs`).
        """
        if cleanup not in (None, 'consumed'):
            raise ValueError(f"Unknown cleanup mode: {cleanup!r} (use 'consumed').")
        if self._process_level == 0:
            self._run_global(reprocess=reprocess)
        else:
            for admin_id_str in self.admin_ids:
                admin_id = AdminId(admin_id_str)
                out_path = get_output_path(self.recipe, admin_id)
                if not reprocess and (
                    out_path.exists() or receipt_justifies_skip(self.recipe, admin_id)
                ):
                    missing = _missing_link_sidecars(self.recipe, admin_id)
                    if not missing:
                        if self.verbose:
                            print(f'[skip] {admin_id}: output exists.')
                        continue
                    if self.verbose:
                        print(
                            f'[rerun] {admin_id}: output exists but '
                            f'{len(missing)} link sidecar(s) are missing '
                            f'({missing[0].name}).'
                        )
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
                if cleanup == 'consumed':
                    cleanup_consumed_inputs(self.recipe, admin_id, verbose=self.verbose)
                release_unused_memory()

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

        # A pipeline step marked 'checkpoint: true' persists the spine
        # after it runs; a later rerun whose config chain up to that
        # step (and the loaded entity_recipe output) is unchanged
        # restores it and runs only the remaining steps. One checkpoint
        # per recipe, placed by the recipe author after the expensive
        # reconcile block: profiling (Nueces, 2026-08-25) puts the
        # load-plus-reconcile prefix at about two-thirds of an
        # attribute rerun. Delete the '_checkpoint.parquet' beside the
        # output to force a full rerun.
        checkpoint_index = next(
            (
                i
                for i, s in enumerate(pipeline)
                if isinstance(s, dict) and s.get('checkpoint')
            ),
            None,
        )
        resume_from = 0
        if checkpoint_index is not None:
            chain = _checkpoint_chain(self.recipe, pipeline, checkpoint_index, admin_id)
            restored = _load_attribute_checkpoint(
                self.recipe, admin_id, chain, verbose=self.verbose
            )
            if restored is not None:
                state.spine = restored
                resume_from = checkpoint_index + 1

        for step_index, step_cfg in enumerate(pipeline):
            if step_index < resume_from:
                continue
            state.step_index = step_index
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
            params = {
                k: v for k, v in step_cfg.items() if k not in ('step', 'checkpoint')
            }
            state = fn(state, **params)
            if step_index == checkpoint_index and resume_from == 0:
                _save_attribute_checkpoint(
                    self.recipe, admin_id, state.spine, chain, verbose=self.verbose
                )

        if state.spine is None:
            warnings.warn(f'Pipeline for {admin_id} produced no spine; nothing saved.')
            return

        # Restore the spine's original index name if resolve_spine renamed it to
        # avoid the spatial-overlay 'parcel_id' reference-level clash.
        restore_name = state.metadata.get('spine_index_name')
        if restore_name is not None and state.spine.index.name != restore_name:
            state.spine.index = state.spine.index.rename(restore_name)

        out_path = get_output_path(self.recipe, admin_id)
        # A source column can be pure numeric in one merged/linked reference
        # (e.g. an all-numeric-looking parcel PIN inferred as float64) and
        # genuine string in another; concatenation then yields a mixed
        # object column pyarrow cannot serialize. Same defensive cast used
        # at ingest save and partition aggregation.
        state.spine = coerce_mixed_object_columns(state.spine)

        # Persist the pipeline metadata a successor attribute-only recipe
        # needs (see load_geospine): in-memory state.metadata does not
        # survive the geometry/attribute recipe split, so the small,
        # JSON-able keys ride in the attribute parquet's footer.
        handoff = {}
        if state.metadata.get('spine_index_name') is not None:
            handoff['spine_index_name'] = state.metadata['spine_index_name']
        for key in ('spine_source_recipe_ids', 'spine_keep_columns'):
            if state.metadata.get(key):
                handoff[key] = sorted(state.metadata[key])
        file_metadata = (
            {HARMONIZE_METADATA_KEY: json.dumps(handoff)} if handoff else None
        )

        # An attribute-only recipe (save_to: geometry: false) writes a plain
        # table; its geometry lives with the entity_recipe predecessor and
        # is resolved by readers through that chain -- never duplicated.
        # Stale sidecars from a pre-split run are deleted, not just ignored:
        # a direct-parquet reader (e.g. qgis/load_joined_parquet) would
        # otherwise join the fresh attribute file's _join_id against the
        # stale sidecar's unrelated ids and get silently wrong geometry.
        if not saves_geometry(self.recipe):
            state.spine = pd.DataFrame(
                state.spine.drop(columns='geometry', errors='ignore')
            )
            for suffix in ('_geo', '_geo_simplified'):
                stale = out_path.with_stem(out_path.stem + suffix)
                stale.unlink(missing_ok=True)
        save_parquet(
            state.spine,
            out_path,
            simplified_geometry=state.simplified_geometry,
            file_metadata=file_metadata,
        )

    def show_random_entity(self):
        """Plot a random entity from the first configured admin unit.

        Delegates to :func:`openplaces.viz.maps.show_random_entity`.
        """
        from openplaces.viz.maps import show_random_entity

        admin_id = self.admin_ids[0] if self.admin_ids else None
        return show_random_entity(self.recipe, admin_id)


def harmonize(
    recipe: str | dict,
    admin_ids: str | list | None = None,
    reprocess: bool = False,
    verbose: bool = False,
    save_statistics: bool = False,
    cleanup: str | None = None,
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
    cleanup : str, optional
        ``'consumed'`` deletes this recipe's direct inputs after each admin
        unit finishes, iff every consumer in the recipe tree is complete.
    """
    Harmonizer(
        recipe,
        admin_ids=admin_ids,
        verbose=verbose,
        save_statistics=save_statistics,
    ).harmonize(reprocess=reprocess, cleanup=cleanup)


__all__ = [
    'HARMONIZE_METADATA_KEY',
    'Harmonizer',
    'HarmonizeState',
    'SourceGeometryType',
    'harmonize',
    '_STEP_REGISTRY',
    '_STEP_PHASES',
    '_register',
    '_record_source',
    '_ensure_object_source_column',
    '_rename_right_index',
]
