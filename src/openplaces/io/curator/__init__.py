"""
Recipe-driven creation of canonical entity datasets.

Curation starts from a harmonized entity, incorporates enrichment evidence,
and applies registered reconciliation, imputation, inference, and filtering
steps before saving a canonical entity dataset.
"""

from __future__ import annotations

import pkgutil as _pkgutil
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module as _import_module

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io import read_parquet, save_parquet
from openplaces.io.readers import get_admin_ids
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_save_admin_level,
)
from openplaces.timing import get_timer


@dataclass
class CurateState:
    """Mutable container passed through every curation pipeline step."""

    recipe: dict
    entity_recipe: dict
    admin_id: AdminId
    verbose: bool
    timer: object | None
    curated: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    save_statistics: bool = False


_STEP_REGISTRY: dict[str, Callable] = {}


def _register(*names: str):
    """Register a curation step under one or more recipe names."""

    def decorator(fn: Callable) -> Callable:
        for name in names:
            _STEP_REGISTRY[name] = fn
        return fn

    return decorator


_steps_loaded = False


def _load_steps() -> None:
    """Import this stage's step submodules so their @_register runs.

    Imports are deferred until a step is first dispatched so that merely
    importing the curator package (e.g. to use CurateState in a test) does not
    pull every step module.
    """
    global _steps_loaded
    if _steps_loaded:
        return
    for _m in _pkgutil.iter_modules(__path__):
        if not _m.ispkg:
            _import_module(f'{__name__}.{_m.name}')
    _steps_loaded = True


class Curator:
    """Create a canonical entity dataset from harmonized and enriched inputs."""

    def __init__(
        self,
        recipe: str | dict,
        admin_ids: str | list | None = None,
        verbose: bool = False,
        save_statistics: bool = False,
        skip_steps: str | list | set | None = None,
    ):
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        if recipe is None:
            raise ValueError(
                '`recipe` must be a recipe dict or a valid recipe ID string.'
            )
        stage = recipe.get('stage', 'ingest')
        if stage != 'curate':
            raise ValueError(
                f"Recipe stage is '{stage}', expected 'curate'. Pass a curation recipe."
            )

        entity_recipe_id = recipe.get('entity_recipe')
        if not entity_recipe_id:
            raise ValueError("Curate recipes require 'entity_recipe'.")

        self.recipe = recipe
        self.entity_recipe = get_recipe_by_id(entity_recipe_id)
        if self.entity_recipe.get('stage') != 'harmonize':
            raise ValueError(
                "Curate recipe 'entity_recipe' must reference a harmonization recipe."
            )
        self.verbose = verbose
        self.save_statistics = save_statistics or bool(
            recipe.get('save_statistics', False)
        )
        if skip_steps is None:
            self.skip_steps: set[str] = set()
        elif isinstance(skip_steps, str):
            self.skip_steps = {skip_steps}
        else:
            self.skip_steps = set(skip_steps)
        self._timer = None
        self.admin_ids = self._resolve_admin_ids(admin_ids)

    @property
    def _process_level(self) -> int:
        process_by = self.recipe.get('process_by') or {}
        if 'admin_level' in process_by:
            return process_by['admin_level']
        return get_save_admin_level(self.entity_recipe)

    def _resolve_admin_ids(self, admin_ids: str | list | None) -> list[str]:
        process_level = self._process_level
        recipe_admin_id = self.recipe['admin_id']

        if admin_ids is None:
            return get_admin_ids(process_level, admin_id=recipe_admin_id)
        if isinstance(admin_ids, str | AdminId):
            admin_ids = [admin_ids]

        expanded: list[str] = []
        for admin_id in admin_ids:
            admin_id = (
                AdminId(admin_id) if not isinstance(admin_id, AdminId) else admin_id
            )
            if admin_id.get_level() < process_level:
                expanded += get_admin_ids(process_level, admin_id=admin_id)
            elif admin_id.get_level() == process_level:
                expanded.append(str(admin_id))
            else:
                raise ValueError(
                    f'admin_id {admin_id} is at level {admin_id.get_level()}, '
                    f'which is deeper than process_level {process_level}.'
                )

        invalid = [
            admin_id
            for admin_id in expanded
            if not recipe_admin_id.is_parent_or_equal_of(AdminId(admin_id))
        ]
        if invalid:
            raise ValueError(
                f'Admin IDs are not children of recipe admin_id '
                f'({recipe_admin_id}): {invalid}'
            )
        return expanded

    def curate(self, reprocess: bool = False) -> None:
        """Run curation for all configured administrative units."""
        for admin_id_str in self.admin_ids:
            admin_id = AdminId(admin_id_str)
            out_path = get_output_path(self.recipe, admin_id)
            if not reprocess and out_path.exists():
                if self.verbose:
                    print(f'[skip] {admin_id}: curation output exists.')
                continue
            if self.verbose:
                print(f'[curate] {admin_id}')
            self._timer = get_timer(
                'Curator',
                admin_id=str(admin_id),
                verbose=self.verbose,
                overwrite=True,
            )
            self._curate_one(admin_id)
            self._timer.finish()

    def _curate_one(self, admin_id: AdminId) -> None:
        pipeline = self.recipe.get('pipeline')
        if not pipeline:
            warnings.warn(
                f"Recipe has no 'pipeline' section; nothing to do for {admin_id}."
            )
            return

        entity_path = get_output_path(self.entity_recipe, admin_id)
        curated = read_parquet(entity_path, geom=True)
        state = CurateState(
            recipe=self.recipe,
            entity_recipe=self.entity_recipe,
            admin_id=admin_id,
            verbose=self.verbose,
            timer=self._timer,
            curated=curated,
            save_statistics=self.save_statistics,
        )

        for step_cfg in pipeline:
            step_name = step_cfg.get('step')
            if not step_name:
                raise ValueError(
                    f"Pipeline entry is missing the 'step' key: {step_cfg!r}"
                )
            # Skip when disabled in the recipe (`enabled: false`) or named at the
            # call site (`skip_steps=[...]`). `enabled` is a control key, never a
            # step argument.
            if step_cfg.get('enabled', True) is False or step_name in self.skip_steps:
                if self.verbose:
                    print(f'  [skip] {step_name}')
                continue
            fn = _STEP_REGISTRY.get(step_name)
            if fn is None:
                _load_steps()
                fn = _STEP_REGISTRY.get(step_name)
            if fn is None:
                raise ValueError(
                    f"Unknown curate step: '{step_name}'. "
                    f'Registered steps: {", ".join(sorted(_STEP_REGISTRY))}.'
                )
            params = {
                key: value
                for key, value in step_cfg.items()
                if key not in ('step', 'enabled')
            }
            state = fn(state, **params)

        save_parquet(state.curated, get_output_path(self.recipe, admin_id))


def curate(
    recipe: str | dict,
    admin_ids: str | list | None = None,
    reprocess: bool = False,
    verbose: bool = False,
    save_statistics: bool = False,
    skip_steps: str | list | set | None = None,
) -> None:
    """Instantiate and run curation for *recipe*.

    When ``save_statistics`` is True (or the recipe sets ``save_statistics:
    true``), curation steps write diagnostic tables to the cache (e.g.
    geometry-indicator quantiles and use-group separability) without changing
    the curated output.

    Pass ``skip_steps`` (a step name or a collection of names) to skip
    computation-intensive pipeline steps for this run without editing the recipe;
    a recipe step may also be disabled persistently with ``enabled: false``.
    """
    Curator(
        recipe,
        admin_ids=admin_ids,
        verbose=verbose,
        save_statistics=save_statistics,
        skip_steps=skip_steps,
    ).curate(reprocess=reprocess)


__all__ = [
    'CurateState',
    'Curator',
    'curate',
    '_STEP_REGISTRY',
    '_register',
]
