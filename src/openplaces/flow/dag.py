"""
RecipeDAG: the recipe dependency graph behind orchestration.

Wraps get_recipe_dependencies and get_output_path so the recipes stay the
single source of truth: adding an enrichment recipe to a curation pipeline
automatically adds its jobs and its image-cache dependency. Consumed by
workflow/Snakefile; imports nothing from snakemake, so the library works
without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io.cleanup import _walk_dag
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_retention,
    get_save_admin_level,
)

STAGES = ('ingest', 'harmonize', 'enrich', 'curate')


@dataclass(frozen=True)
class StageNode:
    """One orchestrated job: a (stage, recipe, admin unit) triple."""

    stage: str
    recipe_id: str
    admin_id: str | None


class RecipeDAG:
    """Dependency graph of every job needed to build a terminal recipe.

    Parameters
    ----------
    target_recipe_id : str
        Terminal recipe (e.g. 'US_footprint-cheer-2026').
    admin_ids : list of str, optional
        Admin units in scope. Coarser IDs are used as-is; each node's
        admin unit is truncated to its recipe's save level. None builds a
        DAG of admin-independent nodes only (auto-discovered references
        stay unresolved).
    """

    def __init__(self, target_recipe_id: str, admin_ids: list[str] | None = None):
        self.target_recipe_id = target_recipe_id
        self._recipes: dict[str, dict] = {}
        target = self._recipe(target_recipe_id)
        target_level = get_save_admin_level(target)
        self.admin_ids = [
            str(AdminId(*AdminId(str(a)).levels[:target_level]))
            for a in (admin_ids or [])
        ] or [None]

        self._nodes: list[StageNode] = []
        seen: set[tuple[str, str | None]] = set()

        def _add(recipe_id, recipe, walk_admin):
            for node_admin in self._node_admins(recipe_id, walk_admin):
                admin_str = str(node_admin) if node_admin is not None else None
                key = (recipe_id, admin_str)
                if key in seen:
                    continue
                seen.add(key)
                self._nodes.append(
                    StageNode(recipe.get('stage', 'ingest'), recipe_id, admin_str)
                )

        for admin_id in self.admin_ids:
            target_admin = AdminId(admin_id) if admin_id else None
            _add(target_recipe_id, target, target_admin)
            for node_id, node_recipe, node_admin in _walk_dag(
                target, target_admin, index=None
            ):
                # _walk_dag truncates finer-saving recipes to the walk
                # admin; _node_admins re-expands them to their save level
                _add(node_id, node_recipe, target_admin)

    def _recipe(self, recipe_id: str) -> dict:
        if recipe_id not in self._recipes:
            self._recipes[recipe_id] = get_recipe_by_id(recipe_id)
        return self._recipes[recipe_id]

    def _node_admin(self, recipe_id: str, admin_id) -> AdminId | None:
        """Truncate an admin unit to a recipe's save level."""
        if admin_id is None:
            return None
        admin_id = AdminId(str(admin_id))
        level = min(get_save_admin_level(self._recipe(recipe_id)), admin_id.get_level())
        if level <= 0:
            return None
        return AdminId(*admin_id.levels[:level])

    def _node_admins(self, recipe_id: str, admin_id) -> list[AdminId | None]:
        """Admin units of a recipe's jobs within one walk admin unit.

        Coarser-saving recipes truncate the walk admin; finer-saving ones
        (e.g. per-town image caches under a county walk) expand into the
        child units at the recipe's save level. Expansion needs the admin
        boundaries on disk; when they are not ingested yet, the recipe's
        jobs are omitted with a warning (ingest admin data first).
        """
        if admin_id is None:
            return [None]
        admin_id = AdminId(str(admin_id))
        save_level = get_save_admin_level(self._recipe(recipe_id))
        if save_level <= admin_id.get_level():
            return [self._node_admin(recipe_id, admin_id)]
        try:
            from openplaces.io.readers import get_admin_ids

            return [
                AdminId(child) for child in get_admin_ids(save_level, admin_id=admin_id)
            ]
        except Exception:
            import warnings

            warnings.warn(
                f'Cannot expand {recipe_id} to admin level {save_level} '
                f'under {admin_id} (admin boundaries not ingested yet); '
                'its jobs are omitted from the DAG.'
            )
            return []

    def nodes(self) -> list[StageNode]:
        """Every job in the DAG (target included), deduplicated."""
        return list(self._nodes)

    def stage_nodes(self, stage: str) -> list[StageNode]:
        """The jobs of one pipeline stage."""
        return [node for node in self._nodes if node.stage == stage]

    def output_path(self, stage: str, recipe_id: str, admin_id=None) -> Path:
        """The primary output parquet of one job."""
        return get_output_path(
            self._recipe(recipe_id), admin_id=self._node_admin(recipe_id, admin_id)
        )

    def extra_outputs(self, stage: str, recipe_id: str, admin_id=None) -> list[Path]:
        """Secondary declared outputs of one job (save_link sidecars)."""
        recipe = self._recipe(recipe_id)
        paths: list[Path] = []
        node_admin = self._node_admin(recipe_id, admin_id)
        for step in recipe.get('pipeline') or []:
            if not (isinstance(step, dict) and step.get('save_link')):
                continue
            from openplaces.io.harmonizer.links import _resolve_reference_recipe

            ref_id, _ = _resolve_reference_recipe(
                step.get('recipe_id'), step.get('entity_type'), node_admin
            )
            if ref_id is not None:
                paths.append(get_entity_link_path(recipe_id, ref_id, node_admin))
        return paths

    def input_paths(self, stage: str, recipe_id: str, admin_id=None) -> list[Path]:
        """The input files of one job: upstream outputs plus link sidecars."""
        recipe = self._recipe(recipe_id)
        node_admin = self._node_admin(recipe_id, admin_id)
        paths: list[Path] = []
        try:
            edges = get_recipe_dependencies(recipe, admin_id=node_admin)
        except Exception:
            edges = []
        seen: set[str] = set()
        for edge in edges:
            upstream_id = edge.upstream_recipe_id
            if not upstream_id or upstream_id in seen:
                continue
            seen.add(upstream_id)
            try:
                upstream = self._recipe(upstream_id)
                for upstream_admin in self._node_admins(upstream_id, node_admin):
                    paths.append(get_output_path(upstream, admin_id=upstream_admin))
                if upstream.get('stage') == 'harmonize':
                    paths.extend(
                        self.extra_outputs('harmonize', upstream_id, node_admin)
                    )
            except Exception:
                continue
        return paths

    def retention(self, stage: str, recipe_id: str, admin_id=None) -> str:
        """The retention class of one job's output (drives temp()/protected())."""
        return get_recipe_retention(self._recipe(recipe_id))

    def bucket(self, recipe_id: str) -> str:
        """The output bucket of a recipe ('share' outputs get protected())."""
        save_to = self._recipe(recipe_id).get('save_to') or {}
        return save_to.get('data_dir', 'cache')

    def target_paths(self) -> list[Path]:
        """The terminal outputs the workflow must produce (rule all inputs)."""
        return [
            self.output_path('curate', self.target_recipe_id, admin_id)
            for admin_id in self.admin_ids
        ]
