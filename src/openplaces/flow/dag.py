"""
RecipeDAG: the recipe dependency graph behind orchestration.

Wraps get_recipe_dependencies and get_output_path so the recipes stay the
single source of truth: adding an enrichment recipe to a curation pipeline
automatically adds its jobs and its image-cache dependency. Consumed by
workflow/Snakefile; imports nothing from snakemake, so the library works
without it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.geo.link import get_entity_link_path
from openplaces.io.cleanup import _relative_posix, _walk_dag
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_retention,
    get_save_admin_level,
)

# 'deliver' is not a recipe stage (recipes rank ingest < harmonize <
# enrich < curate). It is a job this graph derives from the target
# recipe's own `share: delivery:` block: pool the region's curated
# files into the shareable bundle, once, after every county is built.
STAGES = ('ingest', 'harmonize', 'enrich', 'curate', 'deliver')

# Node fill colors per stage in to_mermaid() (pastel, dark text)
_STAGE_COLORS = {
    'ingest': '#cfe2f3',
    'harmonize': '#d9ead3',
    'enrich': '#fff2cc',
    'curate': '#f4cccc',
    'deliver': '#e6d0f0',
}


@dataclass(frozen=True)
class StageNode:
    """One orchestrated job: a (stage, recipe, admin unit) triple."""

    stage: str
    recipe_id: str
    admin_id: str | None


def rule_name(node: StageNode) -> str:
    """Snakemake rule name of one job (used by the Snakefile and --forcerun)."""
    raw = f'{node.stage}_{node.recipe_id}_{node.admin_id or "global"}'
    return re.sub(r'[^0-9a-zA-Z_]', '_', raw)


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
    exclude_recipe_ids : set of str, optional
        Recipe IDs to prune from the graph, along with everything only
        reachable through them (e.g. an opt-in enrichment lane and its own
        upstream ingest recipe). Composes with 'reference_parcel_recipe_id'/
        'image_recipe'/etc. edges automatically -- nothing further needs to
        be excluded by name. See `openplaces.recipe.get_recipe_dependencies`.
    deliver : bool, optional
        Force the delivery job on or off. None (default) decides from scope:
        the bundle is built when the run covers the region the target recipe
        declares, and skipped when it does not, so a scoped debug run leaves
        the shipped files alone. See `_delivery_in_scope`.
    """

    def __init__(
        self,
        target_recipe_id: str,
        admin_ids: list[str] | None = None,
        exclude_recipe_ids: set[str] | None = None,
        deliver: bool | None = None,
    ):
        self.target_recipe_id = target_recipe_id
        self.exclude_recipe_ids = set(exclude_recipe_ids or ())
        # Kept untruncated: the scope test below compares what was asked for,
        # not what it was narrowed to.
        self.requested_admin_ids = [str(a) for a in (admin_ids or [])]
        self._recipes: dict[str, dict] = {}
        target = self._recipe(target_recipe_id)
        target_level = get_save_admin_level(target)
        if not admin_ids and deliver is not False:
            # A recipe that declares a delivery region already says what
            # "everything" means for it, so an unscoped run builds that
            # region rather than a single admin-independent node it has no
            # rule to produce.
            admin_ids = self._delivery_members()
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
                target,
                target_admin,
                index=None,
                exclude_recipe_ids=self.exclude_recipe_ids,
            ):
                # _walk_dag truncates finer-saving recipes to the walk
                # admin; _node_admins re-expands them to their save level
                _add(node_id, node_recipe, target_admin)

        # Node-level edges (upstream -> consumer), for plan() and to_mermaid()
        self._edges: list[tuple[tuple, tuple]] = []
        edge_seen: set[tuple] = set()
        for node in self._nodes:
            consumer_key = (node.recipe_id, node.admin_id)
            for upstream_key in self._upstream_keys(node, seen):
                edge = (upstream_key, consumer_key)
                if edge not in edge_seen:
                    edge_seen.add(edge)
                    self._edges.append(edge)

        # Appended after the walk, not through _add: the delivery job shares
        # its recipe id with the curate jobs and is told apart only by its
        # coarser admin unit, which _add's (recipe_id, admin) dedup would
        # handle correctly but its admin expansion would not.
        self.delivery_nodes = self._build_delivery_nodes(deliver)
        for node, spec in self.delivery_nodes:
            self._nodes.append(node)
            consumer_key = (node.recipe_id, node.admin_id)
            for member in spec['admin_ids']:
                self._edges.append(((target_recipe_id, member), consumer_key))

    @property
    def delivery_node(self):
        """The single delivery job, for recipes that ship exactly one region.

        Kept so callers written against the one-region model still work;
        a recipe shipping several regions has no single node, and reading
        this raises rather than silently returning one of them.
        """
        if len(self.delivery_nodes) > 1:
            raise ValueError(
                f'{self.target_recipe_id!r} builds {len(self.delivery_nodes)} '
                f'delivery nodes; use .delivery_nodes.'
            )
        return self.delivery_nodes[0][0] if self.delivery_nodes else None

    def _delivery_members(self) -> list[str]:
        """Process-level admin units the target recipe's bundles pool.

        Pooled across every declared region: an unscoped run builds all of
        them, so "everything this recipe means" is the union, not one
        region's members.
        """
        from openplaces.io.delivery import delivery_regions

        members: list[str] = []
        for spec in delivery_regions(self._recipe(self.target_recipe_id)):
            members.extend(spec['admin_ids'])
        return list(dict.fromkeys(members))

    def _delivery_in_scope(self, spec: dict, bundle_admin_id) -> bool:
        """Whether this run covers one region the target recipe delivers.

        True when nothing was requested (the whole recipe), when something
        was requested at or above *bundle_admin_id* on its own branch of the
        hierarchy (``US`` or ``US-TX`` for a Texas bundle), or when the
        requested units include every declared member. False for a narrower
        run -- rebuilding one county must not overwrite a shipped regional
        file with a one-county one.

        The containment test is what keeps sibling regions apart: a recipe
        shipping both a Carolina and a Texas bundle must not ship Texas
        because someone asked for ``US-NC``, which a bare level comparison
        (both bundles sit at level 2) would do.
        """
        if not self.requested_admin_ids:
            return True
        bundle_levels = tuple(AdminId(str(bundle_admin_id)).levels)
        for requested in self.requested_admin_ids:
            levels = tuple(AdminId(requested).levels)
            if levels == bundle_levels[: len(levels)]:
                return True
        members = {str(a) for a in spec.get('admin_ids') or []}
        return bool(members) and members <= set(self.admin_ids)

    def _build_delivery_nodes(
        self, deliver: bool | None
    ) -> list[tuple[StageNode, dict]]:
        """One (node, region spec) per declared region this run covers.

        A curation recipe ships as many regions as it declares -- the CHEER
        footprint recipe ships Eastern NC and coastal Texas from identical
        curation logic -- so this is a list, not a single node.
        """
        from openplaces.io.delivery import delivery_admin_id, delivery_regions

        if deliver is False:
            return []
        recipe = self._recipe(self.target_recipe_id)
        nodes = []
        for spec in delivery_regions(recipe):
            admin_id = delivery_admin_id(recipe, region=spec['region_id'])
            ship = self._delivery_in_scope(spec, admin_id)
            # deliver=True overrides the narrow-run veto, but only for a
            # region this run actually touches: forcing a ship from a
            # handful of Texas counties must not also rebuild the
            # Carolina bundle, which those counties contribute nothing to.
            if not ship and deliver is True:
                ship = not self.requested_admin_ids or bool(
                    set(self.admin_ids) & {str(a) for a in spec['admin_ids']}
                )
            if ship:
                nodes.append(
                    (StageNode('deliver', self.target_recipe_id, str(admin_id)), spec)
                )
        return nodes

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

    def _upstream_keys(self, node: StageNode, node_keys: set[tuple]):
        """Yield (recipe_id, admin_str) keys of a node's in-DAG upstreams."""
        node_admin = AdminId(node.admin_id) if node.admin_id else None
        try:
            edges = get_recipe_dependencies(
                self._recipe(node.recipe_id),
                admin_id=node_admin,
                exclude_recipe_ids=self.exclude_recipe_ids,
            )
        except Exception:
            return
        seen: set[str] = set()
        for edge in edges:
            upstream_id = edge.upstream_recipe_id
            if not upstream_id or upstream_id in seen:
                continue
            seen.add(upstream_id)
            if node_admin is None:
                # A global job (no admin split) consumes every in-scope job
                # of the upstream recipe (e.g. the tile grid links to the
                # country-level admin job)
                for key in node_keys:
                    if key[0] == upstream_id:
                        yield key
                continue
            try:
                upstream_admins = self._node_admins(upstream_id, node_admin)
            except Exception:
                continue
            for upstream_admin in upstream_admins:
                key = (
                    upstream_id,
                    str(upstream_admin) if upstream_admin is not None else None,
                )
                if key in node_keys:
                    yield key

    def nodes(self) -> list[StageNode]:
        """Every job in the DAG (target included), deduplicated."""
        return list(self._nodes)

    def stage_nodes(self, stage: str) -> list[StageNode]:
        """The jobs of one pipeline stage."""
        return [node for node in self._nodes if node.stage == stage]

    def output_path(self, stage: str, recipe_id: str, admin_id=None) -> Path:
        """The primary output parquet of one job."""
        if stage == 'deliver':
            return self._delivery_paths(recipe_id, admin_id)['canonical']
        return get_output_path(
            self._recipe(recipe_id), admin_id=self._node_admin(recipe_id, admin_id)
        )

    def _delivery_region(self, recipe_id: str, admin_id=None) -> str | None:
        """The declared region whose bundle lands on *admin_id*.

        A deliver node is told apart from its siblings by the admin unit it
        covers, so that unit is what names the region here -- the reverse of
        `delivery_admin_id`.
        """
        from openplaces.io.delivery import delivery_admin_id, delivery_regions

        recipe = self._recipe(recipe_id)
        regions = delivery_regions(recipe)
        if len(regions) <= 1:
            return regions[0]['region_id'] if regions else None
        for spec in regions:
            if str(delivery_admin_id(recipe, region=spec['region_id'])) == str(
                admin_id
            ):
                return spec['region_id']
        raise KeyError(f'No delivery region of {recipe_id!r} covers {admin_id!r}.')

    def _delivery_paths(self, recipe_id: str, admin_id=None) -> dict:
        """The four bundle files, straight from the writer's own resolver.

        Declaring the orchestrator's expected outputs and writing the actual
        files from one function is what keeps the two from drifting apart.
        """
        from openplaces.io.delivery import delivery_paths

        return delivery_paths(
            self._recipe(recipe_id),
            region=self._delivery_region(recipe_id, admin_id),
        )

    def extra_outputs(self, stage: str, recipe_id: str, admin_id=None) -> list[Path]:
        """Secondary declared outputs of one job.

        Every harmonize `link_to_reference` step persists an n:m link
        sidecar at the canonical get_entity_link_path location unless it
        opts out with `save_link: false` (persistence is default-on: link
        products are first-class tables of the normalized store). Other
        steps declaring a truthy `save_link`, and ingest-level
        `entity_links` entries, persist one the same way.
        """
        if stage == 'deliver':
            bundle = self._delivery_paths(recipe_id, admin_id)
            return [bundle[role] for role in ('point', 'geo', 'evidence')]
        recipe = self._recipe(recipe_id)
        paths: list[Path] = []
        node_admin = self._node_admin(recipe_id, admin_id)
        for step in recipe.get('pipeline') or []:
            if not isinstance(step, dict):
                continue
            if step.get('step') == 'link_to_reference':
                if step.get('save_link') is False:
                    continue
            elif not step.get('save_link'):
                continue
            from openplaces.io.harmonizer.links import _resolve_reference_recipe

            ref_id, _ = _resolve_reference_recipe(
                step.get('recipe_id'), step.get('entity_type'), node_admin
            )
            if ref_id is not None:
                paths.append(get_entity_link_path(recipe_id, ref_id, node_admin))
        for entry in recipe.get('entity_links') or []:
            paths.append(
                get_entity_link_path(recipe_id, entry['recipe_id'], node_admin)
            )
        return paths

    def input_paths(self, stage: str, recipe_id: str, admin_id=None) -> list[Path]:
        """The input files of one job: upstream outputs plus link sidecars."""
        recipe = self._recipe(recipe_id)
        if stage == 'deliver':
            # Every member county's curated file, so the bundle rebuilds
            # whenever any one of them does. Scoped to this bundle's own
            # region: a Texas bundle must not wait on Carolina counties.
            from openplaces.io.delivery import delivery_members

            return [
                get_output_path(recipe, admin_id=member)
                for member in delivery_members(
                    recipe, region=self._delivery_region(recipe_id, admin_id)
                )
            ]
        node_admin = self._node_admin(recipe_id, admin_id)
        paths: list[Path] = []
        try:
            edges = get_recipe_dependencies(
                recipe, admin_id=node_admin, exclude_recipe_ids=self.exclude_recipe_ids
            )
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
                if upstream.get('stage') == 'harmonize' or upstream.get('entity_links'):
                    upstream_stage = upstream.get('stage', 'ingest')
                    paths.extend(
                        self.extra_outputs(upstream_stage, upstream_id, node_admin)
                    )
                    # Under the geometry/attribute split, the link sidecars
                    # this job actually opens are keyed by the *geospine*
                    # (the upstream spine's own entity_recipe), which is not
                    # a direct edge of this node -- resolve the owner the
                    # same way the curate readers do, so the declared inputs
                    # match the files read.
                    from openplaces.geo.link import get_link_owner_recipe_id

                    owner_id = get_link_owner_recipe_id(upstream)
                    if owner_id != upstream_id and owner_id not in seen:
                        seen.add(owner_id)
                        owner = self._recipe(owner_id)
                        paths.extend(
                            self.extra_outputs(
                                owner.get('stage', 'harmonize'), owner_id, node_admin
                            )
                        )
            except Exception:
                continue
        return paths

    def retention(self, stage: str, recipe_id: str, admin_id=None) -> str:
        """The retention class of one job's output (drives temp()/protected())."""
        return get_recipe_retention(self._recipe(recipe_id))

    def bucket(self, recipe_id: str) -> str:
        """The output bucket a recipe writes into ('cache', 'share', ...)."""
        save_to = self._recipe(recipe_id).get('save_to') or {}
        return save_to.get('data_dir', 'cache')

    def target_paths(self) -> list[Path]:
        """The terminal outputs the workflow must produce (rule all inputs).

        Every bundle this run delivers -- each depends on its own member
        counties, so targeting them still builds those -- otherwise the
        per-unit curated files.
        """
        if self.delivery_nodes:
            paths: list[Path] = []
            for node, _ in self.delivery_nodes:
                paths.extend(
                    self._delivery_paths(node.recipe_id, node.admin_id).values()
                )
            return paths
        return [
            self.output_path('curate', self.target_recipe_id, admin_id)
            for admin_id in self.admin_ids
        ]

    def plan(self) -> pd.DataFrame:
        """Preview which jobs would run and why (library-side, stat-only).

        The fast overview for interactive review; the authoritative
        scheduling decision is Snakemake's dry run. Reasons, in priority
        order: 'output missing', 'inputs newer than output' (mirrors the
        mtime rerun trigger), 'upstream scheduled (...)' (propagated along
        the DAG edges), or '' for an up-to-date job. One stat per file, no
        data reads.

        Returns
        -------
        pd.DataFrame
            One row per job: stage, recipe_id, admin_id, output, exists,
            size_mb, will_run, reason.
        """
        rows: dict[tuple, dict] = {}
        for node in self._nodes:
            key = (node.recipe_id, node.admin_id)
            try:
                out_path = self.output_path(node.stage, node.recipe_id, node.admin_id)
            except Exception:
                out_path = None
            exists = out_path is not None and out_path.exists()
            size_mb = (
                round(out_path.stat().st_size / 2**20, 1)
                if exists and out_path.is_file()
                else None
            )
            will_run, reason = False, ''
            if not exists:
                will_run, reason = True, 'output missing'
            else:
                out_mtime = out_path.stat().st_mtime
                try:
                    inputs = self.input_paths(node.stage, node.recipe_id, node.admin_id)
                except Exception:
                    inputs = []
                if any(p.exists() and p.stat().st_mtime > out_mtime for p in inputs):
                    will_run, reason = True, 'inputs newer than output'
            rows[key] = {
                'stage': node.stage,
                'recipe_id': node.recipe_id,
                'admin_id': node.admin_id,
                'output': _relative_posix(out_path) if out_path else None,
                'exists': exists,
                'size_mb': size_mb,
                'will_run': will_run,
                'reason': reason,
            }

        # Propagate scheduling downstream: a job re-runs when any of its
        # upstream jobs will run (fixpoint over the edge list)
        changed = True
        while changed:
            changed = False
            for upstream_key, consumer_key in self._edges:
                upstream, consumer = rows.get(upstream_key), rows.get(consumer_key)
                if upstream is None or consumer is None:
                    continue
                if upstream['will_run'] and not consumer['will_run']:
                    consumer['will_run'] = True
                    consumer['reason'] = f'upstream scheduled ({upstream_key[0]})'
                    changed = True

        stage_rank = {stage: rank for rank, stage in enumerate(STAGES)}
        report = pd.DataFrame(list(rows.values()))
        report['_rank'] = report['stage'].map(stage_rank)
        return (
            report.sort_values(['_rank', 'recipe_id', 'admin_id'])
            .drop(columns='_rank')
            .reset_index(drop=True)
        )

    def to_mermaid(
        self,
        collapse_admin: bool | None = None,
        direction: str = 'LR',
        font_size: int = 24,
        node_spacing: int = 20,
        rank_spacing: int = 35,
        width: int | None = None,
        height: int | None = None,
    ) -> str:
        """Mermaid flowchart source of the job DAG, styled by stage.

        Render with `IPython.display.Markdown` in a mermaid-capable
        frontend (VS Code notebooks), or paste into mermaid.live.

        Parameters
        ----------
        collapse_admin : bool, optional
            Collapse per-admin jobs into one node per recipe (labeled with
            the admin-unit count). None (default) auto-collapses when the
            full graph exceeds 30 nodes.
        direction : str, default 'LR'
            Mermaid flowchart direction: 'LR' (left-to-right), 'TB'
            (top-to-bottom, useful for wide graphs in a narrow notebook),
            'RL', or 'BT'.
        font_size : int, default 24
            Base font size in px (Mermaid `themeVariables.fontSize`).
            Mermaid's own default is 16px; 24 is ~1.5x larger. Also sets
            `flowchart.useMaxWidth: false`, so the diagram renders at its
            natural size instead of being scaled down to fit a narrow
            notebook cell (which would shrink the text back down along with
            everything else) -- the notebook scrolls instead.
        node_spacing : int, default 20
            Horizontal gap between nodes on the same rank (Mermaid
            `flowchart.nodeSpacing`, default 50) -- smaller packs the
            layout tighter.
        rank_spacing : int, default 35
            Gap between ranks along the flow direction (Mermaid
            `flowchart.rankSpacing`, default 50).
        width, height : int, optional
            Force the rendered SVG to a fixed pixel size via an injected
            `themeCSS` rule. Mermaid's flowchart renderer has no
            first-class width/height config, so this is a best-effort CSS
            override -- omit one to let that dimension size itself from
            content (typically: set only `width`, e.g. ~850 for a
            portrait-rotated US-letter-wide screen, and leave `height` to
            preserve the diagram's natural aspect ratio).
        """
        if direction not in ('LR', 'TB', 'RL', 'BT'):
            raise ValueError(f'direction must be one of LR/TB/RL/BT, got {direction!r}')

        if collapse_admin is None:
            collapse_admin = len(self._nodes) > 30

        def _group(key: tuple) -> tuple:
            return (key[0], None) if collapse_admin else key

        groups: dict[tuple, dict] = {}
        for node in self._nodes:
            info = groups.setdefault(
                _group((node.recipe_id, node.admin_id)),
                {'stage': node.stage, 'admins': set()},
            )
            if node.admin_id:
                info['admins'].add(node.admin_id)

        ids: dict[tuple, str] = {}
        used: set[str] = set()
        for group in groups:
            raw = f'{group[0]}_{group[1] or "all"}'
            gid = re.sub(r'[^0-9a-zA-Z_]', '_', raw)
            while gid in used:
                gid += '_'
            used.add(gid)
            ids[group] = gid

        init_config = {
            'theme': 'default',
            'themeVariables': {'fontSize': f'{font_size}px'},
            'flowchart': {
                'useMaxWidth': False,
                'nodeSpacing': node_spacing,
                'rankSpacing': rank_spacing,
            },
        }
        if width is not None or height is not None:
            css = ' '.join(
                f'{prop}: {value}px;'
                for prop, value in (('width', width), ('height', height))
                if value is not None
            )
            init_config['themeCSS'] = f'svg {{ {css} }}'

        lines = [f'%%{{init: {json.dumps(init_config)}}}%%', f'flowchart {direction}']
        for group, info in groups.items():
            label = group[0]
            if not collapse_admin and group[1]:
                label += f'<br/>{group[1]}'
            elif collapse_admin and len(info['admins']) > 1:
                label += f'<br/>({len(info["admins"])} admin units)'
            lines.append(f'    {ids[group]}["{label}"]')

        edge_seen: set[tuple] = set()
        for upstream_key, consumer_key in self._edges:
            pair = (ids[_group(upstream_key)], ids[_group(consumer_key)])
            if pair[0] != pair[1] and pair not in edge_seen:
                edge_seen.add(pair)
                lines.append(f'    {pair[0]} --> {pair[1]}')

        for stage, color in _STAGE_COLORS.items():
            members = [
                ids[group] for group, info in groups.items() if info['stage'] == stage
            ]
            if members:
                lines.append(f'    classDef {stage} fill:{color},stroke:#333;')
                lines.append(f'    class {",".join(members)} {stage};')
        return '\n'.join(lines)
