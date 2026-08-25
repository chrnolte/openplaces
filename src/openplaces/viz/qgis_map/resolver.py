"""Resolve the layers a standardized QGIS map should show for a recipe.

Given a curate-stage recipe and an admin unit, :func:`resolve_layers` walks
the recipe's dependency graph down to its ingest-stage sources and returns
the file paths a QGIS project generator can clone template layers for. This
module is pure Python (no ``qgis.core``) and safe to import in any conda
environment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from openplaces.core.schema import AdminId
from openplaces.recipe import (
    find_entity_recipe_id,
    get_output_path,
    get_recipe_by_id,
    get_recipe_dependencies,
    get_recipe_id,
    get_save_admin_level,
)

# How a cloned template layer should be re-rendered for its data.
# 'default' keeps the template's own symbology, which assumes polygons
# carrying their own attributes.
RENDER_DEFAULT = 'default'
# Recolor the template's categorized fills as point markers: the delivery
# bundle's `_point` file carries the canonical attributes on centroids, so
# it drives every attribute-styled view without loading any polygon.
RENDER_POINTS = 'points'
# Strip symbology down to a plain outline. The bundle's `_geo` file has no
# attributes at all, so it can only be shown, not classified.
RENDER_OUTLINE = 'outline'
# Keep the template's classified fills on polygon geometry, restyled for
# a delivery view: category-colored opaque boundaries over a 30%-opacity
# fill of the same color, attributes hash-joined from the canonical
# table when the polygon file carries none.
RENDER_POLYGONS = 'polygons'


@dataclass(frozen=True)
class LayerSpec:
    """One layer to include in a generated QGIS map.

    Attributes
    ----------
    role : str
        One of 'output' (the curated recipe itself), 'input' (an
        ingest-stage source it depends on), or 'admin' (administrative
        boundary context).
    recipe_id : str
        ID of the recipe this layer's data comes from.
    entity_type : str
        Entity type, e.g. 'footprint'.
    source : str or None
        Entity source, e.g. 'obm'.
    version : str or None
        Entity version.
    admin_id : `openplaces.core.schema.AdminId`
        Admin unit the paths were resolved at (may be coarser than the
        requested unit when the recipe saves at a parent level).
    display_name : str
        Filename stem to give the cloned layer, e.g.
        'US-MA-BOS_footprint-obm-2025'.
    attr_path, geo_path : pathlib.Path
        Resolved attribute and geometry parquet paths. Equal when
        `combined` is True.
    exists : bool
        Whether the underlying file(s) exist on disk.
    depth : int
        Distance from the curate recipe in the dependency graph (0 for the
        output itself).
    combined : bool
        True when the recipe saves attributes and geometry together in one
        file (`save_to: combined: true`, e.g. terminal share-ready
        deliverables) rather than as a joinable attr/`_geo` pair.
    render : str
        How to re-render the cloned template layer: `RENDER_DEFAULT` keeps
        the template's symbology, `RENDER_POINTS` converts its categorized
        fills to point markers, `RENDER_OUTLINE` replaces it with a plain
        boundary line and drops the attribute-driven style variants.
    """

    role: str
    recipe_id: str
    entity_type: str
    source: str | None
    version: str | None
    admin_id: AdminId
    display_name: str
    attr_path: Path
    geo_path: Path
    exists: bool
    depth: int
    combined: bool
    render: str = RENDER_DEFAULT
    # Presentation overrides, used by delivery bundle layers. `label`
    # replaces the file stem as the visible layer name (the stem moves to
    # the end); `group_override` re-homes the layer out of its style's
    # group; the outline fields recolor/reweight a RENDER_OUTLINE layer.
    label: str | None = None
    group_override: str | None = None
    outline_color: str | None = None
    outline_width: str | None = None
    # None defers to the style's default_visible; a bool forces it. Used
    # to keep the joined polygon views unchecked so opening the map does
    # not classify millions of polygons.
    visible: bool | None = None
    # QGIS label expression for the layer (e.g. name over admin id); the
    # size is in points. Labels stay off when the expression is None.
    label_expression: str | None = None
    label_size: str = '9'
    label_opacity: str = '1'
    label_buffer: str = '0.8'
    # None: the label layer's checkbox follows the base layer's
    # visibility. False keeps a label layer available but off at open
    # (the admin4 labels crowd the initial view).
    label_visible: bool | None = None


def _entity_parts(recipe: dict) -> tuple[str, str | None, str | None]:
    entity = recipe.get('entity') or recipe.get('dataset')
    if entity is None:
        return '', None, None
    entity_type = str(getattr(entity, 'entity_type', entity))
    source = entity.source if hasattr(entity, 'source') else None
    source = str(source) if source is not None else None
    version = getattr(entity, 'version', None)
    return entity_type, source, version


def _build_spec(
    recipe: dict,
    resolved_admin_id: AdminId | None,
    *,
    role: str,
    depth: int,
    verbose: bool,
) -> LayerSpec | None:
    """Build a LayerSpec for *recipe*'s output at *resolved_admin_id*.

    *resolved_admin_id* must already be at the recipe's own save admin
    level (callers are responsible for truncating); this function does not
    adjust it further.

    Recipes saved with `save_to: combined: true` (attributes and geometry
    in one file, e.g. terminal share-ready deliverables) have no separate
    `_geo` sidecar; `attr_path` and `geo_path` are the same path in that
    case and only that one file needs to exist.
    """
    combined = bool((recipe.get('save_to') or {}).get('combined', False))
    try:
        attr_path = get_output_path(recipe, admin_id=resolved_admin_id)
        geo_path = (
            attr_path
            if combined
            else get_output_path(recipe, admin_id=resolved_admin_id, geo=True)
        )
    except Exception as exc:
        if verbose:
            warnings.warn(
                f'Could not resolve output path for {get_recipe_id(recipe)}: {exc}'
            )
        return None

    entity_type, source, version = _entity_parts(recipe)
    exists = (
        attr_path.exists() if combined else attr_path.exists() and geo_path.exists()
    )
    return LayerSpec(
        role=role,
        recipe_id=get_recipe_id(recipe),
        entity_type=entity_type,
        source=source,
        version=version,
        admin_id=resolved_admin_id,
        display_name=attr_path.stem,
        attr_path=attr_path,
        geo_path=geo_path,
        exists=exists,
        depth=depth,
        combined=combined,
    )


def _discover_ingest_leaves(
    recipe: dict, admin_id: AdminId, *, verbose: bool
) -> dict[str, int]:
    """BFS `get_recipe_dependencies` down to ingest-stage recipe IDs.

    Returns a mapping of ingest recipe ID to the depth (BFS distance from
    *recipe*) at which it was first discovered. Harmonize/enrich
    intermediates are walked through but not returned; unresolved
    auto-discovery edges are skipped (logged when *verbose*).
    """
    visited = {get_recipe_id(recipe)}
    frontier = [recipe]
    ingest_seen: dict[str, int] = {}
    depth = 0
    while frontier:
        depth += 1
        next_frontier = []
        for node in frontier:
            try:
                edges = get_recipe_dependencies(node, admin_id=admin_id)
            except Exception as exc:
                if verbose:
                    warnings.warn(
                        f'Could not resolve dependencies for '
                        f'{get_recipe_id(node)}: {exc}'
                    )
                continue
            for edge in edges:
                if edge.upstream_recipe_id is None:
                    if verbose:
                        warnings.warn(
                            f'Unresolved auto-discover dependency of '
                            f'{edge.recipe_id!r} at step {edge.step!r}; skipping.'
                        )
                    continue
                if edge.upstream_recipe_id in visited:
                    continue
                visited.add(edge.upstream_recipe_id)
                try:
                    upstream = get_recipe_by_id(edge.upstream_recipe_id)
                except Exception as exc:
                    if verbose:
                        warnings.warn(
                            f'Could not load recipe {edge.upstream_recipe_id!r}: {exc}'
                        )
                    continue
                if upstream.get('stage', 'ingest') == 'ingest':
                    ingest_seen.setdefault(edge.upstream_recipe_id, depth)
                else:
                    next_frontier.append(upstream)
        frontier = next_frontier
    return ingest_seen


def _build_delivery_specs(
    recipe: dict, admin_id: AdminId, *, verbose: bool
) -> list[LayerSpec]:
    """Build the output layers for a recipe's region-wide delivery bundles.

    The curated output is written per process unit, so its own save level
    cannot resolve a path for the coarser unit a bundle covers --
    `get_output_path` rejects it outright. At that scope the map is built
    from the bundle instead, as two layers that each stand alone:

    - the `_point` centroids, which carry the canonical attributes and so
      drive every attribute-styled view;
    - the `_geo` polygons, shown as plain outlines.

    Neither is joined. The alternative -- joining the canonical table onto
    the polygons -- would make QGIS read and index 2.5M polygons before it
    could draw anything classified, for a view whose colors are legible
    from the centroids alone.

    A recipe ships as many regions as it declares, so this emits the pair
    for every region whose own unit is *admin_id* or a descendant of it --
    the same containment test `RecipeDAG._delivery_in_scope` applies, so
    the map and the orchestrator agree on scope. A `US` map of the CHEER
    recipe carries both the Carolina and the Texas bundle, `US-TX` carries
    Texas alone, and a county map carries neither (its per-county curated
    file is the right layer, and the caller falls back to it when this
    returns nothing). Containment rather than a bare level comparison is
    what keeps sibling regions apart: both CHEER bundles sit at level 2,
    but asking for `US-NC` must not ship Texas.
    """
    from openplaces.io.delivery import (
        delivery_admin_id,
        delivery_paths,
        delivery_regions,
    )

    entity_type, source, version = _entity_parts(recipe)
    requested_levels = tuple(admin_id.levels)

    def _spec(
        path: Path,
        render: str,
        bundle_admin: AdminId,
        attr_path: Path | None = None,
        **presentation,
    ) -> LayerSpec:
        return LayerSpec(
            role='output',
            recipe_id=get_recipe_id(recipe),
            entity_type=entity_type,
            source=source,
            version=version,
            admin_id=bundle_admin,
            display_name=path.stem,
            attr_path=attr_path or path,
            geo_path=path,
            exists=path.exists(),
            depth=0,
            combined=True,
            render=render,
            **presentation,
        )

    specs: list[LayerSpec] = []
    for region in delivery_regions(recipe):
        region_id = region.get('region_id')
        bundle_admin = delivery_admin_id(recipe, region=region_id)
        bundle_levels = tuple(bundle_admin.levels)
        if requested_levels != bundle_levels[: len(requested_levels)]:
            continue
        try:
            paths = delivery_paths(recipe, region=region_id)
        except Exception as exc:
            if verbose:
                warnings.warn(f'Could not resolve delivery paths: {exc}')
            continue
        # Buildings split into two sibling groups: every classified view
        # exists once on centroid markers and once on the boundary
        # polygons, with the same category colors. The polygon twins are
        # solid category-colored boundaries over a 30%-opacity fill; the
        # polygon file carries only geometry, so they join the canonical
        # table on the shared entity id.
        specs.append(
            _spec(
                paths['point'],
                RENDER_POINTS,
                bundle_admin,
                label='Occupancy type',
                group_override='Buildings/Point geometries',
            )
        )
        specs.append(
            _spec(
                paths['geo'],
                RENDER_POLYGONS,
                bundle_admin,
                # The point file, not the canonical table: same columns
                # plus the point-only ratios (structure_value_per_area),
                # so every point view has a polygon twin. Subset joins
                # keep the cache cost identical.
                attr_path=paths['point'],
                label='Occupancy type',
                visible=False,
                group_override='Buildings/Polygon geometries',
            )
        )
        specs.append(
            _spec(
                paths['geo'],
                RENDER_OUTLINE,
                bundle_admin,
                label='Polygons',
                group_override='Buildings/Polygon geometries',
            )
        )
        # Administrative context that ships WITH the bundle. The map must
        # stay openable from the delivery folder alone, so county and
        # county-subdivision outlines are aggregate sidecar files written
        # beside the bundle (see `ensure_delivery_admin_outlines`), never
        # references back into this machine's cache tree. Their own layer
        # group sits above the entity groups; purple outlines, the finer
        # (admin4) level thinner than its parent. admin4 is optional --
        # New England has no county subdivisions below its towns -- so a
        # missing sidecar is simply not offered.
        canonical = paths['canonical']
        admin_presentation = {
            'admin3': {
                'label': 'Counties',
                'outline_width': '0.33',
                'label_expression': 'concat("name", char(10), "admin3_id")',
                'label_size': '9',
                'label_opacity': '0.8',
            },
            'admin4': {
                'label': 'County subdivisions',
                'outline_width': '0.18',
                'label_expression': 'concat("name", char(10), "admin4_id")',
                'label_size': '9',
                'label_opacity': '0.8',
                'label_buffer': '0.4',
                'label_visible': False,
            },
        }
        for suffix, pres in admin_presentation.items():
            side = canonical.with_name(f'{canonical.stem}_{suffix}_geo.parquet')
            if side.exists():
                specs.append(
                    _spec(
                        side,
                        RENDER_OUTLINE,
                        bundle_admin,
                        group_override='Administrative units',
                        outline_color='128,0,128,76',
                        **pres,
                    )
                )
    return specs


def resolve_layers(
    recipe: str | dict,
    admin_id: str | AdminId,
    *,
    admin_context_levels: tuple[int, ...] = (1,),
    filter_existing: bool = True,
    include_inputs: bool = True,
    verbose: bool = False,
) -> list[LayerSpec]:
    """Resolve the map layers for a curate recipe at an admin unit.

    Discovers the curated output itself, every ingest-stage recipe it
    depends on (via :func:`openplaces.recipe.get_recipe_dependencies`,
    walked recursively through any harmonize/enrich intermediates), and
    administrative boundary context layers.

    Parameters
    ----------
    recipe : str or dict
        Curate-stage recipe ID or loaded recipe dict.
    admin_id : str or `openplaces.core.schema.AdminId`
        Admin unit to resolve output paths for. Must match the curate
        recipe's own save admin level.
    admin_context_levels : tuple of int, optional
        Admin levels (in addition to *admin_id*'s own level) to include
        boundary context layers for, e.g. ``(1,)`` for country context.
    filter_existing : bool, optional
        If True (default), drop layers whose parquet files do not exist on
        disk for this admin unit.
    include_inputs : bool, optional
        If False, omit the ingest-stage source layers and return only the
        curated output plus administrative context -- the shape wanted for
        a map of the delivered product rather than of how it was built.
        Defaults to True.
    verbose : bool, optional
        If True, warn about unresolved dependencies and skipped layers.

    Returns
    -------
    list of `LayerSpec`
        Ordered by (role, depth, entity_type, source) for deterministic
        output.
    """
    recipe = get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe
    stage = recipe.get('stage', 'ingest')
    if stage != 'curate':
        raise ValueError(
            f"Recipe stage is {stage!r}, expected 'curate'. Pass a curation recipe."
        )
    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)

    specs: list[LayerSpec] = []

    delivery_specs = _build_delivery_specs(recipe, admin_id, verbose=verbose)
    if delivery_specs:
        specs.extend(delivery_specs)
    else:
        output_spec = _build_spec(
            recipe, admin_id, role='output', depth=0, verbose=verbose
        )
        if output_spec is not None:
            specs.append(output_spec)

    ingest_seen = (
        _discover_ingest_leaves(recipe, admin_id, verbose=verbose)
        if include_inputs
        else {}
    )
    for recipe_id, depth in ingest_seen.items():
        try:
            upstream = get_recipe_by_id(recipe_id)
        except Exception as exc:
            if verbose:
                warnings.warn(f'Could not load recipe {recipe_id!r}: {exc}')
            continue
        resolved_admin = admin_id.truncate_to_level(get_save_admin_level(upstream))
        spec = _build_spec(
            upstream, resolved_admin, role='input', depth=depth, verbose=verbose
        )
        if spec is not None:
            specs.append(spec)

    levels = sorted(
        {lvl for lvl in admin_context_levels if 0 < lvl} | {admin_id.get_level()}
    )
    seen_admin_recipe_ids: set[str] = set()
    for level in levels:
        if level > admin_id.get_level():
            continue
        context_admin = admin_id.truncate_to_level(level)
        admin_recipe_id = find_entity_recipe_id(
            context_admin, 'admin', stage='ingest', silent=True
        )
        if admin_recipe_id is None or admin_recipe_id in seen_admin_recipe_ids:
            continue
        seen_admin_recipe_ids.add(admin_recipe_id)
        admin_recipe = get_recipe_by_id(admin_recipe_id)
        resolved_admin = admin_id.truncate_to_level(get_save_admin_level(admin_recipe))
        spec = _build_spec(
            admin_recipe, resolved_admin, role='admin', depth=0, verbose=verbose
        )
        if spec is not None:
            specs.append(spec)

    if filter_existing:
        dropped = [s for s in specs if not s.exists]
        if verbose and dropped:
            for s in dropped:
                warnings.warn(
                    f'Dropping {s.recipe_id!r}: no data on disk for {s.admin_id}.'
                )
        specs = [s for s in specs if s.exists]

    specs.sort(key=lambda s: (s.role, s.depth, s.entity_type, s.source or ''))
    return specs
