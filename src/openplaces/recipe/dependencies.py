"""Recipe dependency edges (DepEdge, get_recipe_dependencies)."""

import re
from functools import cache
from typing import NamedTuple

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.core.schema import (
    AdminId,
    admin_scope_covers,
)
from openplaces.recipe.discovery import (
    find_entity_recipe_id,
)
from openplaces.recipe.loading import (
    get_recipe_id,
)
from openplaces.recipe.naming import (
    version_sort_key,
)


class DepEdge(NamedTuple):
    """One dependency edge: a recipe consuming another recipe's output.

    Attributes
    ----------
    recipe_id : str
        ID of the consuming recipe the edge was extracted from.
    upstream_recipe_id : str or None
        ID of the consumed recipe; None when auto-discovery could not
        resolve a concrete recipe (see `resolved`).
    kind : str
        Reference style: the recipe key the edge came from ('entity_recipe',
        'image_recipe', 'recipe_id', 'admin_recipe_id', 'tile_recipe_id',
        'footprint_recipe_id', ...) or 'auto_discover'.
    step : str or None
        Pipeline step name (or top-level recipe section) where the
        reference was found.
    resolved : bool
        False when an auto-discovered reference could not be resolved to a
        concrete recipe. Consumers of the dependency graph must treat
        unresolved edges as "may consume anything" (fail safe).
    """

    recipe_id: str
    upstream_recipe_id: str | None
    kind: str
    step: str | None = None
    resolved: bool = True


# Recipe keys referencing another recipe's output, e.g. 'recipe_id',
# 'admin_recipe_id', 'tile_recipe_id', 'footprint_recipe_id'. Keys like
# 'remap_id' do not match: value crosswalks are not data dependencies.
_RECIPE_ID_KEY_REGEX = re.compile(r'(^|_)recipe_id$')

# Pipeline steps whose auto-discovery expands to ALL applicable ingest
# recipes (mirroring the harmonizer's _expand_auto_discover), rather than
# the single best match.
_MULTI_DISCOVER_STEPS = ('resolve_spine', 'link_by_id', 'union_spine_sources')

# The multi-discover steps that build a spine from their sources. A
# recipe declaring `supplements:` is never one of their inputs (the
# harmonizer's _expand_auto_discover skips it); link_by_id still joins
# it, so it stays an input there.
_SPINE_BUILDING_STEPS = ('resolve_spine', 'union_spine_sources')


@cache
def _scan_ingest_recipe_ids(entity_type: str) -> tuple[dict, ...]:
    """List ingest recipes of an entity type, most specific and newest first.

    Mirrors the harmonizer's auto-discovery scan
    (io/harmonizer/discover.py) with recipe-layer machinery so dependency
    extraction resolves auto_discover references the same way the pipeline
    does at run time. That includes the dedup key: one recipe per
    (admin_id, source_id, filename_suffix), newest version, exactly as
    :func:`openplaces.io.harmonizer.links._find_admin_scoped_recipe_ids`
    does. Emitting every version instead made the older file a declared
    input, and a fingerprint input, of a job that never opens it, so
    touching or deleting it read as staleness.

    Reads the shared recipe index rather than globbing and parsing the
    tree a second time; the index is parsed once per process.
    """
    from openplaces.diagnostics import find_recipes

    best: dict[tuple[str, str, str], dict] = {}
    for _, row in find_recipes(entity_type, stage='ingest').iterrows():
        if row['exclude_from_auto_discover']:
            # The harmonizer's discovery skips these (a recipe kept out
            # of resolve_spine on purpose); the dependency scan must
            # agree, or a merely-existing excluded recipe changes the
            # dependency edge set - and with it every link-sidecar
            # fingerprint of the entity type, reading as region-wide
            # staleness for data that never changed.
            continue
        admin_id_str = row['admin_id']
        key = (admin_id_str, row['source_id'], row['filename_suffix'])
        version = str(row['version'] or '')
        candidate = {
            'recipe_id': row['recipe_id'],
            'admin_id': admin_id_str,
            'specificity': (len(admin_id_str.split('-')) if admin_id_str else 0),
            'version': version,
            'supplements': str(row.get('supplements') or ''),
            'supplements_key': str(row.get('supplements_key') or ''),
        }
        if key not in best or version_sort_key(version) > version_sort_key(
            best[key]['version']
        ):
            best[key] = candidate
    sources = sorted(
        best.values(),
        key=lambda s: (s['specificity'], version_sort_key(s['version'])),
        reverse=True,
    )
    return tuple(sources)


def get_recipe_dependencies(
    recipe, admin_id=None, exclude_recipe_ids: set[str] | None = None
) -> list[DepEdge]:
    r"""Extract upstream recipe references from a recipe.

    Edge sources (all present in committed recipes today):

    - top-level 'entity_recipe' (curate/enrich -> harmonized spine) and
      'image_recipe' (enrich -> image ingest); enrich recipes without an
      explicit 'entity_recipe' resolve their spine dynamically, mirroring
      the enricher
    - any key matching the suffix 'recipe_id' anywhere in the recipe
      ('recipe_id' in pipeline sources and steps, 'admin_recipe_id',
      'download_by.tile_recipe_id', 'footprint_recipe_id',
      'reference_parcel_recipe_id', merge_enrichments 'recipes' entries,
      ...); keys under a '\*crosswalk' block and 'remap_id' are excluded
      (value crosswalks, not data dependencies) -- unless the block names
      an 'admin_recipe_id', which reads that recipe's output to assign
      admin units and so is a real dependency
    - pipeline steps or source entries with 'auto_discover' or a bare
      'entity_type', resolved per admin unit the same way the pipeline
      resolves them at run time

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    admin_id : str or AdminId, optional
        Admin unit to resolve auto-discovered references for. When None,
        auto-discovered references are returned as unresolved edges.
    exclude_recipe_ids : set of str, optional
        Recipe IDs to prune from the graph. An excluded recipe's own edges
        are never evaluated (it's simply never emitted as an upstream), so
        anything only reachable through it is pruned transitively too,
        without needing to be named -- e.g. excluding an enrich recipe also
        excludes the ingest recipe it names via 'reference_parcel_recipe_id'.

    Returns
    -------
    list of DepEdge
        Unresolved auto-discovery is returned as an edge with
        upstream_recipe_id=None and resolved=False (fail safe: the caller
        must assume such a recipe may consume anything it protects).
    """
    if isinstance(recipe, str):
        recipe = _recipe.get_recipe_by_id(recipe)
    self_id = get_recipe_id(recipe)
    if admin_id is not None and not isinstance(admin_id, AdminId):
        admin_id = AdminId(admin_id)
    admin_str = str(admin_id) if admin_id is not None else None
    exclude_recipe_ids = exclude_recipe_ids or ()

    edges: list[DepEdge] = []
    seen: set[tuple] = set()

    def _add(upstream, kind, step=None, resolved=True):
        if upstream in exclude_recipe_ids:
            return
        key = (upstream, kind, step, resolved)
        if key not in seen:
            seen.add(key)
            edges.append(DepEdge(self_id, upstream, kind, step, resolved))

    # Top-level literal references. The generic *recipe_id walker below only
    # matches keys nested inside a dict/list value (pipeline steps, sources,
    # entity_links entries, ...); a top-level scalar field needs to be
    # listed here explicitly to be seen at all, even though its name already
    # matches _RECIPE_ID_KEY_REGEX.
    for key in ('entity_recipe', 'image_recipe', 'reference_parcel_recipe_id'):
        if recipe.get(key):
            _add(str(recipe[key]), key)

    # Enrich recipes without an explicit entity_recipe resolve their spine
    # dynamically; mirror io/enricher's _resolve_entity_recipe
    if recipe.get('stage') == 'enrich' and not recipe.get('entity_recipe'):
        entity = recipe.get('entity')
        entity_type = str(entity.entity_type) if entity is not None else None
        found = (
            find_entity_recipe_id(
                recipe.get('admin_id'),
                entity_type,
                stage='harmonize',
                source_id='spine',
                silent=True,
            )
            if entity_type
            else None
        )
        _add(found, 'entity_recipe', resolved=found is not None)

    # Generic *recipe_id keys anywhere in the recipe (pipeline steps and
    # sources, download_by/process_by blocks, merge_enrichments entries)
    def _walk(node, context):
        if isinstance(node, dict):
            step_name = node.get('step')
            if isinstance(step_name, str):
                context = step_name
            for key, value in node.items():
                if (
                    isinstance(key, str)
                    and key.endswith('crosswalk')
                    and not (isinstance(value, dict) and value.get('admin_recipe_id'))
                ):
                    # A value crosswalk (a remap table) reads no recipe
                    # output, so it is not a data dependency. An
                    # `admin_id_crosswalk` naming an `admin_recipe_id`
                    # is the opposite: it reads that recipe's output to
                    # decide which admin unit every row belongs to.
                    # Skipping it left `US-NC_parcel-nconemap-2025` with
                    # zero declared inputs, so rebuilding the admin layer
                    # could never invalidate it -- Camden kept a parcel
                    # file full of Columbus County under `mtime`
                    # rerun-triggers, and nothing said so.
                    continue
                # A mapping's keys are not always strings (a value map
                # keyed on an integer code); an int never names a recipe.
                if (
                    isinstance(key, str)
                    and isinstance(value, str)
                    and _RECIPE_ID_KEY_REGEX.search(key)
                ):
                    _add(value, key, step=context)
                else:
                    _walk(value, context)
        elif isinstance(node, list):
            for item in node:
                _walk(item, context)

    for key, value in recipe.items():
        if key in (
            'recipe_id',
            'entity_recipe',
            'image_recipe',
            'reference_parcel_recipe_id',
        ):
            continue
        # A top-level scalar matching the *recipe_id key convention is a
        # dependency too. `_walk` only sees keys nested inside a dict or
        # list value, so `reference_building_recipe_id` (declared at the
        # root by US-NC_footprint_building-cheer-v0 and read by
        # io/enricher/buildings.py) produced no edge at all: the enrich
        # job was neither ordered after that ingest nor listed it as an
        # input, and bundle_terms never reached the reference's licence.
        if (
            isinstance(key, str)
            and isinstance(value, str)
            and _RECIPE_ID_KEY_REGEX.search(key)
        ):
            _add(value, key)
            continue
        _walk(value, context=key)

    # Auto-discovered references in pipeline steps and their sources
    def _default_entity_type():
        entity = recipe.get('entity')
        return str(entity.entity_type) if entity is not None else None

    def _add_discovered(entity_type, step_name, multi, supplements_only=False):
        entity_type = entity_type or _default_entity_type()
        if entity_type is None or admin_id is None:
            _add(None, 'auto_discover', step=step_name, resolved=False)
            return
        additional = _recipe.find_additional_layer_recipes(entity_type, admin_id)
        if multi:
            # All strictly-more-specific ingest recipes covering the admin
            # unit (the harmonizer's _expand_auto_discover semantics); an
            # empty result means the step legitimately has no source here
            recipe_admin_str = str(recipe.get('admin_id') or '')
            # Through the package, like get_recipe_by_id: the tests patch
            # this name on `openplaces.recipe`.
            for src in _recipe._scan_ingest_recipe_ids(entity_type):
                if src.get('supplements') and step_name in _SPINE_BUILDING_STEPS:
                    continue
                # A supplement keyed on its own supplements_key relates
                # only to its roll's rows, so link_by_id joins it in a
                # supplements_only pass and nowhere else; listing it
                # for any other join would make it an input of a job
                # that never reads it.
                if src.get('supplements_key') and not supplements_only:
                    continue
                # Containment by level, mirroring the harmonizer: a raw
                # prefix test would make the pre-2026 'US-NC-WA' (Wake)
                # a dependency of 'US-NC-WAR' (Warren).
                rid = src['admin_id']
                if (
                    rid
                    and rid != recipe_admin_str
                    and admin_scope_covers(rid, admin_str)
                ):
                    _add(src['recipe_id'], 'auto_discover', step=step_name)
            for match in additional:
                _add(match['recipe_id'], 'auto_discover', step=step_name)
        else:
            found = find_entity_recipe_id(
                admin_id, entity_type, stage='ingest', silent=True
            )
            # If no standalone ingest recipe exists, check if a bundled
            # additional layer is available before flagging as unresolved.
            if found is None and additional:
                # Use the newest/best host recipe as the single match
                found = additional[-1]['recipe_id']
            _add(
                found,
                'auto_discover',
                step=step_name,
                resolved=found is not None,
            )

    for step_spec in recipe.get('pipeline') or []:
        if not isinstance(step_spec, dict):
            continue
        step_name = step_spec.get('step')
        multi = step_name in _MULTI_DISCOVER_STEPS
        sources = step_spec.get('sources')
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict) or source.get('recipe_id'):
                    continue
                if source.get('auto_discover') or source.get('entity_type'):
                    _add_discovered(source.get('entity_type'), step_name, multi)
        elif not step_spec.get('recipe_id') and (
            step_spec.get('auto_discover') or step_spec.get('entity_type')
        ):
            _add_discovered(
                step_spec.get('entity_type'),
                step_name,
                multi,
                supplements_only=bool(step_spec.get('supplements_only')),
            )

    return edges
