"""Find recipes for an admin unit: find_entity_recipe_id,
find_additional_layer_recipes, layers, supplemented tables.
"""

import glob
from functools import cache
from pathlib import Path

import yaml

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.core.constants import (
    RECIPE_STAGES,
)
from openplaces.core.schema import (
    AdminId,
    admin_scope_covers,
)
from openplaces.path import recipe_path
from openplaces.recipe.loading import (
    STACKED_UNITS_LAYER_KEY,
    get_recipe_id,
)
from openplaces.recipe.naming import (
    version_sort_key,
)
from openplaces.recipe.tables import (
    get_recipe_output_columns,
    get_table_recipe,
)


@cache
def _recipe_yaml(filepath: str) -> dict:
    """Parse one recipe file, cached for the life of the process.

    The tree is static while a process runs, and the resolvers here read
    the same files repeatedly: measured on a 3-county graph, this pattern
    cost 8,173 parses of 182 distinct files. Same caveat as
    `openplaces.diagnostics._recipe_index`: a session that edits a recipe
    on disk has to call `_recipe_yaml.cache_clear()` to see it. The
    returned dict is shared, so callers read it and never mutate it.

    Parameters
    ----------
    filepath : str
        Path to the recipe .yaml file.

    Returns
    -------
    dict
        The parsed recipe, or an empty dict for an empty file.
    """
    with open(filepath, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def find_entity_recipe_id(
    admin_id,
    entity_type,
    stage: str | None = None,
    source_id: str | None = None,
    filename: str | None = None,
    silent: bool = False,
):
    """Find the most suitable entity recipe.

    Recipes follow the pipeline order ingest, harmonize, enrich, curate unless
    *stage* is specified. Within a stage, prefer the most specific applicable
    administrative scope, then the latest version.

    *source_id* filters rather than ranks. It used to be a preference, so a
    caller asking for a source with no recipe silently received a different
    source's: the enricher's request for the harmonized spine
    (`source_id='spine'`) would fall through to the geospine, which is the
    same entity type at the same stage. Every caller tests only for None, so
    the substitution reached the data instead of the error.

    Parameters
    ----------
    admin_id : str or AdminId
        Admin unit the recipe must cover.
    entity_type : str
        Entity type the recipe must produce.
    stage : str, optional
        Restrict to one pipeline stage; without it the latest stage wins.
    source_id : str, optional
        Restrict to recipes of this source. Returns None when none matches.
    filename : str, optional
        Filename stem to match within the recipe directory.
    silent : bool
        Suppress the message printed when several recipes qualify.
    """
    admin_id = AdminId(admin_id) if not isinstance(admin_id, AdminId) else admin_id
    recipe_paths_found = []
    for level in range(admin_id.get_level(), -1, -1):
        scope_admin_id = AdminId(*admin_id.levels[:level])
        glob_recipe_path = recipe_path(
            scope_admin_id,
            f'{entity_type}-*-*',
            filename=filename,
        )
        recipe_paths_found.extend(glob.glob(str(glob_recipe_path)))
        evidence_recipe_path = recipe_path(
            scope_admin_id,
            entity_type,
            filename=filename or '*',
        )
        recipe_paths_found.extend(glob.glob(str(evidence_recipe_path)))

    candidates = []
    stage_rank = {stage: rank for rank, stage in enumerate(RECIPE_STAGES)}

    for filepath in sorted(set(recipe_paths_found)):
        recipe_data = _recipe_yaml(str(filepath))
        recipe_stage = recipe_data.get('stage') or 'ingest'
        if stage is not None and recipe_stage != stage:
            continue
        # A supplement details another recipe's entities and is never
        # "the" recipe for them. Without this skip it won the final
        # tie-break on its longer filename: Victoria TX's property
        # ingest resolved to its improvement-detail table, not its roll.
        # Asking for it by filename still finds it.
        if filename is None and recipe_data.get('supplements'):
            continue
        # A patch recipe amends another recipe's pipeline for its own
        # scope (apply_recipe_patches). It ranks finer than the recipe
        # it patches, so without this skip a county's two-line patch
        # would be picked as the county's curate recipe.
        if filename is None and recipe_data.get('patches'):
            continue
        entity = recipe_data.get('entity') or {}
        if entity.get('entity_type') != entity_type:
            continue
        recipe_admin_id = AdminId(recipe_data.get('admin_id'))
        if not recipe_admin_id.is_parent_or_equal_of(admin_id):
            continue
        source = entity.get('source') or {}
        recipe_source_id = source.get('source_id', '')
        if source_id is not None and recipe_source_id != source_id:
            continue
        candidates.append(
            (
                stage_rank.get(recipe_stage, -1),
                recipe_admin_id.get_level(),
                version_sort_key(entity.get('version', '')),
                Path(filepath).stem,
            )
        )

    if not candidates:
        return None
    candidates.sort()
    recipe_id = candidates[-1][3]
    if len(candidates) > 1 and not silent:
        print(f'Picked {recipe_id} for {admin_id} ({entity_type}).')
    return recipe_id


#: Parquet footer key naming the patch recipes a table was built with.
def get_layers(recipe: str | dict) -> list[str]:
    """Return the layer names available for a recipe's 'additional_layers'.

    These are the values accepted by the `layer` argument of 'get_entities'
    and 'get_output_path'.

    Parameters
    ----------
    recipe : str or dict
        Recipe dict or recipe ID string.

    Returns
    -------
    list of str
        Entity type strings (e.g. 'property', 'transaction') for each entry
        in 'additional_layers'.
    """
    if isinstance(recipe, str):
        recipe = _recipe.get_recipe_by_id(recipe)
    return [
        str(layer_spec['entity'].entity_type)
        for layer_spec in recipe.get('additional_layers', [])
        if 'entity' in layer_spec
    ]


def _declared_entity_type(recipe: dict) -> str | None:
    """Return a recipe's entity type as a string, or None if it has none."""
    entity = recipe.get('entity')
    entity_type = getattr(entity, 'entity_type', None)
    return str(entity_type) if entity_type is not None else None


def get_supplemented_table(recipe: dict) -> dict | None:
    """Return the table a supplement details, validated against it.

    A supplement (a recipe declaring ``supplements: <recipe id>``) adds
    columns to the entities of another ingest table, its roll, and never
    rows of its own. The roll is usually a recipe in its own right. It
    can also be an ``additional_layers`` entry of a host recipe, which
    has no recipe id: MassGIS's statewide assessing table (L3_ASSESS)
    is the ``property`` layer of ``US-MA_parcel-massgis-2025``. A
    supplement of such a layer names the host and the layer's entity
    type::

        supplements: US-MA_parcel-massgis-2025
        supplements_layer: property

    The host id is what the property spine records for a layer source
    (``union_spine_sources`` adds the host's recipe id to
    ``spine_source_recipe_ids``), so the supplements join matches the
    declaration with no id form of its own for the layer.

    The roll's admin scope must contain the supplement's, tested by
    level with :func:`~openplaces.core.schema.admin_scope_covers`, never
    by string prefix: a city's assessing table may detail a statewide
    roll, as Boston's detailing the MassGIS layer does, and a table
    scoped outside its roll could never match one of its rows.

    Parameters
    ----------
    recipe : dict
        A loaded ingest recipe.

    Returns
    -------
    dict or None
        The roll's table recipe (for a layer, the host merged with its
        layer spec, see :func:`build_table_recipe`), or None when the
        recipe declares no ``supplements``.

    Raises
    ------
    ValueError
        If ``supplements_layer`` is set without ``supplements`` or is
        not a non-empty string; if the host has no additional layer of
        that entity type; if the roll is itself a supplement; if the
        roll's entity type differs from the supplement's (a supplement
        of a host whose own entity type differs must name the layer); or
        if the roll's scope does not contain the supplement's. Entity
        types and scopes are compared only where both recipes declare
        them.
    """
    recipe_id = get_recipe_id(recipe)
    roll_id = recipe.get('supplements')
    layer = recipe.get('supplements_layer')
    if not roll_id:
        if layer is not None:
            raise ValueError(
                f'{recipe_id} declares supplements_layer but no supplements; '
                'the layer names a table of the recipe it supplements.'
            )
        return None
    if layer is not None and (not isinstance(layer, str) or not layer):
        raise ValueError(
            f'{recipe_id}: supplements_layer must be an entity type, got {layer!r}.'
        )

    roll = _recipe.get_recipe_by_id(roll_id)
    if roll.get('supplements'):
        raise ValueError(
            f'{recipe_id} supplements {roll_id}, which is itself a supplement '
            f'(of {roll["supplements"]}); name the roll it details instead.'
        )
    available = get_layers(roll)
    if layer is None:
        table = roll
    elif layer in available:
        table = get_table_recipe(roll, layer)
    else:
        raise ValueError(
            f'{recipe_id} supplements the {layer!r} layer of {roll_id}, which '
            f'has no such additional layer (it has: {available or "none"}).'
        )

    own_type = _declared_entity_type(recipe)
    table_type = _declared_entity_type(table)
    if own_type and table_type and own_type != table_type:
        hint = (
            f' {roll_id} has a {own_type!r} layer: name it with '
            f'supplements_layer: {own_type}.'
            if layer is None and own_type in available
            else ''
        )
        raise ValueError(
            f'{recipe_id} is a {own_type} table but supplements a {table_type} '
            f'table ({roll_id}{f", layer {layer}" if layer else ""}); a '
            f'supplement adds columns to entities of its own type.{hint}'
        )

    own_scope, roll_scope = recipe.get('admin_id'), roll.get('admin_id')
    if (
        own_scope is not None
        and roll_scope is not None
        and not admin_scope_covers(roll_scope, own_scope)
    ):
        raise ValueError(
            f'{recipe_id} (scope {own_scope}) supplements {roll_id} (scope '
            f'{roll_scope}), whose scope does not contain it.'
        )
    return table


def find_additional_layer_recipes(
    layer_entity_type: str,
    admin_id: AdminId | str,
    host_entity_type: str = 'parcel',
) -> list[dict]:
    """Find additional_layers-bundled recipes of *layer_entity_type*.

    Walks every ingest recipe of *host_entity_type* (default ``'parcel'``)
    whose admin scope covers *admin_id* and returns each one that bundles a
    matching *layer_entity_type* entry in its ``additional_layers`` list --
    e.g. MassGIS's L3_ASSESS ``property`` table, bundled inside its
    ``US-MA_parcel-massgis-2025`` parcel recipe rather than registered as its
    own standalone ingest recipe. Neither ``find_recipes`` nor a plain
    ``entity_type`` search ever surfaces a bundled layer this way, since its
    entity type never appears as a directory path component of the host
    recipe's own YAML file. Used by
    :func:`openplaces.io.harmonizer.spine._expand_auto_discover` so
    ``union_spine_sources``/``resolve_spine`` auto-discovery can see a
    bundled layer whose host is a different entity type (e.g. a
    ``property`` table bundled inside a ``parcel`` recipe) -- the same
    additional_layers walk :func:`openplaces.io.harmonizer.links._discover_link_sources`
    already does inline for the *same*-entity-type case (host and layer both
    ``'parcel'``), which that function keeps doing itself rather than calling
    this one.

    Recipes with ``exclude_from_auto_discover: true`` are skipped, and only
    the newest version per (admin_id, source_id, filename_suffix) is kept,
    mirroring :func:`find_entity_recipe_id`'s specificity/version
    precedence and
    :func:`openplaces.io.harmonizer.links._find_admin_scoped_recipe_ids`'s
    dedup key. The suffix belongs in that key: two files sharing a source
    (`CO_parcel-igac-2026_rural` and `_urban`) are different tables
    meant to coexist, not competing versions of one.

    Parameters
    ----------
    layer_entity_type : str
        The bundled entity type to look for (e.g. ``'property'``).
    admin_id : AdminId or str
        Admin unit the host recipe must cover.
    host_entity_type : str, optional
        Entity type of the recipes to search for bundled layers on (default
        ``'parcel'``, the only host type in use today).

    Returns
    -------
    list of dict
        Each entry: ``{'recipe_id': host recipe id, 'layer':
        layer_entity_type, 'label': host source_id, 'layer_key': the join
        key declared on the layer, falling back to ``'parcel_id_local'``}``.
        Ordered oldest-version-first, mirroring
        :func:`find_entity_recipe_id`'s "most recent applied last"
        precedence for callers that fold matches together.
    """
    from openplaces.diagnostics import find_recipes

    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)
    best: dict[tuple[str, str, str], tuple[str, str, str]] = {}
    for _, row in find_recipes(host_entity_type, stage='ingest').iterrows():
        if row['exclude_from_auto_discover']:
            continue
        admin_id_str = row['admin_id']
        if not admin_id_str or not AdminId(admin_id_str).is_parent_or_equal_of(
            admin_id
        ):
            continue
        # Keyed on the filename suffix as well, and reading the recipe
        # id the index already carries rather than rebuilding it: a
        # rebuilt id drops the suffix, so sibling files such as
        # `CO_parcel-igac-2026_rural` and `_urban` collapsed onto one
        # another and then named a `CO_parcel-igac-2026` file that does
        # not exist, raising OSError in every Colombia spine run.
        key = (admin_id_str, row['source_id'], row['filename_suffix'])
        recipe_id = row['recipe_id']
        version = version_sort_key(row['version'])
        if key not in best or version > best[key][0]:
            best[key] = (version, recipe_id, row['source_id'])

    matches = []
    for _version, recipe_id, source_id in sorted(best.values(), key=lambda vrs: vrs[0]):
        recipe = _recipe.get_recipe_by_id(recipe_id)
        for layer_spec in recipe.get('additional_layers') or []:
            entity = layer_spec.get('entity')
            if entity is None or str(entity.entity_type) != layer_entity_type:
                continue
            matches.append(
                {
                    'recipe_id': recipe_id,
                    'layer': layer_entity_type,
                    'label': source_id,
                    'layer_key': layer_spec.get('layer_key', 'parcel_id_local'),
                    # The implicit layer of units split off a parcel
                    # table: a fallback wherever a tax roll describes
                    # the same lots (spine.union_spine_sources).
                    'stacked_units_layer': bool(
                        layer_spec.get(STACKED_UNITS_LAYER_KEY)
                    ),
                }
            )
    return matches


def get_supplements_key(recipe: dict) -> str | None:
    """Return a supplement's declared join key, validated against its roll.

    A supplement (a recipe declaring ``supplements: <roll id>``) joins
    its roll's entities on ``parcel_id_local`` by default. That fails
    where the roll's ``parcel_id_local`` is built from an id the detail
    table does not carry: Travis County TX's roll links to parcels on
    ``geo_id``, while its improvement detail carries only ``prop_id``.
    ``supplements_key: <column>`` names another column both tables carry,
    and the supplement then joins its roll on that column instead,
    leaving the roll's ``parcel_id_local`` free to serve the parcel link.

    Parameters
    ----------
    recipe : dict
        A loaded ingest recipe.

    Returns
    -------
    str or None
        The key column, or None when the recipe declares no
        ``supplements_key``.

    Raises
    ------
    ValueError
        If ``supplements_key`` is set on a recipe that declares no
        ``supplements``, is not a non-empty string, or names a column
        that the supplement or the roll it supplements does not produce
        (see :func:`get_recipe_output_columns`). A recipe that passes
        source columns through unnamed (*keep_unnamed_columns*) cannot be
        checked here; the join itself then raises if the column is
        missing from the data.
    """
    key = recipe.get('supplements_key')
    if key is None:
        return None
    recipe_id = get_recipe_id(recipe)
    roll_id = recipe.get('supplements')
    if not roll_id:
        raise ValueError(
            f'{recipe_id} declares supplements_key but no supplements; the '
            'key names the column it joins its roll on, so it needs a roll.'
        )
    if not isinstance(key, str) or not key:
        raise ValueError(
            f'{recipe_id}: supplements_key must be a column name, got {key!r}.'
        )
    # The table joined may be the roll itself or one of its additional
    # layers (supplements_layer), so the check reads the table the
    # supplement details, not the host recipe.
    supplemented = get_supplemented_table(recipe)
    for name, table in ((recipe_id, recipe), (roll_id, supplemented)):
        produced = get_recipe_output_columns(table)
        if produced is not None and key not in produced:
            raise ValueError(
                f'{recipe_id} joins {roll_id} on supplements_key {key!r}, '
                f'but {name} does not produce that column (neither a '
                '`columns` key nor a transformation output). Both tables '
                'must carry it.'
            )
    return key
