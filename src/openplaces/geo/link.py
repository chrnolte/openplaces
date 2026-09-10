"""
link.py

Entity linking: create ID linkages between two spatial entity datasets.
"""

import os

from openplaces.core.schema import ENTITY_LINK_ORDER, AdminId
from openplaces.geo.overlay import overlay_polygons_with_duckdb
from openplaces.geo.polygon import overlay_polygons
from openplaces.io import read_parquet, save_parquet, to_parquet
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_id,
    get_save_admin_level,
)


def get_link_owner_recipe_id(recipe) -> str:
    """Resolve a recipe to the recipe id whose pipeline ran the link steps.

    Under the geometry/attribute recipe split, link sidecars are keyed by
    the geospine recipe that computed them -- not by the attribute recipe
    downstream consumers name. Follow `entity_recipe` from *recipe* until
    a recipe whose pipeline contains a `link_to_reference` step; an
    unsplit recipe (its own pipeline links) resolves to itself, so
    callers need not know whether a spine was split.
    """
    seen: set[str] = set()
    current = recipe
    while True:
        if isinstance(current, str):
            current = get_recipe_by_id(current)
        pipeline = current.get('pipeline') or []
        if any(
            isinstance(step, dict) and step.get('step') == 'link_to_reference'
            for step in pipeline
        ):
            return get_recipe_id(current)
        predecessor = current.get('entity_recipe')
        if not predecessor or predecessor in seen:
            return get_recipe_id(current)
        seen.add(predecessor)
        current = predecessor


def _entity_link_rank(recipe_id: str) -> int | None:
    """Return the ENTITY_LINK_ORDER rank of a recipe's entity type, or None."""
    recipe = get_recipe_by_id(recipe_id)
    entity = recipe.get('entity')
    entity_type = str(entity.entity_type) if entity is not None else None
    if entity_type in ENTITY_LINK_ORDER:
        return ENTITY_LINK_ORDER.index(entity_type)
    return None


def get_entity_link_path(recipe_id_a, recipe_id_b, admin_id=None):
    """Canonical on-disk path of the entity link between two recipes.

    The link is stored beside the finer entity's output
    (:data:`~openplaces.core.schema.ENTITY_LINK_ORDER`, coarse to fine) as
    ``<finer_stem>_<coarser_recipe_id>.parquet``, so the footprint-parcel
    link IS the parcel-footprint link: both argument orders resolve to the
    same path. Entity types outside the ordering and identical entity
    types fall back to lexicographic recipe-ID order.

    Parameters
    ----------
    recipe_id_a, recipe_id_b : str
        Recipe IDs of the two linked entities, in any order.
    admin_id : str or AdminId, optional
        Admin unit of the link. Truncated to the owning recipe's save
        level to locate its output file, so a county-level admin_id works
        against a state- or country-level recipe.
    """
    rank_a = _entity_link_rank(recipe_id_a)
    rank_b = _entity_link_rank(recipe_id_b)
    if rank_a is None or rank_b is None or rank_a == rank_b:
        owner_id, other_id = sorted([recipe_id_a, recipe_id_b])
    elif rank_a > rank_b:
        owner_id, other_id = recipe_id_a, recipe_id_b
    else:
        owner_id, other_id = recipe_id_b, recipe_id_a

    owner_recipe = get_recipe_by_id(owner_id)
    if admin_id is not None:
        if not isinstance(admin_id, AdminId):
            admin_id = AdminId(admin_id)
        save_level = get_save_admin_level(owner_recipe)
        admin_id = AdminId(*admin_id.levels[:save_level])
    owner_path = get_output_path(owner_recipe, admin_id=admin_id)
    return owner_path.with_name(owner_path.stem + f'_{other_id}.parquet')


def create_entity_link(entity1_recipe_id, entity2_recipe_id, save=True, **kwargs):
    """Create ID linkage (n-to-m) between two entities and save

    Parameters
    ----------
    entity1_recipe_id : str
        Recipe ID of first entity
    entity2_recipe_id : str
        Recipe ID of second entity
    save : bool
        If True, save output at the canonical link path resolved by
        `get_entity_link_path` (beside the finer entity's output,
        regardless of argument order).
    kwargs : dict
        Are passed to `overlay_polygons` or `overlay_polygons_with_duckdb`.
        When ``how='intersection'`` (default), ``iou=False`` (default), and
        ``geom=False`` (default), the DuckDB implementation is used because it
        is faster for Path inputs in that configuration; otherwise geopandas is
        used.
    """

    entity1_path = get_output_path(entity1_recipe_id)
    entity2_path = get_output_path(entity2_recipe_id)

    if not entity1_path.exists():
        print(f'Link creation aborted. Data not found: {entity1_path}')
        return

    if not entity2_path.exists():
        print(f'Link creation aborted. Data not found: {entity2_path}')
        return

    _use_duckdb = (
        kwargs.get('how', 'intersection') == 'intersection'
        and not kwargs.get('iou', False)
        and not kwargs.get('geom', False)
    )
    _fn = overlay_polygons_with_duckdb if _use_duckdb else overlay_polygons

    entity1_entity2_link = _fn(entity1_path, entity2_path, **kwargs)

    if save:
        entity1_entity2_link_path = get_entity_link_path(
            entity1_recipe_id, entity2_recipe_id
        )
        save_parquet(entity1_entity2_link, entity1_entity2_link_path)

    return entity1_entity2_link


def get_scoped_tile_link_path(tile_recipe_id, admin_recipe_id, admin_id):
    """Path of the tile-admin link for one admin unit's own file.

    A tile grid is global, but the question a download asks is local:
    which tiles touch this county. The link therefore lives beside the
    admin unit's file, as ``<admin_unit_stem>_<tile_recipe_id>.parquet``,
    and is built from that unit's polygons alone. A county build then
    needs its own state's admin file and the tile grid, never the world.

    Parameters
    ----------
    tile_recipe_id : str
        The tile grid recipe (``tile-obm-2025``).
    admin_recipe_id : str
        The admin layer recipe whose units the tiles are linked to.
    admin_id : str or AdminId
        Any unit at or below the admin recipe's save level; truncated to
        the save level, which is the file the link sits beside.
    """
    admin_recipe = get_recipe_by_id(admin_recipe_id)
    if not isinstance(admin_id, AdminId):
        admin_id = AdminId(str(admin_id))
    level = get_save_admin_level(admin_recipe)
    unit = admin_id.truncate_to_level(level) if level > 0 else None
    admin_path = get_output_path(admin_recipe, admin_id=unit)
    return admin_path.with_name(f'{admin_path.stem}_{tile_recipe_id}.parquet')


def _input_stamp(path) -> str:
    stat = os.stat(path)
    return f'{stat.st_size}:{int(stat.st_mtime)}'


def ensure_scoped_tile_link(
    tile_recipe_id, admin_recipe_id, admin_id, reprocess=False, verbose=False
):
    """Return the tile-admin link for one admin unit, building it if needed.

    The link is an intersection of the global tile grid with the unit's
    admin polygons, keyed ``(tile_id, admin_id)``. It is rebuilt when
    absent, when *reprocess* is set, or when its footer says it was built
    from a tile grid or admin file other than the ones on disk now (size
    and modification time), so a re-ingested layer is never paired with
    a stale crosswalk.

    Parameters
    ----------
    tile_recipe_id : str
        The tile grid recipe.
    admin_recipe_id : str
        The admin layer recipe.
    admin_id : str or AdminId
        Any unit at or below the admin recipe's save level.
    reprocess : bool
        Rebuild even when the stored link is current.
    verbose : bool
        Report a rebuild.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``(tile_id, admin_id)``.
    """
    admin_recipe = get_recipe_by_id(admin_recipe_id)
    if not isinstance(admin_id, AdminId):
        admin_id = AdminId(str(admin_id))
    level = get_save_admin_level(admin_recipe)
    unit = admin_id.truncate_to_level(level) if level > 0 else None
    tiles_path = get_output_path(get_recipe_by_id(tile_recipe_id))
    admin_path = get_output_path(admin_recipe, admin_id=unit)
    for path in (tiles_path, admin_path):
        if not path.exists():
            raise FileNotFoundError(
                f'Cannot link {tile_recipe_id} to {admin_recipe_id} for '
                f'{unit}: {path} does not exist.'
            )
    link_path = get_scoped_tile_link_path(tile_recipe_id, admin_recipe_id, admin_id)
    stamps = {
        'tile_input': _input_stamp(tiles_path),
        'admin_input': _input_stamp(admin_path),
    }
    if link_path.exists() and not reprocess:
        from openplaces.io.aggregate import read_file_metadata

        stored = read_file_metadata(link_path)
        if all(stored.get(k) == v for k, v in stamps.items()):
            return read_parquet(link_path)
    if verbose:
        print(f'  Linking {tile_recipe_id} to {admin_recipe_id} for {unit}.')
    link = overlay_polygons_with_duckdb(tiles_path, admin_path, verbose=False)
    to_parquet(link, link_path, file_metadata=stamps)
    return link
