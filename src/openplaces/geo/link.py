"""
link.py

Entity linking: create ID linkages between two spatial entity datasets.
"""

from openplaces.core.schema import ENTITY_LINK_ORDER, AdminId
from openplaces.geo.overlay import overlay_polygons_with_duckdb
from openplaces.geo.polygon import overlay_polygons
from openplaces.io import save_parquet
from openplaces.recipe import get_output_path, get_recipe_by_id, get_save_admin_level


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
