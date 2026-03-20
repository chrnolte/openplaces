"""
link.py

Entity linking: create ID linkages between two spatial entity datasets.
"""

from openplaces.geo.overlay import overlay_polygons
from openplaces.io import save_parquet
from openplaces.recipe import get_output_path


def create_entity_link(entity1_recipe_id, entity2_recipe_id, save=True, **kwargs):
    """Create ID linkage (n-to-m) between two entities and save

    Parameters
    ----------
    entity1_recipe_id : str
        Recipe ID of first entity
    entity2_recipe_id : str
        Recipe ID of second entity
    save : bool
        If True, save output as a file, in the folder of `entity1`:
        {entity1_recipe_id}_{entity2_recipe_id}.parquet
    kwargs : dict
        Are passed to `openplaces.geo.overlay.overlay_polygons`
    """

    entity1_path = get_output_path(entity1_recipe_id)
    entity2_path = get_output_path(entity2_recipe_id)

    if not entity1_path.exists():
        print(f'Link creation aborted. Data not found: {entity1_path}')
        return

    if not entity2_path.exists():
        print(f'Link creation aborted. Data not found: {entity2_path}')
        return

    entity1_entity2_link = overlay_polygons(entity1_path, entity2_path, **kwargs)

    if save:
        entity1_entity2_link_path = entity1_path.with_name(
            entity1_path.stem + f'_{entity2_recipe_id}.parquet'
        )
        save_parquet(entity1_entity2_link, entity1_entity2_link_path)

    return entity1_entity2_link
