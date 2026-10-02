"""Retention class, the save-to block, additional-layer table recipes
and output columns.
"""

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.config import cfg
from openplaces.core.constants import (
    RECIPE_PER_TABLE_KEYS,
)
from openplaces.recipe.loading import (
    get_recipe_id,
)


def _get_save_to(recipe):
    """Return (data_dir, filename) from a recipe dict.

    Falls back to the deprecated 'cache_filename' key so that recipes not
    yet migrated to the 'save_to' block continue to work.
    """
    save_to = recipe.get('save_to') or {}
    data_dir = save_to.get('data_dir', 'cache')
    filename = save_to.get('filename') or recipe.get('cache_filename')
    return data_dir, filename


def get_recipe_retention(recipe: str | dict) -> str:
    """Resolve the retention class of a recipe's output.

    Combines the output bucket's default (STANDARD_DIRS), configuration
    overrides, and the recipe's own save_to.retention via
    :meth:`~openplaces.config.OpenPlacesConfig.retention_for`.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID or loaded recipe dictionary.
    """
    if isinstance(recipe, str):
        recipe = _recipe.get_recipe_by_id(recipe)
    data_dir, _ = _get_save_to(recipe)
    save_to = recipe.get('save_to') or {}
    return cfg.retention_for(
        data_dir,
        recipe_id=get_recipe_id(recipe),
        recipe_retention=save_to.get('retention'),
    )


def build_table_recipe(primary_recipe: dict, layer_spec: dict) -> dict:
    """Merge a primary recipe with an additional_layers spec.

    Per-table keys (entity, layer, columns, index config, etc.) are taken
    from `layer_spec` when present, otherwise removed so that primary-only
    values do not bleed into the secondary table.  `process_by` is inherited
    from the primary unless `layer_spec` sets it explicitly (use
    'process_by: null' in the YAML to disable chunking for a specific
    additional table).

    Parameters
    ----------
    primary_recipe : dict
        Loaded primary recipe dictionary.
    layer_spec : dict
        One entry from the primary recipe's 'additional_layers' list.

    Returns
    -------
    dict
        Merged recipe dict for the layer.
    """
    table_recipe = dict(primary_recipe)

    for key in RECIPE_PER_TABLE_KEYS:
        if key in layer_spec:
            table_recipe[key] = layer_spec[key]
        else:
            table_recipe.pop(key, None)

    # entity is required in every additional_layers entry
    table_recipe['entity'] = layer_spec['entity']

    # process_by: inherit unless explicitly overridden (null disables it)
    if 'process_by' in layer_spec:
        if layer_spec['process_by'] is None:
            table_recipe.pop('process_by', None)
        else:
            table_recipe['process_by'] = layer_spec['process_by']

    # No nesting of additional_layers
    table_recipe.pop('additional_layers', None)

    return table_recipe


def get_table_recipe(recipe: str | dict, layer: str) -> dict:
    """Return the merged recipe for a secondary layer identified by entity.

    Parameters
    ----------
    recipe : str or dict
        Primary recipe (ID string or loaded dict).
    layer : str
        Entity type (e.g. 'property') or full entity string
        (e.g. 'property-massgis-2025') of the additional layer.

    Returns
    -------
    dict
        Merged recipe dict for the requested layer.

    Raises
    ------
    KeyError
        If no `additional_layers` entry matching `layer` is found.
    """
    if isinstance(recipe, str):
        recipe = _recipe.get_recipe_by_id(recipe)

    for layer_spec in recipe.get('additional_layers', []):
        entity = layer_spec.get('entity')
        if entity is not None and (
            str(entity) == layer or str(entity.entity_type) == layer
        ):
            return build_table_recipe(recipe, layer_spec)

    primary = recipe.get('entity') or recipe.get('dataset')
    raise KeyError(
        f"No additional_layers entry matching '{layer}' found in recipe for {primary}."
    )


def get_recipe_output_columns(recipe: dict) -> set[str] | None:
    """Return the column names an ingest recipe's output is declared to carry.

    Read from the recipe alone, never from data: the keys of *columns*,
    every transformation *output*, and ``parcel_id_local`` when the
    recipe declares that directive, less anything in *drop_columns*. This
    mirrors what :class:`~openplaces.io.ingester.table_ingester.
    TableIngester` keeps (named columns, columns a transformation added,
    and the matching key it computes last).

    Parameters
    ----------
    recipe : dict
        A loaded ingest recipe.

    Returns
    -------
    set of str or None
        The declared column names, or None for a recipe that sets
        *keep_unnamed_columns*: it passes through source columns the
        recipe never names, so its output cannot be listed statically.
    """
    if recipe.get('keep_unnamed_columns'):
        return None
    produced = set(recipe.get('columns') or {})
    for transformation in recipe.get('transformations') or []:
        if not isinstance(transformation, dict):
            continue
        output = transformation.get('output')
        if isinstance(output, str):
            produced.add(output)
        elif isinstance(output, list):
            produced.update(o for o in output if isinstance(o, str))
    produced -= set(recipe.get('drop_columns') or [])
    if recipe.get('parcel_id_local'):
        produced.add('parcel_id_local')
    return produced
