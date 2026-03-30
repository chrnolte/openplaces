# recipe.py

# Functions to read data ingestion recipes

import glob
import inspect
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from openplaces.config import cfg
from openplaces.core.constants import (
    RECIPE_PER_TABLE_KEYS,
    STANDARD_DIRS,
    STRING_SEPARATOR_BETWEEN_IDS,
)
from openplaces.core.schema import AdminId, DataSet, Entity, Source, sanitize
from openplaces.path import OpenPlacesReference, path, recipe_path


def get_recipe(*args, **kwargs):
    """Load recipe (.yaml, .csv or .xlsx)

    Parameters
    ----------
    args : tuple
        Arguments for `openplaces.path.recipe_path`
    kwargs : dict
        Keywords arguments. Those in `openplaces.path.OpenPlacesReference`
        and `openplaces.path.recipe_path` will be used to find the path,
        the remainder is passed to the reading functions:
        - yaml.safe_load()
        - pd.read_csv()
        - pd.read_excel()
    """

    # Separate keywords: those that don't go to the path go to reading
    recipe_path_kwargs = set(inspect.signature(OpenPlacesReference).parameters) | set(
        inspect.signature(recipe_path).parameters
    )
    path_kwargs = {k: v for k, v in kwargs.items() if k in recipe_path_kwargs}
    read_kwargs = {k: v for k, v in kwargs.items() if k not in recipe_path_kwargs}

    filepath = recipe_path(*args, **path_kwargs)

    if filepath.suffix in ['.csv', '.xlsx', '.xlsx']:
        # To avoid ambiguity between versions, only one tabular format
        # should exist for a given filename. Removing extensions in the
        # arguments for the recipe filepath is one way to enforce that.
        raise Exception(
            f'Remove extensions in filepath when using `get_recipe()`: {filepath.name}'
        )

    if filepath.with_suffix('.yaml').exists():
        return get_recipe_dict(filepath.with_suffix('.yaml'), *args, **kwargs)
    elif filepath.with_suffix('.csv').exists():
        recipe_table = pd.read_csv(filepath.with_suffix('.csv'), **read_kwargs)
    elif filepath.with_suffix('.xlsx').exists():
        recipe_table = pd.read_excel(filepath.with_suffix('.xlsx'), **read_kwargs)
    elif filepath.with_suffix('.xls').exists():
        recipe_table = pd.read_excel(filepath.with_suffix('.xls'), **read_kwargs)
    else:
        raise OSError('Not found: ' + str(filepath.with_suffix('.(yaml|csv|xlsx|xls)')))

    return recipe_table


def _cast_entity(entity):
    """Cast a raw dict (or already-cast Entity) to an Entity object."""
    if isinstance(entity, Entity):
        return entity
    if isinstance(entity.get('source'), dict):
        entity['source'] = Source(**entity['source'])
    return Entity(**entity)


def _cast_dataset(dataset):
    """Cast a raw dict (or already-cast DataSet) to a DataSet object."""
    if isinstance(dataset, DataSet):
        return dataset
    if isinstance(dataset.get('source'), dict):
        dataset['source'] = Source(**dataset['source'])
    return DataSet(**dataset)


def get_recipe_dict(filepath, *args, **kwargs):
    """Read a recipe `.yaml` file as a dictionary, cast it to schema

    Parameters
    ----------
    filepath : pathlib.Path
        Filepath to .yaml file
    args : list
        Passed on from `get_recipe`
    kwargs : dict
        Passed on from `get_recipe`
    """
    with open(filepath, encoding='utf-8') as f:
        recipe_dict = yaml.safe_load(f)

    # Get `admin_id` from arguments
    if len(args) > 0:
        admin_id_arg = args[0]
    elif 'admin_id' in kwargs:
        admin_id_arg = kwargs['admin_id']
    else:
        admin_id_arg = None

    # Cast AdminId from arguments
    if not isinstance(admin_id_arg, AdminId):
        admin_id_arg = AdminId(admin_id_arg)

    # Ensure that 'admin_id' exists in recipe
    if 'admin_id' not in recipe_dict:
        recipe_dict['admin_id'] = admin_id_arg

    # Cast AdminId from .yaml file
    if not isinstance(recipe_dict['admin_id'], AdminId):
        recipe_dict['admin_id'] = AdminId(recipe_dict['admin_id'])

    # Sanity check: are there any conflicting values?
    if admin_id_arg and (str(recipe_dict['admin_id']) != str(admin_id_arg)):
        raise ValueError(
            'Inconsistent `admin_id` in get_recipe(admin_id, ...) and recipe `.yaml`:\n'
            f'get_recipe: {admin_id_arg} {type(admin_id_arg)}\n'
            f'.yaml file: {recipe_dict["admin_id"]} {type(recipe_dict["admin_id"])}'
        )

    # Cast Entity (if there is one)
    if 'entity' in recipe_dict:
        recipe_dict['entity'] = _cast_entity(recipe_dict['entity'])

    # Cast DataSet (if there is one)
    if 'dataset' in recipe_dict:
        recipe_dict['dataset'] = _cast_dataset(recipe_dict['dataset'])

    # Cast additional_layers entities (if any)
    for layer_spec in recipe_dict.get('additional_layers', []):
        if 'entity' in layer_spec:
            layer_spec['entity'] = _cast_entity(layer_spec['entity'])

    # Validate save_to (if present)
    if 'save_to' in recipe_dict and isinstance(recipe_dict['save_to'], dict):
        data_dir = recipe_dict['save_to'].get('data_dir')
        if data_dir is not None:
            if data_dir not in STANDARD_DIRS:
                raise ValueError(
                    f"Recipe 'save_to.data_dir' is '{data_dir}', which is not a "
                    'known openplaces directory. Valid options:\n- '
                    + '\n- '.join(sorted(STANDARD_DIRS))
                )

    return recipe_dict


def get_recipe_by_id(recipe_id, **kwargs):
    """Shortcut to get recipe_id by its parts

    Assumes syntax: {admin_id}_{entity}_{filename}.{extension}

    admin_id or filename can be missing

    (Datasets for non-entities aren't yet supported)

    Parameters
    ----------
    recipe_id : str
        Identifier or a recipe
    kwargs : dict
        Keyword arguments will be passed on to get_recipe()
    """

    n_dots = Counter(recipe_id)['.']
    if n_dots == 0:
        filename_stem = recipe_id
        extension = None
    elif n_dots == 1:
        filename_stem, extension = recipe_id.split('.')
    else:
        raise ValueError('Recipe name cannot contain more than one dot.')

    remaining_parts = filename_stem.split('_')

    # Split off admin_id, if valid
    try:
        admin_id = AdminId(remaining_parts[0])
        remaining_parts = remaining_parts[1:]
    except ValueError:
        admin_id = None

    # Split off entity, if valid
    try:
        entity = Entity(remaining_parts[0])
        remaining_parts = remaining_parts[1:]
    except ValueError:
        entity = None

    # Split off dataset, if valid
    try:
        dataset = DataSet(remaining_parts[0])
        remaining_parts = remaining_parts[1:]
    except (ValueError, IndexError):
        dataset = None

    filename = remaining_parts.pop(0) if remaining_parts else None

    if remaining_parts:
        raise ValueError(f'Cannot interpret recipe_id {recipe_id}')

    if isinstance(filename, str) and isinstance(extension, str):
        filename += '.' + extension

    return get_recipe(
        admin_id,
        entity,
        dataset,
        filename=filename,
        **kwargs,
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
        recipe = get_recipe_by_id(recipe)

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


def find_admin_recipe_id(admin_id, admin_level, silent=False):
    """Find the ID of an administrative data ingestion recipe

    Parameters
    ----------
    admin_id : str
        Administrative unit identifier
    admin_level : int
        Administrative level for which a recipe is sought.
    silent : bool
        If True, suppress the message printed when multiple recipes are found.
    """
    glob_recipe_path = recipe_path(
        admin_id, 'admin-*-*', filename=f'admin{admin_level}'
    )
    recipe_paths_found = glob.glob(str(glob_recipe_path))
    if len(recipe_paths_found) == 0:
        return None
    elif len(recipe_paths_found) == 1:
        recipe_id = Path(recipe_paths_found[0]).name
    else:
        recipe_paths_found = sorted(
            recipe_paths_found, key=lambda p: Path(p).parent.name
        )
        if not silent:
            print(
                f'Multiple admin recipes found for {admin_id} at level {admin_level}:\n'
                + '\n'.join([Path(fp).name for fp in recipe_paths_found])
            )
        recipe_id = Path(recipe_paths_found[-1]).name
        if not silent:
            print(f'Picked last, sorted by version: {recipe_id}')

    return recipe_id


def get_layers(recipe: str | dict) -> list[str]:
    """Return the layer names available for a recipe's 'additional_layers'.

    These are the values accepted by the `layer` argument of 'read_entities'
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
        recipe = get_recipe_by_id(recipe)
    return [
        str(layer_spec['entity'].entity_type)
        for layer_spec in recipe.get('additional_layers', [])
        if 'entity' in layer_spec
    ]


def _get_save_to(recipe):
    """Return (data_dir, filename) from a recipe dict.

    Falls back to the deprecated 'cache_filename' key so that recipes not
    yet migrated to the 'save_to' block continue to work.
    """
    save_to = recipe.get('save_to') or {}
    data_dir = save_to.get('data_dir', 'cache')
    filename = save_to.get('filename') or recipe.get('cache_filename')
    return data_dir, filename


def get_output_path(recipe, admin_id=None, partition_id=None, geo=False, layer=None):
    """Return the path where recipe output is written.

    Mirrors `Ingester._get_output_path` without instantiating an Ingester.
    The output root is determined by 'save_to': 'data_dir' in the recipe
    (default: 'cache'), which must name a directory registered in `STANDARD_DIRS`.

    Parameters
    ----------
    recipe : str or dict
        Recipe identifier (as accepted by `get_recipe_by_id`) or a
        pre-loaded recipe dict.
    admin_id : str or `AdminId`, optional
        Administrative unit for which to resolve the output path.
        Pass `None` for recipes not split by admin unit.
    partition_id : str, optional
        Partition value appended to the filename stem, e.g.
        'US-NC-BS_building-obm-2025_032012.parquet' for a tile partition
        with id '032012'.  Pass `None` (default) to obtain the final,
        merged output path.
    geo : bool, optional
        If True, return the path to the companion '_geo.parquet' file
        instead of the attribute parquet file.
    layer : str, optional
        Entity type (e.g. 'property') or full entity string
        (e.g. 'property-massgis-2025') of a secondary layer defined in
        `additional_layers`. If given, the path for that layer is returned
        instead of the primary entity's path.

    Returns
    -------
    pathlib.Path
        Resolved output path for the recipe data file.
    """
    recipe = get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe

    if layer is not None:
        recipe = get_table_recipe(recipe, layer)

    if admin_id is not None:
        admin_id = AdminId(admin_id) if not isinstance(admin_id, AdminId) else admin_id
        recipe_admin_id = recipe['admin_id']
        if not recipe_admin_id.is_parent_or_equal_of(admin_id):
            raise ValueError(
                f'`admin_id` {admin_id} is not within the scope of '
                f'recipe `admin_id` {recipe_admin_id}.'
            )

        save_level = get_save_admin_level(recipe)
        if admin_id.get_level() != save_level:
            raise ValueError(
                f'`admin_id` {admin_id} is at level {admin_id.get_level()}, '
                f'but the recipe saves data at admin level {save_level}.'
            )

    data_dir, filename = _get_save_to(recipe)

    if partition_id is not None:
        partition_id_save = sanitize(str(partition_id))
        if filename:
            fp = Path(filename)
            filename = fp.with_stem(
                fp.stem + STRING_SEPARATOR_BETWEEN_IDS + partition_id_save
            ).name
        else:
            # path() will join this with the auto-generated prefix via the
            # same separator, e.g. US-NC-BS_building-obm-2025_032012.parquet
            filename = partition_id_save

    p = path(
        admin_id if admin_id else recipe.get('admin_id'),
        recipe.get('entity'),
        recipe.get('dataset'),
        filename=filename,
        root=cfg.get_dir(data_dir),
    )
    if geo:
        p = p.with_stem(p.stem + '_geo')
    return p


def get_save_admin_level(
    recipe, operation_keys=('download_by', 'process_by', 'save_to')
):
    """Return the admin level at which output files are split.

    Takes the maximum 'admin_level' found across the given recipe sections,
    falling back to the recipe's own admin ID depth if none is declared.

    Parameters
    ----------
    recipe : dict
        Loaded recipe dictionary.
    operation_keys : tuple of str
        Recipe section keys to inspect for 'admin_level'. 'save_to' is
        included by default since save_to: admin_level controls output
        granularity. Override when calling from other recipe runners.

    Returns
    -------
    int
        Admin level for output files (0 = no admin split).
    """
    level = recipe['admin_id'].get_level()
    for key in operation_keys:
        if key in recipe and 'admin_level' in recipe[key]:
            level = max(level, recipe[key]['admin_level'])
    # Deprecated / for backward compatibility: cache_by: admin_level
    cache_by = recipe.get('cache_by') or {}
    if 'admin_level' in cache_by:
        level = max(level, cache_by['admin_level'])
    return level


def get_partition_ids(recipe):
    """Return the list of valid partition ID strings for a recipe.

    Returns `[None]` for recipes without a 'download_by': 'partition' key.

    Parameters
    ----------
    recipe : dict
        Loaded recipe dictionary.

    Returns
    -------
    list of str or list of None

    Raises
    ------
    ValueError
        If 'download_by': 'partition' is 'year' but 'first'/'last' are not defined.
    NotImplementedError
        If 'download_by': 'partition' names an unrecognised partition type.
    """
    download_by = recipe.get('download_by') or {}
    partition = download_by.get('partition')

    if not partition:
        return [None]

    if partition == 'year':
        first = download_by.get('first')
        last = download_by.get('last')
        if first is None or last is None:
            raise ValueError(
                "If 'download_by' has 'partition: year', define 'first' and 'last'."
            )
        return [str(year) for year in range(first, last + 1)]

    raise NotImplementedError(
        f'Partition not yet supported by openplaces.recipe.get_partition_ids: '
        f"'{partition}'."
    )


def resolve_output_admin_ids(
    recipe,
    requested_admin_ids,
    operation_keys=('download_by', 'process_by', 'save_to'),
    reprocess=False,
    partition_id=None,
):
    """Return admin IDs for which output files should be (re-)created.

    Parameters
    ----------
    recipe : dict
        Loaded recipe dictionary.
    requested_admin_ids : list of AdminId
        The admin IDs the caller wants to process.
    operation_keys : tuple of str
        Recipe sections to inspect for 'admin_level'. 'save_to' is
        included by default; override when calling from other recipe runners.
    reprocess : bool
        If True, include admin IDs whose output files already exist.
    partition_id : str, optional
        Forwarded to `get_output_path` when checking file existence.

    Returns
    -------
    list of str or list of None
        Admin ID strings at the save level, or `[None]` if not admin-split.
    """
    # Deferred import: openplaces.api imports recipe at module level, so
    # importing api here avoids a circular import at load time.
    from openplaces.api import get_admin

    save_level = get_save_admin_level(recipe, operation_keys)

    if save_level == 0:
        admin_ids_all = [None]
    else:
        admin_ids_all = list(
            dict.fromkeys(get_admin(recipe['admin_id'], save_level).index)
        )

    admin_ids_to_save = [
        admin_id_str
        for admin_id_requested in requested_admin_ids
        for admin_id_str in admin_ids_all
        if admin_id_requested.is_parent_or_equal_of(AdminId(admin_id_str))
        or AdminId(admin_id_str).is_parent_of(admin_id_requested)
    ]
    admin_ids_to_save = list(dict.fromkeys(admin_ids_to_save))

    if not reprocess:
        admin_ids_to_save = [
            admin_id
            for admin_id in admin_ids_to_save
            if not get_output_path(recipe, admin_id, partition_id).exists()
        ]

    return admin_ids_to_save
