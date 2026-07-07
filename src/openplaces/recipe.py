"""
Functions to handle recipes for data ingestion and harmonization

Read and validate recipes, find recipes, build derivatives,
get output paths etc.
"""

import glob
import inspect
from collections import Counter
from functools import cache
from pathlib import Path

import pandas as pd
import yaml

from openplaces.config import cfg
from openplaces.core.constants import (
    RECIPE_PER_TABLE_KEYS,
    RETENTION_CLASSES,
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

    # Record the canonical recipe ID (the file stem), so a loaded recipe can
    # be traced back to its ID even when it carries a filename suffix
    recipe_dict['recipe_id'] = Path(filepath).stem

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

    # Default stage to 'ingest' for recipes that pre-date the stage field
    if 'stage' not in recipe_dict:
        recipe_dict['stage'] = 'ingest'

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
        retention = recipe_dict['save_to'].get('retention')
        if retention is not None and retention not in RETENTION_CLASSES:
            raise ValueError(
                f"Recipe 'save_to.retention' is '{retention}', which is not a "
                'known retention class. Valid options:\n- '
                + '\n- '.join(RETENTION_CLASSES)
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


def get_recipe_id(recipe: str | dict) -> str:
    """Return the canonical recipe ID of a loaded recipe.

    The ID is the recipe file's stem, recorded by get_recipe_dict at load
    time. For recipe dicts constructed without a file (e.g. in tests), the
    ID is rebuilt from admin_id and entity/dataset; filename suffixes cannot
    be recovered in that case.

    Parameters
    ----------
    recipe : str or dict
        Recipe ID string (returned unchanged, minus a .yaml extension) or a
        loaded recipe dictionary.
    """
    if isinstance(recipe, str):
        return recipe.removesuffix('.yaml')
    if 'recipe_id' in recipe:
        return str(recipe['recipe_id']).removesuffix('.yaml')
    parts = []
    admin_id = recipe.get('admin_id')
    if admin_id is not None and len(str(admin_id)) > 0:
        parts.append(str(admin_id))
    entity_or_dataset = recipe.get('entity') or recipe.get('dataset')
    if entity_or_dataset is None:
        raise ValueError('Recipe has neither an entity nor a dataset.')
    parts.append(str(entity_or_dataset))
    return STRING_SEPARATOR_BETWEEN_IDS.join(parts)


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
        recipe = get_recipe_by_id(recipe)
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


def find_recipe_id(admin_id, entity_or_dataset, filename=None, silent=False):
    """Find a recipe ID by admin_id and entity/dataset identifier.

    Parameters
    ----------
    admin_id : str
        Administrative unit identifier.
    entity_or_dataset : str
        Entity or dataset identifier string, may contain glob wildcards
        (e.g. 'parcel-*-*', 'admin-census-2021').
    filename : str, optional
        Filename stem to match within the recipe directory. When None
        (default), matches any .yaml file in the entity directory. A .yaml
        extension is appended automatically if absent.
    silent : bool
        If True, suppress the message printed when multiple recipes are found.
    """
    glob_recipe_path = recipe_path(admin_id, entity_or_dataset, filename=filename)
    recipe_paths_found = glob.glob(str(glob_recipe_path))
    if len(recipe_paths_found) == 0:
        return None
    elif len(recipe_paths_found) == 1:
        return Path(recipe_paths_found[0]).name
    recipe_paths_found = sorted(recipe_paths_found, key=lambda p: Path(p).parent.name)
    if not silent:
        print(
            f'Multiple recipes found for {admin_id} ({entity_or_dataset}):\n'
            + '\n'.join([Path(fp).name for fp in recipe_paths_found])
        )
    recipe_id = Path(recipe_paths_found[-1]).name
    if not silent:
        print(f'Picked last, sorted by version: {recipe_id}')
    return recipe_id


@cache
def iter_entity_sources() -> frozenset:
    """Return the ``(entity_type, source_id)`` pairs across all entity recipes.

    Scans the bundled recipes directory once (cached) and parses each recipe
    filename for its entity token. Dataset/theme recipes and files whose entity
    token does not parse are skipped. Used to auto-generate the provenance suffix
    vocabulary so adding a new source needs no hardcoded list edits.
    """
    root = cfg.code_root.joinpath('src', 'openplaces', 'recipes')
    pairs: set[tuple[str, str | None]] = set()
    for filepath in root.rglob('*.yaml'):
        parts = filepath.stem.split('_')
        try:
            AdminId(parts[0])
            parts = parts[1:]
        except ValueError:
            pass
        if not parts:
            continue
        try:
            entity = Entity(parts[0])
        except (ValueError, IndexError):
            continue
        source_id = getattr(entity.source, 'source_id', None) if entity.source else None
        pairs.add((str(entity.entity_type), source_id))
    return frozenset(pairs)


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
    return find_recipe_id(
        admin_id, 'admin-*-*', filename=f'admin{admin_level}', silent=silent
    )


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
    *stage* is specified. Within a stage, prefer the requested source, the
    most specific applicable administrative scope, and the latest version.
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
    stage_rank = {
        'ingest': 0,
        'harmonize': 1,
        'enrich': 2,
        'curate': 3,
    }

    for filepath in sorted(set(recipe_paths_found)):
        with open(filepath, encoding='utf-8') as f:
            recipe_data = yaml.safe_load(f) or {}
        recipe_stage = recipe_data.get('stage') or 'ingest'
        if stage is not None and recipe_stage != stage:
            continue
        entity = recipe_data.get('entity') or {}
        if entity.get('entity_type') != entity_type:
            continue
        recipe_admin_id = AdminId(recipe_data.get('admin_id'))
        if not recipe_admin_id.is_parent_or_equal_of(admin_id):
            continue
        source = entity.get('source') or {}
        recipe_source_id = source.get('source_id', '')
        version = str(entity.get('version', ''))
        candidates.append(
            (
                stage_rank.get(recipe_stage, -1),
                recipe_source_id == source_id if source_id else False,
                recipe_admin_id.get_level(),
                version,
                Path(filepath).stem,
            )
        )

    if not candidates:
        return None
    candidates.sort()
    recipe_id = candidates[-1][4]
    if len(candidates) > 1 and not silent:
        print(f'Picked {recipe_id} for {admin_id} ({entity_type}).')
    return recipe_id


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


def get_output_path(
    recipe,
    admin_id=None,
    partition_id=None,
    geo=False,
    layer=None,
    entity_recipe_id=None,
):
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
        'US-NC-BS_footprint-obm-2025_032012.parquet' for a tile partition
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
    entity_recipe_id : str or dict, optional
        Concrete entity recipe to enrich when *recipe* has stage 'enrich'.

    Returns
    -------
    pathlib.Path
        Resolved output path for the recipe data file.
    """
    recipe = get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe

    if recipe.get('stage') == 'enrich':
        if entity_recipe_id is None:
            entity_type = str(recipe['entity'].entity_type)
            lookup_admin_id = admin_id or recipe['admin_id']
            entity_recipe_id = find_entity_recipe_id(
                lookup_admin_id,
                entity_type,
                stage='harmonize',
                source_id='spine',
                silent=True,
            )
        entity_recipe = (
            get_recipe_by_id(entity_recipe_id)
            if isinstance(entity_recipe_id, str)
            else entity_recipe_id
        )
        if entity_recipe is None:
            raise ValueError(
                f'No entity recipe found for enrichment recipe {recipe.get("dataset")}.'
            )
        entity_path = get_output_path(
            entity_recipe,
            admin_id=admin_id,
            partition_id=partition_id,
            geo=geo,
            layer=layer,
        )
        dataset = recipe.get('dataset')
        if dataset is None:
            raise ValueError("Enrich recipes require 'dataset'.")
        return entity_path.with_stem(entity_path.stem + '_' + sanitize(str(dataset)))

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
            # same separator, e.g. US-NC-BS_footprint-obm-2025_032012.parquet
            filename = partition_id_save

    is_raster = bool(recipe.get('dataset') and recipe['dataset'].is_raster)
    p = path(
        admin_id if admin_id else recipe.get('admin_id'),
        recipe.get('entity'),
        recipe.get('dataset'),
        filename=filename,
        root=cfg.get_dir(data_dir),
        default_extension='tif' if is_raster else 'parquet',
    )
    if geo:
        p = p.with_stem(p.stem + '_geo')
    return p


def get_save_admin_level(
    recipe, operation_keys=('download_by', 'process_by', 'save_to')
):
    """Return the admin level at which output files are split.

    When `save_to: admin_level` is explicitly set it defines the output
    granularity directly — `process_by` or `download_by` may be finer
    (aggregation) or coarser than this level. When `save_to: admin_level`
    is absent the level is the maximum found across the given operation keys,
    falling back to the recipe's own admin ID depth.

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
    # Explicit save_to: admin_level takes priority.
    save_to = recipe.get('save_to') or {}
    if 'admin_level' in save_to and 'save_to' in operation_keys:
        return save_to['admin_level']

    level = recipe['admin_id'].get_level()
    for key in operation_keys:
        if key == 'save_to':
            continue  # already handled above
        if key in recipe and 'admin_level' in recipe[key]:
            level = max(level, recipe[key]['admin_level'])
    # Deprecated / for backward compatibility: cache_by: admin_level
    cache_by = recipe.get('cache_by') or {}
    if 'admin_level' in cache_by:
        level = max(level, cache_by['admin_level'])
    return level


def get_process_admin_level(recipe):
    """Return the admin level at which data is chunked for processing."""
    return get_save_admin_level(recipe, operation_keys=('download_by', 'process_by'))


def get_download_admin_level(recipe):
    """Return the admin level at which downloads are partitioned."""
    return get_save_admin_level(recipe, operation_keys=('download_by',))


def _year_month_range(first: str, last: str) -> list[str]:
    """Return inclusive YYYYMM strings from *first* to *last*.

    Parameters
    ----------
    first, last : str
        Start and end months as six-digit YYYYMM strings (e.g. '200810').

    Returns
    -------
    list of str
        Consecutive 'YYYYMM' strings, ordered from first to last inclusive.
    """
    first, last = str(first), str(last)
    if len(first) != 6 or len(last) != 6:
        raise ValueError(
            f"'first'/'last' for partition 'year_month' must be YYYYMM "
            f'(six digits); got {first!r} and {last!r}.'
        )
    year, month = int(first[:4]), int(first[4:6])
    end = (int(last[:4]), int(last[4:6]))
    result = []
    while (year, month) <= end:
        result.append(f'{year:04d}{month:02d}')
        month += 1
        if month > 12:
            month = 1
            year += 1
    return result


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
        If 'download_by': 'partition' is 'year' or 'year_month' but
        'first'/'last' are not defined.
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

    elif partition == 'year_month':
        first = download_by.get('first')
        last = download_by.get('last')
        if first is None or last is None:
            raise ValueError(
                "If 'download_by' has 'partition: year_month', "
                "define 'first' and 'last' as YYYYMM (e.g. 200810)."
            )
        return _year_month_range(str(first), str(last))

    elif partition == 'table':
        table_names = download_by.get('table_names')
        if not table_names:
            raise ValueError(
                "If 'download_by' has 'partition: table', "
                "define 'table_names' (list of table names)."
            )
        return [str(tn) for tn in table_names]

    raise NotImplementedError(
        f'Partition not yet supported by openplaces.recipe.get_partition_ids: '
        f"'{partition}'."
    )
