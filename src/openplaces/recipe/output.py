"""Output paths, admin levels and partition ids of a recipe."""

from pathlib import Path

# Looked up through the package at call time, not bound here: the tests
# patch these three names on `openplaces.recipe`, and a function that
# had imported them would keep the originals.
from openplaces import recipe as _recipe
from openplaces.config import cfg
from openplaces.core.constants import (
    STRING_SEPARATOR_BETWEEN_IDS,
)
from openplaces.core.schema import (
    AdminId,
    sanitize,
)
from openplaces.path import path
from openplaces.recipe.discovery import (
    find_entity_recipe_id,
)
from openplaces.recipe.tables import (
    _get_save_to,
    get_table_recipe,
)


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
        'US-NC-BR_footprint-obm-2025_032012.parquet' for a tile partition
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
    recipe = _recipe.get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe

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
            _recipe.get_recipe_by_id(entity_recipe_id)
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
            # same separator, e.g. US-NC-BR_footprint-obm-2025_032012.parquet
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


def saves_geometry(recipe) -> bool:
    """Whether a recipe's output carries its own geometry.

    A recipe may declare `save_to: geometry: false` to write an
    attribute-only table whose geometry lives with its `entity_recipe`
    predecessor (the geospine under the geometry/attribute recipe split).
    Readers resolve geometry through that chain by declaration -- never by
    probing for a `_geo` sidecar, so a stale sidecar from a pre-split run
    is ignored.
    """
    if isinstance(recipe, str):
        recipe = _recipe.get_recipe_by_id(recipe)
    return (recipe.get('save_to') or {}).get('geometry', True) is not False


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
