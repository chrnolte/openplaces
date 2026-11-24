"""
Low-level data-access functions for admin units and entities.
Imported by internal modules (io/*, geo/*). Also re-exported from api.py.
"""

import re
import warnings
from collections.abc import Sequence

import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq

from openplaces.core.attribute_registry import load_registry
from openplaces.core.constants import STRING_SEPARATOR_WITHIN_IDS
from openplaces.core.schema import AdminId
from openplaces.io import read_parquet
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_id,
    get_save_admin_level,
    get_table_recipe,
    resolve_attribute_name,
    saves_geometry,
    version_sort_key,
)
from openplaces.utils import format_list

ADMIN_SOURCE_DEFAULT = 'admin-spine-2026'
REGION_SOURCE_DEFAULT = 'admin-regions-2026'
ADMIN_GEO_SOURCE_DEFAULT = 'admin-openplaces-2026'
# Land-only geometry for drawing, by `get_admin(geom=...)` value: the
# version prefix of the Census cartographic boundary recipes at each
# resolution (500k, 5m and 20m), followed by the vintage year
# (`census-carto500k2025`). Letters lead so that TIGER's `census-2025`
# still sorts last and stays the default census recipe. Only 500k has
# every level; a level or scope the requested resolution lacks reads it.
LAND_GEOMETRY_SOURCE = 'census'
LAND_GEOMETRY_PRODUCTS = {
    'land': 'carto500k',
    'land_coarse': 'carto5m',
    'land_coarsest': 'carto20m',
}
LAND_GEOMETRY_FINEST_PRODUCT = 'carto500k'
LAND_GEOMETRY_LEVELS = (2, 3, 4)
ADMIN_PRIMARY_COLUMNS = {
    1: ['name', 'admin1_id_a3'],
    2: [
        'name',
        'type',
        'admin1_name',
        'admin2_id_admin1',
    ],
    3: [
        'name',
        'name_long',
        'type',
        'admin2_name',
        'admin1_name',
        'admin3_id_admin1',
    ],
    4: [
        'name',
        'name_long',
        'type',
        'admin3_name',
        'admin2_name',
        'admin1_name',
        'admin4_id_admin1',
    ],
}
UNNARROWABLE_POLICIES = ('raise', 'whole')
UNNARROWED_ATTR = 'openplaces_unnarrowed'


class UnnarrowableSourceError(RuntimeError):
    """A per-unit read of a coarser file cannot be narrowed to the unit.

    Raised by :func:`get_entities` when a unit finer than the recipe's
    save level is requested (a county from a statewide file) and the
    rows carry no per-row unit: no admin{level}_id column, no
    geometry to overlay, and the unit is not the sole child of the file's
    unit. Returning the whole file instead gave every county of Indiana
    the state's 1.2 million parcel-sale rows. A RuntimeError, not a
    ValueError, so that a caller skipping an unavailable source on
    ValueError does not mistake it for one.
    """


def unnarrowable_error(recipe, admin_ids, save_level) -> UnnarrowableSourceError:
    """Build the error for a read that cannot be narrowed to its units.

    Parameters
    ----------
    recipe : dict or str
        The recipe read.
    admin_ids : iterable of str
        The requested units the rows could not be narrowed to.
    save_level : int
        The admin level the recipe saves its files at.

    Returns
    -------
    UnnarrowableSourceError
        Naming the recipe, the units, the file's level and the remedies.
    """
    recipe_id = recipe if isinstance(recipe, str) else get_recipe_id(recipe)
    units = format_list(sorted(str(admin_id) for admin_id in admin_ids))
    return UnnarrowableSourceError(
        f'{recipe_id} saves one file per admin{save_level} unit and its rows '
        f'name no finer unit, so a read for {units} would return the whole '
        f'admin{save_level} file. Give the recipe a per-row admin id column '
        '(process_by.admin_id_column with a crosswalk, or a column mapped '
        'to the admin id), or process and save it by the finer unit '
        '(process_by.admin_level). A caller that wants the whole file passes '
        "unnarrowable='whole'."
    )


def _concat_recipe_files(paths, reader=None, **read_kwargs):
    """Read a recipe's per-unit output files and concatenate them.

    A recipe that saves one file per admin unit answers a multi-unit
    request with several files. They share one schema and an admin-id
    index, so a plain concatenation is the whole combination.

    Parameters
    ----------
    paths : list of pathlib.Path
        Output files to read, all of them known to exist.
    reader : callable, optional
        Function that reads one path. Defaults to
        :func:`openplaces.io.read_parquet`.
    **read_kwargs
        Forwarded to the reader.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        The concatenated files, keeping each file's own index.
    """
    reader = read_parquet if reader is None else reader
    # A placeholder written for a unit the source does not cover (an
    # admin level whose file is held or unpublished) holds no rows and
    # no column but the join id. It contributes nothing, and a filter
    # on a column it lacks cannot even be bound to it, so it is left
    # out unless it is all there is.
    real = [path for path in paths if not is_placeholder_output(path)]
    frames = [reader(path, **read_kwargs) for path in real or paths[:1]]
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames)
    if isinstance(frames[0], gpd.GeoDataFrame):
        combined = gpd.GeoDataFrame(combined, crs=frames[0].crs)
    return combined


def is_placeholder_output(path) -> bool:
    """True for an output file with no rows and no column but `_join_id`.

    The harmonize stage writes one for a unit with nothing to build
    from, and the curate of such a unit writes one too: the unit was
    built and has none of the entity. A missing file is not one; it
    means the unit was not built.

    Anything the parquet metadata cannot be read from (a path that is
    not a file, or a reader's own handle in a test) is not a
    placeholder; the reader decides what to do with it.

    Parameters
    ----------
    path : pathlib.Path or str
        An output file.

    Returns
    -------
    bool
        Whether the file is a placeholder.
    """
    try:
        schema = pq.read_schema(path)
        if set(schema.names) - {'_join_id'}:
            return False
        return pq.read_metadata(path).num_rows == 0
    except (TypeError, OSError, ValueError):
        return False


# Footer keys an ingest stamps on every table it writes without rows,
# saying why it has none (`io.ingester.empty_outputs.save_empty_table`).
EMPTY_REASON_KEY = 'openplaces:empty_reason'
EMPTY_DETAIL_KEY = 'openplaces:empty_detail'
# not_published: the source does not publish the unit (or partition).
# not_covered: the source was read and holds rows, none for the unit.
# source_has_none: the source itself holds no rows, and the recipe
#   allows it (empty_download_ok) or does not check that table.
# marker: the rows live elsewhere (the `_all` file or per-partition
#   files); the file only records that the unit was ingested.
# nothing_stacked: the stacked-units layer of a parcel table whose
#   lots each carry one record.
# declined: the user's usage profile or a withheld terms consent
#   declined the download; the source was never read.
# nothing_requested: a series enrichment whose requests named no
#   (entity, period) pair the series answers for the unit.
EMPTY_REASONS = (
    'not_published',
    'not_covered',
    'source_has_none',
    'marker',
    'nothing_stacked',
    'declined',
    'nothing_requested',
)


def _footer(path):
    """A parquet file's footer metadata, or None when it cannot be read."""
    try:
        return pq.read_metadata(path)
    except (TypeError, OSError, ValueError):
        return None


def empty_reason(path) -> str | None:
    """Why an ingest output holds no rows, from its parquet footer.

    Reads the footer only, never a row. The reason is one of
    `EMPTY_REASONS`; the detail beside it is in `EMPTY_DETAIL_KEY`.

    Parameters
    ----------
    path : pathlib.Path or str
        An ingest output file.

    Returns
    -------
    str or None
        The stamped reason, or None for a file without one (a file with
        rows, an unexplained empty file, or one that cannot be read).
    """
    meta = _footer(path)
    raw = (meta.metadata or {}).get(EMPTY_REASON_KEY.encode()) if meta else None
    return raw.decode() if raw is not None else None


def is_unexplained_empty(path) -> bool:
    """True for an ingest output with no rows and no recorded reason.

    Every empty table an ingest writes states why it is empty
    (`empty_reason`). One that does not was written by an older run or
    an interrupted one, and is not evidence that the unit has none (a
    stale empty parcel table, written before an empty download raised,
    let every later run skip the unit). Readers of ingest outputs treat
    it as not built. Not for harmonize or curate outputs, whose
    placeholders carry no reason (`is_placeholder_output`).

    Reads the footer only. A missing or unreadable file is not one.

    Parameters
    ----------
    path : pathlib.Path or str
        An ingest output file (the attribute file, not its `_geo`
        sidecar).

    Returns
    -------
    bool
    """
    meta = _footer(path)
    if meta is None or meta.num_rows:
        return False
    return EMPTY_REASON_KEY.encode() not in (meta.metadata or {})


def get_admin(
    admin_id=None,
    level=None,
    recipe=None,
    geom=False,
    columns=None,
    all_columns=False,
    silent=True,
    allow_empty=False,
):
    """Get admin units of any administrative level

    Parameters
    ----------
    admin_id : str, list, or openplaces.core.schema.AdminId
        Identifier(s) of admin units to return.
        Can include higher-level Admin IDs to select many lower levels
    level : int
        Admin level for which to return units.
        If none, use level of `admin_id` (deepest if a list is passed).
    recipe : str
        Use this recipe to import geometries and additional attributes.
    geom : bool or str
        If False, return a DataFrame without geometries.
        If True, return a GeoDataFrame with full geometries.
        If 'simplified', return a GeoDataFrame with simplified geometries
        from the `_geo_simplified` companion file written by
        `AdminHarmonizer`.
        If 'land', 'land_coarse' or 'land_coarsest', return geometries
        clipped to the shoreline (the Great Lakes and coastal waters
        are not part of a state or county), from the Census cartographic
        boundary files at 1:500,000, 1:5,000,000 and 1:20,000,000. Meant
        for drawing maps of US states (level 2), counties and New
        England towns (level 3) and county subdivisions (level 4), never
        for assigning or clipping data: the polygons are generalized and
        move boundaries by up to a few hundred meters at 500k. The
        Census Bureau also advises against these files for area or
        perimeter calculation, geocoding and precise area relationships.
        Units a
        resolution does not cover (subdivisions and towns exist at 500k
        only) come back at 500k. `recipe` must not be given.
        Measured 2026-10-03 on US-MI: 'land' covers 150,910 km2 against
        250,486 km2 from the TIGER layers, which include the Great Lakes.
    columns : list of str or None
        If a list of strings, will be used to select columns.
    all_columns : bool
        If True, returns not only the most important columns
    silent : bool
        Silence warnings
    allow_empty : bool
        If True, a scope that selects no unit at `level` returns an
        empty table instead of raising. A country with no second-level
        units (Antarctica) or a region with no third-level ones (an
        Andorran parish) is a legitimate empty answer to "which units
        sit at this level within X", which is the question a per-unit
        harmonize run asks. The default raises, because a caller who
        names one unit and gets nothing has usually misspelled it.
    """

    if level is not None and level < 1:
        raise ValueError('The lowest level for admin units is 1. 0 is the planet.')

    # Cast scalar `admin_id` to list
    if isinstance(admin_id, str | AdminId):
        admin_id = [admin_id]
    elif admin_id is not None and not isinstance(admin_id, list):
        raise ValueError(f'AdminID not recognized: {type(admin_id)} {admin_id}.')

    # Cast all entries in the list to AdminId to get level for the recipe
    if admin_id is not None:
        admin_ids = []
        for _admin_id in admin_id:
            if isinstance(_admin_id, AdminId):
                admin_ids += [_admin_id]
            elif isinstance(_admin_id, str):
                admin_ids += [AdminId(_admin_id)]
            else:
                raise ValueError(
                    f'AdminID not recognized: {type(_admin_id)} {_admin_id}.'
                )

    # Cast recipe to a dict if a str (id) is provided
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    if isinstance(recipe, dict) and admin_id is None:
        admin_id = recipe['admin_id']
        admin_ids = [admin_id]

    # Try to infer level from recipe if not explicitly specified
    if not isinstance(level, int) and isinstance(recipe, dict):
        filename = recipe.get('save_to', {}).get('filename', '')
        if filename.startswith('admin') and filename[5:].isdigit():
            level = int(filename[5:])
        elif 'recipe_id' in recipe:
            parts = recipe['recipe_id'].split('_')
            for part in parts:
                if part.startswith('admin') and part[5:].isdigit():
                    level = int(part[5:])
                    break

    # If level is still not specified, use deep level from admin_ids
    if not isinstance(level, int) and admin_id is not None:
        admin_id_level = max(_admin_id.get_level() for _admin_id in admin_ids)
        level = admin_id_level

    land_product = None
    if isinstance(geom, str) and geom in LAND_GEOMETRY_PRODUCTS:
        if recipe is not None:
            raise ValueError(f"geom='{geom}' reads its own recipes; omit `recipe`.")
        if level is None:
            raise ValueError(
                f"geom='{geom}' needs `level` or an admin_id deeper than the "
                f'country: land-only geometry exists for levels '
                f'{format_list(list(LAND_GEOMETRY_LEVELS))}.'
            )
        if level not in LAND_GEOMETRY_LEVELS:
            raise ValueError(
                f"geom='{geom}' covers levels "
                f'{format_list(list(LAND_GEOMETRY_LEVELS))}, not level {level}.'
            )
        land_product = LAND_GEOMETRY_PRODUCTS[geom]

    # Pick default recipe for geometry attributes if None is provided
    if geom and recipe is None and land_product is None:
        if level is None:
            # Default to countries
            level = 1
        # The harmonized spine geometry, which already assigns every
        # unit its most specific source: New England's per-state COUSUB
        # towns (US-MA_admin-census-2025_admin3) as much as a national
        # layer. Reading an ingest recipe here instead, as this did
        # until 2026-09-29, picked a national source with names and no
        # polygons (FR's INSEE, JP's Soumu) and returned no geometry,
        # and fell back to GADM, which the spine may not carry.
        recipe = get_recipe_by_id(f'{ADMIN_GEO_SOURCE_DEFAULT}_admin{level}')

    if admin_id is None and level is None:
        # Default to countries
        level = 1

    if admin_id is not None:
        # Recompute level (as it could have been set by recipe admin ID)
        admin_id_level = max(_admin_id.get_level() for _admin_id in admin_ids)

        if level < admin_id_level:
            # Clip admin_ids to upper level (= load parent admin units)
            admin_ids = [AdminId(*_admin_id.levels[:level]) for _admin_id in admin_ids]
            admin_ids = list(dict.fromkeys(admin_ids))
            if not silent:
                print(f'Inferred Admin IDs: {format_list(admin_ids)}')

    if isinstance(recipe, dict):
        # Resolve every output file the request covers, not just one.
        #
        # A recipe that saves one file per admin unit
        # (`admin-openplaces-2026_admin3`: admin_id NULL, one file per
        # level-2 unit) needs one path per requested unit. Keying on a
        # single deepest id returned the first state's rows and left
        # every other requested state as spine-only rows with null
        # geometry, silently at the default `silent=True`.
        # With no admin_id the request is the recipe's whole scope,
        # which `_get_output_admin_ids` reads off the recipe when
        # handed None. `admin_ids` is bound only when an admin_id was
        # given: the default geometry recipe is picked after the one
        # place that binds it from a recipe.
        recipe_output_admin_ids = _get_output_admin_ids(
            recipe, admin_ids if admin_id is not None else None
        )[0]
        recipe_parquet_paths = [
            get_output_path(recipe, output_admin_id)
            for output_admin_id in recipe_output_admin_ids
        ]
        missing_parquet_paths = [p for p in recipe_parquet_paths if not p.exists()]
        recipe_parquet_paths = [p for p in recipe_parquet_paths if p.exists()]
        if not recipe_parquet_paths:
            raise FileNotFoundError(
                'No output file of recipe '
                f'`{get_recipe_id(recipe)}` exists for '
                f'{format_list(recipe_output_admin_ids)}; first expected '
                f'path: {missing_parquet_paths[0]}'
            )
        if missing_parquet_paths and not silent:
            warnings.warn(
                f'\n\n{len(missing_parquet_paths)} of '
                f'{len(missing_parquet_paths) + len(recipe_parquet_paths)} '
                f'output files of recipe `{get_recipe_id(recipe)}` do not '
                f'exist; their admin units are returned from the spine '
                f'alone. First missing path: {missing_parquet_paths[0]}\n'
            )

    try:
        # Load admin spine from default source
        # `keep_default_na` is `True` to keep `NA` (Namibia)
        admin = get_recipe_by_id(
            f'{ADMIN_SOURCE_DEFAULT}_admin{level}', dtype=str, keep_default_na=False
        )
        if f'admin{level}_id' not in admin:
            raise ValueError(f"'admin{level}_id' does not exist:\n\n" + str(admin))

        admin = admin.set_index(f'admin{level}_id')

    except OSError:
        # Should only happen before the spine exists.
        warnings.warn(
            f'\n\nAdmin spine not found: {ADMIN_SOURCE_DEFAULT}_admin{level}.\n\n'
            'Falling back to `admin_recipe`.\n'
        )
        if isinstance(recipe, dict):
            # Load spine from recipe
            admin = _concat_recipe_files(recipe_parquet_paths, reader=pd.read_parquet)[
                []
            ]
        else:
            raise OSError(
                f'\n\nAdmin spine not found: {ADMIN_SOURCE_DEFAULT}_admin{level}.\n'
            )

    if isinstance(recipe, dict):
        # If a recipe was provided, check whether it contains new IDs

        # Get path of recipe data in filesystem
        admin_ids_in_spine = list(admin.index)

        # Read only the ID column
        admin_ids_from_recipe = _concat_recipe_files(
            recipe_parquet_paths, columns=[]
        ).index
        admin_ids_to_add_to_spine = sorted(
            set(admin_ids_from_recipe) - set(admin_ids_in_spine)
        )
        if admin_ids_to_add_to_spine:
            if not silent:
                txt_warnings = (
                    f'\n\n{len(admin_ids_to_add_to_spine):,d} admin IDs from recipe `'
                    + (
                        f'{recipe["admin_id"]}_'
                        if recipe['admin_id'].get_level() > 1
                        else ''
                    )
                    + f'{recipe.get("entity") or recipe.get("dataset")}_admin{level}`'
                    ' not found in reference Admin IDs:\n\n- '
                    + '\n- '.join(admin_ids_to_add_to_spine[:5])
                    + ('\n- ...' if len(admin_ids_to_add_to_spine) > 5 else '')
                    + '\n\nOptions to silence this warning:\n'
                    '- Add Admin IDs to reference list in '
                    f'`{ADMIN_SOURCE_DEFAULT}_admin{level}.csv`\n'
                    '- Call get_admin() with silent=True.\n'
                )
                warnings.warn(txt_warnings)

            admin_to_add = pd.DataFrame(
                index=pd.Index(admin_ids_to_add_to_spine, name=f'admin{level}_id'),
                columns=admin.columns,
            )
            admin = pd.concat([admin, admin_to_add]).sort_index()

    # Select admin units
    if admin_id is not None:
        mask_select = pd.Series(False, index=admin.index)
        for _admin_id_to_get in admin_ids:
            # Containment has to stop at a level boundary. A raw prefix
            # test lets 'US-NC-WA' (Wake, pre-2026) select 'US-NC-WAR'
            # (Warren). Tested on the index rather than through
            # `AdminId.is_parent_or_equal_of` because this runs on every
            # get_admin() call over a spine of up to a few hundred
            # thousand ids, where per-row parsing is not affordable.
            scope = str(_admin_id_to_get)
            if not scope:
                # Level 0 is the planet and covers every unit. The
                # separator test below would match nothing for it, so a
                # global recipe would resolve to no rows at all.
                mask_select = pd.Series(True, index=admin.index)
                break
            mask_select |= (admin.index == scope) | admin.index.str.startswith(
                scope + STRING_SEPARATOR_WITHIN_IDS
            )
        if not mask_select.any() and not allow_empty:
            raise ValueError(
                'No admin IDs from reference spine found. Perhaps they have not been '
                f'defined at level {level} for `admin_id`: {format_list(admin_ids)}?'
            )
        admin = admin[mask_select].copy()

    if isinstance(recipe, dict):
        if admin_id is None:
            filters = None
        else:
            filters = [(f'admin{level}_id', 'in', sorted(set(admin.index)))]

        # Read attribute data from filesystem
        admin_from_recipe = _concat_recipe_files(
            recipe_parquet_paths, geom=geom, filters=filters
        )

        # Get column order (retain admin, add recipe)
        column_order = list(dict.fromkeys(list(admin) + list(admin_from_recipe)))

        # Join recipe data to spine, overwriting columns from spine
        shared_columns = list(set(admin) & set(admin_from_recipe))
        # Fill empty recipe data with available data from spine
        admin_from_recipe[shared_columns] = admin_from_recipe[shared_columns].fillna(
            admin[shared_columns]
        )
        # Fill columns for spine-only rows (units the recipe lacks) from the
        # spine, so that they come back named rather than nameless: the
        # fillna above only reaches rows the recipe has.
        spine_shared = admin[shared_columns]
        admin = admin.drop(columns=shared_columns).join(admin_from_recipe, how='outer')[
            column_order
        ]
        admin[shared_columns] = admin[shared_columns].fillna(spine_shared)
        # Cast to GeoDataFrame if geometries are included
        if isinstance(admin_from_recipe, gpd.GeoDataFrame):
            admin = gpd.GeoDataFrame(admin, crs=admin_from_recipe.crs)

    if land_product is not None:
        admin = _join_land_geometry(
            admin, land_product, level, admin_ids if admin_id is not None else None
        )

    # Filter columns
    if not all_columns or columns:
        if columns:
            if isinstance(columns, str):
                columns = [columns]
            elif not isinstance(columns, list):
                raise ValueError(f'`columns` must be a string or list: {columns}')
        columns_to_retain = columns if columns else ADMIN_PRIMARY_COLUMNS[level]
        admin = admin[
            [x for x in columns_to_retain + ['geometry'] if x in admin]
        ].copy()

    # Return without empty columns. `geometry` is exempt: dropping
    # it when every selected unit lacks geometry hands a geom=True
    # caller a plain DataFrame, and the callers that then reach for
    # `.geometry` raise KeyError or AttributeError instead of
    # reporting that the units have no geometry.
    column_is_empty = admin.eq('').all() | admin.isnull().all()
    non_empty_columns = list(column_is_empty[~column_is_empty].index)
    if geom and 'geometry' in admin and 'geometry' not in non_empty_columns:
        non_empty_columns.append('geometry')
    return admin[non_empty_columns]


def _land_product(version):
    """The resolution prefix of a cartographic boundary version, or None."""
    match = re.fullmatch(r'(carto\d+[a-z]+)\d{4}', str(version))
    return match.group(1) if match else None


def _land_geometry_recipe_ids(product, level, admin_ids):
    """Pick the land-only recipes that supply `level` units in scope.

    Every scope the cartographic boundary recipes cover at `level` (the
    national county file, each New England state's towns) gets one
    recipe: the newest vintage of `product` where that scope has it,
    else of the finest resolution. Recipes read from the index, not a
    list here, so a state added beside the others needs no code.

    Parameters
    ----------
    product : str
        Version prefix of the requested resolution, such as 'carto5m'.
    level : int
        Admin level of the units.
    admin_ids : list of AdminId or None
        Units asked for; a recipe whose scope neither contains nor sits
        inside one of them is left out. None keeps every recipe.

    Returns
    -------
    list of str
        Recipe ids, sorted.
    """
    from openplaces.diagnostics import find_recipes

    index = find_recipes('admin', stage='ingest')
    products = index['version'].map(_land_product)
    index = index.assign(product=products)[
        index['level'].eq(level)
        & index['source_id'].eq(LAND_GEOMETRY_SOURCE)
        & products.isin([product, LAND_GEOMETRY_FINEST_PRODUCT])
    ]
    chosen = []
    for scope, rows in index.groupby('admin_id'):
        if admin_ids is not None:
            scope_id = AdminId(scope)
            if not any(
                scope_id.is_parent_or_equal_of(unit)
                or unit.is_parent_or_equal_of(scope_id)
                for unit in admin_ids
            ):
                continue
        preferred = rows[rows['product'].eq(product)]
        rows = preferred if len(preferred) else rows
        newest = max(rows.itertuples(), key=lambda r: version_sort_key(r.version))
        chosen.append(newest.recipe_id)
    return sorted(chosen)


def _join_land_geometry(admin, product, level, admin_ids):
    """Attach land-only geometry to the spine rows `get_admin` selected.

    Parameters
    ----------
    admin : pandas.DataFrame
        Selected spine rows, indexed by `admin{level}_id`.
    product : str
        Version prefix of the requested resolution, such as 'carto5m'.
    level : int
        Admin level of the units.
    admin_ids : list of AdminId or None
        The request's scope, or None for the whole level.

    Returns
    -------
    geopandas.GeoDataFrame
        `admin` with a `geometry` column; units without a polygon in the
        cartographic boundary files (the spine has units the Census does
        not draw) keep an empty geometry.
    """
    recipe_ids = _land_geometry_recipe_ids(product, level, admin_ids)
    index_name = f'admin{level}_id'
    wanted = sorted(set(admin.index))
    parts = []
    for recipe_id in recipe_ids:
        recipe = get_recipe_by_id(recipe_id)
        output_ids = _get_output_admin_ids(recipe, admin_ids)[0]
        paths = [get_output_path(recipe, output_id) for output_id in output_ids]
        paths = [path for path in paths if path.exists()]
        if not paths:
            continue
        part = _concat_recipe_files(
            paths, geom=True, filters=[(index_name, 'in', wanted)]
        )
        parts.append(part[['geometry']])
    if not parts:
        raise FileNotFoundError(
            f'No land-only geometry is ingested for level {level}. '
            f'Ingest: {format_list(recipe_ids)}.'
        )
    geometry = gpd.GeoDataFrame(pd.concat(parts), crs=parts[0].crs)
    geometry = geometry[~geometry.index.duplicated()]
    return gpd.GeoDataFrame(admin.join(geometry), crs=geometry.crs)


def get_admin_ids(admin_level, admin_id=None, admin_recipe=None, allow_empty=False):
    """Get list of administrative unit IDs

    Parameters
    ----------
    admin_level : int
        Admin level of the ids to return.
    admin_id : str, list, or AdminId, optional
        Scope; None selects the whole level.
    admin_recipe : str, optional
        Admin recipe id to read the units from.
    allow_empty : bool
        Return an empty list, rather than raise, when no unit sits at
        `admin_level` within `admin_id`. See `get_admin`.
    """
    admin_ids = get_admin(
        admin_id,
        admin_level,
        columns=[],
        recipe=get_recipe_by_id(admin_recipe) if admin_recipe is not None else None,
        allow_empty=allow_empty,
    ).index.tolist()
    return sorted(admin_ids)


def get_regions(region_id=None):
    """Get the named-region registry: a 1:n mapping of region to admin unit.

    A region is any named group of admin units that the admin hierarchy
    cannot express -- a study area, a delivery footprint, a funder's
    geography. The CHEER regions are the motivating case: 45 of North
    Carolina's 100 counties and 42 of Texas's 254, neither of them a
    complete state.

    Kept in one registry rather than beside whichever recipe first needed
    it, because the same grouping is wanted by delivery, by mapping, and by
    ad-hoc analysis, and three copies of a county list drift.

    Parameters
    ----------
    region_id : str, optional
        Return only this region's rows. Raises `KeyError` when it is not
        registered, listing what is.

    Returns
    -------
    pandas.DataFrame
        Columns `region_id`, `name`, `region_admin_id` (the admin unit the
        region rolls up to, which may be blank) and `admin_id`, one row per
        member unit.
    """
    regions = get_recipe_by_id(REGION_SOURCE_DEFAULT, dtype=str, keep_default_na=False)
    if region_id is None:
        return regions
    rows = regions[regions['region_id'] == str(region_id)]
    if rows.empty:
        known = format_list(sorted(regions['region_id'].unique()))
        raise KeyError(f'Unknown region {region_id!r}; registered: {known}.')
    return rows


def get_region_admin_ids(region_id):
    """Get the admin unit IDs a named region groups, in registry order."""
    # The registry is read with `keep_default_na=False`, so a blank cell
    # arrives as an empty string rather than NaN and `dropna` alone lets
    # it through as a member id.
    admin_ids = get_regions(region_id)['admin_id'].dropna()
    return list(dict.fromkeys(admin_ids[admin_ids.astype(str).str.strip() != '']))


def _as_admin_id(value):
    return value if isinstance(value, AdminId) else AdminId(value)


def entity_outputs_exist(recipe, admin_id=None) -> bool:
    """Whether every file `get_entities` would read for *admin_id* exists.

    Tells a unit that was built with no rows of the entity (a saved
    table without rows, or a placeholder) from one that was not built,
    which `get_entities(missing='warn')` returns alike, as a frame
    without rows. A unit outside the recipe's scope, or one whose files
    cannot be resolved, reads as not built.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity.
    admin_id : str or AdminId, optional
        Administrative unit, resolved as `get_entities` resolves it.

    Returns
    -------
    bool
        True when the unit's output files all exist.
    """
    try:
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)
        partition_id = None
        if (recipe.get('aggregate_by') or {}).get('single_file'):
            partition_id = 'all'
        output_admin_ids, _, _ = _get_output_admin_ids(recipe, admin_id)
        paths = [
            get_output_path(recipe, output_admin_id, partition_id=partition_id)
            for output_admin_id in output_admin_ids
        ]
    except (ValueError, KeyError, FileNotFoundError):
        return False
    return bool(paths) and all(path.exists() for path in paths)


def _get_output_admin_ids(recipe, admin_id):
    """Resolve requested admin IDs to the recipe's output granularity.

    Returns ``(output_admin_ids, finer_requested_ids, whole_output_ids)``:
    the deduped save-level AdminIds whose files must be read, the subset of
    originally requested AdminIds strictly finer than the recipe's save level
    (each such saved file may cover admin units the caller didn't ask for, see
    get_entities' post-read filtering), and the save-level units the caller
    asked for whole, which post-read filtering must not narrow.
    """
    save_level = get_save_admin_level(recipe)
    if save_level == 0:
        return [None], [], []

    recipe_admin_id = _as_admin_id(recipe['admin_id'])
    requested = recipe_admin_id if admin_id is None else admin_id
    if isinstance(requested, str | AdminId):
        requested = [requested]

    output_admin_ids = []
    finer_requested_ids = []
    whole_output_ids = []
    for value in requested:
        requested_admin_id = _as_admin_id(value)
        if requested_admin_id.is_parent_or_equal_of(recipe_admin_id):
            requested_admin_id = recipe_admin_id
        elif not recipe_admin_id.is_parent_or_equal_of(requested_admin_id):
            raise ValueError(
                f'{requested_admin_id} is outside recipe scope {recipe_admin_id}'
            )

        if requested_admin_id.get_level() >= save_level:
            output_admin_id = AdminId(*requested_admin_id.levels[:save_level])
            output_admin_ids.append(output_admin_id)
            if requested_admin_id.get_level() > save_level:
                finer_requested_ids.append(requested_admin_id)
            else:
                whole_output_ids.append(output_admin_id)
            continue

        child_admin_ids = [
            _as_admin_id(child_admin_id)
            for child_admin_id in get_admin(
                requested_admin_id,
                save_level,
                columns=[],
            ).index
            if recipe_admin_id.is_parent_or_equal_of(_as_admin_id(child_admin_id))
        ]
        output_admin_ids.extend(child_admin_ids)
        whole_output_ids.extend(child_admin_ids)

    return (
        list(dict.fromkeys(output_admin_ids)),
        finer_requested_ids,
        list(dict.fromkeys(whole_output_ids)),
    )


def get_entities(
    recipe,
    admin_id=None,
    geom=False,
    layer=None,
    partition_id=None,
    columns: Sequence[str] | None = None,
    missing='raise',
    bbox: tuple[float, float, float, float] | None = None,
    unnarrowable='raise',
):
    """Load and combine processed Parquet tables for entities.

    Parent administrative IDs are expanded to the recipe's save level.
    Recipes aggregated to a single file read their ``_all`` output by
    default.

    A unit finer than the save level (a county of a statewide file) is
    read from the file covering it and narrowed to the unit's rows by,
    in this order: the rows' admin{level}_id column; a spatial overlay
    with the unit's boundary when geometry is read; and, where the unit
    is the only one of its level inside the file's unit (the District of
    Columbia in US-DC), the whole file. A read none of these narrows
    raises :class:`UnnarrowableSourceError` unless *unnarrowable* says
    otherwise.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the entity.
    admin_id : str, AdminId, or sequence
        Administrative unit(s) to load. Defaults to the recipe admin ID.
    geom : bool or 'simplified'
        If True, include geometries and return a GeoDataFrame. If
        ``'simplified'``, join simplified geometries from the
        ``_geo_simplified`` sidecar where one exists (see
        :func:`openplaces.io.read_parquet`).
    layer : str, optional
        Secondary layer defined in ``additional_layers``.
    partition_id : str, optional
        Explicit partition value to read.
    columns : sequence of str, optional
        Columns to read.
    missing : {'raise', 'warn', 'ignore'}
        How to handle missing output files.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial bounding box filter in EPSG:4326, forwarded to
        :func:`openplaces.io.read_parquet` for each resolved output file.
        Exploits per-file covering-bbox predicate pushdown, so files whose
        extent doesn't overlap contribute no rows to the combined result:
        this bounds memory use when loading a recipe across many
        administrative units without loading every file in full.
    unnarrowable : {'raise', 'whole'}
        What a read does when a requested unit finer than the save level
        cannot be narrowed (see above). 'raise' (the default) raises
        :class:`UnnarrowableSourceError`. 'whole' returns every row of
        the file covering the unit and records the units it could not
        narrow to in the result's attrs under 'openplaces_unnarrowed'
        (recipe_id, admin_ids, save_level), for a caller that narrows
        by other means (a name column, see
        :func:`openplaces.io.harmonizer.restrict_to_admin_by_name`) or
        genuinely wants the coarser unit.

    Raises
    ------
    FileNotFoundError
        When an output file is missing and *missing* is 'raise'.
    UnnarrowableSourceError
        When a requested finer unit cannot be narrowed to and
        *unnarrowable* is 'raise'.
    """
    if missing not in {'raise', 'warn', 'ignore'}:
        raise ValueError("missing must be 'raise', 'warn', or 'ignore'")
    if unnarrowable not in UNNARROWABLE_POLICIES:
        raise ValueError("unnarrowable must be 'raise' or 'whole'")
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    if layer is not None:
        recipe = get_table_recipe(recipe, layer)

    # Kept separate from the derived 'all' below: a partition the caller
    # named applies to a predecessor recipe too, one this recipe's own
    # aggregate_by implies does not.
    requested_partition_id = partition_id
    if partition_id is None and (recipe.get('aggregate_by') or {}).get('single_file'):
        partition_id = 'all'

    # An attribute-only recipe (save_to: geometry: false) has no geometry
    # of its own: read its files plain and resolve geometry through the
    # entity_recipe chain afterwards. Driven by the declaration, never by
    # probing for a `_geo` sidecar, so a stale sidecar from a pre-split
    # run beside the attribute output is ignored.
    geometry_recipe = None
    requested_geom = geom
    if geom and not saves_geometry(recipe):
        geometry_recipe = recipe
        seen_ids = set()
        while not saves_geometry(geometry_recipe):
            predecessor = geometry_recipe.get('entity_recipe')
            if not predecessor or predecessor in seen_ids:
                raise ValueError(
                    f'{get_recipe_id(recipe)} declares save_to: geometry: '
                    'false but its entity_recipe chain reaches no recipe '
                    'that saves geometry.'
                )
            seen_ids.add(predecessor)
            geometry_recipe = get_recipe_by_id(predecessor)
        geom = False

    save_level = get_save_admin_level(recipe)
    admin_col = f'admin{save_level}_id' if save_level > 0 else None
    if isinstance(columns, str):
        columns = [columns]
    read_columns = columns
    if admin_col is not None and columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        if admin_col in columns:
            read_columns = [col for col in columns if col != admin_col]
        else:
            admin_col = None

    output_admin_ids, finer_requested_ids, whole_output_ids = _get_output_admin_ids(
        recipe, admin_id
    )
    finer_by_level: dict[int, set[str]] = {}
    for finer_admin_id in finer_requested_ids:
        finer_by_level.setdefault(finer_admin_id.get_level(), set()).add(
            str(finer_admin_id)
        )

    # Narrowing needs columns the caller may not have asked for: each
    # file's own unit, and the finer unit ids its rows carry. A read
    # naming its columns used to leave them out and so returned the
    # whole file. They are dropped again before returning.
    save_col = f'admin{save_level}_id' if save_level > 0 else None
    fill_col = admin_col
    added_cols: set[str] = set()
    if finer_by_level and save_col is not None and admin_col is None:
        fill_col = save_col
        added_cols.add(save_col)
    narrow_cols = [f'admin{level}_id' for level in sorted(finer_by_level)]

    frames = []
    output_paths = []
    missing_paths = []
    for output_admin_id in output_admin_ids:
        path = get_output_path(
            recipe,
            output_admin_id,
            partition_id=partition_id,
        )
        if not path.exists():
            missing_paths.append(path)
            continue
        # A placeholder has none of the requested columns, and asking
        # for one raises instead of returning its no rows.
        placeholder = read_columns is not None and is_placeholder_output(path)
        file_columns = None if placeholder else read_columns
        if file_columns is not None and narrow_cols:
            present = _parquet_column_names(path)
            extra = [c for c in narrow_cols if c in present and c not in file_columns]
            if extra:
                file_columns = [*file_columns, *extra]
                added_cols.update(extra)
        df = read_parquet(path, geom=geom, columns=file_columns, bbox=bbox)
        if fill_col is not None and output_admin_id is not None:
            if fill_col not in df.columns:
                df[fill_col] = str(output_admin_id)
        frames.append(df)
        output_paths.append(path)

    if missing_paths:
        message = (
            f'{len(missing_paths)} recipe output file(s) do not exist; '
            f'first missing path: {missing_paths[0]}'
        )
        if missing == 'raise':
            raise FileNotFoundError(message)
        if missing == 'warn':
            warnings.warn(message, stacklevel=2)

    if not frames:
        data = gpd.GeoDataFrame() if geom else pd.DataFrame()
    elif len(frames) == 1:
        data = frames[0]
    else:
        # Renumbering is safe only when nothing later joins by index.
        # The geometry chain joins on exactly this index, so renumbering
        # a RangeIndex-saved attribute recipe silently gave each row the
        # geometry that happens to sit at its new position.
        ignore_index = geometry_recipe is None and all(
            isinstance(frame.index, pd.RangeIndex) for frame in frames
        )
        data = pd.concat(frames, ignore_index=ignore_index)
        if geometry_recipe is not None and not data.index.is_unique:
            # Reported rather than refused: a read that only wants the
            # attributes still works, and the pipeline stage entries are
            # where an entity id is guaranteed unique.
            warnings.warn(
                f'{get_recipe_id(recipe)} output files repeat index labels '
                f'across admin units, so the geometry joined from '
                f'{get_recipe_id(geometry_recipe)} cannot be attributed to '
                'the right rows. Give the recipe a unique entity id.',
                stacklevel=2,
            )
        if geom:
            data = gpd.GeoDataFrame(data, crs=frames[0].crs)

    if admin_col is not None and admin_col in data.columns:
        data[admin_col] = data[admin_col].astype('category')

    if geometry_recipe is not None and len(data) == 0:
        # No row to join a geometry to (a unit built with none of the
        # entity); the result still carries the geometry it was asked
        # for, as an empty layer.
        from openplaces.config import cfg

        data = gpd.GeoDataFrame(
            data.assign(geometry=gpd.GeoSeries([], index=data.index, crs=cfg.crs)),
            crs=cfg.crs,
        )
        geom = requested_geom
    elif geometry_recipe is not None and not data.empty:
        predecessor = get_entities(
            geometry_recipe,
            admin_id=admin_id,
            geom=requested_geom,
            layer=layer,
            partition_id=requested_partition_id,
            missing=missing,
            bbox=bbox,
            unnarrowable=unnarrowable,
        )
        # `columns` is deliberately not forwarded even though only
        # the geometry is wanted: a split attribute plus `_geo` pair
        # is joined through a `_join_id`/`geo_id` column read from
        # the attribute file, and a narrowed read would drop the very
        # column that join needs.
        if 'geometry' not in predecessor:
            # The predecessor's own files were missing and
            # `missing` let that pass, so there is no geometry to
            # join. Honor the same policy here rather than raising
            # KeyError, and keep the result a GeoDataFrame so a
            # caller can ask and be told.
            message = (
                f'{get_recipe_id(geometry_recipe)} returned no geometry for '
                f'{admin_id}, so {get_recipe_id(recipe)} is returned with an '
                'empty geometry column.'
            )
            if missing == 'raise':
                raise FileNotFoundError(message)
            if missing == 'warn':
                warnings.warn(message, stacklevel=2)
            data = gpd.GeoDataFrame(
                data,
                geometry=gpd.GeoSeries([None] * len(data), index=data.index),
                crs=getattr(predecessor, 'crs', None),
            )
        else:
            geometry = predecessor['geometry']
            n_repeated = int(geometry.index.duplicated().sum())
            if n_repeated:
                # Silent before: the join would otherwise fan every
                # attribute row out across the repeated geometries.
                # Report rather than raise, so a read that only wants
                # to draw still works.
                warnings.warn(
                    f'{n_repeated} repeated {geometry.index.name} '
                    f'label(s) in the geometry of '
                    f'{get_recipe_id(geometry_recipe)} for {admin_id}; '
                    'kept the first geometry of each.',
                    stacklevel=2,
                )
                geometry = geometry[~geometry.index.duplicated()]
            data = gpd.GeoDataFrame(
                data.join(geometry, how='left'), crs=predecessor.crs
            )
            # A bbox read of a geometry-bearing recipe returns only the
            # rows inside the box (the attributes join onto the filtered
            # geometry); match that here rather than keeping out-of-box
            # rows with null geometry.
            if bbox is not None:
                data = data[data.geometry.notna()]
        geom = requested_geom

    # A requested admin_id finer than the recipe's save level (e.g. a town
    # when the recipe saves one file per county) gets truncated to its
    # save-level ancestor above, so a single read file can cover admin units
    # the caller didn't ask for. Narrow back down to exactly what was
    # requested, using a stored per-row id column when the recipe happens to
    # carry one, or a spatial fallback (join to just the requested units'
    # boundary polygons) otherwise. A unit that is the only one of its
    # level inside the file's unit (US-DC-DC in US-DC) is the whole
    # file. What none of these narrows raises, unless the caller asked
    # for the whole file explicitly: returning it silently gave every
    # Indiana county's transaction spine the whole state's sales.
    #
    # Everything the caller asked for is kept, so the levels combine
    # as a union and units requested whole at the save level survive.
    # Applying each level as a successive filter instead returned an
    # empty frame for a two-level request, and dropped every row of a
    # county the caller had named alongside a town.
    unnarrowed_ids: set[str] = set()
    if finer_by_level and not data.empty:
        keep = pd.Series(False, index=data.index)
        whole_ids = {str(admin) for admin in whole_output_ids}
        # Each file's unit is filled in above whenever narrowing is
        # needed, so a unit asked for whole is always identifiable.
        if whole_ids:
            keep |= data[save_col].astype(str).isin(whole_ids)
        for level, ids in finer_by_level.items():
            col = f'admin{level}_id'
            if col not in data.columns and geom:
                from openplaces.geo.overlay import overlay_admin_ids

                boundaries = get_admin(sorted(ids), level, geom=True)['geometry']
                # overlay_admin_ids joins without aligning CRSs,
                # and a projected frame then raises inside the join.
                if data.crs is not None and boundaries.crs is not None:
                    boundaries = boundaries.to_crs(data.crs)
                data = overlay_admin_ids(data, admin_geometries=boundaries)
            keep = keep.reindex(data.index, fill_value=False)
            if col in data.columns:
                keep |= data[col].astype(str).isin(ids)
                continue
            sole = _sole_child_parents(ids, level, save_level)
            keep |= data[save_col].astype(str).isin(sole.values())
            unnarrowed_ids.update(set(ids) - set(sole))
        if unnarrowed_ids and unnarrowable == 'raise':
            raise unnarrowable_error(recipe, unnarrowed_ids, save_level)
        if unnarrowed_ids:
            # Asked for explicitly: the file covering each such unit
            # stays whole, and the result says so.
            parents = {
                str(AdminId(*AdminId(admin_id).levels[:save_level]))
                for admin_id in unnarrowed_ids
            }
            keep |= data[save_col].astype(str).isin(parents)
        data = data[keep]

    if added_cols:
        data = data.drop(columns=[col for col in added_cols if col in data.columns])

    partition_ids = set()
    if partition_id == 'all':
        from openplaces.io.aggregate import read_partition_coverage

        for path in output_paths:
            partition_ids.update(read_partition_coverage(path))

    data.attrs['openplaces_output_paths'] = [str(path) for path in output_paths]
    data.attrs['openplaces_missing_paths'] = [str(path) for path in missing_paths]
    data.attrs['openplaces_partition_ids'] = sorted(partition_ids)
    if unnarrowed_ids:
        data.attrs[UNNARROWED_ATTR] = {
            'recipe_id': get_recipe_id(recipe),
            'admin_ids': sorted(unnarrowed_ids),
            'save_level': save_level,
        }
    return data


def _parquet_column_names(path) -> set[str]:
    """Column names stored in a parquet file, or none if unreadable."""
    try:
        return set(pq.read_schema(path).names)
    except (OSError, ValueError, TypeError):
        return set()


def _sole_child_parents(ids, level, save_level) -> dict[str, str]:
    """The requested units that are the only one of their level in their file.

    Parameters
    ----------
    ids : iterable of str
        Requested units at *level*.
    level : int
        Their admin level.
    save_level : int
        The level of the file covering them.

    Returns
    -------
    dict of str to str
        Each such unit mapped to its save-level ancestor. That ancestor
        has no other unit at *level* its file's rows could belong to,
        so the whole file is the unit's.
    """
    parents = {
        admin_id: str(AdminId(*AdminId(admin_id).levels[:save_level]))
        for admin_id in ids
    }
    sole = {}
    for admin_id, parent in parents.items():
        try:
            children = get_admin(parent, level, columns=[]).index
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if [str(child) for child in children] == [admin_id]:
            sole[admin_id] = parent
    return sole


def describe_recipe(recipe, admin_id=None, layer=None, partition_id=None):
    """Describe a recipe's output columns without reading its data.

    Reads the parquet schema of one output file, so a consumer can check
    what a recipe supplies across a whole state at the cost of a footer
    read per unit rather than a table read. Each column is joined to the
    attribute registry through :func:`openplaces.recipe.resolve_attribute_name`,
    so a provenance-suffixed evidence column (``elevation_fmv2026``)
    reports its registered base attribute, unit, data type and default
    aggregation.

    Parameters
    ----------
    recipe : str or dict
        Recipe id or loaded recipe.
    admin_id : str or AdminId, optional
        Unit whose output file to describe. Defaults to the recipe's own
        admin id. When the unit is finer than the recipe's save level,
        the file covering it is described.
    layer : str, optional
        Secondary layer defined in ``additional_layers``.
    partition_id : str, optional
        Explicit partition value to read.

    Returns
    -------
    pandas.DataFrame
        One row per column, indexed by column name: ``arrow_type`` (the
        stored type), ``attribute`` (the registered base name, or None),
        and the registry's ``data_type``, ``unit``, ``aggregation`` and
        ``stage`` for it. The frame's ``attrs`` carry ``recipe_id``,
        ``admin_id``, ``path`` and ``n_rows``.

    Raises
    ------
    FileNotFoundError
        If the unit's output file does not exist.
    """
    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    if layer is not None:
        recipe = get_table_recipe(recipe, layer)
    if partition_id is None and (recipe.get('aggregate_by') or {}).get('single_file'):
        partition_id = 'all'
    output_admin_ids, _finer, _whole = _get_output_admin_ids(recipe, admin_id)
    if len(output_admin_ids) != 1:
        raise ValueError(
            f'describe_recipe describes one output file; {admin_id!r} resolves '
            f'to {len(output_admin_ids)} for {get_recipe_id(recipe)}.'
        )
    output_admin_id = output_admin_ids[0]
    path = get_output_path(recipe, output_admin_id, partition_id=partition_id)
    if not path.exists():
        raise FileNotFoundError(f'Recipe output file does not exist: {path}')
    metadata = pq.read_metadata(path)
    schema = pq.read_schema(path)
    registry = load_registry()
    rows = []
    for field in schema:
        attribute = resolve_attribute_name(field.name)
        if attribute not in registry.index:
            attribute = None
        entry = registry.loc[attribute] if attribute is not None else None
        rows.append(
            {
                'column': field.name,
                'arrow_type': str(field.type),
                'attribute': attribute,
                'data_type': None if entry is None else entry['data_type'],
                'unit': None if entry is None else entry['unit'],
                'aggregation': None if entry is None else entry['aggregation'],
                'stage': None if entry is None else entry['stage'],
            }
        )
    described = pd.DataFrame(rows).set_index('column').astype(object)
    described = described.where(described.notna(), None)
    described.attrs.update(
        recipe_id=get_recipe_id(recipe),
        admin_id=None if output_admin_id is None else str(output_admin_id),
        path=str(path),
        n_rows=metadata.num_rows,
    )
    return described


def get_dataset(recipe, admin_id=None, partition_id=None, geom=False):
    """Load a processed dataset by recipe.

    Handles both raster and tabular dataset recipes. For raster datasets
    (Cloud Optimized GeoTIFFs written by `fetch_rasters_by_admin`), returns
    the path to the .tif file so the caller controls resource management
    (e.g. with rasterio or xarray). For tabular datasets, returns a
    DataFrame or GeoDataFrame exactly as `get_entities` does.

    Parameters
    ----------
    recipe : str or dict
        Recipe that defines the dataset. Can be a loaded recipe (dict) or
        a string recipe ID.
    admin_id : str or AdminId, optional
        Administrative unit for which to load the data. If None, uses
        the admin_id from the recipe.
    partition_id : str, optional
        Partition value to locate a specific partition file, e.g. '2020'
        for a year-partitioned recipe. Pass None (default) for recipes
        without partitioning.
    geom : bool
        If True, include geometries and return a GeoDataFrame.
        Ignored for raster datasets.

    Returns
    -------
    Path
        Path to the .tif file, for raster datasets.
    pandas.DataFrame or geopandas.GeoDataFrame
        Loaded tabular data, for non-raster datasets.

    Raises
    ------
    ValueError
        If the recipe does not have a 'dataset' key. Use `get_entities`
        for entity recipes.
    """

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)

    if 'dataset' not in recipe:
        raise ValueError(
            "Recipe does not have a 'dataset' key. "
            'Use `get_entities` for entity recipes.'
        )

    if admin_id is None:
        admin_id = recipe['admin_id']

    output_path = get_output_path(recipe, admin_id, partition_id=partition_id)

    if recipe['dataset'].is_raster:
        return output_path

    return read_parquet(output_path, geom=geom)
