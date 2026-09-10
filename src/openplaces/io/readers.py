"""
Low-level data-access functions for admin units and entities.
Imported by internal modules (io/*, geo/*). Also re-exported from api.py.
"""

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
    find_admin_recipe_id,
    get_output_path,
    get_recipe_by_id,
    get_recipe_id,
    get_save_admin_level,
    get_table_recipe,
    resolve_attribute_name,
    saves_geometry,
)
from openplaces.utils import format_list

ADMIN_SOURCE_DEFAULT = 'admin-spine-2026'
REGION_SOURCE_DEFAULT = 'admin-regions-2026'
ADMIN_GEO_SOURCE_DEFAULT = 'admin-gadm-4~1'
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
    frames = [reader(path, **read_kwargs) for path in paths]
    if len(frames) == 1:
        return frames[0]
    combined = pd.concat(frames)
    if isinstance(frames[0], gpd.GeoDataFrame):
        combined = gpd.GeoDataFrame(combined, crs=frames[0].crs)
    return combined


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
    geom : bool or 'simplified'
        If False, return a DataFrame without geometries.
        If True, return a GeoDataFrame with full geometries.
        If 'simplified', return a GeoDataFrame with simplified geometries
        from the `_geo_simplified` companion file written by
        `AdminHarmonizer`.
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

    # Pick default recipe for geometry attributes if None is provided
    if geom and recipe is None:
        if level is None:
            # Default to countries
            level = 1
        # Prefer the most specific admin recipe scope the requested IDs
        # share, walking up to the country; fall back to the global GADM
        # default otherwise. The walk matters where a level's geometry
        # ships per state rather than nationally: New England admin3 is
        # towns, excluded from the national county layer and provided by
        # per-state COUSUB recipes (US-MA_admin-census-2025_admin3), so a
        # request scoped inside one state must find that state's recipe.
        in_country_recipe_id = None
        if admin_id is not None:
            common = admin_ids[0].levels
            for _aid in admin_ids[1:]:
                n = 0
                while (
                    n < len(common)
                    and n < len(_aid.levels)
                    and common[n] == _aid.levels[n]
                ):
                    n += 1
                common = common[:n]
            for scope_level in range(len(common), 0, -1):
                in_country_recipe_id = find_admin_recipe_id(
                    AdminId(*common[:scope_level]), level, silent=True
                )
                if in_country_recipe_id:
                    break
        recipe = in_country_recipe_id or f'{ADMIN_GEO_SOURCE_DEFAULT}_admin{level}'
        if isinstance(recipe, str):
            recipe = get_recipe_by_id(recipe)

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
        recipe_output_admin_ids = _get_output_admin_ids(recipe, admin_ids)[0]
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
):
    """Load and combine processed Parquet tables for entities.

    Parent administrative IDs are expanded to the recipe's save level.
    Recipes aggregated to a single file read their ``_all`` output by
    default.

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
    """
    if missing not in {'raise', 'warn', 'ignore'}:
        raise ValueError("missing must be 'raise', 'warn', or 'ignore'")
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
        df = read_parquet(path, geom=geom, columns=read_columns, bbox=bbox)
        if admin_col is not None and output_admin_id is not None:
            if admin_col not in df.columns:
                df[admin_col] = str(output_admin_id)
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

    if geometry_recipe is not None and not data.empty:
        predecessor = get_entities(
            geometry_recipe,
            admin_id=admin_id,
            geom=requested_geom,
            layer=layer,
            partition_id=requested_partition_id,
            missing=missing,
            bbox=bbox,
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
    # boundary polygons) otherwise.
    #
    # Everything the caller asked for is kept, so the levels combine
    # as a union and units requested whole at the save level survive.
    # Applying each level as a successive filter instead returned an
    # empty frame for a two-level request, and dropped every row of a
    # county the caller had named alongside a town.
    if finer_by_level and not data.empty:
        keep = pd.Series(False, index=data.index)
        save_col = f'admin{save_level}_id' if save_level > 0 else None
        whole_ids = {str(admin) for admin in whole_output_ids}
        # A unit asked for whole has to be identifiable before
        # anything is narrowed, or narrowing drops all of its rows.
        unresolved_whole = bool(whole_ids) and (
            save_col is None or save_col not in data.columns
        )
        if whole_ids and not unresolved_whole:
            keep |= data[save_col].astype(str).isin(whole_ids)
        unresolved_levels = []
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
            if col in data.columns:
                keep = keep.reindex(data.index, fill_value=False)
                keep |= data[col].astype(str).isin(ids)
            else:
                unresolved_levels.append(level)
        if unresolved_levels or unresolved_whole:
            unresolved = unresolved_levels or [save_level]
            warnings.warn(
                'Cannot restrict output to the requested '
                f'{format_list([f"admin{i}" for i in unresolved])} IDs: '
                "this recipe's saved data has no such column and geom=False "
                'rules out a spatial fallback; returning unfiltered '
                f'admin{save_level}-level data instead.',
                stacklevel=2,
            )
        if not unresolved_whole and len(unresolved_levels) < len(finer_by_level):
            data = data[keep]

    partition_ids = set()
    if partition_id == 'all':
        from openplaces.io.aggregate import read_partition_coverage

        for path in output_paths:
            partition_ids.update(read_partition_coverage(path))

    data.attrs['openplaces_output_paths'] = [str(path) for path in output_paths]
    data.attrs['openplaces_missing_paths'] = [str(path) for path in missing_paths]
    data.attrs['openplaces_partition_ids'] = sorted(partition_ids)
    return data


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
