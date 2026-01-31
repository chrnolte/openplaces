"""
src/openplaces/io.py

Input/output utilities

"""

import glob
import importlib
import os
import re
import urllib
import warnings
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd

from openplaces.api import get_admin_by_level, get_admin_ids
from openplaces.config import cfg
from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
    REGEX_FILENAME_IN_URL,
    REGEX_HAS_GLOB_WILDCARDS,
    ZIP_EXTENSIONS,
)
from openplaces.core.schema import AdminId
from openplaces.geo.vector import (
    overlay_admin_ids,
)
from openplaces.io import download, find_latest_file_or_gdb, to_parquet, unzip
from openplaces.io.admin import find_admin_recipe  # , get_admin_id_crosswalk
from openplaces.io.parcel import drop_problematic_parcels
from openplaces.io.transform import apply_transformations, get_crosswalk
from openplaces.path import cache_path, external_dir, heap_dir, recipe_path
from openplaces.timing import get_timer, log_step

__all__ = [
    'ingest_recipe_data',
    'get_recipe_data',
]


# Downloads


def _catch_missing_partition_keys_error(recipe, admin_id, partition_id):
    # Error checks
    download_by = recipe['download_by']
    if download_by.startswith('admin') and admin_id is None:
        raise ValueError(
            f"Download of `{recipe['entity'].source}` "
            f"{recipe['entity'].entity_type}s is by `{download_by}`.\n\n"
            f"Use `admin_id` argument to identify the `{download_by}` unit"
            f' to download.'
        )
    elif not download_by.startswith('admin') and partition_id is None:
        raise ValueError(
            f"Download of `{recipe['entity'].source}` "
            f"{recipe['entity'].entity_type} is by `{download_by}`.\n\n"
            f"Use `partition_id` argument to identify the "
            f"`{download_by}` partition to download."
        )


def _get_admin_partition_key(recipe, placeholder, admin_id):
    """Get the key of an administrative unit used by a dataset partition

    Used to obtain the correct filename for partitioned datasets.

    Example: 'US-ND' -> 'NorthDakota'

    Parameters
    ----------
    recipe : dict
        Data ingestion recipe. Needs to have 'entity' and 'download_url'
    placeholder: str
        Data partition placeholder as used in the `download_url`
    admin_id : AdminId or None
        Administrative level for which to resolve partition key.
    """
    # Get partition key from admin data
    admin_level = int(recipe['download_by'].replace('admin', ''))

    # Translate placeholders to admin columns / identifiers by cutting
    # off 'adminX_' prefixes unless the prefix is 'adminX_id'
    # 'admin1_name' -> 'name'
    # 'admin1_id_leaf', 'admin1_id_admin0' -> keep as is
    if placeholder.startswith(
        f"{recipe['download_by']}_"
    ) and not placeholder.startswith(f"{recipe['download_by']}_id"):
        column = placeholder.replace(f"{recipe['download_by']}_", '')
    else:
        column = placeholder
    if column == f"{recipe['download_by']}_id_leaf":
        partition_key = AdminId(admin_id).levels[-1]
    else:
        try:
            admin_recipe = find_admin_recipe(recipe['admin_id'], admin_level)
            partition_key = get_admin_by_level(
                admin_level,
                admin_id,
                recipe=admin_recipe,
                columns=column,
            ).iloc[0, 0]
        except IndexError:
            partition_key = get_admin_by_level(
                admin_level, admin_id, columns=column
            ).iloc[0, 0]

    # Transform IDs if needed
    if (
        'download_partition_key_transform' in recipe
        and placeholder in recipe['download_partition_key_transform']
    ):
        key_transform = recipe['download_partition_key_transform'][placeholder]
        if key_transform == 'remove_spaces':
            partition_key = partition_key.replace(' ', '')

    return partition_key


def get_placeholders(url):
    """Extract placeholders ('{placeholder}') from URL."""
    return list(dict.fromkeys(re.findall(r'\{([a-zA-Z0-9_-]+)\}', url)))


def _resolve_placeholders(
    recipe, admin_id, partition_id, url_or_path, partition_keys=None
):
    """Resolve placeholders (partition keys) in a URL or filepath

    Example: {admin1_name}.geojson.zip => NorthCarolina.geojson.zip
    """
    if partition_keys is None:
        partition_keys = {}
    placeholders = get_placeholders(url_or_path)
    for placeholder in placeholders:
        if placeholder in partition_keys:
            partition_key = partition_keys[placeholder]
        else:
            if not placeholder.startswith('admin'):
                if placeholder == recipe['download_by']:
                    partition_key = partition_id
                else:
                    raise NotImplementedError(
                        "URL placeholders different from `download_by` are only"
                        f'implemented for administrative units:\n\n'
                        f'Placeholder: `{placeholder}`, '
                        f"`download_by`: `{recipe['download_by']}`"
                    )
            elif placeholder == recipe['download_by'] + '_id':
                # Special case: `adminX_id`
                # (unlikely, unless coming from `openplaces`)
                partition_key = admin_id
            else:
                # Get partition key from admin data with transforms
                partition_key = _get_admin_partition_key(recipe, placeholder, admin_id)

            partition_keys[placeholder] = partition_key

        url_or_path = url_or_path.replace('{' + placeholder + '}', str(partition_key))

    return url_or_path, partition_keys


def _resolve_download_url(
    recipe, admin_id=None, partition_id=None, return_partition_keys=False
):
    """Resolve the download URL of a recipe (add in partition keys)

    Parameters
    ----------
    recipe : dict
        Data ingestion recipe. Needs to have 'entity' and 'download_url'
    admin_id : AdminId or None
        Administrative level for which to resolve URL.
    partition_id : str or None
        Partition key (other than administrative) to identify the sub-
        dataset, e.g. a value '2025-09' to replace 'year-month'
    return_partition_keys : bool
        If True, return tuple of (download_url, partition_keys dictionary)
    """

    if 'entity' not in recipe:
        raise ValueError('recipe needs an `entity` with a `source` for the download.')

    if recipe['entity'].source.download_url is not None:
        download_url = recipe['entity'].source.download_url

        # Shortcut: if there are no partitions, just return the URL
        if 'download_by' not in recipe or not recipe['download_by']:
            # Catch error if the URL has placeholder
            placeholders_in_url = get_placeholders(download_url)
            if placeholders_in_url:
                raise ValueError(
                    'Set `download_by` to resolve partition placeholders in download '
                    f'URL:\n{placeholders_in_url}'
                )
            partition_keys = None
        else:
            _catch_missing_partition_keys_error(recipe, admin_id, partition_id)

            download_url, partition_keys = _resolve_placeholders(
                recipe, admin_id, partition_id, download_url
            )
    elif recipe['entity'].source.download_url_source is not None:
        if not recipe['download_by']:
            raise ValueError(
                '`download_url_source` provided, but `download_by` not set.'
            )
        # Scrape website providing download URLs
        with urllib.request.urlopen(recipe['entity'].source.download_url_source) as fp:
            html = fp.read().decode("utf8")

        download_url_source_regex, partition_keys = _resolve_placeholders(
            recipe,
            admin_id,
            partition_id,
            recipe['entity'].source.download_url_source_regex,
        )
        download_url = re.compile(download_url_source_regex).findall(html)[0]
    else:
        download_url = None
        partition_keys = None

    if return_partition_keys:
        return download_url, partition_keys
    else:
        return download_url


def _resolve_downloaded_and_data_paths(
    recipe, admin_id=None, partition_id=None, download_url=None, partition_keys=None
):
    """Get the resolvable paths for the data ingestion files of a recipe

    Returns a tuple of:
    - downloaded_path: the path where the downloaded file is stored
      (will return `None` if `data_path` exists: no download needed)
    - data_path: the path of the readable data file (can be compressed)

    Parameters
    ----------
    recipe : dict
        Data ingestion recipe. Needs to have 'entity' and 'download_url'
    admin_id : AdminId
        Administrative level for which to resolve the URL.
    partition_id : str
        Partition key (other than administrative) to identify the sub-
        dataset, e.g. a value '2025-09' to replace 'year-month'
    download_url : str
        Resolved download URL (from _resolve_download_url) can be passed
        here to reduce impact on filesystem.
    partition_keys : str
        Partition keys (from _resolve_download_url) can be passed here
        to reduce impact on filesystem.
    """

    if 'download_by' in recipe:
        if download_url is None or partition_keys is None:
            download_url, partition_keys = _resolve_download_url(
                recipe, admin_id, partition_id, return_partition_keys=True
            )

    compressed_file_name = None
    if 'compressed_file_name' in recipe:
        if 'download_by' in recipe:
            compressed_file_name, _ = _resolve_placeholders(
                recipe,
                admin_id,
                partition_id,
                recipe['compressed_file_name'],
                partition_keys=partition_keys,
            )
        else:
            compressed_file_name = recipe['compressed_file_name']

    uncompressed_file_name = None
    if 'uncompressed_file_name' in recipe:
        if 'download_by' in recipe:
            uncompressed_file_name, _ = _resolve_placeholders(
                recipe,
                admin_id,
                partition_id,
                recipe['uncompressed_file_name'],
                partition_keys=partition_keys,
            )
        else:
            uncompressed_file_name = recipe['uncompressed_file_name']

    recipe_heap_dir = heap_dir(recipe['admin_id'], recipe['entity'])
    recipe_external_dir = external_dir(recipe['admin_id'], recipe['entity'])

    # Identify path of file to import (to see whether it's already saved)
    if compressed_file_name is not None:
        if uncompressed_file_name is not None:
            # Assume that extracted file will be read (in heap folder)
            data_path = recipe_heap_dir / uncompressed_file_name
        else:
            # Assume that compressed file will be read
            data_path = recipe_external_dir / compressed_file_name
    elif uncompressed_file_name is not None:
        data_path = recipe_external_dir / uncompressed_file_name
    else:
        data_path = None

    # Identify path of file that has been downloaded
    downloaded_path = None
    if data_path is None or not data_path.exists():
        # Find the path of the downloaded file.
        if compressed_file_name is not None:
            downloaded_path = recipe_external_dir / compressed_file_name
        elif uncompressed_file_name is not None:
            downloaded_path = recipe_external_dir / uncompressed_file_name
        elif download_url:
            # Try to extract filename from URL
            re_match = re.search(REGEX_FILENAME_IN_URL, download_url)
            if re_match:
                filename = re_match.group(1)
                print(f'Name of downloaded file inferred from URL: {filename}')
                downloaded_path = recipe_external_dir / filename

        # If the downloaded path contains wildcards, search for it
        if (
            downloaded_path is not None
            and not downloaded_path.exists()
            and re.search(REGEX_HAS_GLOB_WILDCARDS, str(downloaded_path))
        ):
            filepaths = glob.glob(str(downloaded_path))
            if len(filepaths) > 0:
                downloaded_path = Path(max(filepaths, key=os.path.getmtime))
                if len(filepaths) > 1:
                    print(
                        'Found more than one file. Selected most recent one:\n\n'
                        f'{downloaded_path}\n\n'
                        'Others:\n\n'
                        + '\n'.join([x for x in filepaths if x != downloaded_path])
                    )

    return downloaded_path, data_path


def _catch_missing_download_url_error(recipe, downloaded_path):
    if (
        not recipe['entity'].source.download_url
        and not recipe['entity'].source.download_url_source
    ):
        error_message = ''
        if downloaded_path:
            error_message += (
                '\n\nDownloaded file not found in the `openplaces` filesystem:'
                f'\n\n{downloaded_path.relative_to(cfg.data_root)}\n\n'
            )
        error_message += (
            f'Recipe for `{recipe["entity"].source}` has no direct download URL.\n\n'
            '1. Download the data manually here:\n\n'
            + f'{recipe["entity"].source.portal_url}'
            + '\n\n2. Save it in this location:\n\n'
            + str(downloaded_path)
            + '\n\n3. Re-run this data ingestion script.'
        )
        raise FileNotFoundError(error_message)


def download_and_unzip_recipe_data(
    recipe, download_url, downloaded_path, data_path, redownload=False, timer=None
):
    """Download and unzip dataset from the original source

    Parameters
    ----------
    recipe : dict
        Data ingestion recipe
    download_url : str
        URL from which to download the data
    downloaded_path : str
        Path of the downloaded data file (might be compressed)
        Can be None, if `data_path` is not None and exists.
    data_path : str
        Path of the file to read (compressed or uncompressed)
    redownload : bool
        Set to True to skip checking for existing files (overwrite)
    timer : openplaces.timing.Timer or None
        Timer
    """

    if timer is None:
        timer = get_timer('download_and_unzip_recipe_data', verbose=True)

    if data_path is not None and data_path.exists() and not redownload:
        return data_path

    recipe_external_dir = external_dir(recipe['admin_id'], recipe['entity'])
    recipe_heap_dir = heap_dir(recipe['admin_id'], recipe['entity'])

    # Download if necessary
    if (
        (not downloaded_path or not downloaded_path.exists())
        and (not data_path or not data_path.exists())
    ) or redownload:
        _catch_missing_download_url_error(recipe, downloaded_path)

        with log_step('Download', timer=timer):
            downloaded_path = download(download_url, recipe_external_dir)

    # Unzip if necessary
    if (
        'compressed_file_path' in recipe
        or (
            downloaded_path is not None
            and Path(downloaded_path).suffix.lower() in ZIP_EXTENSIONS
        )
        or redownload
    ):
        with log_step('Unzip', timer=timer):
            unzip(downloaded_path, recipe_heap_dir)

    # Pick last extracted file to load if no filepath is provided
    if (
        data_path is None or re.search(REGEX_HAS_GLOB_WILDCARDS, str(data_path))
    ) and recipe_heap_dir.exists():
        data_path = find_latest_file_or_gdb(recipe_heap_dir)
        if data_path is None:
            raise ValueError(
                'find_latest_file_or_gdb(recipe_heap_dir) did not find a valid dataset'
            )
        print(f'Inferred file to read:\n{data_path.relative_to(recipe_heap_dir)}')

    if data_path is not None and not data_path.exists():
        raise FileNotFoundError(
            f'Did not succeed in downloading / unzipping:\n{data_path}'
        )

    return data_path


def _admin_id_is_below_dataset_admin_id(recipe, admin_id):
    admin_id_level = len(AdminId(admin_id).levels) - 1
    if 'download_by' in recipe and recipe['download_by'].startswith('admin'):
        dataset_level = int(recipe['download_by'].replace('admin', ''))
    else:
        dataset_level = len(recipe['admin_id'].levels) - 1
    return admin_id_level > dataset_level


def read_recipe_data(recipe, data_path, admin_id=None, timer=None):
    """Read a data file from a recipe

    Parameters
    -----------
    recipe : dict
        Data ingestion recipe (needed to read a specific layer)
    data_path : pathlib.Path
        Path to data file
    admin_id : str
        Identifier of the administrative unit to read.
        Used to query (access subsets of) large data files.
    timer : openplaces.timing.Timer or None
        Timer
    """
    if timer is None:
        timer = get_timer('read_recipe_data', verbose=True)

    if admin_id is not None and _admin_id_is_below_dataset_admin_id(recipe, admin_id):
        if 'process_by' not in recipe:
            raise NotImplementedError(
                f"`admin_id` is {admin_id} but no 'process_by' in `recipe`."
            )
        admin_id_col = recipe['process_by']['admin_id_col']

        # Attribute entities to administrative unit IDs
        if 'admin_id_crosswalk' in recipe['process_by']:
            # Use custom crosswalk
            admin_id_crosswalk_dict = recipe['process_by']['admin_id_crosswalk']
        elif 'admin_id_crosswalk' in recipe:
            # Use recipe crosswalk (backup)
            admin_id_crosswalk_dict = recipe['admin_id_crosswalk']
        else:
            admin_id_crosswalk_dict = None

        if admin_id_crosswalk_dict:
            admin_id_crosswalk_dict['admin_id'] = recipe['admin_id']
            admin_id_crosswalk = get_crosswalk(admin_id_crosswalk_dict)
            admin_id_value = admin_id_crosswalk.loc[admin_id]
        else:
            admin_id_value = admin_id

        if isinstance(admin_id_value, str):
            where = f"{admin_id_col} = '{admin_id_value}'"
        else:
            where = f"{admin_id_col} = {admin_id_value}"
    else:
        where = None

    layer = recipe['layer'] if 'layer' in recipe else None

    # Silence warnings from reading complex polygons
    warnings.filterwarnings('ignore', 'received a polygon with more than 100 parts')

    if data_path.suffix in GEOPANDAS_EXTENSIONS:
        if data_path.suffix == 'parquet':
            gdf = gpd.read_parquet(data_path)
            timer.mark('Read parquet file', path=data_path)
        else:
            try:
                gdf = gpd.read_file(data_path, layer=layer, where=where)
            except (RuntimeWarning, Exception) as e:
                if 'CDF' in str(e) or 'Permission denied' in str(e):
                    gdf = gpd.read_file(
                        data_path, layer=layer, where=where, engine='fiona'
                    )
                else:
                    raise
            timer.mark('Read vector file', path=data_path)
    elif data_path.suffix in PANDAS_EXTENSIONS:
        gdf = pd.read_file(data_path)
        timer.mark('Read data table', where=where, path=data_path)
    elif data_path.suffix in ZIP_EXTENSIONS:
        try:
            # Try to read with `geopandas`
            gdf = gpd.read_file(data_path, layer=layer, where=where)
            timer.mark('Read compressed file', path=data_path)
        except (RuntimeWarning, Exception):
            recipe_heap_dir = heap_dir(recipe['admin_id'], recipe['entity'])
            if not recipe_heap_dir.exists():
                unzip(data_path, recipe_heap_dir)

            data_path = find_latest_file_or_gdb(recipe_heap_dir)
            if data_path is None:
                raise OSError(
                    f'`geopandas` could not read compressed file:\n\n{data_path}.\n\n'
                    f'After unzipping, could not find a valid dataset in:\n\n{heap_dir}'
                )

            gdf = gpd.read_file(data_path, layer=layer, where=where)
            timer.mark('Read unzipped file', path=data_path)
    else:
        raise ValueError(f'Filepath suffix not yet interpreted: {data_path.suffix}')

    warnings.filterwarnings('default', 'received a polygon with more than 100 parts')

    return gdf


def get_labels(recipe, column):
    """Get dictionary of codes > labels for a column in a recipe

    Parameters
    ----------
    recipe : dict
        openplaces recipe
    column : str
        Column for which to find a CSV file with labels near the recipe
        Underscores will be converted to dashes.
        Example: if the column is 'purpose_group', the label CSV is:
        '<recipe_id>_purpose-group-labels.csv'
    """
    labels_path = recipe_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=column.replace('_', '-') + '-labels.csv',
    )
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        labels = labels.set_index(labels.columns[0])[labels.columns[1]].to_dict()
        return labels
    else:
        return None


def preprocess_recipe_data(df, recipe, timer=True):
    """Preprocess imported dataset

    Handles column renaming, indexing, querying, null value filling

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Dataframe with unprocessed data.
    recipe : dict
        `openplaces` recipe
    timer : openplaces.timing.Timer or None
        Timer
    """

    if timer is True:
        timer = get_timer('preprocess_recipe_data', verbose=True)

    # Replace known NA value strings with `None`.
    if 'null_value_strings' in recipe:
        for col, na_value in product(df.columns, recipe['null_value_strings']):
            i_has_na_value = df[col].eq(na_value)
            if i_has_na_value.sum():
                df.loc[i_has_na_value, col] = None

    # Rename columns
    if 'columns' in recipe:
        # Rename columns
        df = df.rename(columns={v: k for k, v in recipe['columns'].items()})

    # Filter rows
    if 'query' in recipe:
        df = df.query(recipe['query'])

    # Cast columns to categorical
    if 'columns_to_categorical' in recipe:
        columns_to_cast = [v for v in recipe['columns_to_categorical'] if v in df]
        for column_to_cast in columns_to_cast:
            # If labels are provided, use labels as categories
            # (more human-readable)
            labels = get_labels(recipe, column_to_cast)
            if labels is not None:
                values = df[column_to_cast].replace(labels)
                categories = labels.values()
                ordered = True
            else:
                values = df[column_to_cast]
                categories = None
                ordered = False
            df[column_to_cast] = pd.Series(
                pd.Categorical(values, categories, ordered),
                index=values.index,
            )

    # Set index
    if 'set_index' in recipe:
        # Set column as index
        if recipe['set_index'] not in df:
            raise ValueError(
                'Column not found to use as index: ' + str(recipe['set_index'])
            )
        if df[recipe['set_index']].duplicated().any():
            raise ValueError(
                f"Duplicates found in '{recipe['set_index']}'. Choose other index.\n\n"
                + str(
                    df[df[recipe['set_index']].duplicated(keep=False)][
                        recipe['set_index']
                    ]
                    .sort_values()
                    .head(5)
                )
            )
        df = df.set_index(recipe['set_index'])
    elif 'create_index' in recipe:
        if recipe['create_index']['method'] == 'prefix':
            df.index = pd.Index(
                recipe['create_index']['prefix'] + df[recipe['create_index']['column']],
                name=recipe['create_index']['name'],
            )
    elif 'index_function' in recipe:
        # Create index with custom function

        def load_function(path):
            module, name = path.rsplit('.', 1)
            return getattr(importlib.import_module(module), name)

        with log_step('Generate indices', timer=timer):
            index_function = load_function(recipe['index_function'])

            df = index_function(df)

    # Drop observations by index
    if 'drop' in recipe:
        df = df.drop(recipe['drop'])

    # Double-check that the index has no duplicates
    if df.index.duplicated().any():
        raise ValueError(
            'Duplicated indices are not allowed in imported data.\n'
            'Change `index_function`, `create_index` or `set_index` column:\n'
            + str(df[df.index.duplicated(keep=False)].sort_index().head().iloc[:, :5])
        )

    # Reorder columns
    if 'columns' in recipe:
        cols_order = [c for c in recipe['columns'] if c in df]
        if 'keep_unnamed_columns' in recipe and recipe['keep_unnamed_columns']:
            cols_order += [
                c for c in df if c not in (list(recipe['columns']) + ['geometry'])
            ]
        if 'geometry' in df:
            cols_order += ['geometry']
        df = df[cols_order]

    if 'transformations' in recipe:
        df = apply_transformations(df, recipe)

    return df


def get_recipe_data(
    recipe,
    admin_id=None,
    partition_id=None,
    timer=True,
    redownload=True,
):
    """Get data described in a recipe.

    Handles downloading (to `external` directory), unzipping, reading,
    and preprocessing (columns, indices, queries, null/na values).

    Parameters
    ----------
    recipe : dict
        `openplaces` recipe. Should be a dictionary (not a DataFrame).
    admin_id : str
        Identifier of the administrative unit to download / process.
        Required if data downloads are partitioned by admin unit.
        Can be used to query (access subsets of) large data files.
    partition_id : str
        Identifier of other partitions to download (e.g. year-month)
    timer : openplaces.timing.Timer or None
        Timer
    redownload : bool
        If True, will re-download and unzip original data
    """
    if timer is True:
        timer = get_timer('get_recipe_data', verbose=True)

    if not isinstance(recipe, dict):
        raise TypeError(
            f'recipe needs to be a dictionary but is {type(recipe)}:\n{str(recipe)}'
        )

    # Resolve download URL
    download_url, partition_keys = _resolve_download_url(
        recipe, admin_id, return_partition_keys=True
    )

    # Resolve downloaded path and data path
    downloaded_path, data_path = _resolve_downloaded_and_data_paths(
        recipe, admin_id, partition_id, download_url, partition_keys=partition_keys
    )

    # Download and unzip the data (if it has not happened yet)
    if data_path is None or not data_path.exists() or redownload:
        data_path = download_and_unzip_recipe_data(
            recipe, download_url, downloaded_path, data_path, redownload
        )

    # Read data
    gdf = read_recipe_data(recipe, data_path, admin_id=admin_id, timer=timer)

    # Reproject vector data to default CRS
    if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs != cfg.crs:
        gdf = gdf.to_crs(cfg.crs)
        timer.mark(f'Reproject to {cfg.crs}')

    # Preprocess recipe data
    # (column names, indices, remapping, NAs, categoricals)
    gdf = preprocess_recipe_data(gdf, recipe, timer=timer)

    # Attribute entities to administrative unit IDs
    if 'admin_id_crosswalk' in recipe:
        admin_id_crosswalk_dict = recipe['admin_id_crosswalk']
        admin_id_crosswalk_dict['admin_id'] = recipe['admin_id']
        admin_id_crosswalk = get_crosswalk(admin_id_crosswalk_dict, flip=True)
        gdf = gdf.join(admin_id_crosswalk, on=admin_id_crosswalk.index.name)
    elif 'overlay_admin_ids' in recipe:
        gdf = overlay_admin_ids(
            gdf, admin_id, **recipe['overlay_admin_ids'], timer=timer
        )
        timer.mark('Overlay admin IDs')

    if 'drop_problematic_parcels' in recipe:
        gdf = drop_problematic_parcels(
            gdf,
            recipe['columns'].keys() if 'columns' in recipe else None,
            **recipe['drop_problematic_parcels'],
        )
        timer.mark('Drop problematic parcels')

    return gdf


def save_recipe_data(
    gdf,
    recipe,
    admin_id=None,
    partition_id=None,
):
    """Save data from an ingested recipe

    Parameters
    ----------
    gdf : GeoDataFrame or DataFrame
        Data, ready to be saved
    recipe : dict
        `openplaces` recipe. Should be a dictionary (not a DataFrame).
    admin_id : str
        Identifier of the administrative unit of the GeoDataFrame

        Required if the `admin_id` cannot be inferred from the recipe,
        because the recipe partitions data downloads by admin unit.

        If the recipe splits the data by lower-level administrative
        units (i.e., if 'cache_by_admin_level' is set) and `admin_id`
        belongs to that level, save only data for that `admin_id`.
    partition_id : str
        Identifier of other partitions to save (e.g. year-month)
    """
    if partition_id is not None:
        raise NotImplementedError(f'Not yet implemented: partition_id={partition_id}')

    if 'cache_by_admin_level' in recipe:
        admin_level = recipe['cache_by_admin_level']
        admin_id_col = f"admin{admin_level}_id"
        if admin_id_col not in gdf:
            raise ValueError(
                f"'cache_by_admin_level' is set, but column "
                f"'{admin_id_col}' does not exist in DataFrame:\n\n"
                + str(gdf.sample(1).T)
            )
        admin_ids_to_save = sorted(set(gdf[admin_id_col].dropna()))

        # If the passed `admin_id` has the same level as the cached
        # partitions, save only the data for that unit (lightweight)
        if admin_id is not None and len(AdminId(admin_id).levels) == admin_level + 1:
            if admin_id not in admin_ids_to_save:
                raise ValueError(
                    f'`admin_id` {admin_id} not found in column `{admin_id_col}`.'
                    'Choose from:\n' + ', '.join(admin_ids_to_save)
                )
            admin_ids_to_save = [admin_id]

    elif admin_id:
        admin_ids_to_save = [admin_id]
    elif 'download_by' not in recipe or not recipe['download_by'].startswith('admin'):
        admin_ids_to_save = [recipe['admin_id']]
    else:
        raise ValueError(
            'Recipe downloads are partitioned by administrative unit.'
            '`admin_id` is required.'
        )

    for admin_id_to_save in admin_ids_to_save:
        if 'cache_by_admin_level' in recipe:
            gdf_to_save = (
                gdf[gdf[admin_id_col].eq(admin_id_to_save)]
                .copy()
                .drop(columns=admin_id_col)
            )
        else:
            gdf_to_save = gdf.copy()

        # Create space-efficient integer ID to join data tables
        gdf_to_save['_join_id'] = range(1, len(gdf_to_save) + 1)

        parquet_path = cache_path(
            admin_id_to_save,
            recipe['entity'],
            filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
        )

        if isinstance(gdf_to_save, gpd.GeoDataFrame):
            # Create two files for tabular and geospatial ('_geo') data
            to_parquet(
                gdf_to_save[[v for v in gdf_to_save if v != 'geometry']], parquet_path
            )

            parquet_geo_path = cache_path(
                admin_id_to_save,
                recipe['entity'],
                filename='_'.join(
                    ([recipe['cache_filename']] if 'cache_filename' in recipe else [])
                    + ['geo']
                ),
            )

            to_parquet(
                gdf_to_save.set_index('_join_id')[['geometry']], parquet_geo_path
            )
        else:
            to_parquet(gdf_to_save, parquet_path)


def ingest_recipe_data(
    recipe,
    admin_id=None,
    partition_id=None,
    timer=None,
    return_result=False,
    redownload=False,
):
    """Execute the ingestion recipe for the dataset.

    Might involve download, reading, cleanup, geoprocessing.

    Creates a dataset in the cache folder for subsequent processing.

    Parameters
    ----------
    recipe : dict
        Data ingestion recipe
    admin_id : str
        Identifier of the administrative unit.
        Required if recipe partitions data downloads by admin unit.
        Can be used to query (access subsets of) large data files.
    partition_id : str
        Identifier of other partitions to download (e.g. year-month)
    timer : openplaces.timing.Timer
        Timer.
    return_result : bool
        Should result be returned (should they be loaded if they exist)?
    redownload : bool
        If True, will re-download and unzip original data
    """

    if timer is None:
        timer = get_timer('ingest_recipe_data', verbose=True)

    if redownload:
        reprocess = True

    # Resolve which administrative units will be processed (chunking)
    if 'process_by' in recipe:
        _admin_recipe_id = (
            recipe['process_by']['admin_recipe_id']
            if 'admin_recipe_id' in recipe['process_by']
            else None
        )
        admin_ids_to_process = get_admin_ids(
            recipe['process_by']['admin_level'],
            recipe['admin_id'],
            _admin_recipe_id,
        )
        if admin_id is not None:
            if admin_id in admin_ids_to_process:
                admin_ids_to_process = [admin_id]
            else:
                raise ValueError(
                    f'{admin_id} not found. Admin recipe: {_admin_recipe_id}`.'
                )
    else:
        admin_ids_to_process = [admin_id]

    for admin_id_to_process in admin_ids_to_process:
        if len(admin_ids_to_process) > 1:
            print(f'Processing {admin_id_to_process}...')

        # Get data from recipe (may involve downloads, unzipping, reading)
        gdf = get_recipe_data(
            recipe, admin_id_to_process, partition_id, timer, redownload
        )
        timer.mark('Get recipe data', admin_id=admin_id_to_process)

        # Save recipe data
        save_recipe_data(gdf, recipe, admin_id_to_process, partition_id)
        timer.mark('Save recipe data', admin_id=admin_id_to_process)

    if return_result:
        if len(admin_ids_to_process) > 1:
            warnings.warn(
                'Multiple administrative units were processed. '
                'Returning data for {admin_id_to_process}.'
            )
        return gdf
