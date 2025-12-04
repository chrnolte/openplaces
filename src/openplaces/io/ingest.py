"""
src/openplaces/io.py

Input/output utilities

"""

import glob
import importlib
import os
import re
import shutil
import tempfile
from itertools import product
from pathlib import Path
from urllib.parse import unquote, urlparse
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from openplaces.config import cfg
from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
    REGEX_FILENAME_IN_URL,
    REGEX_HAS_GLOB_WILDCARDS,
    ZIP_EXTENSIONS,
)
from openplaces.geo.vector import add_geometry_derivatives, get_simplified_geometries
from openplaces.io import download, find_latest_file, to_parquet, unzip
from openplaces.path import cache_path, external_dir, heap_dir
from openplaces.timing import get_timer, log_step

__all__ = [
    'ingest_recipe',
    'get_recipe_data',
]


def ingest_recipe(recipe, timer=None, return_result=False, redo=False):
    """
    Execute the ingestion recipe for the dataset.

    Might involve download, reading, cleanup, geoprocessing.

    Creates a dataset in the cache folder for subsequent processing.

    Parameters
    ----------
    recipe : dict
        Data ingestion recipe. From `openplaces.recipes.get_recipe()`
    timer : openplaces.timing.Timer
        Timer. From `openplaces.timing.get_timer()`
    return_result : bool
        Should result be returned (should they be loaded if they exist)?
    redo : bool
        If True, will overwrite existing files
    """
    if timer is None:
        timer = get_timer('import_recipe', verbose=True)

    # Outcome path
    parquet_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=recipe['cache_filename'] if 'cache_filename' in recipe else None,
    )

    parquet_geo_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=(recipe['cache_filename'] if 'cache_filename' in recipe else '')
        + '_geo',
    )

    if not parquet_path.exists() or redo:
        # Get data from recipe (may involve downloads, unzipping, reading)
        gdf = get_recipe_data(recipe, timer=timer)
        timer.mark('Get recipe data')

        if (
            isinstance(gdf, gpd.GeoDataFrame)
            and 'add_geometry_derivatives' in recipe
            and recipe['add_geometry_derivatives']
        ):
            with log_step('Add geometry derivatives', timer=timer):
                gdf = add_geometry_derivatives(gdf, timer=timer, **recipe)

        # Reorder columns
        if 'columns' in recipe:
            cols_order = [c for c in recipe['columns'] if c in gdf]
            cols_order += [
                c for c in gdf if c not in (list(recipe['columns']) + ['geometry'])
            ]
            if 'geometry' in gdf:
                cols_order += ['geometry']
            gdf = gdf[cols_order]

        if isinstance(gdf, gpd.GeoDataFrame):
            # Create two files for tabular and geospatial ('_geo') data
            to_parquet(gdf[[v for v in gdf if v != 'geometry']], parquet_path)
            timer.mark('Save attributes as parquet')

            to_parquet(gdf[['geometry']], parquet_geo_path)
            timer.mark('Save geometries as parquet')
        else:
            to_parquet(gdf, parquet_path)
            timer.mark('Save as parquet')

        if (
            isinstance(gdf, gpd.GeoDataFrame)
            and 'simplify_coverage_tolerance' in recipe
            and isinstance(recipe['simplify_coverage_tolerance'], float)
            and recipe['simplify_coverage_tolerance'] > 0
        ):
            # Create and save geometry-simplified versions
            save_simplified_geometries(gdf, recipe, timer)

    elif return_result:
        gdf = pd.read_parquet(parquet_path)
        if parquet_geo_path.exists():
            # Join geometries if they exist
            gdf = gpd.read_parquet(parquet_geo_path).join(gdf)[
                list(gdf.columns) + ['geometry']
            ]
        timer.mark('Loaded parquet')

    if return_result:
        return gdf


def save_simplified_geometries(gdf, recipe, timer=None):
    """Saves the recipe geodataframe with simplified geometries"""
    if timer is None:
        timer = get_timer('save_simplified_geometries', verbose=True)

    # Catching argument errors: enforce positive float
    if 'simplify_coverage_tolerance' not in recipe:
        raise KeyError('`simplify_coverage_tolerance` not found in recipe.')
    if not isinstance(recipe['simplify_coverage_tolerance'], float):
        raise ValueError(
            '`simplify_coverage_tolerance` is not a float: '
            + str(recipe['simplify_coverage_tolerance'])
        )
    if not recipe['simplify_coverage_tolerance'] > 0:
        raise ValueError(
            '`simplify_coverage_tolerance` is not positive: '
            + str(recipe['simplify_coverage_tolerance'])
        )

    if timer.verbose:
        print('Creating simplified geometries. This can take some time.')
    gdf_simplified = get_simplified_geometries(
        gdf, recipe['simplify_coverage_tolerance']
    )
    timer.mark('Simplify coverage')

    parquet_simplified_path = cache_path(
        recipe['admin_id'],
        recipe['entity'],
        filename=(recipe['cache_filename'] if 'cache_filename' in recipe else '')
        + '_geo_simplified',
    )
    to_parquet(gdf_simplified[['geometry']], parquet_simplified_path)
    timer.mark('Save simplified coverage as geoparquet')


def get_recipe_data(recipe, timer=True):
    """Get data described in a recipe

    Handles downloading (to `external` directory), unzipping, reading,
    column renaming, indexing, querying, null value filling.

    Parameters
    ----------
    recipe : dict
        `openplaces` recipe. Should be a dictionary (not a DataFrame).
    timer : True, openplaces.timing.Timer, or None
        True: create timer
        openplaces.timing.Timer: use that one.
    """
    if timer is True:
        timer = get_timer('save_simplified_coverage', verbose=True)

    if not isinstance(recipe, dict):
        raise TypeError(
            f'recipe needs to be a dictionary but is {type(recipe)}:\n{str(recipe)}'
        )

    recipe_heap_dir = heap_dir(recipe['admin_id'], recipe['entity'])
    recipe_external_dir = external_dir(recipe['admin_id'], recipe['entity'])

    # Identify path of file to import (to see whether it's already saved)
    if 'compressed_file_name' in recipe:
        if 'uncompressed_file_name' in recipe:
            # Assume that uncompressed file will be read
            filepath = recipe_heap_dir / recipe['uncompressed_file_name']
        else:
            # Assume that compressed file will be read
            filepath = recipe_external_dir / recipe['compressed_file_name']
    elif 'uncompressed_file_name' in recipe:
        filepath = recipe_external_dir / recipe['uncompressed_file_name']
    else:
        filepath = None

    if filepath is None or not filepath.exists():
        # Find the path of the downloaded file.
        if 'compressed_file_name' in recipe:
            downloaded_path = recipe_external_dir / recipe['compressed_file_name']
        elif 'uncompressed_file_name' in recipe:
            downloaded_path = recipe_external_dir / recipe['uncompressed_file_name']
        elif recipe['entity'].source.download_url:
            # Try to extract filename from URL
            re_match = re.search(
                REGEX_FILENAME_IN_URL, recipe['entity'].source.download_url
            )
            if re_match:
                filename = re_match.group(1)
                print(f'Name of downloaded file inferred from URL: {filename}')
                downloaded_path = recipe_external_dir / filename
            else:
                downloaded_path = None
        else:
            downloaded_path = None

        # If no filepath is provided
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
                        f'Found more than one file. Selected most recent one:'
                        '{download_path}'
                    )

        # Download if necessary
        if not downloaded_path or not downloaded_path.exists():
            if not recipe['entity'].source.download_url:
                error_message = ''
                if downloaded_path:
                    error_message += (
                        '\n\nSource data file not found in the `openplaces` filesystem:'
                        f'\n\n{downloaded_path.relative_to(cfg.data_root)}\n\n'
                    )
                error_message += (
                    f'Source `{recipe["entity"].source}` has no direct download URL. '
                    'Download the data manually here:\n\n'
                    + f'{recipe["entity"].source.portal_url}'
                    + f'\n\nSave it in this location:\n\n'
                    + str(downloaded_path)
                    + '\n\nThen re-run this data ingestion script.'
                )
                raise FileNotFoundError(error_message)

            with log_step('Download', timer=timer):
                downloaded_path = download(
                    recipe['entity'].source.download_url, recipe_external_dir
                )

        # Unzip if necessary
        if 'compressed_file_path' in recipe or (
            downloaded_path is not None
            and Path(downloaded_path).suffix.lower() in ZIP_EXTENSIONS
        ):
            with log_step('Unzip', timer=timer):
                unzip(downloaded_path, recipe_heap_dir)

        # Pick last extracted file to load if no filepath is provided
        if filepath is None or re.search(REGEX_HAS_GLOB_WILDCARDS, str(filepath)):
            filepath = find_latest_file(recipe_heap_dir)
            print(f'Inferred file to read:\n{filepath.relative_to(recipe_heap_dir)}')
        else:
            if not filepath.exists():
                raise FileNotFoundError(
                    f'Did not succeed in downloading / unzipping:\n{filepath}'
                )

    # Read file
    if filepath.suffix in GEOPANDAS_EXTENSIONS:
        if filepath.suffix == 'parquet':
            with log_step('Read parquet', timer=timer, path=filepath):
                gdf = gpd.read_parquet(filepath)
        else:
            with log_step('Read vector file', timer=timer, path=filepath):
                gdf = gpd.read_file(
                    filepath, layer=recipe['layer'] if 'layer' in recipe else None
                )
    elif filepath.suffix in PANDAS_EXTENSIONS:
        with log_step('Read data file', timer=timer, path=filepath):
            gdf = pd.read_file(filepath)
    else:
        raise ValueError(f'Filepath suffix {filepath.suffix} not yet interpreted.')

    # Replacing known NA value strings with `None`.
    if 'null_value_strings' in recipe:
        for col, na_value in product(gdf.columns, recipe['null_value_strings']):
            i_has_na_value = gdf[col].eq(na_value)
            if i_has_na_value.sum():
                gdf.loc[i_has_na_value, col] = None

    # Indexing and filtering
    if 'columns' in recipe:
        # Rename columns
        gdf = gdf.rename(columns={v: k for k, v in recipe['columns'].items()})

    if 'query' in recipe:
        gdf = gdf.query(recipe['query'])

    if 'set_index' in recipe:
        # Set column as index
        if recipe['set_index'] not in gdf:
            raise ValueError(
                'Column not found to use as index: ' + str(recipe['set_index'])
            )
        gdf = gdf.set_index(recipe['set_index'])
    elif 'create_index' in recipe:
        if recipe['create_index']['method'] == 'prefix':
            gdf.index = pd.Index(
                recipe['create_index']['prefix']
                + gdf[recipe['create_index']['column']],
                name=recipe['create_index']['name'],
            )
    elif 'index_function' in recipe:
        # Create index with custom function

        def load_function(path):
            module, name = path.rsplit('.', 1)
            return getattr(importlib.import_module(module), name)

        with log_step('Generate indices', timer=timer):
            index_function = load_function(recipe['index_function'])

        gdf = index_function(gdf)

    # Sort by index
    gdf = gdf.sort_index()

    if 'drop' in recipe:
        gdf = gdf.drop(recipe['drop'])

    # Double-check that the index has no duplicates
    if gdf.index.duplicated().any():
        raise ValueError(
            'Duplicated indices are not allowed in imported data.\n'
            'Change index function or column:\n'
            + str(gdf[gdf.index.duplicated(keep=False)].sort_index().head().iloc[:, :5])
        )

    # Reorder columns
    if 'columns' in recipe:
        cols_order = [c for c in recipe['columns'] if c in gdf]
        if 'keep_unnamed_columns' in recipe and recipe['keep_unnamed_columns']:
            cols_order += [
                c for c in gdf if c not in (list(recipe['columns']) + ['geometry'])
            ]
        if 'geometry' in gdf:
            cols_order += ['geometry']
        gdf = gdf[cols_order]

    return gdf
