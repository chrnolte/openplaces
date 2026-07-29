"""
Processes a single table (layer) from an already-resolved source file
into an openplaces entity output file.
"""

import importlib
import shutil
import warnings
from itertools import product
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyogrio.errors import DataSourceError

from openplaces.config import cfg
from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
    ZIP_EXTENSIONS,
)
from openplaces.core.schema import AdminId
from openplaces.geo import get_crs
from openplaces.geo.ids import get_geo_ids
from openplaces.geo.overlay import overlay_admin_ids
from openplaces.geo.polygon import (
    clean_polygons,
    fix_polygons,
    reproject,
    resolve_overlapping_polygons,
)
from openplaces.io import (
    coerce_mixed_object_columns,
    find_latest_file_or_gdb,
    read_gdb_with_domains,
    save_parquet,
    unzip,
)
from openplaces.io.readers import get_admin
from openplaces.io.transform import (
    add_unique_suffix,
    apply_transformation,
    apply_transformations,
    get_crosswalk,
)
from openplaces.path import recipe_path
from openplaces.recipe import get_output_path, get_recipe


class TableIngester:
    """
    Processes a single table from an already-resolved source file into an
    openplaces entity output file.

    Handles reading, preprocessing, and saving one layer or table (a named
    layer in a GDB or GeoPackage, or the entirety of a single-table file).
    Does not own download or chunking logic — those belong to the parent
    Ingester.

    Parameters
    ----------
    table_recipe : dict
        Recipe for this specific table. For the primary entity this is the
        full recipe dict; for additional entities it is the result of
        build_table_recipe().
    download_partition : dict
        Shared mutable state from the parent Ingester (data_path,
        admin_id_to_download, admin_id_crosswalk, table_fids,
        admin_geometries, etc.). Mutated in place to cache results.
    processing_chunk : dict
        Mutable dict holding 'admin_id_to_process'. Shared with Ingester;
        mutated in place by the tile-partition loop so that TableIngester
        always sees the current admin unit.
    recipe_heap_dir : Path
        Heap directory for the primary entity. Used as the unzip target in
        the compressed-file fallback path of _read_recipe_data.
    timer : Timer
    verbose : bool
    admin_ids_to_save : list
        Admin ID strings (or [None]) at the save level. Required only when
        the recipe uses save_to: admin_level to split output by admin unit.
    """

    def __init__(
        self,
        table_recipe: dict,
        download_partition: dict,
        processing_chunk: dict,
        recipe_heap_dir: Path,
        timer,
        verbose: bool = False,
        admin_ids_to_save: list = None,
    ):
        self.recipe = table_recipe
        self.download_partition = download_partition
        self.processing_chunk = processing_chunk
        self.recipe_heap_dir = recipe_heap_dir
        self.timer = timer
        self.verbose = verbose
        self.admin_ids_to_save = admin_ids_to_save or []

    @property
    def table_name(self) -> str:
        """Stable key for this table in per-table caches (e.g. table_fids)."""
        return self.recipe.get('layer') or str(
            self.recipe.get('entity') or self.recipe.get('dataset')
        )

    @staticmethod
    def _mark_suffix(*parts):
        present = [str(p) for p in parts if p is not None]
        return (': ' + ' | '.join(present)) if present else ''

    # Public entry point

    def process(self, process_in_chunks: bool = False, bbox=None):
        """Read, preprocess, and save data for this table.

        Parameters
        ----------
        process_in_chunks : bool
            If True, filter data to the current
            processing_chunk['admin_id_to_process'].
        bbox : tuple, optional
            Bounding-box filter (minx, miny, maxx, maxy) used instead of
            FID-based filtering when process_in_chunks is True.
        """
        admin_id_to_process = self.processing_chunk['admin_id_to_process']
        partition_id = self.download_partition.get('partition_id_to_download')
        suffix = self._mark_suffix(admin_id_to_process, partition_id)

        read_kwargs = {}
        if process_in_chunks:
            if bbox is not None:
                read_kwargs['bbox'] = bbox
            else:
                self._ensure_table_fid_filter()
                fids_series = self.download_partition['table_fids'][self.table_name]
                read_kwargs['fids'] = list(
                    fids_series[fids_series.eq(admin_id_to_process)].index
                )

        gdf = self._read_recipe_data(**read_kwargs)

        if isinstance(gdf, gpd.GeoDataFrame) and gdf.crs != cfg.crs:
            gdf = reproject(gdf, cfg.crs)
            self.timer.mark(f'Reproject to {cfg.crs}{suffix}')

        gdf = self._preprocess_recipe_data(gdf)
        self.timer.mark(f'Preprocess{suffix}')

        self._save_recipe_data(gdf)
        self.timer.mark(f'Save{suffix}')

    # FID filter helpers (per-table, cached in download_partition)

    def _ensure_table_fid_filter(self):
        """Build FID filter for this table if not already cached."""
        if 'admin_id_crosswalk' not in self.download_partition:
            self._prepare_admin_id_crosswalk()
        if 'table_fids' not in self.download_partition:
            self.download_partition['table_fids'] = {}
        if self.table_name not in self.download_partition['table_fids']:
            self._prepare_table_fid_filter()

    def _prepare_admin_id_crosswalk(self):
        """Build crosswalk from source admin column values to AdminIds.

        Uses process_by.admin_id_crosswalk from the recipe. The result is
        stored in download_partition and shared across all tables in the
        same download partition.
        """
        process_by = self.recipe.get('process_by', {})
        if 'admin_id_crosswalk' not in process_by:
            raise ValueError('No crosswalk recipe found in process_by.')
        admin_id_crosswalk_dict = dict(process_by['admin_id_crosswalk'])
        admin_id_crosswalk_dict['admin_id'] = self.download_partition[
            'admin_id_to_download'
        ]
        self.download_partition['admin_id_crosswalk'] = get_crosswalk(
            admin_id_crosswalk_dict, flip=True
        )

    def _prepare_table_fid_filter(self):
        """Build FID → admin_id mapping for this table's layer.

        FIDs are layer-specific in multi-layer formats (e.g. GDB), so each
        table builds and caches its own mapping under
        download_partition['table_fids'][table_name].
        """
        admin_level_to_process = self.recipe['process_by']['admin_level']
        admin_id_column_source = self.recipe['process_by']['admin_id_column']

        admin_id_filter = self._read_recipe_data(
            columns=[admin_id_column_source],
            read_geometry=False,
            fid_as_index=True,
        )

        if 'admin_id_transformation' in self.recipe['process_by']:
            transforms = self.recipe['process_by']['admin_id_transformation']
            if isinstance(transforms, list):
                transforms[0].setdefault('input', admin_id_column_source)
                for t in transforms:
                    admin_id_filter = apply_transformation(admin_id_filter, t)
                join_column = transforms[-1]['output']
            else:
                transforms['input'] = admin_id_column_source
                admin_id_filter = apply_transformation(admin_id_filter, transforms)
                join_column = transforms['output']
        else:
            join_column = admin_id_column_source

        admin_id_filter = admin_id_filter.join(
            self.download_partition['admin_id_crosswalk'],
            on=join_column,
        )

        self.download_partition['table_fids'][self.table_name] = admin_id_filter[
            f'admin{admin_level_to_process}_id'
        ]

    # Admin geometry loading (cached in download_partition)

    def _load_admin_geometries(self):
        """Load admin geometries for spatial overlay or spatial mask.

        Reads admin unit boundaries and stores them in download_partition
        for reuse across admin chunks and tables.
        """
        if (
            'process_by' in self.recipe
            and 'use_spatial_index' in self.recipe['process_by']
            and self.recipe['process_by']['use_spatial_index']
        ):
            admin_specs = self.recipe['process_by']
        elif 'overlay_admin_ids' in self.recipe:
            admin_specs = self.recipe['overlay_admin_ids']
        else:
            raise ValueError(
                'Cannot load admin geometries: recipe has neither '
                'process_by.use_spatial_index nor overlay_admin_ids.'
            )

        admin_id = self.download_partition['admin_id_to_download'] or self.recipe.get(
            'admin_id'
        )
        admin_geometries = get_admin(
            admin_id,
            admin_specs['admin_level'],
            recipe=admin_specs.get('admin_recipe_id'),
            geom=True,
        )['geometry']

        if 'admin_ids_in_tile' in self.download_partition:
            admin_ids_in_tile = self.download_partition['admin_ids_in_tile']
            admin_geometries = admin_geometries.loc[
                [aid for aid in admin_ids_in_tile if aid in admin_geometries.index]
            ]

        data_crs = get_crs(
            self.download_partition['data_path'], layer=self.recipe.get('layer')
        )
        if data_crs != admin_geometries.crs:
            admin_geometries = admin_geometries.to_crs(data_crs)
        self.download_partition['admin_geometries'] = admin_geometries

    # Read

    def _read_recipe_data(self, columns=None, **kwargs):
        """Read data from the resolved data path for this table's layer.

        Parameters
        ----------
        columns : list, optional
            Column names to read. Enables lightweight reads for FID prep.
        kwargs : dict
            Passed to the underlying reader (e.g. fids, bbox,
            read_geometry, fid_as_index).
        """
        warnings.filterwarnings('ignore', 'received a polygon with more than 100 parts')

        if 'encoding' in self.recipe:
            kwargs['encoding'] = self.recipe['encoding']

        if columns:
            timer_suffix = (
                ', '
                + str(len(columns))
                + ' column'
                + ('(s)' if len(columns) > 1 else '')
            )
        else:
            timer_suffix = ''

        layer = self.recipe.get('layer')
        data_path = self.download_partition['data_path']

        if data_path.suffix == '.parquet':
            if 'fids' in kwargs:
                raise ValueError('`fid`-based selection might not work with `parquet`.')
            try:
                gdf = gpd.read_parquet(data_path, columns=columns, **kwargs)
            except ValueError as e:
                # A plain (non-geo) parquet partition -- e.g. one of several
                # per-admin-unit tables meant to be joined later, only one of
                # which carries geometry (see io.aggregate.join_partitions_by_index).
                if 'Missing geo metadata' not in str(e):
                    raise
                gdf = pd.read_parquet(data_path, columns=columns)
            self.timer.mark('Read parquet file' + timer_suffix, path=data_path)
        elif data_path.suffix == '.gdb':
            try:
                gdf = read_gdb_with_domains(
                    data_path, columns=columns, layer=layer, **kwargs
                )
            except DataSourceError as e:
                if 'Permission denied' in str(e):
                    print(
                        f'\n\033[33mPermission denied reading:\033[0m\n'
                        f'  {data_path}\n\n'
                        'This usually means the .gdb folder is locked by a file sync '
                        'app (e.g. Dropbox) or was only partially deleted.\n'
                    )
                    answer = input('Try to delete it now? [y/n] ').strip().lower()
                    if answer == 'y':
                        try:
                            shutil.rmtree(data_path)
                            raise RuntimeError('Deleted. Please re-run the ingestion.')
                        except Exception as del_err:
                            raise RuntimeError(
                                f'Could not delete: {del_err}\n'
                                'Remove it manually and re-run.'
                            )
                    raise RuntimeError('Remove the folder manually and re-run.')
                raise
            self.timer.mark('Read GDB file' + timer_suffix, path=data_path)
        elif data_path.suffix in GEOPANDAS_EXTENSIONS:
            try:
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
            except DataSourceError:
                raise OSError(
                    f'Failed to read data file:\n\n{data_path}\n\n'
                    'Possibly an incompletely unzipped file? '
                    'If so, delete manually, and re-run unzipping.'
                )
            self.timer.mark(
                f'Read vector file ({data_path.suffix})' + timer_suffix,
                path=data_path,
            )
        elif data_path.suffix in PANDAS_EXTENSIONS:
            # `csv_dtype: str` reads every column as text — the robust choice
            # for messy flat dumps where a column mixes ints and strings.
            # A dict maps specific columns to dtypes.
            csv_dtype = self.recipe.get('csv_dtype')
            if csv_dtype == 'str':
                dtype = str
            elif isinstance(csv_dtype, dict):
                dtype = csv_dtype
            else:
                dtype = None

            if self.recipe.get('fixed_width'):
                gdf = self._read_fixed_width(data_path, dtype)
            elif data_path.suffix in {'.xlsx', '.xls'}:
                header = self.recipe.get('header', 'infer')
                gdf = pd.read_excel(
                    data_path,
                    sheet_name=self.recipe.get('sheet_name', 0),
                    header=None if header in (None, 'none') else header,
                    names=self.recipe.get('names'),
                    dtype=dtype,
                )
            else:
                # low_memory=False avoids per-chunk dtype inference (the source
                # of mixed-type object columns that then fail Parquet writes).
                read_kwargs = {
                    'delimiter': self.recipe.get('delimiter', ','),
                    'low_memory': False,
                }
                if dtype is not None:
                    read_kwargs['dtype'] = dtype
                gdf = pd.read_csv(data_path, usecols=columns, **read_kwargs)
            self.timer.mark(
                'Read data table' + timer_suffix,
                path=data_path,
            )
        elif data_path.suffix in ZIP_EXTENSIONS:
            try:
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
                self.timer.mark('Read compressed file' + timer_suffix, path=data_path)
            except (RuntimeWarning, Exception):
                unzip(data_path, self.recipe_heap_dir)
                data_path = find_latest_file_or_gdb(self.recipe_heap_dir)
                self.download_partition['data_path'] = data_path
                if data_path is None:
                    raise OSError(
                        f'`geopandas` could not read compressed file:\n\n{data_path}.'
                        '\n\n'
                        'Could not find a dataset after unzipping to:\n\n'
                        f'{self.recipe_heap_dir}'
                    )
                gdf = gpd.read_file(data_path, layer=layer, columns=columns, **kwargs)
                self.timer.mark('Read unzipped file' + timer_suffix, path=data_path)
        else:
            raise ValueError(f'Filepath suffix not yet interpreted: {data_path.suffix}')

        warnings.filterwarnings(
            'default', 'received a polygon with more than 100 parts'
        )
        return gdf

    def _read_fixed_width(self, data_path, dtype):
        """Read a fixed-width flat file using the recipe's ``fixed_width`` layout.

        ``fixed_width`` is an ordered list of ``[field_name, width]`` covering the
        whole record. Fields named ``filler`` are padding and are dropped after
        reading. Field values are stripped of their fixed-width padding.

        Parameters
        ----------
        data_path : Path
            Path to the flat file.
        dtype : type, dict, or None
            Passed to :func:`pandas.read_fwf` (defaults to ``str`` so zero-padded
            identifiers keep their leading zeros).
        """
        spec = self.recipe['fixed_width']
        names, widths = [], []
        for i, field in enumerate(spec):
            name, width = field[0], int(field[1])
            names.append(f'filler_{i}' if name == 'filler' else name)
            widths.append(width)

        df = pd.read_fwf(data_path, widths=widths, names=names, dtype=dtype or str)
        df = df.drop(columns=[c for c in df.columns if c.startswith('filler_')])
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.strip()
        return df

    # Preprocess

    def _preprocess_recipe_data(self, df):
        """Rename columns, filter rows, apply transformations, set index.

        Parameters
        ----------
        df : DataFrame or GeoDataFrame
            Raw data read from the source file.
        """
        # Rename columns
        if 'columns' in self.recipe:
            df = df.rename(columns={v: k for k, v in self.recipe['columns'].items()})

        # Replace known NA value strings with None
        if 'null_value_strings' in self.recipe:
            columns_to_convert = [
                v for v in df.columns if not v.startswith('admin1_id')
            ]
            for col, na_value in product(
                columns_to_convert, self.recipe['null_value_strings']
            ):
                i_has_na_value = df[col].eq(na_value)
                if i_has_na_value.sum():
                    df.loc[i_has_na_value, col] = None

        # Filter rows
        if 'query' in self.recipe:
            df = df.query(self.recipe['query'])

        # Drop duplicate rows (some source dumps repeat exact rows). `true`
        # drops full-row duplicates; a list of column names dedupes on a subset.
        if self.recipe.get('drop_duplicates'):
            subset = self.recipe['drop_duplicates']
            df = df.drop_duplicates(subset=None if subset is True else subset)

        # Apply variable transformations
        # (Before categorical casting and crosswalks, so that derived columns
        # can be cast to categorical and used in overlap resolution)
        if 'transformations' in self.recipe:
            cols_before = list(df)
            df = apply_transformations(df, self.recipe)
            cols_added = [v for v in df if v not in cols_before]
        else:
            cols_added = []

        # Cast columns to categorical
        # Each item is either a plain column name (unordered) or a single-key
        # dict {column: [cat1, cat2, ...]} for an inline ordered categorical.
        if 'columns_to_categorical' in self.recipe:
            for item in self.recipe['columns_to_categorical']:
                if isinstance(item, dict):
                    column_to_cast, inline_categories = next(iter(item.items()))
                else:
                    column_to_cast, inline_categories = item, None

                if column_to_cast not in df:
                    continue

                if inline_categories is not None:
                    labels = self._get_labels(column_to_cast)
                    if labels is not None and all(
                        c in labels for c in inline_categories
                    ):
                        # inline_categories are raw codes → remap and preserve order
                        values = df[column_to_cast].replace(labels)
                        categories = [labels[c] for c in inline_categories]
                    else:
                        # inline_categories are already the final label strings
                        values = df[column_to_cast]
                        categories = inline_categories
                    ordered = True
                else:
                    labels = self._get_labels(column_to_cast)
                    if labels is not None:
                        values = df[column_to_cast].replace(labels)
                        categories, ordered = labels.values(), True
                    else:
                        values, categories, ordered = df[column_to_cast], None, False

                df[column_to_cast] = pd.Series(
                    pd.Categorical(values, categories, ordered),
                    index=values.index,
                )

        admin_id_to_process = self.processing_chunk.get('admin_id_to_process')
        self.timer.mark('Transform' + self._mark_suffix(admin_id_to_process))

        # Clean geometries and resolve overlapping polygons (parcels, buildings).
        # Cleaning must precede the overlap test: invalid geometries cause
        # TopologyExceptions in shapely intersection.
        # Runs after transformations and categorical casting so that
        # 'prefer_higher' can reference a transformed or categorical column.
        if isinstance(df, gpd.GeoDataFrame):
            _suffix = self._mark_suffix(admin_id_to_process)

            if self.recipe.get('force_2d', False):
                import shapely

                df['geometry'] = shapely.force_2d(df['geometry'])

            if self.recipe.get('add_geometry_derivatives', False):
                from openplaces.geo.polygon import add_geometry_derivatives

                df = add_geometry_derivatives(df, self.timer)

            if self.recipe.get('add_tile_utm_derivatives', False):
                from openplaces.geo.tiles import add_tile_utm_derivatives

                cfg_utm = self.recipe['add_tile_utm_derivatives']
                df = add_tile_utm_derivatives(
                    df,
                    tile_id_col=cfg_utm.get('tile_id_col', 'tile_id'),
                    tile_type=cfg_utm.get('tile_type'),
                )

            if (~df.geometry.is_valid).any():
                df = fix_polygons(df)
            self.timer.mark(f'Clean geometries{_suffix}')

            if self.recipe.get('resolve_overlaps', False):
                df = clean_polygons(df)
                keep = self.recipe.get('keep_overlapping_polygons', None)
                recipe_col_names = list(self.recipe.get('columns', {}) or {})
                skip = {c for c in df.columns if '_id' in c} | {'geometry'}
                compare_cols = [c for c in df.columns if c not in skip]
                snippet_cols = (
                    [c for c in recipe_col_names if c in compare_cols]
                    + [c for c in compare_cols if c not in set(recipe_col_names)]
                )[:5]
                df = resolve_overlapping_polygons(
                    df,
                    keep=keep,
                    compare_cols=compare_cols,
                    snippet_cols=snippet_cols,
                )
                self.timer.mark(f'Resolve overlaps{_suffix}')

        # Attribute entities to administrative unit IDs via crosswalk
        # (Before admin ID index creation, which needs parent Admin ID)
        use_spatial_mask = (
            'process_by' in self.recipe
            and 'use_spatial_mask' in self.recipe['process_by']
            and self.recipe['process_by']['use_spatial_mask']
        )
        if 'admin_id_crosswalk' in self.recipe:
            admin_id_crosswalk_dict = self.recipe['admin_id_crosswalk']
            admin_id_crosswalk_dict['admin_id'] = self.processing_chunk[
                'admin_id_to_process'
            ]
            admin_id_crosswalk = get_crosswalk(admin_id_crosswalk_dict, flip=True)

            missing_crosswalk_ids = set(df[admin_id_crosswalk.index.name]) - set(
                admin_id_crosswalk.index
            )
            if missing_crosswalk_ids:
                mask_unmatched = df[admin_id_crosswalk.index.name].isin(
                    missing_crosswalk_ids
                )
                if self.verbose:
                    warnings.warn(
                        f'\n\nImperfect crosswalk: {mask_unmatched.sum():,d} '
                        'admin IDs were not linked and will be dropped:\n\n'
                        + str(df[mask_unmatched][[v for v in df if 'name' in v]])
                        + '\n'
                    )
            df = df.join(
                admin_id_crosswalk, on=admin_id_crosswalk.index.name, how='inner'
            )
            cols_added += (
                [admin_id_crosswalk.name]
                if isinstance(admin_id_crosswalk, pd.Series)
                else list(admin_id_crosswalk)
            )
            self.timer.mark('Attribute admin IDs: crosswalk join')

        elif use_spatial_mask or ('overlay_admin_ids' in self.recipe):
            if self.verbose:
                print(
                    'Overlaying polygons with administrative boundaries. '
                    'This can take a while.'
                )
            if 'admin_geometries' not in self.download_partition:
                self._load_admin_geometries()
            if use_spatial_mask:
                admin_specs = self.recipe['process_by']
                admin_geometries = (
                    self.download_partition['admin_geometries']
                    .loc[[self.processing_chunk['admin_id_to_process']]]
                    .copy()
                )
            else:
                admin_specs = self.recipe['overlay_admin_ids']
                admin_geometries = self.download_partition['admin_geometries']

            kwargs_overlay = {
                k: v for k, v in admin_specs.items() if k != 'admin_recipe_id'
            }
            cols_before = set(df.columns)
            df = overlay_admin_ids(
                df,
                admin_geometries=admin_geometries,
                timer=self.timer,
                **kwargs_overlay,
            )
            cols_added += [v for v in df.columns if v not in cols_before]

        # Set index
        _entity = self.recipe.get('entity')
        _has_custom_index = (
            'set_index' in self.recipe
            or 'create_index' in self.recipe
            or 'index_function' in self.recipe
        )
        if (
            _entity is not None
            and str(_entity.entity_type) == 'parcel'
            and isinstance(df, gpd.GeoDataFrame)
            and not _has_custom_index
            and df.index.name != 'geo_id'
        ):
            df['geo_id'] = get_geo_ids(df, handle_duplicates=False)
            df.index = pd.Index(add_unique_suffix(df['geo_id']), name='parcel_id')
        elif 'set_index' in self.recipe:
            if self.recipe['set_index'] not in df:
                raise ValueError(
                    'Column not found to use as index: ' + str(self.recipe['set_index'])
                )
            if df[self.recipe['set_index']].duplicated().any():
                raise ValueError(
                    f"Duplicates found in '{self.recipe['set_index']}'. "
                    'Choose other index.\n\n'
                    + str(
                        df[df[self.recipe['set_index']].duplicated(keep=False)][
                            self.recipe['set_index']
                        ]
                        .sort_values()
                        .head(5)
                    )
                )
            df = df.set_index(self.recipe['set_index'])
        elif 'create_index' in self.recipe:
            if 'function' in self.recipe['create_index']:
                if not self.recipe['create_index']['function'].startswith(
                    'openplaces.'
                ):
                    raise ValueError(
                        'Function in `create_index` must start with `openplaces.`\n'
                        'Changing this would create a security risk (run any function).'
                    )
                index_function = self._load_function(
                    self.recipe['create_index']['function']
                )
                index_function_kwargs = self.recipe['create_index'].get('args', {})
                df = index_function(df, **index_function_kwargs)
            elif 'method' in self.recipe['create_index']:
                if self.recipe['create_index']['method'] == 'prefix':
                    df.index = pd.Index(
                        self.recipe['create_index']['prefix']
                        + df[self.recipe['create_index']['column']],
                        name=self.recipe['create_index']['name'],
                    )
        elif 'index_function' in self.recipe:
            if not self.recipe['index_function'].startswith('openplaces.'):
                raise ValueError(
                    'Function in `index_function` must start with `openplaces.`\n'
                    'Changing this would create a security risk (run any function).'
                )
            index_function = self._load_function(self.recipe['index_function'])
            df = index_function(df)
            self.timer.mark('Generate indices')

        # Drop observations by index
        if 'drop' in self.recipe:
            df = df.drop(self.recipe['drop'])

        # Double-check that the index has no duplicates
        if df.index.duplicated().any():
            raise ValueError(
                'Duplicated indices are not allowed in imported data.\n'
                'Change `index_function`, `create_index` or `set_index` column:\n'
                + str(df[df.index.duplicated(keep=False)].sort_index().head())
            )

        # Reorder columns
        if 'columns' in self.recipe:
            named_cols = [c for c in list(self.recipe['columns']) if c in df]
            cols_order = (
                named_cols
                + [
                    c
                    for c in df
                    if c.startswith('admin')
                    and c.endswith('_id_source')
                    and c not in named_cols
                ]
                + cols_added
            )
            if self.recipe.get('keep_unnamed_columns'):
                cols_order += [
                    c for c in df if c not in cols_order + ['geo_id', 'geometry']
                ]
            for geo_col in ['geo_id', 'geometry']:
                if geo_col in df:
                    cols_order += [geo_col]
            df = df[cols_order]

        # Standardized cross-comparable parcel matching key (after reorder so it
        # is always retained).
        df = self._add_parcel_id_local(df)

        return df

    def _load_parcel_id_overrides(self, kind: str) -> dict | None:
        """Load the recipe-tree id-conversion override table, if present.

        Returns an ``{admin_id: {pattern, conv}}`` dict in the shape
        :func:`~openplaces.geo.ids.compute_parcel_id_local`'s ``instruction``
        parameter expects, built from rows of
        ``{country}_{entity_type}_id-overrides.csv``
        (``recipes/{country}/_all/{entity_type}/_all/``) matching *kind* and
        this recipe's ``source_id`` (a blank ``source_id`` row matches any
        source at that ``admin_id``; an exact-source row at the same
        ``admin_id`` takes precedence). Admin-hierarchy walking from a
        specific admin id to a broader one is handled by
        :func:`~openplaces.geo.ids._resolve_instruction`, which already
        knows how to fall back within an ``instruction`` dict — this only
        builds the dict. Returns ``None`` when no override table exists for
        this recipe's country/entity_type (the common case today).
        """
        entity = self.recipe.get('entity')
        admin_id = self.recipe.get('admin_id')
        if entity is None or not admin_id or not admin_id.levels:
            return None
        country = AdminId(admin_id.levels[0])
        try:
            table = get_recipe(
                country,
                str(entity.entity_type),
                filename='id-overrides',
                dtype=str,
                keep_default_na=False,
            )
        except OSError:
            return None

        table = table[table['kind'] == kind]
        source_id = str(entity.source) if entity.source else ''
        overrides: dict[str, dict] = {}
        for _, row in table[table['source_id'] == ''].iterrows():
            if row['admin_id']:
                overrides[row['admin_id']] = {
                    'pattern': row['pattern'],
                    'conv': row['conv'],
                }
        for _, row in table[table['source_id'] == source_id].iterrows():
            if row['admin_id']:
                overrides[row['admin_id']] = {
                    'pattern': row['pattern'],
                    'conv': row['conv'],
                }
        return overrides or None

    def _add_parcel_id_local(self, df):
        """Add `parcel_id_local` from `parcel_id_assessor` per the recipe directive.

        Recipe directive::

            parcel_id_local:
              source: parcel_id_assessor   # raw column to standardize
              kind: parcel                 # parcel | tax (selects default conv)
              admin_id_column: admin4_id   # optional: per-row admin unit (MA towns)
              instruction: {<admin_id>: {pattern: ..., conv: ...}}   # optional override

        The conversion is admin-unit-specific: a recipe-inline `instruction`
        wins, then the recipe-tree override table
        (:meth:`_load_parcel_id_overrides`), then the bundled default table
        (see :func:`openplaces.geo.ids.compute_parcel_id_local`), and is
        hardened so it never adds duplicates beyond those already in
        `parcel_id_assessor`.
        """
        spec = self.recipe.get('parcel_id_local')
        if not spec:
            return df
        from openplaces.geo.ids import compute_parcel_id_local

        source = spec.get('source', 'parcel_id_assessor')
        if source not in df.columns:
            warnings.warn(
                f"parcel_id_local: source column '{source}' not found; skipping.",
                stacklevel=2,
            )
            return df
        kind = spec.get('kind', 'parcel')
        instruction = {
            **(self._load_parcel_id_overrides(kind) or {}),
            **(spec.get('instruction') or {}),
        } or None
        admin_col = spec.get('admin_id_column')

        if admin_col and admin_col in df.columns:
            # Per-row admin-unit-specific conversion (e.g. Massachusetts towns).
            result = pd.Series(pd.NA, index=df.index, dtype='string')
            for admin_id, group in df.groupby(admin_col):
                result.loc[group.index] = compute_parcel_id_local(
                    group[source],
                    admin_unit_id=admin_id,
                    instruction=instruction,
                    kind=kind,
                )
            df['parcel_id_local'] = result
        else:
            admin_id = self.processing_chunk.get('admin_id_to_process')
            df['parcel_id_local'] = compute_parcel_id_local(
                df[source],
                admin_unit_id=admin_id,
                instruction=instruction,
                kind=kind,
            )
        return df

    # Save

    def _save_recipe_data(self, gdf):
        """Save processed data to the entity's output path.

        Parameters
        ----------
        gdf : DataFrame or GeoDataFrame
            Preprocessed data ready to be saved.
        """
        gdf = coerce_mixed_object_columns(gdf)

        save_to = self.recipe.get('save_to') or {}
        admin_level = save_to.get('admin_level') or (
            self.recipe.get('cache_by') or {}
        ).get('admin_level')
        split_dataset_by_admin = admin_level is not None

        if split_dataset_by_admin:
            admin_id_col = f'admin{admin_level}_id'
            if admin_id_col not in gdf:
                raise ValueError(
                    f"Recipe says 'save_to: admin_level: {admin_level}', but column "
                    f"'{admin_id_col}' does not exist in DataFrame:\n\n"
                    + str(gdf.sample(1).T)
                )
            admin_ids_in_data = sorted(set(gdf[admin_id_col].dropna()))
            admin_ids_to_save_expected = [
                admin_id
                for admin_id in self.admin_ids_to_save
                if admin_id.startswith(self.processing_chunk['admin_id_to_process'])
            ]
            admin_ids_to_save_in_data = [
                admin_id
                for admin_id in admin_ids_to_save_expected
                if admin_id in admin_ids_in_data
            ]

            missing_admin_ids = set(admin_ids_to_save_expected) - set(
                admin_ids_to_save_in_data
            )
            if missing_admin_ids:
                warnings.warn(
                    f'\n\n{len(missing_admin_ids)} AdminIds to save not found in data:'
                    '\n' + ', '.join(sorted(missing_admin_ids)[:15]) + '\n'
                )
        else:
            admin_ids_to_save_in_data = [self.processing_chunk['admin_id_to_process']]

        if split_dataset_by_admin and admin_ids_to_save_in_data:
            print('Saving ' + ', '.join(admin_ids_to_save_in_data))

        for admin_id_to_save in admin_ids_to_save_in_data:
            if split_dataset_by_admin:
                redundant_admin_id_columns = [
                    v for v in gdf if v.startswith(f'admin{admin_level}_id')
                ]
                gdf_to_save = (
                    gdf[gdf[admin_id_col].eq(admin_id_to_save)]
                    .copy()
                    .drop(columns=redundant_admin_id_columns)
                )
            else:
                gdf_to_save = gdf.copy()

            output_path = get_output_path(
                self.recipe,
                admin_id_to_save,
                self.download_partition.get('partition_id_to_download'),
            )

            if output_path.suffix == '.parquet':
                save_parquet(gdf_to_save, output_path)
            else:
                raise NotImplementedError(
                    f'Output file type not yet supported: {output_path.suffix}'
                )

    # Utilities

    def _get_labels(self, column):
        """Get code → label dict for a categorical column.

        Looks for a CSV file named '<column-with-dashes>-labels.csv' next
        to the recipe file.
        """
        labels_recipe_path = recipe_path(
            self.recipe['admin_id'],
            self.recipe.get('entity') or self.recipe.get('dataset'),
            filename=column.replace('_', '-') + '-labels.csv',
        )
        if labels_recipe_path.exists():
            labels = pd.read_csv(labels_recipe_path)
            return labels.set_index(labels.columns[0])[labels.columns[1]].to_dict()
        return None

    def _load_function(self, path):
        """Import and return a function by dotted module path."""
        module, name = path.rsplit('.', 1)
        return getattr(importlib.import_module(module), name)
