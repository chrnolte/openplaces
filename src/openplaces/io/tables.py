"""
Read and write tables: parquet with the footer metadata this
project relies on, plus CSV, GeoPackage and KMZ writers and the
geometry-sidecar convention behind save_parquet and read_parquet.
"""

import json
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd
import pandas as pd
import pyarrow


def _remove_if_exists(filepath: Path) -> None:
    """Remove file if it exists (needed for some formats like gpkg)."""
    if filepath.exists():
        filepath.unlink()


def _categoricals_to_string(
    df: pd.DataFrame | gpd.GeoDataFrame,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Cast categorical columns to a string (object) dtype for writing.

    Pandas categoricals serialize to Arrow ``dictionary<string,int>``, which
    GDAL/QGIS expose as integer codes plus a per-file code->label domain whose
    mapping shifts whenever the category set changes. Writing them as a plain
    string logical type avoids that: GDAL/QGIS read stable string values, and
    Parquet still dictionary-encodes the column physically (RLE_DICTIONARY), so
    the file stays just as compact. ``astype(object)`` (not ``astype(str)``)
    preserves missing values as nulls rather than the literal string ``'nan'``.
    """
    cat_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.CategoricalDtype)]
    if not cat_cols:
        return df
    df = df.copy()
    for col in cat_cols:
        df[col] = df[col].astype(object)
    return df


def parquet_columns(parquet_path: str | Path) -> list[str]:
    """Return a parquet file's column names, reading the schema only.

    Cheap enough to call per file in a loop: no row group is touched. Use it
    to decide what to request from `read_parquet`, which errors on a column
    the file does not have.

    Parameters
    ----------
    parquet_path : str or Path
        Filepath of the Parquet file.
    """
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(parquet_path).schema_arrow.names)


def to_parquet(
    df: pd.DataFrame | gpd.GeoDataFrame,
    filepath: str | Path,
    *,
    file_metadata: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Save dataframe to Parquet format.

    Categorical columns are written as a string logical type (Parquet still
    dictionary-encodes them physically, so files stay compact) so GDAL/QGIS read
    stable string values rather than a per-file integer code->label mapping.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Data to save
    filepath : str or Path
        Output parquet path (should end in .parquet)
    file_metadata : dict of str to str, optional
        Key-value pairs written into the Parquet footer (file-level) metadata,
        merged with the metadata pandas/pyarrow already attaches. Read back via
        pyarrow.parquet.read_metadata() without scanning rows. Only supported
        for plain (non-geo) DataFrames; ignored for GeoDataFrames.
    **kwargs
        Additional arguments passed to to_parquet(). When file_metadata is
        given the write goes through pyarrow directly, which accepts none of
        pandas' keywords: only index is honored there, and anything else
        raises rather than being dropped.

    Raises
    ------
    TypeError
        If file_metadata is combined with a keyword other than index.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    df = _categoricals_to_string(df)

    if not isinstance(df, gpd.GeoDataFrame) and set(kwargs) <= {'index'}:
        # Every attribute table records which recipe roots the build
        # read from (bundled, at which version or commit, plus any
        # installed recipe package): auto-discovery picks the most
        # specific recipe those roots hold, so a file is not
        # reproducible without them. Geometry sidecars are written by
        # geopandas, which takes no footer, and stay as they are.
        from openplaces.path import RECIPE_ROOTS_METADATA_KEY, recipe_roots_footer

        file_metadata = {
            RECIPE_ROOTS_METADATA_KEY: recipe_roots_footer(),
            **(file_metadata or {}),
        }
    if isinstance(df, gpd.GeoDataFrame):
        with warnings.catch_warnings():
            kwargs.setdefault('write_covering_bbox', True)
            warnings.filterwarnings('ignore', '.*initial implementation of Parquet.*')
            df.to_parquet(filepath, **kwargs)
    elif file_metadata:
        import pyarrow.parquet as pq

        # pq.write_table takes none of pandas' to_parquet keywords, so a
        # dropped index=False wrote an index level column here and not
        # on the branch below: the same call produced two schemas.
        index = kwargs.pop('index', None)
        if kwargs:
            raise TypeError(
                'to_parquet() cannot pass '
                + ', '.join(sorted(kwargs))
                + ' through with file_metadata; only `index` is honored '
                'on the footer-metadata path.'
            )
        table = pyarrow.Table.from_pandas(df, preserve_index=index)
        merged = dict(table.schema.metadata or {})
        merged.update(
            {
                (k.encode() if isinstance(k, str) else k): (
                    v.encode() if isinstance(v, str) else v
                )
                for k, v in file_metadata.items()
            }
        )
        pq.write_table(table.replace_schema_metadata(merged), filepath)
    else:
        df.to_parquet(filepath, **kwargs)


def to_csv(
    df: pd.DataFrame | gpd.GeoDataFrame,
    filepath: str | Path,
    index: bool = False,
    **kwargs,
) -> None:
    """Save dataframe as CSV file.

    Automatically drops the active geometry column if present, under
    whatever name the frame gives it.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        DataFrame to save
    filepath : str or Path
        Output CSV path
    index : bool, default False
        Whether to write row index
    **kwargs
        Additional arguments passed to df.to_csv()
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Drop geometry if present. The active geometry column is not
    # always named 'geometry', and it can also be unset entirely.
    if isinstance(df, gpd.GeoDataFrame):
        geometry_column = df.active_geometry_name
        if geometry_column is not None and geometry_column in df.columns:
            df = df.drop(columns=geometry_column)

    df.to_csv(filepath, index=index, **kwargs)


def to_gpkg(
    gdf: gpd.GeoDataFrame, filepath: str | Path, layer: str = None, **kwargs
) -> None:
    """Save geodataframe as GeoPackage.

    Removes existing file before writing (GeoPackage format requirement).

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe to save
    filepath : str or Path
        Output geopackage path
    layer : str, optional
        Layer name within geopackage
    **kwargs
        Additional arguments passed to to_file()
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(filepath)

    if not isinstance(gdf, gpd.GeoDataFrame):
        warnings.warn('Object is not a GeoDataFrame.', UserWarning)

    gdf.to_file(filepath, driver='GPKG', layer=layer, **kwargs)


def to_kmz(gdf: gpd.GeoDataFrame, filepath: str | Path) -> None:
    """Save geodataframe as KMZ file (zipped KML).

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe to save
    filepath : str or Path
        Output KMZ path
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)
    _remove_if_exists(filepath)

    if not isinstance(gdf, gpd.GeoDataFrame):
        warnings.warn('Object is not a GeoDataFrame.', UserWarning)

    # Save as KML temporarily
    kml_path = filepath.with_suffix('.kml')
    gdf.to_file(kml_path, driver='KML')

    # Convert to KMZ (zipped KML)
    with ZipFile(filepath, 'w', ZIP_DEFLATED) as kmz:
        kmz.write(kml_path, kml_path.name)

    # Clean up temporary KML
    kml_path.unlink()


def save(df: pd.DataFrame | gpd.GeoDataFrame, filepath: str | Path, **kwargs) -> None:
    """Save dataframe with format auto-detected from file extension.

    Supported formats:
    - .parquet: Parquet (or GeoParquet if GeoDataFrame with geometry)
    - .gpkg: GeoPackage (GeoDataFrame only)
    - .csv: CSV (geometry dropped if present)
    - .kmz: KMZ (GeoDataFrame only)

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Data to save
    filepath : str or Path
        Output path with extension
    **kwargs
        Additional arguments passed to format-specific save function

    Examples
    --------
    >>> save(gdf, 'data/core/parcels.parquet')
    >>> save(df, 'data/out/results.csv', index=True)
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == '.parquet':
        to_parquet(df, filepath, **kwargs)
    elif ext == '.gpkg':
        to_gpkg(df, filepath, **kwargs)
    elif ext == '.csv':
        to_csv(df, filepath, **kwargs)
    elif ext == '.kmz':
        to_kmz(df, filepath)
    else:
        raise ValueError(
            f'Unsupported file extension: {ext}. Supported: .parquet, .gpkg, .csv, .kmz'
        )


def coerce_mixed_object_columns(df):
    """Cast mixed-type object columns to a clean nullable string dtype.

    Flat-file sources and partition rollups can yield an object column that
    mixes Python str and numeric values — e.g. a mostly-text ``grantor`` column
    with a stray numeric cell, or a ``book`` column stored as int in one
    partition file and str in another. pyarrow cannot serialize such a column to
    Parquet. Only the genuinely mixed columns are cast to pandas ``'string'``
    (null-preserving); columns already typed as numeric, datetime, categorical,
    or geometry are left untouched. Used both at ingest save and at partition
    aggregation so either write path is robust to dtype heterogeneity.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Table about to be written to Parquet. Mutated in place and returned.
    """
    for col in df.columns:
        if col == 'geometry' or df[col].dtype != object:
            continue
        if pd.api.types.infer_dtype(df[col], skipna=True) in ('mixed', 'mixed-integer'):
            df[col] = df[col].astype('string')
    return df


def save_parquet(
    gdf, parquet_path, simplified_geometry=None, combined=False, file_metadata=None
):
    """Save parquet file (with geometries in joinable geoparquet file)

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        Data to save
    parquet_path : str
        Filepath of Parquet file
    simplified_geometry : GeoSeries or None
        When provided, a companion ``_geo_simplified.parquet`` sidecar is
        written alongside the standard ``_geo.parquet``, containing only the
        join-id column and the simplified geometries.  Intended for
        visualization use; readable via ``read_parquet(path, geom='simplified')``.
        Ignored when *combined* is True.
    combined : bool
        If True and *gdf* is a GeoDataFrame, write a single geoparquet file
        that includes all attribute columns and the geometry column together,
        with no ``_geo`` sidecar and no ``_join_id``.  Use this when
        downstream consumers expect a standard geoparquet rather than the
        split two-file layout.
    file_metadata : dict of str to str, optional
        Key-value pairs written into the attribute parquet's footer (file-level)
        metadata. Applied to plain frames and to the attribute half of the
        split two-file layout (never to the `_geo` sidecar, whose geoparquet
        writer manages its own footer); see `to_parquet`.
    """
    if isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)

    # Break link to avoid warnings of setting on slice
    gdf = gdf.copy()

    if combined and isinstance(gdf, gpd.GeoDataFrame):
        to_parquet(gdf, parquet_path, schema_version='1.1.0')
        return

    if isinstance(gdf, gpd.GeoDataFrame) and (
        'geo_id' in gdf or gdf.index.name == 'geo_id'
    ):
        join_id_column = 'geo_id'
    else:
        # Create space-efficient integer ID to join source table
        # to geospatial data and other attribute tables
        join_id_column = '_join_id'
        if gdf.index.name != join_id_column and join_id_column not in gdf:
            gdf[join_id_column] = range(1, len(gdf) + 1)

    if isinstance(gdf, gpd.GeoDataFrame):
        # Save attribute table without geometry
        to_parquet(
            gdf[[v for v in gdf if v != 'geometry']],
            parquet_path,
            file_metadata=file_metadata,
        )

        # Save non-duplicate geometries separately
        if join_id_column == gdf.index.name:
            gdf_geo = gdf.reset_index()
        else:
            gdf_geo = gdf
        mask_unique_join_id = ~gdf_geo[join_id_column].duplicated()
        to_parquet(
            gdf_geo[mask_unique_join_id][[join_id_column, 'geometry']],
            parquet_path.with_stem(parquet_path.stem + '_geo'),
            index=False,
            schema_version='1.1.0',
            write_covering_bbox=True,
        )

        if simplified_geometry is not None:
            gdf_geo_simp = gdf_geo[[join_id_column]].copy()
            gdf_geo_simp['geometry'] = simplified_geometry.values
            gdf_geo_simp = gpd.GeoDataFrame(gdf_geo_simp, crs=gdf.crs)
            to_parquet(
                gdf_geo_simp[mask_unique_join_id][[join_id_column, 'geometry']],
                parquet_path.with_stem(parquet_path.stem + '_geo_simplified'),
                index=False,
                schema_version='1.1.0',
                write_covering_bbox=True,
            )
    else:
        to_parquet(gdf, parquet_path, file_metadata=file_metadata)


def delete_parquet(parquet_path):
    """Delete a parquet file and its geoparquet sidecar if it exists.

    Mirrors the two-file structure written by `save_parquet`: an attribute
    table at `parquet_path` and an optional geometry table at
    `parquet_path.stem + '_geo' + parquet_path.suffix`.

    Parameters
    ----------
    parquet_path : str or Path
        Path to the attribute parquet file.
    """
    parquet_path = Path(parquet_path)
    geo_path = parquet_path.with_stem(parquet_path.stem + '_geo')
    for _path in (parquet_path, geo_path):
        if _path.exists():
            _path.unlink()


def _covering_bbox_column(schema) -> str | None:
    """Return the name of a geoparquet file's covering-bbox column.

    GeoParquet records the struct column holding per-row bounding boxes
    (written by write_covering_bbox=True) in the file's `geo` metadata
    rather than under a fixed name. It is an internal spatial index: no
    attribute registry entry and no `order_columns` step knows it, so a
    read that skips geometry must skip it too.

    Parameters
    ----------
    schema : pyarrow.Schema
        Arrow schema of the parquet file, carrying its `geo` metadata.

    Returns
    -------
    str or None
        The column name, or None when the file declares no covering.
    """
    raw = (schema.metadata or {}).get(b'geo')
    if raw is None:
        return None
    try:
        geo = json.loads(raw)
    except (TypeError, ValueError):
        return None
    column = geo.get('primary_column')
    spec = (geo.get('columns') or {}).get(column) or {}
    corner = ((spec.get('covering') or {}).get('bbox') or {}).get('xmin')
    if isinstance(corner, list | tuple) and corner:
        return corner[0]
    return None


def read_parquet(
    parquet_path,
    geom=False,
    drop_join_id=True,
    filters=None,
    bbox: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Read parquet file from filesystem (with optional geometries).

    Parameters
    ----------
    parquet_path : str
        Filepath of Parquet file
    geom : bool or 'simplified'
        If True, join full geometries from the ``_geo`` sidecar.
        If ``'simplified'``, join simplified geometries from the
        ``_geo_simplified`` sidecar written by ``save_parquet``.
        For a combined file (written with ``save_parquet(..., combined=True)``
        — geometry already merged in, no sidecar), *geom* only controls
        whether the (always-present) ``geometry`` column is kept or dropped;
        ``'simplified'`` is not supported, since no simplified sidecar exists.
    drop_join_id : bool
        Drop column '_join_id' if it exists.
    filters : list of filters, optional
        Passed to pd.read_parquet for the attribute table. Also applied to
        the geo file as a join-id filter when bbox is not provided.
    bbox : tuple of (minx, miny, maxx, maxy), optional
        Spatial bounding box filter in EPSG:4326. When provided and geom=True,
        exploits covering bbox columns written by write_covering_bbox=True for
        Parquet predicate pushdown on the geo file — bypasses the join-id filter.
    **kwargs
        Additional keyword arguments passed to pd.read_parquet() (e.g. columns).
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            'Could not read file from `openplaces` filesystem:\n' + str(parquet_path)
        )

    import pyarrow.parquet as pq

    # A single column name is widened here, before either branch can
    # append the join id or the geometry column to it and unpack the
    # string into its characters.
    if isinstance(kwargs.get('columns'), str):
        kwargs['columns'] = [kwargs['columns']]

    # A combined file (save_parquet(..., combined=True)) has geometry baked
    # into the same file as the attributes -- no `_geo` sidecar, no join-id
    # column. Detected via a cheap schema-only peek, before deciding whether
    # to read through pandas or geopandas.
    schema = pq.ParquetFile(parquet_path).schema_arrow
    schema_names = schema.names
    if 'geometry' in schema_names:
        if geom == 'simplified':
            raise ValueError(
                f'{parquet_path} is a combined geoparquet file (geometry merged '
                "into the attribute table, no `_geo` sidecar); a 'simplified' "
                'geometry sidecar was never written for it.'
            )
        columns = kwargs.pop('columns', None)
        bbox_column = _covering_bbox_column(schema)
        read_filters = filters
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            field = bbox_column or 'bbox'
            bbox_filter = (
                (pyarrow.compute.field(field, 'xmin') <= maxx)
                & (pyarrow.compute.field(field, 'ymin') <= maxy)
                & (pyarrow.compute.field(field, 'xmax') >= minx)
                & (pyarrow.compute.field(field, 'ymax') >= miny)
            )
            if read_filters is None:
                read_filters = bbox_filter
            else:
                # Both predicates have to survive: replacing one with the
                # other returns rows the caller excluded (e.g. narrowing
                # to one state, then also passing a bbox).
                if not isinstance(read_filters, pyarrow.compute.Expression):
                    read_filters = pq.filters_to_expression(read_filters)
                read_filters = read_filters & bbox_filter
        if geom:
            if columns is not None and 'geometry' not in columns:
                columns = [*columns, 'geometry']
            df = gpd.read_parquet(
                parquet_path, filters=read_filters, columns=columns, **kwargs
            )
        else:
            # geom is the ultimate decision on whether geometry is read at
            # all: skip gpd.read_parquet (which requires a geometry column
            # present to build a GeoDataFrame) and the WKB decode it implies,
            # reading everything else straight through pandas instead. The
            # covering-bbox struct goes with it: gpd.read_parquet drops it,
            # a plain pandas read would hand it to downstream writes.
            skip = {'geometry'} | ({bbox_column} if bbox_column else set())
            requested = schema_names if columns is None else columns
            columns = [c for c in requested if c not in skip]
            df = pd.read_parquet(
                parquet_path, filters=read_filters, columns=columns, **kwargs
            )
        if drop_join_id and '_join_id' in df:
            df = df.drop(columns='_join_id')
        return df

    columns = kwargs.pop('columns', None)
    join_id_column = None
    join_column_added = False

    if geom:
        # Resolved from the schema rather than from the frame, so a
        # caller's explicit column list can be widened before the read.
        if '_join_id' in schema_names:
            join_id_column = '_join_id'
        elif 'geo_id' in schema_names:
            join_id_column = 'geo_id'
        else:
            raise ValueError('Could not identify column to join GeoParquet.')

        if columns is not None and join_id_column not in columns:
            columns = [*columns, join_id_column]
            join_column_added = True

    df = pd.read_parquet(parquet_path, filters=filters, columns=columns, **kwargs)

    if 'geometry' in df:
        raise ValueError(
            "'geometry' column found in:\n\n"
            + str(parquet_path)
            + '\n\n`read_parquet` expects split attribute (parquet) + '
            'geometry (geoparquet) tables.'
        )

    if geom:
        geo_suffix = '_geo_simplified' if geom == 'simplified' else '_geo'
        geoparquet_path = parquet_path.with_stem(parquet_path.stem + geo_suffix)

        if df.empty:
            # A predicate matching no rows would build an empty `in` list
            # below, which pyarrow rejects against the sidecar's string
            # column. Read zero sidecar rows instead, so the result still
            # carries the geometry column and the sidecar's CRS.
            geoparquet_filters = pyarrow.compute.scalar(False)
        elif bbox is not None:
            minx, miny, maxx, maxy = bbox
            geoparquet_filters = (
                (pyarrow.compute.field('bbox', 'xmin') <= maxx)
                & (pyarrow.compute.field('bbox', 'ymin') <= maxy)
                & (pyarrow.compute.field('bbox', 'xmax') >= minx)
                & (pyarrow.compute.field('bbox', 'ymax') >= miny)
            )
        elif filters is not None:
            geoparquet_filters = [(join_id_column, 'in', df[join_id_column].tolist())]
        else:
            geoparquet_filters = None

        gdf = gpd.read_parquet(geoparquet_path, filters=geoparquet_filters)
        df = gpd.GeoDataFrame(
            df.join(
                gdf.set_index(join_id_column),
                on=join_id_column,
                how='inner' if bbox is not None else 'left',
            ),
            crs=gdf.crs,
        )

    if join_column_added and join_id_column in df:
        df = df.drop(columns=join_id_column)

    if drop_join_id and '_join_id' in df:
        df = df.drop(columns='_join_id')

    return df
