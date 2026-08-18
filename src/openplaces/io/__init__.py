"""
Input/output utilities
"""

import bz2
import gc
import math
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import warnings
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZIP_LZMA, ZIP_STORED, BadZipFile, ZipFile

import geopandas as gpd
import pandas as pd
import pyarrow
import requests
from tqdm import tqdm

from openplaces.config import cfg
from openplaces.core.constants import (
    GEOPANDAS_EXTENSIONS,
    PANDAS_EXTENSIONS,
    SHAPEFILE_EXTENSIONS,
)
from openplaces.core.schema import AdminId

__all__ = [
    'compress',
    'delete_image_caches',
    'download',
    'read_parquet',
    'release_unused_memory',
    'save',
    'save_parquet',
    'share',
    'to_csv',
    'to_drive',
    'to_parquet',
    'to_gpkg',
    'to_kmz',
    'unzip',
]

_CONTENT_TYPE_EXT = {
    'application/geo+json': '.geojson',
    'application/json': '.geojson',  # ambiguous but common for GeoJSON APIs
    'application/geopackage+sqlite3': '.gpkg',
    'application/x-sqlite3': '.gpkg',
    'application/zip': '.zip',
    'application/x-zip-compressed': '.zip',
    'application/vnd.apache.parquet': '.parquet',
}


def release_unused_memory() -> None:
    """Return freed Python and pyarrow allocations to the OS.

    Geometry-heavy GeoDataFrames and pyarrow's arena allocator do not
    reliably release freed memory back to the OS through refcounting
    alone. Call this after finishing work on one admin unit in a
    multi-admin-unit run to keep peak resident memory bounded.
    """
    gc.collect()
    pyarrow.default_memory_pool().release_unused()


def _content_type_to_ext(content_type: str) -> str | None:
    """Return file extension for a Content-Type header value, or None."""
    mime = content_type.split(';')[0].strip().lower()
    return _CONTENT_TYPE_EXT.get(mime)


def download(from_url, to_path, chunk_size=8192, timeout=None, verify_ssl=True):
    """Download file from URL with progress bar.

    Parameters
    ----------
    from_url : str
        Source URL
    to_path : str or Path
        Target file path or directory
        If a directory is passed (.suffix == ''), filename is inferred
        from response headers or url.
    chunk_size : int, default 8192
        Download chunk size in bytes
    timeout : int, optional
        Request timeout in seconds (uses cfg.download_timeout if None)

    Returns
    -------
    Path
        Path to downloaded file

    Raises
    ------
    requests.RequestException
        If download fails
    """

    to_path = Path(to_path)

    if to_path.suffix == '':
        # Assumption: `to_path` refers to a directory
        to_path.mkdir(parents=True, exist_ok=True)
    else:
        to_path.parent.mkdir(parents=True, exist_ok=True)

    timeout = timeout or cfg.download_timeout

    # Some servers reject the default python-requests User-Agent (404/403);
    # present a browser UA, as the page-scraping path already does.
    headers = {'User-Agent': 'Mozilla/5.0'}

    # Get file size for progress bar
    try:
        response = requests.head(
            from_url, timeout=timeout, verify=verify_ssl, headers=headers
        )
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
    except Exception:
        total_size = 0

    # Download with progress bar
    response = requests.get(
        from_url, stream=True, timeout=timeout, verify=verify_ssl, headers=headers
    )
    response.raise_for_status()

    if to_path.suffix == '':
        # Assumption: `to_path` refers to a directory
        # Extract filename and add it to `to_path`

        # Try Content-Disposition header first if response provided
        if response and 'content-disposition' in response.headers:
            content_disp = response.headers['content-disposition']
            if 'filename=' in content_disp:
                filename_part = content_disp.split('filename=')[1]
                filename = filename_part.split(';')[0].strip('"\'')
                to_path /= unquote(filename)
        else:
            # Fall back to URL parsing
            parsed = urlparse(from_url)
            path = unquote(parsed.path)
            filename = Path(path).name
            to_path /= filename

    # If extension is still unknown, sniff from Content-Type then first chunk
    first_chunk = b''
    content_iter = response.iter_content(chunk_size=chunk_size)

    if to_path.suffix == '':
        ext = _content_type_to_ext(response.headers.get('content-type', ''))

        if ext is None:
            first_chunk = next(content_iter, b'')
            ext = _sniff_ext(first_chunk)

        if ext is not None:
            to_path = to_path.with_suffix(ext)

    # Download to temp location first, then move to final destination
    temp_path = Path(tempfile.gettempdir()) / f'{to_path.name}.part'

    try:
        with open(temp_path, 'wb') as f:
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc='\u2913 ' + to_path.name,
            ) as pbar:
                if first_chunk:
                    f.write(first_chunk)
                    pbar.update(len(first_chunk))
                for chunk in content_iter:
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Move completed download to final destination
        shutil.move(str(temp_path), str(to_path))

    except Exception:
        # Clean up partial download on any error
        if temp_path.exists():
            temp_path.unlink()
        raise

    return to_path


def _sniff_ext(chunk: bytes) -> str | None:
    """Detect file format from a leading byte chunk.

    Checks (in order): GeoParquet, GeoPackage, Shapefile ZIP, GeoJSON.

    Returns
    -------
    str | None
        File extension including dot, or None if unrecognised.
    """

    # Parquet: magic bytes PAR1 at offset 0
    if chunk[:4] == b'PAR1':
        return '.parquet'

    # GeoPackage / SQLite: magic string at offset 0
    if chunk[:16] == b'SQLite format 3\x00':
        return '.gpkg'

    # Shapefile delivered as ZIP: PK magic bytes
    if chunk[:2] == b'PK':
        return '.zip'

    # GeoJSON: no need for a balanced parse — just check the type field
    text = chunk.decode('utf-8', errors='replace').strip()
    if text.startswith('{'):
        match = re.search(r'"type"\s*:\s*"(\w+)"', text)
        if match and match.group(1) in {
            'FeatureCollection',
            'Feature',
            'Point',
            'MultiPoint',
            'LineString',
            'MultiLineString',
            'Polygon',
            'MultiPolygon',
            'GeometryCollection',
        }:
            return '.geojson'

    print('openplaces.io._sniff_ext() failed to infer file type.')

    return None


def unzip(in_path, out_dir=None, members=None, verbose=True):
    """Extract files from a zip archive.

    Supports standard ZIP (deflate) and Deflate64 ZIP files. Deflate64
    extraction requires 7z to be installed (see dev.py ensure_7zip()).

    Parameters
    ----------
    in_path : str or Path
        Path to input zip file
    out_dir : str or Path, optional
        Output directory. If None, extracts to directory named after
        the zip file (without extension) in the same location.
        Example: 'data.zip' -> 'data/'
    members : list of str, optional
        Specific files to extract. If None, extracts all files.
        Note: ignored when falling back to 7z.
    verbose:
        If True, might print warnings, e.g. when switching to 7z

    Returns
    -------
    Path
        Path to output directory

    Examples
    --------
    >>> unzip('data/raw/parcels.zip')  # -> data/raw/parcels/
    >>> unzip('data.zip', 'data/heap')  # -> data/heap/
    >>> unzip('data.zip', members=['file1.txt', 'file2.csv'])
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f'Zip file not found: {in_path}')
    if out_dir is None:
        out_dir = in_path.parent / in_path.stem
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [s.lower() for s in in_path.suffixes]
    _is_tar_gz = in_path.suffix.lower() == '.tgz' or (
        len(suffixes) >= 2 and suffixes[-2] == '.tar' and suffixes[-1] == '.gz'
    )
    if _is_tar_gz:
        with tarfile.open(in_path, 'r:gz') as tar:
            tar.extractall(out_dir)
        return out_dir

    if in_path.suffix in {'.bz2', '.tbz2'}:
        # Handle .tar.bz2 / .tbz2
        if in_path.stem.endswith('.tar') or in_path.suffix == '.tbz2':
            with tarfile.open(in_path, 'r:bz2') as tar:
                tar.extractall(out_dir)
        # Handle bare .bz2 (single compressed file)
        else:
            out_file = out_dir / in_path.stem
            with bz2.open(in_path, 'rb') as src, out_file.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        return out_dir

    if _needs_7z(in_path):
        return _unzip_with_7z(in_path, out_dir, verbose)
    return _unzip_standard(in_path, out_dir, members)


def _needs_7z(zip_path):
    """Return True if any entry uses a compression type unsupported by zipfile."""
    with ZipFile(zip_path, 'r') as z:
        return any(
            info.compress_type
            not in {
                ZIP_STORED,
                ZIP_DEFLATED,
                ZIP_BZIP2,
                ZIP_LZMA,
            }
            for info in z.infolist()
        )


def _unzip_standard(in_path, out_dir, members):
    """Extract using Python's zipfile (deflate and store)."""
    with ZipFile(in_path, 'r') as z:
        all_members = z.namelist()
        member_list = members or all_members

        strip_prefix = _get_strip_prefix(member_list)

        total_size = sum(z.getinfo(m).file_size for m in member_list)
        with tqdm(
            total=total_size, unit='B', unit_scale=True, desc='Extracting'
        ) as pbar:
            for member in member_list:
                if strip_prefix and member == strip_prefix.rstrip('/'):
                    continue
                if strip_prefix and member.startswith(strip_prefix):
                    relative_path = member.removeprefix(strip_prefix)
                    if not relative_path:
                        continue
                    target = out_dir / relative_path
                else:
                    target = out_dir / member
                if member.endswith('/'):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, open(target, 'wb') as dst:
                        while chunk := src.read(8192):
                            dst.write(chunk)
                            pbar.update(len(chunk))
    return out_dir


def _find_7z():
    """Return path to 7z executable, checking known install locations."""
    path = shutil.which('7z')
    if path:
        return path
    if sys.platform == 'win32':
        default = r'C:\Program Files\7-Zip\7z.exe'
        if Path(default).exists():
            return default
    return None


def _unzip_with_7z(in_path, out_dir, verbose=True):
    """Extract using 7z for compression types unsupported by zipfile."""
    sz = _find_7z()
    if not sz:
        raise RuntimeError(
            f'Archive {in_path.name} uses a compression type `zipfile` cannot deflate. '
            'Install 7z: brew install sevenzip (macOS), '
            'winget install 7zip.7zip (Windows), or sudo apt install 7zip (Linux).'
        )
    if verbose:
        print('Extracting with 7z...')
    result = subprocess.run(
        [sz, 'x', str(in_path), f'-o{out_dir}', '-y'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BadZipFile(f'7z failed to extract {in_path.name}:\n{result.stderr}')
    return out_dir


def _get_strip_prefix(member_list):
    """Return common top-level directory prefix to strip, or None."""
    top_level_items = set()
    for m in member_list:
        if m:
            top_level_items.add(m.split('/')[0])
    if len(top_level_items) == 1:
        common_prefix = next(iter(top_level_items))
        has_nested = any('/' in m for m in member_list)
        if has_nested and not common_prefix.endswith('.gdb'):
            return f'{common_prefix}/'
    return None


def find_latest_file_or_gdb(
    directory: str, extensions: list[str] = GEOPANDAS_EXTENSIONS | PANDAS_EXTENSIONS
) -> Path | None:
    """
    Find the most recently modified file or .gdb directory in a directory.

    Parameters
    ----------
    directory : str
        Path to the directory to search
    extensions : list[str]
        List of accepted file extensions (e.g., ['.csv', '.txt', '.json'])

    Returns
    -------
    Optional[Path]
        Path to the most recent file or .gdb directory, or None if no matches found
    """
    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f'Directory does not exist: {directory}')

    # Normalize extensions to include leading dot
    normalized_exts = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    # Find all files with matching extensions
    matching_files = [
        f for f in dir_path.iterdir() if f.is_file() and f.suffix in normalized_exts
    ]

    # Find all .gdb directories
    gdb_dirs = [d for d in dir_path.iterdir() if d.is_dir() and d.suffix == '.gdb']

    # Combine files and .gdb directories
    all_matches = matching_files + gdb_dirs

    if not all_matches:
        return None

    # Return the item with the most recent modification time
    return max(all_matches, key=lambda f: f.stat().st_mtime)


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
        Additional arguments passed to to_parquet()
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)

    filepath.parent.mkdir(parents=True, exist_ok=True)

    df = _categoricals_to_string(df)

    if isinstance(df, gpd.GeoDataFrame):
        with warnings.catch_warnings():
            kwargs.setdefault('write_covering_bbox', True)
            warnings.filterwarnings('ignore', '.*initial implementation of Parquet.*')
            df.to_parquet(filepath, **kwargs)
    elif file_metadata:
        import pyarrow.parquet as pq

        table = pyarrow.Table.from_pandas(df)
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

    Automatically drops 'geometry' column if present.

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

    # Drop geometry if present
    if isinstance(df, gpd.GeoDataFrame):
        df = df.drop(columns='geometry')

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
        metadata. Only applied to plain (non-geo) frames; see `to_parquet`.
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

    # A combined file (save_parquet(..., combined=True)) has geometry baked
    # into the same file as the attributes -- no `_geo` sidecar, no join-id
    # column. Detected via a cheap schema-only peek, before deciding whether
    # to read through pandas or geopandas.
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names
    if 'geometry' in schema_names:
        if geom == 'simplified':
            raise ValueError(
                f'{parquet_path} is a combined geoparquet file (geometry merged '
                "into the attribute table, no `_geo` sidecar); a 'simplified' "
                'geometry sidecar was never written for it.'
            )
        columns = kwargs.pop('columns', None)
        read_filters = filters
        if bbox is not None:
            minx, miny, maxx, maxy = bbox
            read_filters = (
                (pyarrow.compute.field('bbox', 'xmin') <= maxx)
                & (pyarrow.compute.field('bbox', 'ymin') <= maxy)
                & (pyarrow.compute.field('bbox', 'xmax') >= minx)
                & (pyarrow.compute.field('bbox', 'ymax') >= miny)
            )
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
            # reading everything else straight through pandas instead.
            columns = [c for c in (columns or schema_names) if c != 'geometry']
            df = pd.read_parquet(
                parquet_path, filters=read_filters, columns=columns, **kwargs
            )
        if drop_join_id and '_join_id' in df:
            df = df.drop(columns='_join_id')
        return df

    df = pd.read_parquet(parquet_path, filters=filters, **kwargs)

    if 'geometry' in df:
        raise ValueError(
            "'geometry' column found in:\n\n"
            + str(parquet_path)
            + '\n\n`read_parquet` expects split attribute (parquet) + '
            'geometry (geoparquet) tables.'
        )

    if geom:
        if '_join_id' in df:
            join_id_column = '_join_id'
        elif 'geo_id' in df:
            join_id_column = 'geo_id'
        else:
            raise ValueError('Could not identify column to join GeoParquet.')

        geo_suffix = '_geo_simplified' if geom == 'simplified' else '_geo'
        geoparquet_path = parquet_path.with_stem(parquet_path.stem + geo_suffix)

        if bbox is not None:
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

    if drop_join_id and '_join_id' in df:
        df = df.drop(columns='_join_id')

    return df


class DataDeletionError(OSError):
    """Raised when an unzipped dataset could not be fully deleted.

    A partial deletion, typically caused by a file sync app (e.g. Dropbox)
    locking files mid-removal, leaves a corrupt copy on disk that a later
    ingest would silently reuse. Deletion failures therefore interrupt the run
    rather than warn.
    """


def _deletion_interrupted_error(path: Path, *, is_dir: bool) -> DataDeletionError:
    """Build a DataDeletionError naming *path* with a clickable file link.

    The ``file://`` URI is rendered as a clickable link by most terminals and
    Jupyter, opening the location in the OS file browser (e.g. Explorer) so the
    leftover, partially deleted dataset can be removed by hand.

    Parameters
    ----------
    path : Path
        Path whose deletion was interrupted.
    is_dir : bool
        Whether the path is a directory (e.g. a geodatabase) or a single file.
    """
    resolved = Path(path).resolve()
    target = 'directory' if is_dir else 'file'
    return DataDeletionError(
        f'\n\nInterrupted while deleting an unzipped {target}.\n\n'
        'A file sync app (e.g., Dropbox) is most likely locking files here,\n'
        'leaving a partially deleted, corrupt copy that the next ingest would '
        'silently reuse.\n\n'
        f'  {resolved}\n'
        f'  {resolved.as_uri()}\n\n'
        'Pause or quit the sync app, delete the path above, then re-run.\n\n'
    )


def delete_data(data_path, delete_empty_parent_dirs=True):
    """Delete dataset from openplaces filesystem

    Handles geodatabases and shapefiles with compantion files

    Parameters
    ----------
    data_path : Path
        Path to file to be deleted. Extension determines how deletion
        occurs (e.g. '.shp' files and '.gdb' folders are handled)
    delete_empty_parent_dirs : bool
        Deletes any parent directories that are now empty.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f'File to delete not found: {data_path.relative_to(cfg.data_root)}'
        )

    if data_path.suffix == '.gdb':
        try:
            shutil.rmtree(data_path)
        except OSError as error:
            raise _deletion_interrupted_error(data_path, is_dir=True) from error
        # rmtree can stop partway when a lock is released mid-walk; confirm the
        # directory is actually gone so a partial geodatabase is never reused.
        if data_path.exists():
            raise _deletion_interrupted_error(data_path, is_dir=True)

    elif data_path.suffix == '.shp':
        for shapefile_extension in SHAPEFILE_EXTENSIONS:
            data_path.with_suffix(shapefile_extension).unlink(missing_ok=True)
    else:
        try:
            data_path.unlink()
        except OSError as error:
            raise _deletion_interrupted_error(data_path, is_dir=False) from error

    if delete_empty_parent_dirs:
        current_dir = data_path.parent
        while True:
            if current_dir == cfg.data_root:
                break

            # Check if directory is empty
            if current_dir.exists() and not any(current_dir.iterdir()):
                try:
                    current_dir.rmdir()
                except PermissionError:
                    warnings.warn(
                        '\n\nUnable to delete empty directory due to permission error:'
                        + f'\n\n{current_dir}\n\n'
                        'Is a file sync app running (e.g., Dropbox)? '
                        'If so, quit and retry, or remove the directory manually.\n\n'
                    )
                current_dir = current_dir.parent
            else:
                break


def delete_image_caches(
    admin_ids: str | list | None = None,
    source: str | None = None,
    version: str | None = None,
    dry_run: bool = True,
) -> pd.DataFrame:
    """Delete location-specific image caches from the external directory.

    Parameters
    ----------
    admin_ids : str, AdminId, list, or None
        Admin units whose caches to delete; a coarser unit (e.g. a county)
        matches all caches of its children. None matches all locations.
    source : str or None
        Restrict to one image source (e.g. 'googlesatellite').
    version : str or None
        Restrict to one recipe version (e.g. 'z20').
    dry_run : bool
        If True (default), only report what would be deleted. If False,
        remove each matched cache directory, including images and the
        image metadata parquet.

    Returns
    -------
    pd.DataFrame
        The matched caches: admin_id, source, version, n_files, size_mb,
        path.
    """
    from openplaces.diagnostics import list_image_caches

    caches = list_image_caches()
    if caches.empty:
        print('No image caches found.')
        return caches

    if admin_ids is not None:
        if isinstance(admin_ids, str | AdminId):
            admin_ids = [admin_ids]
        selectors = [AdminId(str(a)) for a in admin_ids]
        caches = caches[
            [
                any(
                    str(sel) == str(aid) or sel.is_parent_of(aid)
                    for sel in selectors
                    for aid in [AdminId(cache_admin_id)]
                )
                for cache_admin_id in caches['admin_id']
            ]
        ]
    if source is not None:
        caches = caches[caches['source'] == source]
    if version is not None:
        caches = caches[caches['version'] == str(version)]
    caches = caches.reset_index(drop=True)

    total_mb = caches['size_mb'].sum()
    if dry_run:
        print(
            f'Dry run: would delete {len(caches)} image cache(s), '
            f'{total_mb:,.1f} MB total. Pass dry_run=False to delete.'
        )
        return caches

    for cache_path in caches['path']:
        shutil.rmtree(cache_path, ignore_errors=True)
    print(f'Deleted {len(caches)} image cache(s), {total_mb:,.1f} MB total.')
    return caches


def get_gdb_domains(gdb_path: str) -> dict[str, dict]:
    """Extract coded value domains from a GDB's internal metadata table."""
    domains = {}
    try:
        items = gpd.read_file(gdb_path, layer='GDB_Items', engine='pyogrio')
    except Exception:
        return domains

    for _, row in items.iterrows():
        definition = row.get('Definition')
        if not definition or (isinstance(definition, float) and math.isnan(definition)):
            continue
        try:
            root = ElementTree.fromstring(definition)
        except ElementTree.ParseError:
            continue

        # Coded value domains have a <CodedValueDomain> or <GPCodedValueDomain2> element
        for cv_domain in root.iter('CodedValue'):
            domain_name = root.findtext('DomainName')
            if not domain_name:
                continue
            if domain_name not in domains:
                domains[domain_name] = {}
            code = cv_domain.findtext('Code')
            name = cv_domain.findtext('Name')
            if code is not None and name is not None:
                domains[domain_name][code] = name

    return domains


def get_gdb_field_domain_map(gdb_path: str, layer: str) -> dict[str, dict]:
    """Map field names to their coded value domain for a given layer.

    Parameters
    ----------
    gdb_path : str
        Path to geodatabase
    layer : str
        Layer to read
    """
    field_domains = {}
    try:
        items = gpd.read_file(gdb_path, layer='GDB_Items', engine='pyogrio')
    except Exception:
        return field_domains

    all_domains = get_gdb_domains(gdb_path)

    for _, row in items.iterrows():
        if row.get('Name') != layer:
            continue
        definition = row.get('Definition')
        if not definition or (isinstance(definition, float) and math.isnan(definition)):
            continue
        try:
            root = ElementTree.fromstring(definition)
        except ElementTree.ParseError:
            continue
        for field in root.iter('GPFieldInfoEx'):
            fname = field.findtext('Name')
            dname = field.findtext('DomainName')
            if fname and dname and dname in all_domains:
                field_domains[fname] = all_domains[dname]

    return field_domains


def read_gdb_with_domains(
    gdb_path: str, columns: list = None, layer: str = None, **kwargs
) -> gpd.GeoDataFrame:
    """Read a Geodatabase while resolving categorical label mappings"""
    gdf = gpd.read_file(
        gdb_path, columns=columns, layer=layer, engine='pyogrio', **kwargs
    )
    field_domains = get_gdb_field_domain_map(gdb_path, layer)

    for col, mapping in field_domains.items():
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).map(mapping).fillna(gdf[col])

    return gdf


def compress(
    filepaths: str | Path | list[str] | set[str],
    zip_filepath: str | None = None,
    delete_original: bool = False,
) -> None:
    """Compress one or more files.

    Parameters
    ----------
    filepaths : str or list of str
        Single filepath or list of filepaths
    zip_filepath : str, optional
        Output ZIP filepath. If None, derived from the first entry in filepaths.
    delete_original : bool
        If True, deletes the original file(s) after compression.
    """
    if isinstance(filepaths, str | Path):
        filepaths = [filepaths]
    elif not isinstance(filepaths, list | set):
        raise ValueError(f'`filepaths` argument not understood: {filepaths}')

    paths = [Path(fp) for fp in filepaths]

    if zip_filepath is None:
        zip_path = paths[0].parent / f'{paths[0].stem}_{paths[0].suffix[1:]}.zip'
    else:
        zip_path = Path(zip_filepath)
        if zip_path.suffix != '.zip':
            zip_path = zip_path.parent / f'{zip_path.stem.replace(".", "_")}.zip'

    paths_to_compress: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f'Warning: file does not exist: {path}')
            continue
        if path.suffix == '.shp':
            paths_to_compress.extend(
                path.with_suffix(ext)
                for ext in SHAPEFILE_EXTENSIONS
                if path.with_suffix(ext).exists()
            )
        else:
            paths_to_compress.append(path)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, 'w', ZIP_DEFLATED) as z:
        for p in paths_to_compress:
            z.write(p, arcname=p.name)

    if delete_original:
        for p in paths_to_compress:
            p.unlink()


def to_drive(filepath, directory, remote='budrive', verbose=True):
    """Copy file to Google Drive

    Uses `rclone`. Remote 'drive' must exist: https://rclone.org/drive/

    Parameters
    ----------
    filepath : str
        Path of file to copy
    directory : str
        Drive folder to copy to
    remote : str
        Name of the `rclone` remote to copy to
    verbose : bool
        If True, print progress
    """

    cmd = ['rclone', 'copy', filepath, f'{remote}:{directory}']
    if verbose and sys.stdout.isatty():
        cmd += ['--progress']
    subprocess.run(cmd)


def share(df, filepath, drive_dir=None, delete_original=True, verbose=True):
    """Shortcut for saving, compressing, and uploading to Drive

    File format is deduced from filepath extension.

    Drive folder is deduced from filepath and assumed to be in the
    `share` data directory (openplaces.config.cfg.share_dir)

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Dataset to be saved
    filepath : pathlib.Path
        Filepath used for saving (and for the compressed ZIP file).
    delete_original : bool
        If True, deletes the unzipped file after compression
    verbose : bool
        If True, prints statements ('Saving', 'compressing', etc.)
    """

    if verbose:
        print('Saving...', end='')
    save(df, filepath)

    if verbose:
        print(' compressing...', end='')
    zip_path = filepath.parent / f'{filepath.stem}_{filepath.suffix[1:]}.zip'
    compress(filepath, zip_path, delete_original=delete_original)

    if drive_dir is None:
        drive_dir = filepath.parent.relative_to(cfg.share_dir)
    if verbose:
        print(f' uploading to: {drive_dir} ...', end='')
    to_drive(zip_path, drive_dir, verbose=verbose)
    if verbose:
        print(' done!')

    if delete_original:
        zip_path.unlink()
