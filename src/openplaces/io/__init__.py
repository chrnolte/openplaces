"""
src/openplaces/io.py

Input/output utilities

"""

import shutil
import tempfile
import warnings
from pathlib import Path
from urllib.parse import unquote, urlparse
from zipfile import BadZipFile, ZipFile

import geopandas as gpd
import pandas as pd
import requests
from tqdm import tqdm

from openplaces.config import cfg
from openplaces.core.constants import GEOPANDAS_EXTENSIONS, PANDAS_EXTENSIONS

__all__ = [
    'download',
    'read_parquet',
    'save',
    'save_parquet',
    'to_csv',
    'to_parquet',
    'to_gpkg',
    'to_kmz',
    'unzip',
]


def download(from_url, to_path, chunk_size=8192, timeout=None):
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

    # Get file size for progress bar
    try:
        response = requests.head(from_url, timeout=timeout)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
    except Exception:
        total_size = 0

    # Download with progress bar
    response = requests.get(from_url, stream=True, timeout=timeout)
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
                for chunk in response.iter_content(chunk_size=chunk_size):
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


def unzip(in_path, out_dir=None, members=None):
    """Extract files from a zip archive.

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

    Returns
    -------
    Path
        Path to output directory

    Raises
    ------
    BadZipFile
        If the file is not a valid zip archive
    FileNotFoundError
        If input file doesn't exist

    Examples
    --------
    >>> unzip('data/raw/parcels.zip')  # -> data/raw/parcels/
    >>> unzip('data.zip', 'data/heap')  # -> data/heap/
    >>> unzip('data.zip', members=['file1.txt', 'file2.csv'])
    """
    in_path = Path(in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Zip file not found: {in_path}")

    if out_dir is None:
        out_dir = in_path.parent / in_path.stem
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        with ZipFile(in_path, 'r') as z:
            all_members = z.namelist()
            member_list = members or all_members

            # Determine if we should strip a common prefix
            strip_prefix = None

            # Get top-level items from member_list
            # Items with '/' are directories, items without are files
            top_level_items = set()
            for m in member_list:
                if not m:
                    continue
                # Get first path component
                first_component = m.split('/')[0]
                top_level_items.add(first_component)

            # If all members share a single top-level directory (not file)
            # Check if there's more than just the top-level item itself
            if len(top_level_items) == 1:
                common_prefix = list(top_level_items)[0]
                # Only strip if it's actually a directory (has nested content)
                # and is not a geodatabase
                has_nested = any('/' in m for m in member_list)
                if has_nested and not common_prefix.endswith('.gdb'):
                    strip_prefix = f"{common_prefix}/"

            total_size = sum(z.getinfo(m).file_size for m in member_list)

            with tqdm(
                total=total_size, unit='B', unit_scale=True, desc='Extracting'
            ) as pbar:
                for member in member_list:
                    # Skip the top-level folder itself if stripping
                    if strip_prefix and member == strip_prefix.rstrip('/'):
                        continue

                    # Determine target path
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

    except BadZipFile as e:
        raise BadZipFile(f"Invalid zip file: {in_path}") from e

    return out_dir


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
        raise ValueError(f"Directory does not exist: {directory}")

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


def _ensure_parent_dir(filepath: Path):
    """Ensure parent directory exists."""
    filepath.parent.mkdir(parents=True, exist_ok=True)


def _remove_if_exists(filepath: Path) -> None:
    """Remove file if it exists (needed for some formats like gpkg)."""
    if filepath.exists():
        filepath.unlink()


def to_parquet(
    df: pd.DataFrame | gpd.GeoDataFrame, filepath: str | Path, **kwargs
) -> None:
    """Save dataframe to Parquet format.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        Data to save
    filepath : str or Path
        Output parquet path (should end in .parquet)
    **kwargs
        Additional arguments passed to to_parquet()
    """
    _ensure_parent_dir(filepath)

    if isinstance(df, gpd.GeoDataFrame):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*initial implementation of Parquet.*')
            df.to_parquet(filepath, **kwargs)
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
    _ensure_parent_dir(filepath)

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
    _ensure_parent_dir(filepath)
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
    import zipfile

    _ensure_parent_dir(filepath)
    _remove_if_exists(filepath)

    if not isinstance(gdf, gpd.GeoDataFrame):
        warnings.warn('Object is not a GeoDataFrame.', UserWarning)

    # Save as KML temporarily
    kml_path = filepath.with_suffix('.kml')
    gdf.to_file(kml_path, driver='KML')

    # Convert to KMZ (zipped KML)
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as kmz:
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
            f"Unsupported file extension: {ext}. "
            f"Supported: .parquet, .gpkg, .csv, .kmz"
        )


def save_parquet(gdf, parquet_path):
    """Save parquet file (with geometries in joinable geoparquet file)

    Parameters
    ----------
    gdf : DataFrame or GeoDataFrame
        Data to save
    parquet_path : str
        Filepath of Parquet file
    """
    if isinstance(parquet_path, str):
        parquet_path = Path(parquet_path)

    # Break link to avoid warnings of setting on slice
    gdf = gdf.copy()

    if isinstance(gdf, gpd.GeoDataFrame) and (
        'geo_id' in gdf or gdf.index.name == 'geo_id'
    ):
        join_id_column = 'geo_id'
    else:
        join_id_column = '_join_id'
        if gdf.index.name != join_id_column and join_id_column not in gdf:
            # Create space-efficient integer ID to join source table
            # to geospatial data and other attribute tables
            gdf[join_id_column] = range(1, len(gdf) + 1)

    if isinstance(gdf, gpd.GeoDataFrame):
        # Save attribute table without geometry
        to_parquet(
            gdf[[v for v in gdf if v != 'geometry']],
            parquet_path,
        )

        # Save geometries separately, with the join ID as the index
        if join_id_column == gdf.index.name:
            gdf_geo = gdf
        else:
            gdf_geo = gdf.set_index(join_id_column)[['geometry']]
        to_parquet(gdf_geo, parquet_path.with_stem(parquet_path.stem + '_geo'))
    else:
        to_parquet(gdf, parquet_path)


def read_parquet(parquet_path, geom=False, drop_join_id=True):
    """Read parquet file from filesystem (with optional geometries)

    Parameters
    ----------
    parquet_path : str
        Filepath of Parquet file
    geom : bool
        If True, join geometries from Geoparquet file
        (`parquet_path` with '_geo' suffix)
    drop_join_id : bool
        Drop column '_join_id' if it exists.
    """
    parquet_path = Path(parquet_path)

    df = pd.read_parquet(parquet_path)
    if 'geometry' in df:
        raise ValueError(
            "'geometry' column found in:\n\n"
            + parquet_path
            + '\n\n`read_parquet` expects split attribute (parquet) + '
            'geometry (geoparquet) tables.'
        )

    if geom:
        geoparquet_path = parquet_path.with_stem(parquet_path.stem + '_geo')
        # Join polygons to table

        gdf = gpd.read_parquet(geoparquet_path)
        df = gpd.GeoDataFrame(df.join(gdf, on=gdf.index.name), crs=cfg.crs)

    if drop_join_id and '_join_id' in df:
        df = df.drop(columns='_join_id')

    return df
