"""
Read and write tables: parquet with the footer metadata this
project relies on, plus CSV, GeoPackage and KMZ writers and the
geometry-sidecar convention behind save_parquet and read_parquet.
"""

import contextlib
import contextvars
import functools
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import geopandas as gpd
import pandas as pd
import pyarrow


class UnredactedShareWriteError(PermissionError):
    """A table write into the share bucket that no redaction preceded."""


# Set only while `io.delivery.export_delivery` writes a bundle it has
# redacted. A context variable rather than a flag argument, so no
# caller can pass its way past the check by keyword.
_REDACTED_SHARE_WRITE = contextvars.ContextVar(
    'openplaces_redacted_share_write', default=False
)


@contextlib.contextmanager
def redacted_share_writes():
    """Allow table writes into the share bucket inside this block.

    For `io.delivery.export_delivery` alone, around the writes of a
    bundle it has already redacted (restricted sources withheld,
    personal columns refused). Any other writer reaching the share
    bucket is refused by `refuse_unredacted_share_write`.
    """
    token = _REDACTED_SHARE_WRITE.set(True)
    try:
        yield
    finally:
        _REDACTED_SHARE_WRITE.reset(token)


def _share_root() -> Path | None:
    """The configured share bucket, resolved, or None without one."""
    from openplaces.config import cfg

    try:
        return Path(cfg.get_dir('share')).resolve()
    except KeyError:
        return None


def refuse_unredacted_share_write(filepath: str | Path) -> None:
    """Raise when *filepath* lies in the share bucket outside a delivery.

    The share bucket is the folder other people receive (`shared` in
    `STANDARD_DIRS`), so a table reaches it only as a bundle that
    `export_delivery` has redacted. A team folder is no exception: a
    team bundle withholds every source not cleared for the team, which
    only `export_delivery` does.

    Parameters
    ----------
    filepath : str or Path
        The file about to be written.

    Raises
    ------
    UnredactedShareWriteError
        When the file is inside the share bucket and the write is not
        inside `redacted_share_writes`.
    """
    # Curated per-county files used to be saved into the share bucket
    # directly, and one carried a restricted county roll's attributes
    # unredacted (Fairfax VA, 2026-10-21): nothing between the curate
    # step and the folder applied the withholding rules.
    if _REDACTED_SHARE_WRITE.get():
        return
    root = _share_root()
    if root is None:
        return
    target = Path(filepath).resolve()
    if target == root or root in target.parents:
        raise UnredactedShareWriteError(
            f'Refusing to write {target}: it is in the share bucket ({root}), '
            'which receives only bundles io.delivery.export_delivery has '
            'redacted. Write curated or intermediate tables to a private '
            "bucket ('core', 'cache', 'out') and ship them with "
            'export_delivery.'
        )


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


# True while one of this module's own reads runs, so the pyarrow
# calls it makes are not logged a second time as reads of every
# column (see `start_read_log`).
_OWN_READ = contextvars.ContextVar('openplaces_own_read', default=False)


def _own_read(function):
    """Mark *function*'s pyarrow calls as this module's own reads."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        token = _OWN_READ.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _OWN_READ.reset(token)

    return wrapper


@_own_read
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

    _log_read(parquet_path, [SCHEMA])
    return list(pq.ParquetFile(parquet_path).schema_arrow.names)


# Footer keys: per-column content digests of the attribute file, and
# of the geometry sidecars written beside it (geopandas writes those
# and takes no footer of their own), keyed by the sidecar's suffix.
COLUMN_DIGESTS_KEY = 'openplaces:column_digests'
SIDECAR_DIGESTS_KEY = 'openplaces:sidecar_digests'
# Pseudo columns: a file's list of column names, for a reader that
# branches on which columns exist, and its footer metadata, for one
# that reads the footer (keys every rewrite changes left out).
SCHEMA = '__schema__'
METADATA = '__metadata__'
_VOLATILE_METADATA = (
    # pyarrow's copy of the schema, footer included; types are in
    # the column digests
    b'ARROW:schema',
    b'openplaces:recipe_roots',
    COLUMN_DIGESTS_KEY.encode(),
    SIDECAR_DIGESTS_KEY.encode(),
)


def _hashable(series: pd.Series) -> pd.Series:
    """Return *series* in a form `hash_pandas_object` hashes in full."""
    if isinstance(series.dtype, gpd.array.GeometryDtype):
        import shapely

        return pd.Series(shapely.to_wkb(series.values), dtype=object)
    if series.dtype == object:
        sample = series.dropna()
        if len(sample) and not isinstance(sample.iloc[0], str | bytes):
            # pandas falls back to str() for unhashable cells, which
            # abbreviates a long array with '...': two different
            # arrays would hash the same.
            def _text(value):
                if hasattr(value, 'tolist'):
                    value = value.tolist()
                return json.dumps(value, default=str)

            return series.map(_text, na_action='ignore')
    return series


def column_digest(series: pd.Series) -> str:
    """Return a digest of a column's dtype and values, in row order.

    Two columns get the same digest when they hold the same values in
    the same order under the same dtype, whatever the parquet writer
    put around them (footer, row groups, compression).

    Parameters
    ----------
    series : Series
        The column.
    """
    # hash_pandas_object hashes each value with a fixed key, so the
    # result is the same in every process. Hashing the row hashes in
    # order makes the digest depend on row order, as a reader may.
    digest = hashlib.blake2b(str(series.dtype).encode(), digest_size=12)
    hashed = pd.util.hash_pandas_object(_hashable(series), index=False)
    digest.update(hashed.to_numpy().tobytes())
    return digest.hexdigest()


def frame_digests(df: pd.DataFrame, index: bool | None = None) -> dict[str, str]:
    """Return the digest of every column a parquet write of *df* stores.

    Parameters
    ----------
    df : DataFrame or GeoDataFrame
        The frame as it is written.
    index : bool or None
        The write's `index` argument: None stores a non-default index as
        columns (pandas' rule), True any index, False none.
    """
    if index or (index is None and not isinstance(df.index, pd.RangeIndex)):
        df = df.reset_index()
    return {str(c): column_digest(df[c]) for c in df.columns}


def _schema_digest(names) -> str:
    return hashlib.blake2b('\x1f'.join(names).encode(), digest_size=12).hexdigest()


def _sidecar_owner(path: Path) -> tuple[Path, str] | None:
    """Return the attribute file and suffix of a geometry sidecar."""
    for suffix in ('_geo_simplified', '_geo'):
        if path.stem.endswith(suffix):
            return path.with_stem(path.stem[: -len(suffix)]), suffix
    return None


@_own_read
def recorded_digests(parquet_path: str | Path) -> dict[str, str]:
    """Return the column digests a file's write recorded, if any.

    A geometry sidecar's digests are read from its attribute file's
    footer, and count only while the sidecar is not newer than that
    file (they are written together). Reads the footer only.

    Parameters
    ----------
    parquet_path : str or Path
        A parquet file written by `to_parquet` or `save_parquet`.

    Returns
    -------
    dict of str to str
        Column name to digest; empty when the write recorded none.
    """
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    owner = _sidecar_owner(path)
    if owner is not None:
        attributes, suffix = owner
        # save_parquet moves the sidecar into place just after its
        # attribute file; one written much later is not the one the
        # footer describes.
        if not attributes.exists() or (
            path.stat().st_mtime > attributes.stat().st_mtime + 60
        ):
            return {}
        source, key = attributes, SIDECAR_DIGESTS_KEY
    else:
        source, key = path, COLUMN_DIGESTS_KEY
    try:
        meta = pq.read_metadata(source).metadata or {}
        recorded = json.loads(meta.get(key.encode(), b'{}'))
    except (OSError, ValueError, pyarrow.ArrowException):
        return {}
    return recorded.get(suffix, {}) if owner is not None else recorded


@_own_read
def file_digests(parquet_path: str | Path, columns=None) -> dict[str, str]:
    """Return the digests of a parquet file's columns, as it is now.

    Digests its write recorded are read from the footer; any other
    column is read and digested, so a file written before digests were
    recorded still compares. `SCHEMA` among *columns* gives a digest of
    the file's column names.

    Parameters
    ----------
    parquet_path : str or Path
        Parquet file.
    columns : iterable of str, optional
        Columns to digest; None for every column and the schema.

    Returns
    -------
    dict of str to str
        Column name to digest. A requested column the file lacks is
        absent. `METADATA` gives a digest of the footer's key-value
        metadata, without the keys every rewrite changes.
    """
    import pyarrow.parquet as pq

    path = Path(parquet_path)
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    wanted = [*names, SCHEMA, METADATA] if columns is None else list(columns)
    recorded = recorded_digests(path)
    digests = {c: recorded[c] for c in wanted if c in recorded}
    if SCHEMA in wanted:
        digests[SCHEMA] = _schema_digest(names)
    if METADATA in wanted:
        footer = parquet.metadata.metadata or {}
        digests[METADATA] = _schema_digest(
            f'{k!r}={v!r}'
            for k, v in sorted(footer.items())
            if k not in _VOLATILE_METADATA
        )
    missing = [c for c in wanted if c not in digests and c in names]
    if missing:
        if _sidecar_owner(path) is not None or 'geometry' in names:
            frame = gpd.read_parquet(path, columns=missing)
        else:
            frame = pd.read_parquet(path, columns=missing)
        digests.update(frame_digests(frame[missing], index=False))
    return digests


# The reads of one job, started by `start_read_log` (run_stage): file
# path to the set of columns read, or None for every column. None
# outside a job, so library use pays nothing.
_READ_LOG: dict[str, set[str] | None] | None = None


def start_read_log() -> dict[str, set[str] | None]:
    """Start recording which columns of which files this process reads.

    Every `read_parquet` and `parquet_columns` call after this adds to
    the returned dict: resolved path to the set of columns read (with
    `SCHEMA` for a look at the column names), or None where every
    column was read. Columns a row filter names count as read. A read
    of geometry counts the whole geometry sidecar.

    Reads that bypass this module are logged too, at the pyarrow calls
    pandas and geopandas go through: `pyarrow.parquet.read_table` by
    its columns, `read_schema` as `SCHEMA`, `read_metadata` as
    `SCHEMA` and `METADATA`, and `ParquetFile`, `ParquetDataset` and
    `pyarrow.dataset.dataset` as every column. Reads of other formats
    are not seen.

    Returns
    -------
    dict
        The log, filled as the process reads.
    """
    global _READ_LOG
    _watch_pyarrow()
    _READ_LOG = {}
    return _READ_LOG


def _log_source(source, columns) -> None:
    """Log a pyarrow read of *source* (a path or a list of them)."""
    if _READ_LOG is None or _OWN_READ.get():
        return
    sources = source if isinstance(source, list | tuple) else [source]
    for item in sources:
        # pandas opens a local file itself and hands pyarrow the handle
        if not isinstance(item, str | os.PathLike):
            item = getattr(item, 'name', None)
        if isinstance(item, str | os.PathLike):
            _log_read(item, columns)


def _watch_pyarrow() -> None:
    """Wrap the pyarrow parquet entry points to log reads (once)."""
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if getattr(pq.read_table, '_openplaces_watch', False):
        return

    def _read_table_columns(columns=None, filters=None, **_):
        named = _filter_columns(filters)
        if columns is None or named is None:
            return None
        return [*columns, *named]

    def wrap(module, name, columns_of):
        original = getattr(module, name)

        def wrapper(source, *args, **kwargs):
            _log_source(source, columns_of(*args, **kwargs))
            # read_table builds a ParquetDataset: one read, logged once
            token = _OWN_READ.set(True)
            try:
                return original(source, *args, **kwargs)
            finally:
                _OWN_READ.reset(token)

        wrapper._openplaces_watch = True
        setattr(module, name, wrapper)

    wrap(pq, 'read_table', _read_table_columns)
    wrap(pq, 'read_schema', lambda *a, **k: [SCHEMA])
    wrap(pq, 'read_metadata', lambda *a, **k: [SCHEMA, METADATA])
    wrap(ds, 'dataset', lambda *a, **k: None)
    for cls in (pq.ParquetFile, pq.ParquetDataset):
        original_init = cls.__init__

        def init(self, source, *args, _original=original_init, **kwargs):
            _log_source(source, None)
            token = _OWN_READ.set(True)
            try:
                _original(self, source, *args, **kwargs)
            finally:
                _OWN_READ.reset(token)

        cls.__init__ = init


def stop_read_log() -> None:
    """Stop recording reads (see `start_read_log`)."""
    global _READ_LOG
    _READ_LOG = None


def _log_read(path, columns) -> None:
    """Add a read of *columns* (None: all) of *path* to the read log."""
    if _READ_LOG is None:
        return
    key = str(Path(path).resolve())
    if key in _READ_LOG and _READ_LOG[key] is None:
        return
    if columns is None:
        _READ_LOG[key] = None
    else:
        _READ_LOG.setdefault(key, set()).update(columns)


def _filter_columns(filters) -> list[str] | None:
    """Return the columns a read's row filter names; None if unknown."""
    if filters is None:
        return []
    if isinstance(filters, pyarrow.compute.Expression):
        return None
    names = []
    for item in filters:
        terms = item if isinstance(item, list) else [item]
        for term in terms:
            if not isinstance(term, tuple) or not isinstance(term[0], str):
                return None
            names.append(term[0])
    return names


# Waits between attempts to move a written file into place (about 60 s
# in all). Windows refuses to replace a file another process holds open,
# and Dropbox holds files while it syncs them: 10 s was too short for
# geospine parcel sidecars during the 2026-10-27 wave.
REPLACE_BACKOFF_S = (0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0)


def _temp_path(filepath: Path) -> Path:
    """Where a write lands before it replaces *filepath*.

    Beside the target, so the rename stays on one volume, and with a name
    no reader's `*.parquet` glob matches.
    """
    return filepath.with_name(f'{filepath.name}.tmp-{os.getpid()}')


# Set by `run_stage --keep-if-equal`, the rerun that checks whether a
# code change alters a job's output (`flow.code_staleness --verify`).
KEEP_EQUAL_ENV = 'OPENPLACES_KEEP_EQUAL'


@_own_read
def same_table(path_a: str | Path, path_b: str | Path) -> bool:
    """Return whether two parquet files hold the same table.

    Column names, types, order and every value must agree; footer
    metadata (the build commit, pandas' schema notes) is ignored, so a
    rewrite by newer code of identical content counts as the same.

    Parameters
    ----------
    path_a, path_b : str or Path
        Parquet files.
    """
    import pyarrow.parquet as pq

    a, b = pq.read_table(path_a), pq.read_table(path_b)
    return a.schema.remove_metadata().equals(b.schema.remove_metadata()) and a.equals(b)


def _replace(temp: Path, target: Path) -> None:
    """Move *temp* onto *target*, retrying while the target is held.

    Under `OPENPLACES_KEEP_EQUAL`, a *temp* holding the same table as
    *target* is discarded instead, so the target keeps its mtime and
    nothing downstream sees a change.
    """
    if os.environ.get(KEEP_EQUAL_ENV) and target.exists():
        try:
            equal = same_table(temp, target)
        except Exception:
            equal = False
        if equal:
            temp.unlink(missing_ok=True)
            return
    for wait in (*REPLACE_BACKOFF_S, None):
        try:
            os.replace(temp, target)
            return
        except PermissionError:
            if wait is None:
                temp.unlink(missing_ok=True)
                raise PermissionError(
                    f'Could not replace {target}: another process holds it '
                    f'(tried {len(REPLACE_BACKOFF_S) + 1} times).'
                ) from None
            time.sleep(wait)


def _write_to_temp(df, filepath: Path, **kwargs) -> Path:
    """Write *df* to *filepath*'s temp path and return it."""
    refuse_unredacted_share_write(filepath)
    temp = _temp_path(filepath)
    try:
        _write_parquet(df, temp, **kwargs)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return temp


def to_parquet(
    df: pd.DataFrame | gpd.GeoDataFrame,
    filepath: str | Path,
    *,
    file_metadata: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """Save dataframe to Parquet format.

    The file is written beside its target under a temporary name and
    then moved into place, so a reader never sees a half-written file
    and a failed or killed write leaves the previous file as it was.
    This matters because the workflow keeps a failed batch job's other
    outputs (`keep-incomplete`), and a downstream job reads them. The
    move retries for about 10 s while another process holds the target
    (Windows, Dropbox), then raises.

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
    PermissionError
        If the target stays held by another process.
    """
    if isinstance(filepath, str):
        filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp = _write_to_temp(df, filepath, file_metadata=file_metadata, **kwargs)
    _replace(temp, filepath)


def _write_parquet(df, filepath: Path, *, file_metadata=None, **kwargs) -> None:
    """Write *df* to exactly *filepath* (see `to_parquet`)."""
    df = _categoricals_to_string(df)

    if not isinstance(df, gpd.GeoDataFrame) and set(kwargs) <= {'index'}:
        # Every attribute table records which recipe roots the build
        # read from (bundled, at which version or commit, plus any
        # installed recipe package): auto-discovery picks the most
        # specific recipe those roots hold, so a file is not
        # reproducible without them. Geometry sidecars are written by
        # geopandas, which takes no footer, and stay as they are.
        from openplaces.path import RECIPE_ROOTS_METADATA_KEY, recipe_roots_footer

        # Per-column digests let a job tell, from the footer alone,
        # whether the columns it reads changed (`file_digests`).
        file_metadata = {
            RECIPE_ROOTS_METADATA_KEY: recipe_roots_footer(),
            COLUMN_DIGESTS_KEY: json.dumps(
                frame_digests(df, index=kwargs.get('index'))
            ),
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

    refuse_unredacted_share_write(filepath)
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

    refuse_unredacted_share_write(filepath)
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

    refuse_unredacted_share_write(filepath)
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
        # Every file of the set is written under its temporary name
        # first and only then moved into place, so the attribute file
        # and its `_geo` sidecar change together as closely as the file
        # system allows. It is not one atomic step: a crash between the
        # moves leaves a new attribute file beside the old sidecar.
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        if join_id_column == gdf.index.name:
            gdf_geo = gdf.reset_index()
        else:
            gdf_geo = gdf
        mask_unique_join_id = ~gdf_geo[join_id_column].duplicated()
        writes = [
            (
                gdf[[v for v in gdf if v != 'geometry']],
                parquet_path,
                {'file_metadata': file_metadata},
            ),
            (
                gdf_geo[mask_unique_join_id][[join_id_column, 'geometry']],
                parquet_path.with_stem(parquet_path.stem + '_geo'),
                {
                    'index': False,
                    'schema_version': '1.1.0',
                    'write_covering_bbox': True,
                },
            ),
        ]
        if simplified_geometry is not None:
            gdf_geo_simp = gdf_geo[[join_id_column]].copy()
            gdf_geo_simp['geometry'] = simplified_geometry.values
            gdf_geo_simp = gpd.GeoDataFrame(gdf_geo_simp, crs=gdf.crs)
            writes.append(
                (
                    gdf_geo_simp[mask_unique_join_id][[join_id_column, 'geometry']],
                    parquet_path.with_stem(parquet_path.stem + '_geo_simplified'),
                    {
                        'index': False,
                        'schema_version': '1.1.0',
                        'write_covering_bbox': True,
                    },
                )
            )
        writes[0][2]['file_metadata'] = {
            SIDECAR_DIGESTS_KEY: json.dumps(
                {
                    target.stem[len(parquet_path.stem) :]: frame_digests(
                        frame, index=False
                    )
                    for frame, target, _ in writes[1:]
                }
            ),
            **(file_metadata or {}),
        }
        temps = []
        try:
            for frame, target, kwargs in writes:
                temps.append((_write_to_temp(frame, target, **kwargs), target))
        except BaseException:
            for temp, _ in temps:
                temp.unlink(missing_ok=True)
            raise
        for temp, target in temps:
            _replace(temp, target)
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


@_own_read
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
    if _READ_LOG is not None:
        filtered = _filter_columns(filters)
        requested = kwargs.get('columns')
        _log_read(
            parquet_path,
            None if requested is None or filtered is None else [*requested, *filtered],
        )
        if geom:
            suffix = '_geo_simplified' if geom == 'simplified' else '_geo'
            sidecar = parquet_path.with_stem(parquet_path.stem + suffix)
            if sidecar.exists():
                _log_read(sidecar, None)

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
        if geom or bbox is not None:
            _log_read(parquet_path, None)
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
        _log_read(parquet_path, [join_id_column])

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
