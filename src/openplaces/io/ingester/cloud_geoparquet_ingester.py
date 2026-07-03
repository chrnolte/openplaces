"""
Helper functions for the `latlon_tile` partition mode in `Ingester`.
"""

import math
import xml.etree.ElementTree as ET

import geopandas as gpd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import requests

from openplaces.io.readers import get_admin


def tile_str(lat: int, lon: int) -> str:
    return f'lat{lat:+04d}_lon{lon:+05d}'


def tile_bounds(
    tile_id: str, tile_size_deg: float
) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) in EPSG:4326 for a tile_id string."""
    lat = int(tile_id[3:7])
    lon = int(tile_id[11:])
    return (lon, lat, lon + tile_size_deg, lat + tile_size_deg)


def tile_ids_for_admin(admin_id: str, tile_size_deg: float = 1.0) -> list[str]:
    """Return all tile IDs whose bbox overlaps the admin polygon's bounding box."""
    admin = get_admin(admin_id, geom=True).to_crs('EPSG:4326')
    minx, miny, maxx, maxy = admin.total_bounds
    tiles = []
    for lat in range(math.floor(miny / tile_size_deg), math.ceil(maxy / tile_size_deg)):
        for lon in range(
            math.floor(minx / tile_size_deg), math.ceil(maxx / tile_size_deg)
        ):
            tiles.append(tile_str(lat, lon))
    return tiles


class _HTTPRangeFile:
    """Seekable file-like object backed by HTTP Range requests.

    PyArrow's ParquetFile reads the footer (end of file) then individual
    row groups via seek+read — exactly what HTTP Range requests support.
    """

    def __init__(self, url: str, session):
        self._url = url
        self._session = session
        self._pos = 0
        self._size: int | None = None

    def read(self, n: int = -1) -> bytes:
        size = self._get_size()
        start = self._pos
        end = (size - 1) if n < 0 else min(self._pos + n - 1, size - 1)
        if start > end:
            return b''
        resp = self._session.get(self._url, headers={'Range': f'bytes={start}-{end}'})
        resp.raise_for_status()
        data = resp.content
        self._pos = start + len(data)
        return data

    def seek(self, pos: int, whence: int = 0) -> int:
        size = self._get_size()
        if whence == 0:
            self._pos = pos
        elif whence == 1:
            self._pos += pos
        elif whence == 2:
            self._pos = size + pos
        self._pos = max(0, min(self._pos, size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return False

    def _get_size(self) -> int:
        if self._size is None:
            r = self._session.head(self._url)
            r.raise_for_status()
            self._size = int(r.headers['Content-Length'])
        return self._size


def _resolve_s3_latest_release(url: str, region: str | None = None) -> str:
    """Replace a literal 'release/latest/' path segment with the newest
    dated release folder actually present in the bucket.

    Buckets such as Overture's rotate out old dated releases, so a release
    date pinned in a recipe eventually points at deleted data.
    """
    if '/release/latest/' not in url:
        return url

    from urllib.parse import urlparse

    parsed = urlparse(url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/').split('release/latest/')[0] + 'release/'
    region = region or 'us-east-1'
    base = f'https://{bucket}.s3.{region}.amazonaws.com'

    resp = requests.get(f'{base}/?list-type=2&prefix={prefix}&delimiter=/')
    resp.raise_for_status()
    _ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    root = ET.fromstring(resp.content)
    releases = [
        cp.find('s3:Prefix', _ns).text.rstrip('/').rsplit('/', 1)[-1]
        for cp in root.findall('s3:CommonPrefixes', _ns)
    ]
    if not releases:
        raise ValueError(f'No releases found under s3://{bucket}/{prefix}')

    def _release_key(name: str) -> tuple[str, int]:
        date_part, _, version_part = name.partition('.')
        return (date_part, int(version_part or 0))

    latest = max(releases, key=_release_key)
    return url.replace('/release/latest/', f'/release/{latest}/')


def fetch_latlon_tile_to_cache(
    download_url: str,
    tile_id: str,
    tile_size_deg: float,
    bbox_column: str,
    cache_path,
    s3_anonymous: bool = False,
    s3_region: str | None = None,
    redownload: bool = False,
    verbose: bool = False,
) -> None:
    """Fetch a bbox-filtered slice of a cloud GeoParquet dataset and save to cache.

    Parameters
    ----------
    download_url
        S3 or https base path to the parquet dataset (directory or single file).
    tile_id
        Tile identifier string (e.g. 'lat+035_lon-077').
    tile_size_deg
        Tile size in degrees.
    bbox_column
        Name of the parquet column containing bbox struct fields
        ``xmin``, ``ymin``, ``xmax``, ``ymax``.
    cache_path
        Local path to write the cached GeoParquet tile.
    s3_anonymous
        If True, use anonymous S3 credentials for public buckets.
    s3_region
        AWS region hint (e.g. 'us-west-2').
    redownload
        If True, overwrite existing cache.
    verbose
        Print progress messages.
    """
    from pathlib import Path

    cache_path = Path(cache_path)
    if cache_path.exists() and not redownload:
        return

    if download_url.startswith('s3://'):
        download_url = _resolve_s3_latest_release(download_url, s3_region)

    minx, miny, maxx, maxy = tile_bounds(tile_id, tile_size_deg)
    col = bbox_column

    bbox_filter = (
        (pc.field(col, 'xmin') <= maxx)
        & (pc.field(col, 'ymin') <= maxy)
        & (pc.field(col, 'xmax') >= minx)
        & (pc.field(col, 'ymax') >= miny)
    )

    if verbose:
        print(f'Reading tile {tile_id} from {download_url} ...')

    if download_url.startswith('s3://'):
        table = _read_s3_parquet(download_url, bbox_filter, s3_region, verbose)
    else:
        filesystem, path = pafs.FileSystem.from_uri(download_url.rstrip('/'))
        infos = filesystem.get_file_info(pafs.FileSelector(path, recursive=True))
        paths = [
            i.path
            for i in infos
            if i.type == pafs.FileType.File and i.path.endswith('.parquet')
        ]
        table = ds.dataset(paths, filesystem=filesystem, format='parquet').to_table(
            filter=bbox_filter
        )

    if len(table) == 0:
        if verbose:
            print(f'  No rows in tile {tile_id} — skipping cache write.')
        return

    df = table.to_pandas()
    if 'geometry' in df.columns:
        geometry = gpd.GeoSeries.from_wkb(df['geometry'], crs='EPSG:4326')
        gdf = gpd.GeoDataFrame(
            df.drop(columns=['geometry']), geometry=geometry, crs='EPSG:4326'
        )
    else:
        gdf = gpd.GeoDataFrame.from_arrow(table)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(cache_path, schema_version='1.1.0')
    if verbose:
        print(f'  Cached {len(gdf):,d} rows → {cache_path.name}')


def _read_s3_parquet(url, bbox_filter, region, verbose):
    """Read S3 parquet files via HTTPS range requests with row-group filtering."""
    from urllib.parse import quote, urlparse

    parsed = urlparse(url)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    region = region or 'us-east-1'
    base = f'https://{bucket}.s3.{region}.amazonaws.com'

    list_url = f'{base}/?list-type=2&prefix={quote(prefix, safe="/")}'
    resp = requests.get(list_url)
    resp.raise_for_status()
    _ns = 'http://s3.amazonaws.com/doc/2006-03-01/'
    keys = [
        e.text
        for e in ET.fromstring(resp.content).iter(f'{{{_ns}}}Key')
        if e.text and e.text.endswith('.parquet')
    ]

    session = requests.Session()
    parts = []
    for key in keys:
        table = pq.read_table(
            _HTTPRangeFile(f'{base}/{key}', session),
            filters=bbox_filter,
            use_pandas_metadata=False,
        )
        if len(table):
            parts.append(table)

    return pa.concat_tables(parts) if parts else pa.table({})
