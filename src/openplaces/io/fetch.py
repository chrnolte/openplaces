"""
Fetch a source file over HTTP: the request headers that identify
openplaces to a server, the download itself (resumable, with the
extension sniffed from the content when the URL does not say), and
the content-type table behind that.
"""

import re
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from tqdm import tqdm

from openplaces.config import cfg

_CONTENT_TYPE_EXT = {
    'application/geo+json': '.geojson',
    'application/json': '.geojson',  # ambiguous but common for GeoJSON APIs
    'application/geopackage+sqlite3': '.gpkg',
    'application/x-sqlite3': '.gpkg',
    'application/zip': '.zip',
    'application/x-zip-compressed': '.zip',
    'application/vnd.apache.parquet': '.parquet',
}


def _content_type_to_ext(content_type: str) -> str | None:
    """Return file extension for a Content-Type header value, or None."""
    mime = content_type.split(';')[0].strip().lower()
    return _CONTENT_TYPE_EXT.get(mime)


def request_headers(extra: dict | None = None) -> dict:
    """Headers for every outbound request openplaces makes.

    Carries `cfg.user_agent`, which names the project, links its page, and
    reports the operator's chosen nickname and place. Use this rather than
    letting a request go out under the bare `python-requests` agent: some
    servers reject that outright, and a named agent gives an operator
    seeing unexpected load someone to contact.

    Parameters
    ----------
    extra : dict, optional
        Additional headers, merged over the defaults.

    Returns
    -------
    dict
        Header mapping to pass to requests or urllib.
    """
    headers = {'User-Agent': cfg.user_agent}
    if extra:
        headers.update(extra)
    return headers


def download(
    from_url,
    to_path,
    chunk_size=8192,
    timeout=None,
    verify_ssl=True,
    headers=None,
):
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
    headers : dict, optional
        Extra request headers, merged over the defaults. Needed for APIs
        that content-negotiate rather than taking a format in the query
        string: Wikidata's SPARQL endpoint ignores `format=csv` outright
        and serves XML unless asked for CSV in an `Accept` header.

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

    # A recipe may add or override headers via `download_headers`, for
    # sources that reject anything but a specific agent.
    headers = request_headers(headers)

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
        filename = ''

        # Try Content-Disposition header first if response provided
        if response and 'content-disposition' in response.headers:
            content_disp = response.headers['content-disposition']
            if 'filename=' in content_disp:
                filename_part = content_disp.split('filename=')[1]
                filename = unquote(filename_part.split(';')[0].strip('"\''))

        if not filename:
            # The fallback fires whenever no name was extracted, not
            # only when the header was absent: a Content-Disposition
            # carrying no filename= would otherwise leave `to_path`
            # pointing at the directory itself, and the suffix sniffing
            # below would rename that directory.
            parsed = urlparse(from_url)
            filename = Path(unquote(parsed.path)).name

        to_path /= filename or 'download'

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

    # Download to temp location first, then move to final destination.
    # The staging directory is unique per call: keyed on the basename
    # alone, two concurrent jobs fetching per-county archives that share
    # a name (parcels.zip) wrote the same file and each moved a
    # half-written mix into place.
    temp_dir = Path(tempfile.mkdtemp(prefix='openplaces-download-'))
    temp_path = temp_dir / f'{to_path.name}.part'

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

    finally:
        # Removes a partial download on error, and the empty staging
        # directory on success.
        shutil.rmtree(temp_dir, ignore_errors=True)

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


# Archive members that unzip() opens in turn instead of leaving in the
# output directory. Only .zip: a .kmz is a KML container GDAL reads as
# it is, and a .jar is never data. No recipe reads a .zip member in
# place (checked over the recipe tree on 2026-09-16), so extracting
# them cannot take away a file a recipe names.
