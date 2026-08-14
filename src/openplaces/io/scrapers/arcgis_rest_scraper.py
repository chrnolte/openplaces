"""
Generic downloader for a single ArcGIS REST (FeatureServer/MapServer) layer.
Not tied to any recipe or dataset: the layer's query endpoint
(`scraper_options.layer_url`) is supplied by the recipe.

County-hosted parcel layers are commonly capped at 1,000-2,000 features per
query (the service's `maxRecordCount`), well below a county's parcel count.
This scraper reads that cap from the layer's metadata, pages through the
layer with `resultOffset`/`resultRecordCount` until exhausted, and writes the
combined result as a single GeoJSON `FeatureCollection` to `target_path` --
letting the recipe's normal `columns:` mapping read it like any other vector
file. Esri's `f=geojson` output is always reprojected to WGS84 (EPSG:4326)
server-side regardless of the layer's native spatial reference, so the
combined file needs no separate reprojection step.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import requests

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3
DEFAULT_PAGE_SIZE = 2000


def _log(message: str) -> None:
    print(f'  {message}')


def _get_json(
    url: str, params: dict, *, timeout: float, retries: int, verbose: bool, label: str
) -> dict:
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get('error'):
                raise RuntimeError(f'{label}: server returned {data["error"]}')
            return data
        except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
            last_exc = exc
            if verbose:
                more = 'retrying...' if attempt < retries else 'giving up'
                _log(f'{label}: attempt {attempt}/{retries} failed ({exc}), {more}')
            if attempt < retries:
                time.sleep(2**attempt)
    raise last_exc


def _layer_page_size(
    layer_url: str, *, timeout: float, retries: int, verbose: bool, label: str
) -> int:
    """Return the layer's `maxRecordCount`, capped at `DEFAULT_PAGE_SIZE`.

    Some on-prem ArcGIS Server instances report a `maxRecordCount` far above
    what a single query actually returns; capping keeps requests a
    predictable size regardless.
    """
    meta = _get_json(
        layer_url,
        {'f': 'json'},
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=f'{label}: layer metadata',
    )
    max_record_count = meta.get('maxRecordCount') or DEFAULT_PAGE_SIZE
    return min(int(max_record_count), DEFAULT_PAGE_SIZE)


def _layer_count(
    query_url: str,
    params: dict,
    *,
    timeout: float,
    retries: int,
    verbose: bool,
    label: str,
) -> int:
    count_page = _get_json(
        query_url,
        {**params, 'returnCountOnly': 'true'},
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=f'{label}: record count',
    )
    return int(count_page['count'])


def _fetch_range(
    query_url: str,
    params: dict,
    offset: int,
    count: int,
    *,
    timeout: float,
    retries: int,
    verbose: bool,
    label: str,
) -> list:
    """Fetch features [offset, offset+count) with automatic bisection.

    Some county-hosted services return a 400 for one specific record inside
    an otherwise fine page (e.g. Robeson County: a working service that
    fails only on `resultOffset=50000`, no error detail given) rather than
    for a data range as a whole. A failed range is bisected and each half
    retried; a single unrecoverable record (count == 1) is skipped with a
    warning rather than aborting the whole download.
    """
    try:
        page = _get_json(
            query_url,
            {**params, 'resultOffset': offset, 'resultRecordCount': count},
            timeout=timeout,
            retries=1,
            verbose=False,
            label=label,
        )
        return page.get('features', [])
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
        if count <= 1:
            warnings.warn(
                f'{label}: skipping unfetchable record at offset {offset} ({exc})'
            )
            return []
        if verbose:
            _log(
                f'{label}: range [{offset}, {offset + count}) failed, '
                'bisecting to isolate the bad record'
            )
        left = count // 2
        return _fetch_range(
            query_url,
            params,
            offset,
            left,
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=label,
        ) + _fetch_range(
            query_url,
            params,
            offset + left,
            count - left,
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=label,
        )


def fetch(
    partition_id=None,
    target_path=None,
    portal_url=None,
    *,
    admin_id_to_download=None,
    layer_url=None,
    where='1=1',
    out_fields='*',
    extra_params=None,
    label=None,
    redownload=False,
    verbose=False,
    timeout=DEFAULT_TIMEOUT,
    retries=DEFAULT_RETRIES,
) -> Path:
    """Download every feature of one ArcGIS REST layer to a GeoJSON file.

    Parameters
    ----------
    partition_id : optional
        Unused (this scraper is for single-file, unpartitioned recipes).
        Accepted for interface compatibility with
        `Ingester._run_download_scraper`'s shared `fetch` contract.
    target_path : str or pathlib.Path
        Where to write the combined GeoJSON `FeatureCollection`.
    portal_url : str, optional
        Unused; the layer's own query endpoint is `layer_url`. Accepted for
        interface compatibility with the shared `fetch` contract.
    admin_id_to_download : optional
        Unused (this scraper is for single-file, unpartitioned recipes).
    layer_url : str
        Base URL of the FeatureServer/MapServer layer, without a trailing
        `/query` (e.g. `'.../FeatureServer/0'`).
    where : str, default '1=1'
        SQL `WHERE` clause passed to every query (the default selects every
        feature).
    out_fields : str, default '*'
        Comma-separated field list, or `'*'` for every field.
    extra_params : dict, optional
        Additional query-string parameters merged into every page request
        (e.g. `{'outSR': 4326}`).
    label : str, optional
        Human-readable prefix for progress messages.
    redownload : bool, default False
        Re-fetch every page even if `target_path` already exists.
    verbose : bool, default False
        Print progress messages.
    timeout : float, default 60.0
        Seconds to wait on each page request before treating it as failed
        and retrying.
    retries : int, default 3
        Attempts per page request before giving up and letting the
        underlying error propagate.

    Returns
    -------
    pathlib.Path
        Path to the written GeoJSON file (equal to `target_path`).
    """
    if not layer_url:
        raise ValueError(
            "arcgis_rest_scraper requires 'layer_url' (recipe scraper_options)."
        )
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = label or 'arcgis_rest_scraper'

    if target_path.exists() and not redownload:
        if verbose:
            _log(f'{prefix}: already downloaded ({target_path.name})')
        return target_path

    page_size = _layer_page_size(
        layer_url, timeout=timeout, retries=retries, verbose=verbose, label=prefix
    )
    query_url = f'{layer_url.rstrip("/")}/query'
    params = {'where': where, 'outFields': out_fields, 'f': 'geojson'}
    if extra_params:
        params.update(extra_params)

    total = _layer_count(
        query_url,
        params,
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=prefix,
    )
    features = []
    for offset in range(0, total, page_size):
        batch = _fetch_range(
            query_url,
            params,
            offset,
            min(page_size, total - offset),
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=prefix,
        )
        features.extend(batch)
        if verbose:
            _log(f'{prefix}: fetched {len(features)} features so far')

    part_path = target_path.with_name(target_path.name + '.part')
    part_path.write_text(
        json.dumps({'type': 'FeatureCollection', 'features': features})
    )
    part_path.replace(target_path)
    if verbose:
        _log(f'{prefix}: wrote {len(features)} features -> {target_path.name}')
    return target_path
