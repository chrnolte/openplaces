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

Two optional modes exist for the common case where an agency publishes a
bulk export that is cheaper to fetch but poorer than its live service:

`bulk_url`
    Fetch a ready-made export (the agency's own GeoJSON download) instead
    of paging geometry out of the service. One request rather than
    hundreds.

`attribute_join`
    Page a REST layer for ATTRIBUTES ONLY and join them onto the features
    by a shared key. Geometry is what makes these services fall over --
    measured on DeKalb County, 2,000 attribute-only rows return in ~0.4s
    at any offset, while 500 rows carrying polygons and 95 columns kill
    the connection at a 60s server-side timeout. Pulling only the columns
    the bulk export lacks turns a multi-hour geometry scrape into about a
    minute of light queries.

Together they let a recipe take geometry from the sanctioned bulk file and
top it up with the handful of fields that exist only on the live service.
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


def _download_bulk(
    url: str, target: Path, *, timeout: float, retries: int, verbose: bool, label: str
) -> None:
    """Stream an agency's ready-made export to `target`.

    Chunked so a multi-hundred-MB GeoJSON never lands in memory.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                n_bytes = 0
                with target.open('wb') as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
                        n_bytes += len(chunk)
            if verbose:
                _log(f'{label}: bulk download {n_bytes / 1e6:.1f}MB -> {target.name}')
            return
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if verbose:
                more = 'retrying...' if attempt < retries else 'giving up'
                _log(f'{label}: bulk attempt {attempt}/{retries} failed, {more}')
            if attempt < retries:
                time.sleep(2**attempt)
    raise last_exc


def _fetch_attribute_table(
    layer_url: str,
    *,
    key: str,
    fields: list,
    where: str,
    page_size: int,
    timeout: float,
    retries: int,
    verbose: bool,
    label: str,
) -> dict:
    """Page a layer for attributes only, keyed by `key`.

    `returnGeometry=false` with a short `outFields` list is what makes this
    viable on services that cannot serve geometry in bulk.
    """
    query_url = f'{layer_url.rstrip("/")}/query'
    params = {
        'where': where,
        'outFields': ','.join([key, *fields]),
        'returnGeometry': 'false',
        'f': 'json',
    }
    total = _layer_count(
        query_url,
        params,
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=label,
    )
    table = {}
    for offset in range(0, total, page_size):
        page = _get_json(
            query_url,
            {**params, 'resultOffset': offset, 'resultRecordCount': page_size},
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=f'{label}: attributes',
        )
        for feature in page.get('features', []):
            attributes = feature.get('attributes') or {}
            key_value = attributes.get(key)
            if key_value is not None:
                table[key_value] = {f: attributes.get(f) for f in fields}
        if verbose:
            _log(f'{label}: joined {len(table)} attribute rows so far')
    return table


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


def _apply_attribute_join(
    path: Path,
    table: dict,
    *,
    key: str,
    fields: list,
    min_match: float,
    verbose: bool,
    label: str,
) -> None:
    """Merge an attribute table into the vector file at `path`, in place.

    Read through pyogrio rather than `json` -- it parses at C level, which
    matters on the hundred-MB exports this mode exists to serve.
    """
    # Imported lazily so the plain paging path stays a requests-only
    # module with no geo stack to load.
    import geopandas as gpd
    import pandas as pd

    gdf = gpd.read_file(path)
    if key not in gdf.columns:
        raise ValueError(
            f'{label}: attribute_join key {key!r} is not a column of the '
            f'downloaded file. Available: {sorted(gdf.columns)[:20]}'
        )

    attributes = pd.DataFrame.from_dict(table, orient='index')
    attributes.index.name = key
    attributes = attributes.reset_index()

    # A key whose dtype differs between the two sides silently matches
    # nothing -- a space-padded string parcel key on one side against an
    # integer on the other. Align on string form, which is what these
    # assessor keys always are.
    if gdf[key].dtype != attributes[key].dtype:
        gdf[key] = gdf[key].astype('string')
        attributes[key] = attributes[key].astype('string')

    n_matched = int(gdf[key].isin(set(attributes[key])).sum())
    match_rate = n_matched / len(gdf) if len(gdf) else 0.0
    if match_rate < min_match:
        raise RuntimeError(
            f'{label}: attribute join matched only {n_matched:,} of '
            f'{len(gdf):,} features ({match_rate:.1%}), below the required '
            f'{min_match:.0%}. Check that {key!r} is the right join key and '
            'that both sides use the same identifier form.'
        )

    # Never let joined columns silently overwrite the bulk file's own.
    collisions = [f for f in fields if f in gdf.columns]
    if collisions:
        raise ValueError(
            f'{label}: attribute_join fields already exist in the '
            f'downloaded file: {collisions}. Rename or drop them.'
        )

    merged = gdf.merge(attributes, on=key, how='left')
    merged.to_file(path, driver='GeoJSON')
    if verbose:
        _log(
            f'{label}: joined {len(fields)} column(s) onto '
            f'{n_matched:,}/{len(gdf):,} features ({match_rate:.1%})'
        )


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
    skipped: list,
) -> list:
    """Fetch features [offset, offset+count) with automatic bisection.

    Some county-hosted services return a 400 for one specific record inside
    an otherwise fine page (e.g. Robeson County: a working service that
    fails only on `resultOffset=50000`, no error detail given) rather than
    for a data range as a whole. A failed range is bisected and each half
    retried; a single unrecoverable record (count == 1) is skipped, its
    offset recorded in `skipped`, rather than aborting the whole download.

    Each range gets the caller's full `retries` budget (with the backoff in
    `_get_json`) BEFORE any bisection. This distinction matters on flaky
    servers: DeKalb County's ArcGIS Server intermittently drops heavy
    requests at a 60s server-side timeout, with the same range succeeding
    on a later attempt. Bisecting on the first failure treats that
    transient timeout as a permanent bad record, halving down to single
    records and skipping each -- which silently returned 1,789 of 246,063
    features on a run that reported success.
    """
    try:
        page = _get_json(
            query_url,
            {**params, 'resultOffset': offset, 'resultRecordCount': count},
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=label,
        )
        return page.get('features', [])
    except (requests.exceptions.RequestException, RuntimeError, ValueError) as exc:
        if count <= 1:
            skipped.append(offset)
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
            skipped=skipped,
        ) + _fetch_range(
            query_url,
            params,
            offset + left,
            count - left,
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=label,
            skipped=skipped,
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
    page_size=None,
    allow_partial=False,
    bulk_url=None,
    attribute_join=None,
    join_min_match=0.9,
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
        underlying error propagate. Each range gets this full budget
        before bisection begins.
    page_size : int, optional
        Features per request. Defaults to the layer's own
        `maxRecordCount`, capped at 2,000. Lower it for services that
        drop heavy requests: DeKalb County served 250-feature pages in
        ~10s but closed the connection on 2,000-feature pages.
    allow_partial : bool, default False
        Save the file even when fewer features were retrieved than the
        service reported. Off by default so a short download fails loudly
        instead of yielding a plausible but truncated layer.
    bulk_url : str, optional
        URL of a ready-made vector export to fetch instead of paging the
        layer for geometry. Use the agency's own published download where
        one exists; it is one request rather than hundreds, and it is the
        access route the agency intends.
    attribute_join : dict, optional
        Enrich the downloaded features with columns pulled attribute-only
        from a REST layer. Keys:
        `layer_url` (defaults to the scraper's own `layer_url`), `key`
        (join field, must exist on both sides), `fields` (list of columns
        to add), and optionally `where` and `page_size` for that query.
        Geometry is never requested, which is what keeps it cheap.
    join_min_match : float, default 0.9
        Minimum share of features the join must match. Below this the
        fetch raises rather than returning mostly-null columns, since a
        wrong key silently produces exactly that.

    Raises
    ------
    RuntimeError
        If fewer features were written than the layer reported and
        `allow_partial` is False.

    Returns
    -------
    pathlib.Path
        Path to the written GeoJSON file (equal to `target_path`).
    """
    if not layer_url and not bulk_url:
        raise ValueError(
            "arcgis_rest_scraper requires 'layer_url' or 'bulk_url' "
            '(recipe scraper_options).'
        )
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = label or 'arcgis_rest_scraper'

    if target_path.exists() and not redownload:
        if verbose:
            _log(f'{prefix}: already downloaded ({target_path.name})')
        return target_path

    if bulk_url:
        # The agency's own export: one request, and the access route it
        # publishes for whole-layer use.
        part_path = target_path.with_name(target_path.name + '.part')
        try:
            _download_bulk(
                bulk_url,
                part_path,
                timeout=timeout,
                retries=retries,
                verbose=verbose,
                label=prefix,
            )
        except BaseException:
            part_path.unlink(missing_ok=True)
            raise
        part_path.replace(target_path)
        _maybe_attribute_join(
            target_path,
            attribute_join,
            default_layer_url=layer_url,
            join_min_match=join_min_match,
            timeout=timeout,
            retries=retries,
            verbose=verbose,
            label=prefix,
        )
        return target_path

    if page_size is None:
        page_size = _layer_page_size(
            layer_url, timeout=timeout, retries=retries, verbose=verbose, label=prefix
        )
    page_size = max(1, int(page_size))
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

    # Stream each page straight to disk rather than accumulating the
    # whole layer. json.dumps() on a full FeatureCollection builds one
    # file-sized str beside the already-resident feature list, so peak
    # memory was ~2x the output in a single allocation; a 250k-parcel
    # layer with polygon geometry runs to hundreds of MB. Writing per
    # feature keeps peak at one page.
    part_path = target_path.with_name(target_path.name + '.part')
    n_written = 0
    skipped: list[int] = []
    try:
        with part_path.open('w', encoding='utf-8') as handle:
            handle.write('{"type": "FeatureCollection", "features": [')
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
                    skipped=skipped,
                )
                for feature in batch:
                    # Separator driven by the running written count, not
                    # the page index: a bisected range can legitimately
                    # come back empty, which would emit '[,'.
                    if n_written:
                        handle.write(',')
                    handle.write(json.dumps(feature, separators=(',', ':')))
                    n_written += 1
                if verbose:
                    _log(f'{prefix}: fetched {n_written} features so far')
            handle.write(']}')
    except BaseException:
        part_path.unlink(missing_ok=True)
        raise

    # Refuse to hand back a silently short download. Skipping an
    # unfetchable record is a deliberate escape hatch for one bad row,
    # but without this check a flaky server turns it into wholesale data
    # loss that looks like success: DeKalb County once yielded 1,789 of
    # 246,063 features and still exited 0, writing a plausible parquet.
    if n_written < total and not allow_partial:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(
            f'{prefix}: incomplete download -- wrote {n_written:,} of '
            f'{total:,} features ({n_written / total:.1%}), '
            f'{len(skipped):,} record(s) skipped as unfetchable.\n'
            'The service reported more features than it served, so the file '
            'was discarded rather than saved as a partial layer. Retry; if '
            'it persists, lower `page_size` or raise `timeout` in the '
            "recipe's scraper_options, or pass allow_partial=True to accept "
            'a known-incomplete extract.'
        )

    part_path.replace(target_path)
    if verbose:
        _log(f'{prefix}: wrote {n_written} features -> {target_path.name}')
        if skipped:
            _log(f'{prefix}: {len(skipped)} record(s) skipped as unfetchable')

    _maybe_attribute_join(
        target_path,
        attribute_join,
        default_layer_url=layer_url,
        join_min_match=join_min_match,
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=prefix,
    )
    return target_path


def _maybe_attribute_join(
    target_path: Path,
    attribute_join: dict | None,
    *,
    default_layer_url: str | None,
    join_min_match: float,
    timeout: float,
    retries: int,
    verbose: bool,
    label: str,
) -> None:
    """Pull and apply an attribute-only join, if the recipe asked for one."""
    if not attribute_join:
        return

    join_layer = attribute_join.get('layer_url') or default_layer_url
    key = attribute_join.get('key')
    fields = list(attribute_join.get('fields') or [])
    if not join_layer or not key or not fields:
        raise ValueError(
            f'{label}: attribute_join needs `key`, `fields`, and a '
            '`layer_url` (or the scraper-level layer_url).'
        )

    join_page_size = max(1, int(attribute_join.get('page_size') or DEFAULT_PAGE_SIZE))
    table = _fetch_attribute_table(
        join_layer,
        key=key,
        fields=fields,
        where=attribute_join.get('where', '1=1'),
        page_size=join_page_size,
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        label=label,
    )
    _apply_attribute_join(
        target_path,
        table,
        key=key,
        fields=fields,
        min_match=float(join_min_match),
        verbose=verbose,
        label=label,
    )
