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

A third, independent mode (`layer_url`'s `{admin_key}` placeholder,
with `admin_key_column`/`admin_key_transform`) is for a statewide
source published as one genuinely separate FeatureServer per admin
unit -- not one shared service filtered by an in-data column --
e.g. one service per county, named after the county.
`download_by: {admin_level: N}` then drives one `fetch()` call per
admin unit, each resolving its own URL.

A fourth mode (`where_admin_column`, paired with the same
`admin_key_column`/`admin_key_transform`) is the complementary case:
one *shared* service for every admin unit, distinguished only by an
in-data filter column (e.g. a statewide parcel layer with its own
`countyfips` field, and a `maxRecordCount` too low to page the whole
state in one pass). `fetch` builds the `where` clause itself from the
current admin unit rather than a recipe needing a dedicated wrapper
module per source.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import requests

from openplaces.io import request_headers
from openplaces.io.readers import get_admin

DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 3
DEFAULT_PAGE_SIZE = 2000

# Minimum gap between two successful requests to the same service. Paging a
# county parcel layer is hundreds of queries against one server that is
# usually a single machine in a county IT room, and backing off only after
# a failure means the load that caused the failure was already applied.
# 0.2s costs about 20 seconds over a 100-page scrape and keeps openplaces
# below the rate at which these services start shedding requests.
DEFAULT_REQUEST_INTERVAL_S = 0.2

_last_request_at = 0.0


def _pace(interval_s: float) -> None:
    """Sleep until *interval_s* has passed since the last request.

    Module-level rather than per-call state: the point is to bound the rate
    openplaces hits one server across a whole paging loop, not within a
    single function.
    """
    global _last_request_at
    wait = interval_s - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _log(message: str) -> None:
    print(f'  {message}')


def _resolve_admin_key(
    admin_id_to_download: str,
    admin_key_column: str,
    admin_key_transform: str | None,
):
    """Look up one admin unit's own value of `admin_key_column`.

    Shared by `{admin_key}` URL substitution (`_resolve_layer_url`) and
    `where_admin_column` filter-building (`fetch`) -- both need the same
    "what is this admin unit called/coded in the source" lookup, just to
    plug the result into a different place (a URL segment vs. a SQL
    `where` clause). Mirrors `Ingester._get_admin_partition_key`'s own
    `{adminN_name}`-style resolution rather than reimplementing it.
    """
    from openplaces.core.schema import AdminId

    admin_id = AdminId(admin_id_to_download)
    key = get_admin(admin_id, admin_id.get_level(), columns=admin_key_column).iloc[0, 0]
    if admin_key_transform == 'remove_spaces':
        return key.replace(' ', '')
    if admin_key_transform:
        raise NotImplementedError(
            f'admin_key_transform == {admin_key_transform!r} not supported.'
        )
    return key


def _resolve_layer_url(
    layer_url: str,
    *,
    admin_id_to_download: str | None,
    admin_key_column: str | None,
    admin_key_transform: str | None,
) -> str:
    """Substitute an optional `{admin_key}` placeholder in `layer_url`.

    Lets one recipe fan out over N admin units that each live on a
    genuinely different REST service (not a shared service filtered
    by an in-data column) -- e.g. one FeatureServer per county,
    named after the county.
    """
    if '{admin_key}' not in layer_url:
        return layer_url
    if not (admin_id_to_download and admin_key_column):
        raise ValueError(
            "'{admin_key}' in layer_url requires both admin_id_to_download "
            'and scraper_options.admin_key_column.'
        )
    key = _resolve_admin_key(
        admin_id_to_download, admin_key_column, admin_key_transform
    )
    return layer_url.format(admin_key=key)


def _resolve_where(
    where: str,
    *,
    admin_id_to_download: str | None,
    admin_key_column: str | None,
    admin_key_transform: str | None,
    where_admin_column: str | None,
) -> str:
    """Fold an admin unit's own filter value into `where`, if configured.

    Lets one recipe fan out over N admin units that all live on the
    *same* shared REST service, distinguished only by an in-data column
    (e.g. a statewide parcel layer with a `countyfips` field) -- the
    complementary shape to `{admin_key}` in `_resolve_layer_url`, which
    is for N genuinely different services instead. `where_admin_column`
    names that in-data column; the value to filter on is looked up the
    same way `{admin_key}` is. Any recipe-supplied `where` is combined
    with `AND` rather than replaced, so a source-side row filter (e.g.
    excluding a bad `LOT_TYPE`) still applies alongside the admin-unit
    filter.
    """
    if not where_admin_column:
        return where
    if not (admin_id_to_download and admin_key_column):
        raise ValueError(
            "'where_admin_column' requires both admin_id_to_download and "
            'scraper_options.admin_key_column.'
        )
    key = _resolve_admin_key(
        admin_id_to_download, admin_key_column, admin_key_transform
    )
    admin_clause = f"{where_admin_column} = '{key}'"
    if where and where != '1=1':
        return f'({where}) AND ({admin_clause})'
    return admin_clause


def _download_bulk(
    url: str, target: Path, *, timeout: float, retries: int, verbose: bool, label: str
) -> None:
    """Stream an agency's ready-made export to `target`.

    Chunked so a multi-hundred-MB GeoJSON never lands in memory.
    """
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            _pace(DEFAULT_REQUEST_INTERVAL_S)
            with requests.get(
                url, stream=True, timeout=timeout, headers=request_headers()
            ) as response:
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

    `key` may name one field or several; several are assembled into one
    string, the same way :func:`_composite_key` assembles the geometry
    side, so the two always agree.

    `returnGeometry=false` with a short `outFields` list is what makes this
    viable on services that cannot serve geometry in bulk.
    """
    query_url = f'{layer_url.rstrip("/")}/query'
    key_fields = _key_fields(key)
    params = {
        'where': where,
        'outFields': ','.join([*key_fields, *fields]),
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
            parts = [attributes.get(f) for f in key_fields]
            if any(part is None for part in parts):
                continue
            key_value = _KEY_PART_SEPARATOR.join(str(part) for part in parts)
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
            _pace(DEFAULT_REQUEST_INTERVAL_S)
            response = requests.get(
                url, params=params, timeout=timeout, headers=request_headers()
            )
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


# Column the join key is assembled in. Temporary: dropped before the
# file is written, so the source's own columns survive untouched.
_JOIN_KEY_COLUMN = '_join_key'

# Separator between the parts of a composite key. A unit separator
# cannot occur in an assessor identifier, so joining on it can never
# make two different (part, part) pairs collide the way a hyphen could
# ('1' + '2-3' against '1-2' + '3'). `pipe` renders it as a boundary
# like any other separator, which is one more reason to prefer `pipe`
# over `simple` here: `simple` deletes separators and would reintroduce
# exactly that ambiguity.
_KEY_PART_SEPARATOR = '\x1f'


def _key_fields(key) -> list:
    """Return the join key's source fields, from a name or a list of them."""
    if isinstance(key, str):
        return [key]
    return list(key)


def _composite_key(frame, fields: list, *, label: str, side: str):
    """Assemble the join key from *fields* of *frame*.

    A single field is used as-is, so a recipe naming one field behaves
    exactly as it did before composite keys existed.
    """
    missing = [f for f in fields if f not in frame.columns]
    if missing:
        raise ValueError(
            f'{label}: attribute_join key field(s) {missing} are not columns '
            f'of the {side}. Available: {sorted(frame.columns)[:20]}'
        )
    parts = [frame[f].astype('string').fillna('') for f in fields]
    key = parts[0]
    for part in parts[1:]:
        key = key + _KEY_PART_SEPARATOR + part
    return key


def _apply_attribute_join(
    path: Path,
    table: dict,
    *,
    key: str,
    fields: list,
    min_match: float,
    verbose: bool,
    label: str,
    key_conv: str | None = None,
) -> None:
    """Merge an attribute table into the vector file at `path`, in place.

    Read through pyogrio rather than `json` -- it parses at C level, which
    matters on the hundred-MB exports this mode exists to serve.

    With *key_conv*, both sides are compared through
    :func:`~openplaces.geo.ids.convert_parcel_id` instead of literally.
    Two submissions of the same parcel roll routinely agree on the
    identifier and disagree on how they punctuate it: measured on Maine's
    assessor table 2026-08-25, Kennebunk's geometry writes a map-lot as
    `999-999` where its own assessor rows write `999_999`, and a literal
    comparison matches none of 7,183 parcels where `pipe` matches 78%.
    `pipe` is the conversion to reach for, because it collapses each run
    of separators to a single marker rather than deleting it, so
    `12-3` and `1-23` stay distinct.
    """
    # Imported lazily so the plain paging path stays a requests-only
    # module with no geo stack to load.
    import geopandas as gpd
    import pandas as pd

    gdf = gpd.read_file(path)
    key_fields = _key_fields(key)

    # The attribute side arrives already keyed on the assembled string
    # (`_fetch_attribute_table` builds it the same way), so only the
    # geometry side has to be assembled here. Both are string by
    # construction, which also settles the dtype mismatch that used to
    # make a space-padded key silently match nothing against an integer.
    attributes = pd.DataFrame.from_dict(table, orient='index')
    attributes.index.name = _JOIN_KEY_COLUMN
    attributes = attributes.reset_index()
    attributes[_JOIN_KEY_COLUMN] = attributes[_JOIN_KEY_COLUMN].astype('string')
    gdf[_JOIN_KEY_COLUMN] = _composite_key(
        gdf, key_fields, label=label, side='downloaded file'
    )

    literal_matched = int(
        gdf[_JOIN_KEY_COLUMN].isin(set(attributes[_JOIN_KEY_COLUMN])).sum()
    )
    dropped = 0
    if key_conv:
        gdf, attributes, dropped = _normalize_join_key(
            gdf, attributes, key_conv=key_conv
        )

    matched = gdf[_JOIN_KEY_COLUMN].isin(set(attributes[_JOIN_KEY_COLUMN].dropna()))
    n_matched = int(matched.sum())
    if key_conv and n_matched < literal_matched:
        raise RuntimeError(
            f'{label}: attribute_join key_conv={key_conv!r} matched '
            f'{n_matched:,} features where comparing {key_fields} literally '
            f'matched {literal_matched:,}. A conversion that loses matches '
            'is the wrong conversion for this source; drop it or pick '
            'another.'
        )

    match_rate = n_matched / len(gdf) if len(gdf) else 0.0
    if match_rate < min_match:
        raise RuntimeError(
            f'{label}: attribute join matched only {n_matched:,} of '
            f'{len(gdf):,} features ({match_rate:.1%}), below the required '
            f'{min_match:.0%}. Check that {key_fields} is the right join key '
            'and that both sides use the same identifier form.'
        )

    # Never let joined columns silently overwrite the bulk file's own.
    collisions = [f for f in fields if f in gdf.columns]
    if collisions:
        raise ValueError(
            f'{label}: attribute_join fields already exist in the '
            f'downloaded file: {collisions}. Rename or drop them.'
        )

    merged = gdf.merge(attributes, on=_JOIN_KEY_COLUMN, how='left')
    merged = merged.drop(columns=[_JOIN_KEY_COLUMN])
    merged.to_file(path, driver='GeoJSON')
    if verbose:
        gain = (
            f' ({n_matched - literal_matched:+,} vs a literal comparison)'
            if key_conv
            else ''
        )
        lost = f', {dropped:,} ambiguous attribute rows dropped' if dropped else ''
        _log(
            f'{label}: joined {len(fields)} column(s) onto '
            f'{n_matched:,}/{len(gdf):,} features ({match_rate:.1%})'
            f'{gain}{lost}'
        )


def _normalize_join_key(gdf, attributes, *, key_conv):
    """Convert the assembled key on both sides, dropping ambiguous rows.

    Normalizing makes keys collide that did not collide before, and a
    collision on the *attribute* side is the dangerous one: two assessor
    rows reaching one normalized key would each attach to every parcel
    carrying it, inventing rows and picking arbitrarily between
    conflicting values. Those keys are dropped rather than resolved,
    which costs the parcels behind them their attributes and never gives
    them another parcel's. A collision on the geometry side is harmless:
    two parcels may legitimately resolve to one assessor row.

    Returns
    -------
    tuple
        (geometry frame, attribute frame, number of attribute rows dropped)
    """
    from openplaces.geo.ids import convert_parcel_id

    gdf = gdf.copy()
    attributes = attributes.copy()
    for frame in (gdf, attributes):
        frame[_JOIN_KEY_COLUMN] = convert_parcel_id(
            frame[_JOIN_KEY_COLUMN], None, key_conv
        )

    ambiguous = attributes[_JOIN_KEY_COLUMN].notna() & attributes.duplicated(
        _JOIN_KEY_COLUMN, keep=False
    )
    dropped = int(ambiguous.sum())
    if dropped:
        attributes = attributes[~ambiguous]
    return gdf, attributes, dropped


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
    # `f` must be forced back to 'json' here. The caller's params carry
    # `f=geojson` for the feature pages, but most servers ignore
    # `returnCountOnly` in GeoJSON mode and answer with a normal (here,
    # unbounded) FeatureCollection instead of `{"count": N}` -- verified
    # against Cumberland, Wilson and Gates counties, NC, all of which
    # return `{'type', 'properties', 'features'}` for `f=geojson` and a
    # real count for `f=json`. Reading `count` off that response raised
    # KeyError and failed the whole download. A minority of older
    # on-prem ArcGIS Server instances (e.g. Hertford County, NC) do
    # honor it in either mode, which is why this went unnoticed.
    count_page = _get_json(
        query_url,
        {**params, 'returnCountOnly': 'true', 'f': 'json'},
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
    admin_key_column=None,
    admin_key_transform=None,
    where_admin_column=None,
) -> Path:
    """Download every feature of one ArcGIS REST layer to a GeoJSON file.

    Parameters
    ----------
    partition_id : optional
        Unused (this scraper is for single-file, unpartitioned recipes, or
        recipes partitioned by admin unit -- see `admin_id_to_download`
        below). Accepted for interface compatibility with
        `Ingester._run_download_scraper`'s shared `fetch` contract.
    target_path : str or pathlib.Path
        Where to write the combined GeoJSON `FeatureCollection`.
    portal_url : str, optional
        Unused; the layer's own query endpoint is `layer_url`. Accepted for
        interface compatibility with the shared `fetch` contract.
    admin_id_to_download : str, optional
        The current admin unit (e.g. `'US-UT-SL'`), supplied by the
        Ingester when the recipe's `download_by` is `admin_level` rather
        than a partition key. Used only to resolve an `{admin_key}`
        placeholder in `layer_url` -- see `admin_key_column` below.
        Unused for a single-file, unpartitioned recipe.
    layer_url : str
        Base URL of the FeatureServer/MapServer layer, without a trailing
        `/query` (e.g. `'.../FeatureServer/0'`). May contain a single
        `{admin_key}` placeholder for the case where each admin unit is a
        genuinely different service rather than a shared one filtered by
        an in-data column (e.g. one FeatureServer per county, named after
        the county) -- see `admin_key_column`.
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
        `key` is a field name, or a list of them when the identifier has
        to be rebuilt from its parts. `key_conv` is an optional
        :func:`~openplaces.geo.ids.convert_parcel_id` conversion applied
        to both sides before they are compared, for a source whose two
        submissions punctuate the same identifier differently.
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
    admin_key_column : str, optional
        Column name looked up on `admin_id_to_download`'s own admin table
        (via `get_admin`) to resolve `{admin_key}` in `layer_url`, e.g.
        `'name'` on a Census admin3 table (`'Salt Lake'`). Required only
        when `layer_url` uses the placeholder.
    admin_key_transform : str, optional
        Transform applied to the looked-up key before substitution. Only
        `'remove_spaces'` is supported (matches the name and behavior of
        `Ingester`'s own `download_by.admin_key_transform`), e.g. turning
        `'Salt Lake'` into `'SaltLake'` for a service name with no spaces.
    where_admin_column : str, optional
        Complementary to `{admin_key}` in `layer_url`: for a *shared*
        service filtered by an in-data admin column (e.g. a statewide
        parcel layer with a `countyfips` field), names that column so
        `fetch` builds `f"{where_admin_column} = '{key}'"` from the
        current `admin_id_to_download` (via `admin_key_column`) and
        ANDs it onto `where`, rather than requiring one recipe per
        admin unit to hand-write its own `where` string.

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
    if layer_url:
        layer_url = _resolve_layer_url(
            layer_url,
            admin_id_to_download=admin_id_to_download,
            admin_key_column=admin_key_column,
            admin_key_transform=admin_key_transform,
        )
    where = _resolve_where(
        where,
        admin_id_to_download=admin_id_to_download,
        admin_key_column=admin_key_column,
        admin_key_transform=admin_key_transform,
        where_admin_column=where_admin_column,
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
        key_conv=attribute_join.get('key_conv'),
    )
