"""Concatenate several JSON list endpoints into one table.

A statistical agency commonly publishes one endpoint per tier of its
classification, and a recipe sometimes needs a tier the agency never
publishes as a single list. The Philippine PSGC is the worked case: its
level-2 equivalent is the 81 provinces plus the National Capital Region,
which has no provinces of its own, and those live in two endpoints.

Concatenating in the recipe rather than after ingest matters because
`create_index` mints admin ids per ingested file. Two files would mint
two independent id sets under the same parent and could collide, so the
rows have to meet before the ingester sees them.

Nothing here knows about any particular agency: the endpoint list, the
column that records which endpoint a row came from, and any filtering
all live in the recipe (`scraper_options`, then `query`).
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd

from openplaces.io import request_headers

# Endpoints of this kind are small index listings rather than bulk data,
# but they are still somebody else's server, so requests are made one at
# a time and identified (see AGENTS.md, "Talking to other people's
# servers").
DEFAULT_TIMEOUT_S = 180


def _read_endpoint(url: str, timeout: float) -> pd.DataFrame:
    """Read one JSON endpoint returning a list of records.

    Parameters
    ----------
    url : str
        Endpoint to read.
    timeout : float
        Socket timeout in seconds.
    """
    request = urllib.request.Request(url, headers=request_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    if isinstance(payload, dict):
        # Some APIs wrap the list in an envelope; take the sole
        # list value rather than guessing at a key name.
        lists = [value for value in payload.values() if isinstance(value, list)]
        if len(lists) != 1:
            raise ValueError(
                f'Expected a list of records from {url}, got an object with '
                f'{len(lists)} list-valued keys.'
            )
        payload = lists[0]

    if not isinstance(payload, list):
        raise ValueError(f'Expected a list of records from {url}.')

    return pd.DataFrame(payload)


def fetch(
    partition_id=None,
    target_path=None,
    portal_url=None,
    admin_id_to_download=None,
    label=None,
    redownload=False,
    verbose=False,
    urls=None,
    source_column=None,
    source_names=None,
    timeout=DEFAULT_TIMEOUT_S,
    **_ignored,
):
    """Write the concatenated rows of several JSON endpoints.

    Parameters
    ----------
    target_path : pathlib.Path
        Where to write the combined table. The suffix picks the format:
        `.csv` writes CSV, anything else writes JSON records.
    urls : list of str
        Endpoints to read, in order. Their rows are concatenated with the
        union of their columns; a column one endpoint lacks is missing
        for its rows.
    source_column : str, optional
        Name of a column recording which endpoint each row came from, so
        the recipe's `query` can keep or drop a whole endpoint's rows.
    source_names : list of str, optional
        Labels written into `source_column`, one per URL. Defaults to the
        last non-empty path segment of each URL.
    timeout : float
        Socket timeout per request, in seconds.
    partition_id, portal_url, admin_id_to_download, label, redownload,
    verbose
        Standard scraper arguments; unused beyond progress reporting.
    """
    if not urls:
        raise ValueError(
            'A `json_endpoints` scraper needs `scraper_options: {urls: [...]}`.'
        )
    if source_names is not None and len(source_names) != len(urls):
        raise ValueError(
            f'`source_names` has {len(source_names)} entries for {len(urls)} urls.'
        )

    frames = []
    for position, url in enumerate(urls):
        if verbose:
            print(f'   reading {url}', flush=True)
        frame = _read_endpoint(url, timeout)
        if source_column:
            if source_names is not None:
                name = source_names[position]
            else:
                name = url.rstrip('/').rsplit('/', 1)[-1]
            frame[source_column] = name
        frames.append(frame)
        if verbose:
            print(f'      {len(frame):,} rows', flush=True)

    combined = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f'   combined: {len(combined):,} rows', flush=True)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.suffix.lower() == '.csv':
        combined.to_csv(target_path, index=False)
    else:
        combined.to_json(target_path, orient='records')

    return target_path
