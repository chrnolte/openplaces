"""Every table an ingest writes without rows says why.

An empty saved table is an observation only when something says what
was observed. A stale empty parcel table, written before an empty
download raised, read as "built" to every later run: the ingest skipped
the unit and every downstream table was built from no rows without an
error. Each ingest writer of an empty table therefore goes through
`save_empty_table`, which stamps the reason (`EMPTY_REASONS`) and a
short detail in the parquet footer, and a table with no rows and no
reason is treated as not built (`readers.is_unexplained_empty`).
"""

from __future__ import annotations

import geopandas as gpd

from openplaces.config import cfg
from openplaces.io import save_parquet
from openplaces.io.readers import (
    EMPTY_DETAIL_KEY,
    EMPTY_REASON_KEY,
    EMPTY_REASONS,
)

# A detail is a line in a footer, not a log.
_DETAIL_MAX_CHARS = 1000


class UnexplainedEmptySourceError(RuntimeError):
    """An ingest output holds no rows and records no reason why.

    Raised where a consumer would read the file as "the unit has none".
    The file was written before ingest outputs recorded their reason,
    or by a run that stopped; rerun the unit's ingest, which treats the
    file as not built.
    """


class UnexplainedEmptyIngestError(RuntimeError):
    """An ingest ended with a unit that has no output and no known reason.

    Raised instead of writing an empty table nothing can account for.
    """


def empty_table_metadata(reason: str, detail: str) -> dict[str, str]:
    """Footer metadata for a table without rows.

    Parameters
    ----------
    reason : str
        One of `EMPTY_REASONS`.
    detail : str
        A short free text: what was read, which filter, where the rows
        are for a marker.

    Returns
    -------
    dict of str to str

    Raises
    ------
    ValueError
        For a reason outside `EMPTY_REASONS`.
    """
    if reason not in EMPTY_REASONS:
        raise ValueError(
            f'Unknown empty-table reason {reason!r}; expected one of {EMPTY_REASONS}.'
        )
    return {
        EMPTY_REASON_KEY: reason,
        EMPTY_DETAIL_KEY: str(detail)[:_DETAIL_MAX_CHARS],
    }


def save_empty_table(path, reason: str, detail: str, frame=None) -> None:
    """Write an ingest table without rows, stamped with why it has none.

    Parameters
    ----------
    path : pathlib.Path
        The output path.
    reason : str
        One of `EMPTY_REASONS`.
    detail : str
        A short free text (see `empty_table_metadata`).
    frame : DataFrame or GeoDataFrame, optional
        The empty frame to write, keeping its columns; an empty
        GeoDataFrame in the project CRS when omitted.

    Raises
    ------
    ValueError
        For a frame with rows, or an unknown reason.
    """
    metadata = empty_table_metadata(reason, detail)
    if frame is None:
        frame = gpd.GeoDataFrame(
            {'geometry': gpd.GeoSeries([], dtype='geometry')}, crs=cfg.crs
        )
    elif len(frame):
        raise ValueError(f'save_empty_table got {len(frame)} rows for {path}.')
    # A plain frame writes no `_geo` sidecar, so one left by an earlier
    # run with rows would otherwise sit beside the empty table.
    for old in (path, path.with_stem(path.stem + '_geo')):
        old.unlink(missing_ok=True)
    save_parquet(frame, path, file_metadata=metadata)
