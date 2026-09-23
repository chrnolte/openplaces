"""
Input/output utilities.

The package file re-exports what its six modules define, so every
name that could be imported from openplaces.io before 2026-09-22 still
can be: download, archives, tables, deletion, geodatabase, transfer.
"""

import gc

import pyarrow

from openplaces.io.archives import find_latest_file_or_gdb as find_latest_file_or_gdb
from openplaces.io.archives import unzip
from openplaces.io.deletion import DataDeletionError as DataDeletionError
from openplaces.io.deletion import delete_data as delete_data
from openplaces.io.deletion import delete_image_caches
from openplaces.io.fetch import download, request_headers
from openplaces.io.geodatabase import get_gdb_domains as get_gdb_domains
from openplaces.io.geodatabase import (
    get_gdb_field_domain_map as get_gdb_field_domain_map,
)
from openplaces.io.geodatabase import read_gdb_with_domains as read_gdb_with_domains
from openplaces.io.tables import (
    coerce_mixed_object_columns as coerce_mixed_object_columns,
)
from openplaces.io.tables import delete_parquet as delete_parquet
from openplaces.io.tables import parquet_columns as parquet_columns
from openplaces.io.tables import (
    read_parquet,
    save,
    save_parquet,
    to_csv,
    to_gpkg,
    to_kmz,
    to_parquet,
)
from openplaces.io.transfer import DriveTransferError, compress, share, to_drive

__all__ = [
    'DriveTransferError',
    'compress',
    'delete_image_caches',
    'download',
    'read_parquet',
    'release_unused_memory',
    'request_headers',
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


def release_unused_memory() -> None:
    """Return freed Python and pyarrow allocations to the OS.

    Geometry-heavy GeoDataFrames and pyarrow's arena allocator do not
    reliably release freed memory back to the OS through refcounting
    alone. Call this after finishing work on one admin unit in a
    multi-admin-unit run to keep peak resident memory bounded.
    """
    gc.collect()
    pyarrow.default_memory_pool().release_unused()
