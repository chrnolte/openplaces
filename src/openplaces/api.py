"""
Public API for openplaces.

Re-exports data access and recipe-driven pipeline functions.
"""

from openplaces.io.aggregate import (  # noqa: F401
    aggregate_files,
    aggregate_partitions,
)
from openplaces.io.curator import curate  # noqa: F401
from openplaces.io.enricher import enrich  # noqa: F401
from openplaces.io.harmonizer import harmonize  # noqa: F401
from openplaces.io.ingester import ingest  # noqa: F401
from openplaces.io.readers import (  # noqa: F401
    get_admin,
    get_admin_ids,
    get_dataset,
    get_entities,
)
from openplaces.utils import inspect_table  # noqa: F401

__all__ = [
    'get_admin',
    'get_admin_ids',
    'get_entities',
    'ingest',
    'inspect_table',
    'harmonize',
    'enrich',
    'curate',
    'get_dataset',
    'aggregate_files',
    'aggregate_partitions',
]
