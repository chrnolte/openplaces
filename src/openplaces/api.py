"""
src/openplaces/api.py

Public API for openplaces. Re-exports data-access primitives from
io/readers and will grow to include higher-level query, enrichment,
and analysis functions.
"""

from openplaces.io.readers import get_admin, get_admin_ids, get_entities  # noqa: F401

# Backward-compatible alias — prefer get_entities in new code
read_entities = get_entities  # noqa: F401
