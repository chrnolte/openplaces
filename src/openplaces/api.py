"""
Public API for openplaces.

Re-exports data access and recipe-driven pipeline functions. Each name resolves
lazily on first access (PEP 562), so importing this module pulls nothing heavy
and, e.g., accessing ``curate`` imports only the curator stage.
"""

from importlib import import_module

# Maps each public name to the 'module:attribute' that implements it. Resolved
# on first access by __getattr__ so the stages load only when actually used.
_SOURCES = {
    'get_admin': 'openplaces.io.readers:get_admin',
    'get_admin_ids': 'openplaces.io.readers:get_admin_ids',
    'get_regions': 'openplaces.io.readers:get_regions',
    'get_region_admin_ids': 'openplaces.io.readers:get_region_admin_ids',
    'get_entities': 'openplaces.io.readers:get_entities',
    'get_dataset': 'openplaces.io.readers:get_dataset',
    'ingest': 'openplaces.io.ingester:ingest',
    'harmonize': 'openplaces.io.harmonizer:harmonize',
    'enrich': 'openplaces.io.enricher:enrich',
    'curate': 'openplaces.io.curator:curate',
    'aggregate_files': 'openplaces.io.aggregate:aggregate_files',
    'aggregate_partitions': 'openplaces.io.aggregate:aggregate_partitions',
    'export_delivery': 'openplaces.io.delivery:export_delivery',
    'cleanup': 'openplaces.io.cleanup:cleanup',
    'compact': 'openplaces.io.cleanup:compact',
    'inspect_table': 'openplaces.utils:inspect_table',
    'export_qgis_map': 'openplaces.viz.qgis_map:export_qgis_map',
}

__all__ = list(_SOURCES)


def __getattr__(name: str):
    source = _SOURCES.get(name)
    if source is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_path, attr = source.split(':')
    value = getattr(import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__():
    return __all__
