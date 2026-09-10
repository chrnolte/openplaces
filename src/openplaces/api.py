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
    # For connectors (spokes), which may read the hub only through this
    # module: the attribute registry and how a column name resolves to
    # it, admin containment and ancestry, and a recipe's columns
    # without reading its data.
    'get_attribute_registry': 'openplaces.core.attribute_registry:load_registry',
    'get_agg_func': 'openplaces.core.attribute_registry:get_agg_func',
    'resolve_attribute_name': 'openplaces.recipe:resolve_attribute_name',
    'admin_scope_covers': 'openplaces.core.schema:admin_scope_covers',
    'get_admin_ancestor': 'openplaces.core.schema:admin_ancestor',
    'describe_recipe': 'openplaces.io.readers:describe_recipe',
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
