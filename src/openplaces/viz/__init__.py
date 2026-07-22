"""Visualization tools for geospatial data."""

import urllib
from importlib import import_module

from openplaces.viz.axes import add_log_ticks
from openplaces.viz.colors import (
    CATEGORY_COLORS,
    adjust_brightness,
    continuous_to_rgba,
    match_palette,
    to_rgba_array,
)
from openplaces.viz.maps import (
    show_building,
    show_building_imagery,
    show_geometry_context,
    show_ingested_geometries,
    show_random_entity,
)
from openplaces.viz.tabulation import plot_tabulation, tabulate

# Lazily resolved (PEP 562): show_entities_interactive/raster depend on the
# `viz-fast` extra (lonboard, datashader), which isn't installed for plain
# `viz` (matplotlib-only) users. Importing this package must not require
# `viz-fast`; only calling one of these two names does.
_LAZY_SOURCES = {
    'DEFAULT_ELEVATION_SCALE': 'openplaces.viz.terrain:DEFAULT_ELEVATION_SCALE',
    'get_admin_boundary_layer': 'openplaces.viz.interactive:get_admin_boundary_layer',
    'get_basemap_layer': 'openplaces.viz.interactive:get_basemap_layer',
    'show_entities_interactive': 'openplaces.viz.interactive:show_entities_interactive',
    'show_entities_raster': 'openplaces.viz.raster:show_entities_raster',
    'show_value_terrain_layer': 'openplaces.viz.terrain:show_value_terrain_layer',
}

__all__ = [
    'CATEGORY_COLORS',
    'DEFAULT_ELEVATION_SCALE',
    'add_log_ticks',
    'adjust_brightness',
    'continuous_to_rgba',
    'create_street_view_link',
    'get_admin_boundary_layer',
    'get_basemap_layer',
    'match_palette',
    'plot_tabulation',
    'show_building',
    'show_building_imagery',
    'show_entities_interactive',
    'show_entities_raster',
    'show_geometry_context',
    'show_ingested_geometries',
    'show_random_entity',
    'show_value_terrain_layer',
    'tabulate',
    'to_rgba_array',
]


def __getattr__(name: str):
    source = _LAZY_SOURCES.get(name)
    if source is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module_path, attr = source.split(':')
    value = getattr(import_module(module_path), attr)
    globals()[name] = value
    return value


def __dir__():
    return __all__


def create_street_view_link(address):
    """Create a Google street view link from a sequence

    Parameters
    ----------
    address: list | pd.Series
        Address, ideally in the order: street, city, zip, state, country
    """

    # Combine the address components into a single string
    full_address = ', '.join(address)

    # URL encode the string (e.g., spaces become '+')
    encoded_address = urllib.parse.quote_plus(full_address)

    # Construct the final URL
    base_url = 'https://www.google.com/maps/search/?api=1&query='
    return f'{base_url}{encoded_address}'


# Patch lonboard.Map to automatically inject Javascript that expands
# the "Feature properties" side panel and prevents name abbreviation.
try:
    import lonboard

    _original_map_repr = lonboard.Map._repr_mimebundle_

    def _custom_map_repr(self, *args, **kwargs):
        js_code = """
<script>
(function() {
    function expandPanel() {
        function search(node) {
            if (!node) return;
            if (node.classList && node.classList.contains('w-96')) {
                node.classList.remove('w-96');
                node.classList.add('w-[480px]');
                node.style.width = '480px';
            }
            if (
                node.classList &&
                node.classList.contains('truncate') &&
                node.title &&
                node.parentNode &&
                node.parentNode.tagName === 'TD'
            ) {
                node.classList.remove('truncate');
                node.classList.add('break-words');
                node.style.overflow = 'visible';
                node.style.textOverflow = 'clip';
                node.style.whiteSpace = 'normal';
            }
            if (node.children) {
                for (let child of node.children) search(child);
            }
            if (node.shadowRoot) search(node.shadowRoot);
        }
        search(document.body);
    }
    setInterval(expandPanel, 500);
})();
</script>
"""
        try:
            from IPython.display import HTML, display

            display(HTML(js_code))
        except (ImportError, NameError):
            pass
        return _original_map_repr(self, *args, **kwargs)

    lonboard.Map._repr_mimebundle_ = _custom_map_repr
except Exception:
    pass
