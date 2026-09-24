"""
Read an Esri file geodatabase with its coded-value domains applied,
so a layer's categorical fields arrive as labels rather than codes.
"""

import math
from xml.etree import ElementTree

import geopandas as gpd


def get_gdb_domains(gdb_path: str) -> dict[str, dict]:
    """Extract coded value domains from a GDB's internal metadata table."""
    domains = {}
    try:
        items = gpd.read_file(gdb_path, layer='GDB_Items', engine='pyogrio')
    except Exception:
        return domains

    for _, row in items.iterrows():
        definition = row.get('Definition')
        if not definition or (isinstance(definition, float) and math.isnan(definition)):
            continue
        try:
            root = ElementTree.fromstring(definition)
        except ElementTree.ParseError:
            continue

        # Coded value domains have a <CodedValueDomain> or <GPCodedValueDomain2> element
        for cv_domain in root.iter('CodedValue'):
            domain_name = root.findtext('DomainName')
            if not domain_name:
                continue
            if domain_name not in domains:
                domains[domain_name] = {}
            code = cv_domain.findtext('Code')
            name = cv_domain.findtext('Name')
            if code is not None and name is not None:
                domains[domain_name][code] = name

    return domains


def get_gdb_field_domain_map(gdb_path: str, layer: str) -> dict[str, dict]:
    """Map field names to their coded value domain for a given layer.

    Parameters
    ----------
    gdb_path : str
        Path to geodatabase
    layer : str
        Layer to read
    """
    field_domains = {}
    try:
        items = gpd.read_file(gdb_path, layer='GDB_Items', engine='pyogrio')
    except Exception:
        return field_domains

    all_domains = get_gdb_domains(gdb_path)

    for _, row in items.iterrows():
        if row.get('Name') != layer:
            continue
        definition = row.get('Definition')
        if not definition or (isinstance(definition, float) and math.isnan(definition)):
            continue
        try:
            root = ElementTree.fromstring(definition)
        except ElementTree.ParseError:
            continue
        for field in root.iter('GPFieldInfoEx'):
            fname = field.findtext('Name')
            dname = field.findtext('DomainName')
            if fname and dname and dname in all_domains:
                field_domains[fname] = all_domains[dname]

    return field_domains


def read_gdb_with_domains(
    gdb_path: str, columns: list = None, layer: str = None, **kwargs
) -> gpd.GeoDataFrame:
    """Read a Geodatabase while resolving categorical label mappings"""
    gdf = gpd.read_file(
        gdb_path, columns=columns, layer=layer, engine='pyogrio', **kwargs
    )
    field_domains = get_gdb_field_domain_map(gdb_path, layer)

    for col, mapping in field_domains.items():
        if col in gdf.columns:
            gdf[col] = gdf[col].astype(str).map(mapping).fillna(gdf[col])

    return gdf
