"""Regenerate tests/viz/qgis_map/fixtures/tiny_template.qgz.

A small, hand-built (via this script, not the QGIS GUI) QGIS project
mirroring the real structure `openplaces.viz.qgis_map.generator` expects:
`<maplayer>` entries in `<projectlayers>`, joined attr/geo pairs via
`<vectorjoins>/<join>`, a `<layer-tree-group>` tree, a legacy
`<legend>/<legendgroup>/<legendlayer>` tree, and a `<mapcanvas>` with
`<extent>`/`<destinationsrs>`. Structure was reverse-engineered from a real
QGIS 3.44 project file (`US-NC-CE.qgz`) authored via
`qgis/load_joined_parquet.py`; kept intentionally minimal (most metadata
subtrees real QGIS writes are omitted) since the generator only reads/writes
the elements documented in its own module docstring.

Run with ``python build_tiny_template.py`` from this directory to
regenerate ``tiny_template.qgz`` after changing this script.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

_HERE = Path(__file__).parent

_WKT_3395 = (
    'PROJCRS["WGS 84 / World Mercator",'
    'BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],ANGLEUNIT["degree",0.0174532925199433]],'
    'CONVERSION["World Mercator",METHOD["Mercator (variant A)"],'
    'PARAMETER["Latitude of natural origin",0],'
    'PARAMETER["Longitude of natural origin",0],'
    'PARAMETER["Scale factor at natural origin",1],'
    'PARAMETER["False easting",0],PARAMETER["False northing",0]],'
    'CS[Cartesian,2],AXIS["easting (E)",east],AXIS["northing (N)",north],'
    'LENGTHUNIT["metre",1],ID["EPSG",3395]]'
)


def _maplayer(
    *,
    layer_id: str,
    layername: str,
    datasource: str,
    join_to: str | None = None,
    extra_xml: str = '',
) -> str:
    join_xml = (
        f'<vectorjoins><join joinFieldName="_join_id" memoryCache="1" '
        f'customPrefix="" dynamicForm="0" joinLayerId="{join_to}" '
        f'hasCustomPrefix="0" upsertOnEdit="0" targetFieldName="_join_id" '
        f'cascadedDelete="0" editable="0"/></vectorjoins>'
        if join_to
        else '<vectorjoins/>'
    )
    return (
        '<maplayer type="vector" wkbType="Polygon" geometry="Polygon">'
        f'<id>{layer_id}</id>'
        f'<datasource>{datasource}</datasource>'
        f'<layername>{layername}</layername>'
        '<provider encoding="UTF-8">ogr</provider>'
        f'{join_xml}'
        f'{extra_xml}'
        '</maplayer>'
    )


# A categorized-symbol renderer with baked "stale" categories, standing in
# for a template author's reference-county category list -- used to test
# that `generator._regenerate_categories` replaces it from real run data.
_DYNAMIC_CAT_RENDERER = (
    '<renderer-v2 type="categorizedSymbol" attr="conflict_field" forceraster="0" '
    'symbollevels="0" enableorderby="0">'
    '<categories>'
    '<category value="old_a" label="old_a" symbol="0" render="true"/>'
    '<category value="old_b" label="old_b" symbol="1" render="true"/>'
    '</categories>'
    '<symbols>'
    '<symbol type="fill" name="0" alpha="1">'
    '<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
    '<Option type="Map">'
    '<Option type="QString" name="color" value="10,10,10,255"/>'
    '</Option>'
    '</layer>'
    '</symbol>'
    '<symbol type="fill" name="1" alpha="1">'
    '<layer class="SimpleFill" enabled="1" locked="0" pass="0">'
    '<Option type="Map">'
    '<Option type="QString" name="color" value="20,20,20,255"/>'
    '</Option>'
    '</layer>'
    '</symbol>'
    '</symbols>'
    '</renderer-v2>'
)


def _tree_layer(*, layer_id: str, name: str, source: str, checked: bool) -> str:
    checked_str = 'Qt::Checked' if checked else 'Qt::Unchecked'
    return (
        f'<layer-tree-layer legend_exp="" patch_size="-1,-1" source="{source}" '
        f'id="{layer_id}" name="{name}" providerKey="ogr" checked="{checked_str}" '
        f'legend_split_behavior="0" expanded="1"/>'
    )


def _legend_layer(*, layer_id: str, name: str, checked: bool) -> str:
    checked_str = 'Qt::Checked' if checked else 'Qt::Unchecked'
    return (
        f'<legendlayer showFeatureCount="0" name="{name}" open="true" '
        f'checked="{checked_str}" drawingOrder="-1">'
        '<filegroup hidden="false" open="true">'
        f'<legendlayerfile layerid="{layer_id}" isInOverview="0" visible="0"/>'
        '</filegroup></legendlayer>'
    )


# (group name, geo id, geo name, attr id-or-None, checked)
_PROTOTYPES = [
    ('Curated', 'proto_output_id', 'proto_output', 'proto_output_attr_id', True),
    ('Curated', 'proto_output_combined_id', 'proto_output_combined', None, True),
    ('Inputs', 'proto_input_id', 'proto_input', 'proto_input_attr_id', False),
    ('Admin', 'proto_admin_id', 'proto_admin', 'proto_admin_attr_id', True),
    ('Basemaps', 'proto_basemap_id', 'proto_basemap', None, True),
    (
        'Unstyled',
        'proto_fallback_id',
        'proto_fallback',
        'proto_fallback_attr_id',
        False,
    ),
]

# Style-variant prototypes: geometry-only, joined to an *existing* prototype's
# attr layer above rather than getting one of their own.
# (group name, geo id, geo name, join-to attr id, checked)
_VARIANTS = [
    (
        'Curated',
        'proto_output_by_roof_shape_id',
        'proto_output_by_roof_shape',
        'proto_output_attr_id',
        False,
    ),
]

# Style-variant prototypes for a *combined* base (no attr sibling to join at
# all): geometry-only, no join.
# (group name, geo id, geo name, checked)
_COMBINED_VARIANTS = [
    (
        'Curated',
        'proto_output_combined_by_zone_id',
        'proto_output_combined_by_zone',
        False,
    ),
]


def build_qgs() -> str:
    maplayers = []
    groups: dict[str, list[str]] = {}
    legend_groups: dict[str, list[str]] = {}

    for group, geo_id, geo_name, attr_id, checked in _PROTOTYPES:
        datasource = f'/fixtures/{geo_name}.parquet'
        if attr_id:
            attr_name = f'{geo_name}_attr'
            maplayers.append(
                _maplayer(
                    layer_id=attr_id,
                    layername=attr_name,
                    datasource=f'/fixtures/{geo_name}.parquet',
                )
            )
            maplayers.append(
                _maplayer(
                    layer_id=geo_id,
                    layername=geo_name,
                    datasource=datasource,
                    join_to=attr_id,
                )
            )
        else:
            maplayers.append(
                _maplayer(layer_id=geo_id, layername=geo_name, datasource=datasource)
            )

        groups.setdefault(group, []).append(
            _tree_layer(
                layer_id=geo_id, name=geo_name, source=datasource, checked=checked
            )
        )
        legend_groups.setdefault(group, []).append(
            _legend_layer(layer_id=geo_id, name=geo_name, checked=checked)
        )

    for group, geo_id, geo_name, join_to, checked in _VARIANTS:
        datasource = f'/fixtures/{geo_name}.parquet'
        maplayers.append(
            _maplayer(
                layer_id=geo_id,
                layername=geo_name,
                datasource=datasource,
                join_to=join_to,
            )
        )
        groups.setdefault(group, []).append(
            _tree_layer(
                layer_id=geo_id, name=geo_name, source=datasource, checked=checked
            )
        )
        legend_groups.setdefault(group, []).append(
            _legend_layer(layer_id=geo_id, name=geo_name, checked=checked)
        )

    for group, geo_id, geo_name, checked in _COMBINED_VARIANTS:
        datasource = f'/fixtures/{geo_name}.parquet'
        maplayers.append(
            _maplayer(layer_id=geo_id, layername=geo_name, datasource=datasource)
        )
        groups.setdefault(group, []).append(
            _tree_layer(
                layer_id=geo_id, name=geo_name, source=datasource, checked=checked
            )
        )
        legend_groups.setdefault(group, []).append(
            _legend_layer(layer_id=geo_id, name=geo_name, checked=checked)
        )

    # A single combined (no attr sibling) base prototype styled with a
    # categorized-symbol renderer, for testing `_regenerate_categories`.
    dynamic_cat_datasource = '/fixtures/proto_dynamic_cat.parquet'
    maplayers.append(
        _maplayer(
            layer_id='proto_dynamic_cat_id',
            layername='proto_dynamic_cat',
            datasource=dynamic_cat_datasource,
            extra_xml=_DYNAMIC_CAT_RENDERER,
        )
    )
    groups.setdefault('Curated', []).append(
        _tree_layer(
            layer_id='proto_dynamic_cat_id',
            name='proto_dynamic_cat',
            source=dynamic_cat_datasource,
            checked=True,
        )
    )
    legend_groups.setdefault('Curated', []).append(
        _legend_layer(
            layer_id='proto_dynamic_cat_id', name='proto_dynamic_cat', checked=True
        )
    )

    tree_groups_xml = ''.join(
        f'<layer-tree-group groupLayer="" name="{name}" '
        'checked="Qt::Checked" expanded="1">'
        '<customproperties><Option/></customproperties>'
        f'{"".join(layers)}'
        '</layer-tree-group>'
        for name, layers in groups.items()
    )
    legend_groups_xml = ''.join(
        f'<legendgroup name="{name}" open="true" checked="Qt::Checked">'
        f'{"".join(layers)}'
        '</legendgroup>'
        for name, layers in legend_groups.items()
    )

    return (
        "<?xml version='1.0' encoding='UTF-8'?>\n"
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis saveDateTime="2026-01-01T00:00:00" saveUser="test" '
        'version="3.44.11-Solothurn" projectname="">'
        '<title></title>'
        '<projectlayers>'
        f'{"".join(maplayers)}'
        '</projectlayers>'
        f'<layer-tree-group>{tree_groups_xml}</layer-tree-group>'
        f'<legend updateDrawingOrder="true">{legend_groups_xml}</legend>'
        '<properties></properties>'
        '<mapcanvas annotationsVisible="1" name="theMapCanvas">'
        '<units>meters</units>'
        '<extent><xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax></extent>'
        '<rotation>0</rotation>'
        '<destinationsrs><spatialrefsys nativeFormat="Wkt">'
        f'<wkt>{_WKT_3395}</wkt>'
        '</spatialrefsys></destinationsrs>'
        '</mapcanvas>'
        '</qgis>'
    )


def main() -> None:
    qgs_text = build_qgs()
    out_path = _HERE / 'tiny_template.qgz'
    with zipfile.ZipFile(out_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('tiny_template.qgs', qgs_text)
        zf.writestr(
            'tiny_template_styles.db', b'fake-style-db-bytes-for-pass-through-test'
        )
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
