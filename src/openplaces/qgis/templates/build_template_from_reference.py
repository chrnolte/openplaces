"""Build ``openplaces_template.qgz`` from a hand-authored reference project.

Extracts real, already-styled layers out of the user's exploratory QGIS
project (``US-NC-AR.qgz``, not part of this repo — lives under
``Dropbox/Data/world/show/qgis``), renames/relocates them into the packaged
template's ``template_*`` naming and ``Buildings/{openplaces,inputs}`` /
``Parcels`` / ``Admin`` / ``Basemaps`` group convention documented in
``README.md``, and authors the pieces that have no real-world example to
copy from: an open/no-API-key basemap (cloned structurally from the
reference's Google raster layer, URL swapped), a print layout (title bound
to the generator's project variables, legend, scale bar, north arrow), and
the reserved unstyled ``_fallback`` prototype.

Layer extraction preserves each copied ``<maplayer>``'s own ``<id>``
unchanged (already a globally-unique UUID in the source project) and only
rewrites ``<layername>`` — so a base geo layer's existing
``<vectorjoins><join joinLayerId=...>`` continues to point at the correct
(also copied, also id-unchanged) attr layer without any join rewriting
needed. Everything downstream that varies per run (ids, datasources,
joins to the *run's* clones) is handled by
:func:`openplaces.viz.qgis_map.generator.generate_qgz`, not here.

Run with ``python build_template_from_reference.py <path to US-NC-AR.qgz>``
to regenerate ``openplaces_template.qgz`` in this directory. Re-run whenever
the reference project gets updated with better/newer styling to pull in.
"""

from __future__ import annotations

import argparse
import copy
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

_HERE = Path(__file__).parent
_OUTPUT_PATH = _HERE / 'openplaces_template.qgz'


@dataclass(frozen=True)
class Variant:
    template_name: str
    source_name: str
    label: str


@dataclass(frozen=True)
class BaseLayer:
    template_name: str
    source_geo_name: str
    source_attr_name: str | None
    group: str
    checked: bool
    variants: tuple[Variant, ...] = field(default_factory=tuple)
    #: True for a recipe whose own `save_to.combined` is true (attributes and
    #: geometry live in one file, e.g. `US_footprint-openplaces-2026`) — no attr
    #: sibling is extracted, and the join is stripped from every clone.
    combined: bool = False


BASE_LAYERS: tuple[BaseLayer, ...] = (
    BaseLayer(
        template_name='template_footprint_output',
        source_geo_name='US-NC-AR_footprint-cheer-2026',
        source_attr_name=None,
        group='Buildings/openplaces',
        checked=True,
        combined=True,  # US_footprint-openplaces-2026 recipe: save_to.combined: true
        variants=(
            Variant(
                'template_footprint_output__roof_shape',
                'roof_type - US-NC-AR_footprint-cheer-2026',
                'Roof shape',
            ),
            Variant(
                'template_footprint_output__n_stories',
                'n_stories - US-NC-AR_footprint-cheer-2026',
                'Number of stories',
            ),
            Variant(
                'template_footprint_output__occupancy_type_conflict',
                'occupancy_type_conflict - US-NC-AR_footprint-cheer-2026',
                'Occupancy type (evidence conflicts)',
            ),
            Variant(
                'template_footprint_output__priority_on_parcel',
                'role - US-NC-AR_footprint-cheer-2026',
                'Priority on parcel',
            ),
            Variant(
                'template_footprint_output__geometry_source',
                'source - US-NC-AR_footprint-cheer-2026 copy copy',
                'Geometry source',
            ),
            Variant(
                'template_footprint_output__structure_value',
                'structure_value - US-NC-AR_footprint-cheer-2026 copy',
                'Structure value (NSI, per area)',
            ),
            Variant(
                'template_footprint_output__improvement_value',
                'improvement_value - US-NC-AR_footprint-cheer-2026',
                'Improvement value (parcel, per area)',
            ),
        ),
    ),
    BaseLayer(
        'template_parcel_output',
        'US-NC-AR_parcel-openplaces-2026',
        'US-NC-AR_parcel-openplaces-2026_attr',
        'Parcels',
        True,
    ),
    BaseLayer(
        'template_footprint_obm_input',
        'US-NC-AR_footprint-obm-2025',
        'US-NC-AR_footprint-obm-2025_attr',
        'Buildings/inputs',
        False,
    ),
    BaseLayer(
        'template_footprint_fema_input',
        'US-NC-AR_footprint-fema-2023',
        'US-NC-AR_footprint-fema-2023_attr',
        'Buildings/inputs',
        False,
    ),
    BaseLayer(
        'template_footprint_microsoft_input',
        'US-NC-AR_footprint-microsoft-v2',
        'US-NC-AR_footprint-microsoft-v2_attr',
        'Buildings/inputs',
        False,
    ),
    BaseLayer(
        'template_footprint_ncdps_input',
        'US-NC-AR_footprint-ncdps-2023',
        'US-NC-AR_footprint-ncdps-2023_attr',
        'Buildings/inputs',
        False,
    ),
    BaseLayer(
        'template_dwelling_overture_input',
        'US-NC-AR_dwelling-overture-2025',
        'US-NC-AR_dwelling-overture-2025_attr',
        'Buildings/inputs',
        False,
    ),
    BaseLayer(
        'template_building_nsi_input',
        'US-NC-AR_building-nsi-2022',
        'US-NC-AR_building-nsi-2022_attr',
        'Buildings/inputs',
        False,
        variants=(
            Variant(
                'template_building_nsi_input__occupancy_type',
                'US-NC-AR_building-nsi-2022 occupancy_type',
                'Occupancy type',
            ),
        ),
    ),
    BaseLayer(
        'template_parcel_nconemap_input',
        'US-NC-AR_parcel-nconemap-2025',
        'US-NC-AR_parcel-nconemap-2025_attr',
        'Parcels',
        False,
        variants=(
            Variant(
                'template_parcel_nconemap_input__land_use_type',
                'land_use_type - US-NC-AR_parcel-nconemap-2025',
                'Land use (detailed)',
            ),
        ),
    ),
    BaseLayer(
        'template_admin_boundary',
        'admin-openplaces-2026_admin2',
        'admin-openplaces-2026_admin2_attr',
        'Admin',
        True,
    ),
)

# (template_name, source_name) - single-file raster basemaps, no attr sibling.
RASTER_BASEMAPS: tuple[tuple[str, str], ...] = (
    ('template_basemap_google_satellite', 'Google Satellite'),
    ('template_basemap_google_roads', 'Google Roads'),
)

# CARTO Positron, not OSM standard: OSM's default style renders building
# footprints prominently, which visually competes with the recipe's own
# footprint layer drawn on top. Positron renders buildings as flat light-gray
# fill instead. No API key required (same free-tier XYZ pattern as OSM);
# attribution "(c) OpenStreetMap contributors (c) CARTO" applies.
_OSM_DATASOURCE = (
    'type=xyz&url=https://basemaps.cartocdn.com/light_all/%7Bz%7D/%7Bx%7D/%7By%7D.png'
    '&zmax=20&zmin=0&crs=EPSG3857'
)


def _extract_doctype(raw: bytes) -> bytes | None:
    import re

    match = re.search(rb'<!DOCTYPE[^>]*>', raw[:1000])
    return match.group(0) if match else None


def _index_by_layername(projectlayers: ET.Element) -> dict[str, ET.Element]:
    index: dict[str, ET.Element] = {}
    for maplayer in projectlayers.findall('maplayer'):
        ln = maplayer.find('layername')
        if ln is not None and ln.text:
            index[ln.text] = maplayer
    return index


def _renamed_clone(
    source_index: dict[str, ET.Element],
    source_name: str,
    new_name: str,
    *,
    placeholder_datasource: bool = False,
) -> ET.Element:
    if source_name not in source_index:
        raise KeyError(f'Reference project is missing expected layer {source_name!r}.')
    clone = copy.deepcopy(source_index[source_name])
    clone.find('layername').text = new_name
    if placeholder_datasource:
        _set_placeholder_datasource(clone, new_name)
    return clone


def _set_placeholder_datasource(elem: ET.Element, name: str) -> None:
    """Replace *elem*'s `<datasource>` with an unambiguous placeholder.

    The generator always overwrites cloned layers' `<datasource>` at
    generation time and prunes the original template prototype it cloned
    from out of the generated output (see `generator.py`'s pruning step), so
    this value is only ever visible when opening this template file
    directly, never in generated output. Setting it to an
    obviously-synthetic `./{name}.parquet` — rather than carrying over the
    reference project's real-but-wrong-from-here relative path — avoids it
    reading as an almost-correct broken path.
    """
    elem.find('datasource').text = f'./{name}.parquet'


def _set_text(elem: ET.Element, tag: str, text: str) -> ET.Element:
    child = ET.SubElement(elem, tag)
    child.text = text
    return child


def _tree_layer(*, layer_id: str, name: str, source: str, checked: bool) -> ET.Element:
    return ET.Element(
        'layer-tree-layer',
        {
            'legend_exp': '',
            'patch_size': '-1,-1',
            'source': source,
            'id': layer_id,
            'name': name,
            'providerKey': 'ogr',
            'checked': 'Qt::Checked' if checked else 'Qt::Unchecked',
            'legend_split_behavior': '0',
            'expanded': '1',
        },
    )


def _legend_layer(*, layer_id: str, name: str, checked: bool) -> ET.Element:
    legendlayer = ET.Element(
        'legendlayer',
        {
            'showFeatureCount': '0',
            'name': name,
            'open': 'true',
            'checked': 'Qt::Checked' if checked else 'Qt::Unchecked',
            'drawingOrder': '-1',
        },
    )
    filegroup = ET.SubElement(
        legendlayer, 'filegroup', {'hidden': 'false', 'open': 'true'}
    )
    ET.SubElement(
        filegroup,
        'legendlayerfile',
        {'layerid': layer_id, 'isInOverview': '0', 'visible': '0'},
    )
    return legendlayer


def _get_or_create_group(root_group: ET.Element, group_path: str) -> ET.Element:
    current = root_group
    for segment in [s for s in group_path.split('/') if s]:
        found = next(
            (
                c
                for c in current.findall('layer-tree-group')
                if c.get('name') == segment
            ),
            None,
        )
        if found is None:
            found = ET.SubElement(
                current,
                'layer-tree-group',
                {
                    'groupLayer': '',
                    'name': segment,
                    'checked': 'Qt::Checked',
                    'expanded': '1',
                },
            )
        current = found
    return current


def _get_or_create_legendgroup(root_legend: ET.Element, group_path: str) -> ET.Element:
    current = root_legend
    for segment in [s for s in group_path.split('/') if s]:
        found = next(
            (c for c in current.findall('legendgroup') if c.get('name') == segment),
            None,
        )
        if found is None:
            found = ET.SubElement(
                current,
                'legendgroup',
                {'name': segment, 'open': 'true', 'checked': 'Qt::Checked'},
            )
        current = found
    return current


def _fallback_layers(srs_element: ET.Element | None) -> tuple[ET.Element, ET.Element]:
    """A minimal, deliberately generic single-symbol geo+attr pair.

    *srs_element* is a real data layer's ``<srs>`` block (from a layer
    already cloned out of the reference project), reused as-is so the
    fallback layer -- unlike the rest of the template -- isn't hand-authored
    from scratch, has a real CRS too, instead of QGIS treating it as having
    none.
    """
    attr = ET.Element('maplayer', {'type': 'vector', 'geometry': 'NoGeometry'})
    _set_text(attr, 'id', f'template_fallback_attr_{uuid.uuid4().hex}')
    _set_text(attr, 'datasource', './template_fallback.parquet')
    _set_text(attr, 'layername', 'template_fallback_attr')
    provider = ET.SubElement(attr, 'provider', {'encoding': 'UTF-8'})
    provider.text = 'ogr'

    geo = ET.Element(
        'maplayer', {'type': 'vector', 'wkbType': 'Polygon', 'geometry': 'Polygon'}
    )
    geo_id = f'template_fallback_{uuid.uuid4().hex}'
    _set_text(geo, 'id', geo_id)
    _set_text(geo, 'datasource', './template_fallback.parquet')
    _set_text(geo, 'layername', 'template_fallback')
    if srs_element is not None:
        geo.append(copy.deepcopy(srs_element))
    provider = ET.SubElement(geo, 'provider', {'encoding': 'UTF-8'})
    provider.text = 'ogr'
    vectorjoins = ET.SubElement(geo, 'vectorjoins')
    ET.SubElement(
        vectorjoins,
        'join',
        {
            'joinFieldName': '_join_id',
            'targetFieldName': '_join_id',
            'joinLayerId': attr.find('id').text,
            'memoryCache': '1',
            'customPrefix': '',
            'hasCustomPrefix': '0',
            'dynamicForm': '0',
            'upsertOnEdit': '0',
            'cascadedDelete': '0',
            'editable': '0',
        },
    )
    ET.SubElement(
        geo,
        'renderer-v2',
        {
            'type': 'singleSymbol',
            'forceraster': '0',
            'symbollevels': '0',
            'enableorderby': '0',
        },
    )
    return geo, attr


def _osm_basemap(google_raster: ET.Element) -> ET.Element:
    """Clone the structure of a working Google XYZ raster layer for OSM."""
    clone = copy.deepcopy(google_raster)
    clone.find('id').text = f'template_basemap_open_{uuid.uuid4().hex}'
    clone.find('datasource').text = _OSM_DATASOURCE
    clone.find('layername').text = 'template_basemap_open'
    return clone


def _print_layout() -> ET.Element:
    """Best-effort minimal print layout: title label, legend, scale bar, north arrow.

    No real-world example exists in any of the user's reference projects to
    copy exact QGIS layout-item XML from (every one checked has an empty
    ``<Layouts/>``), so this is authored from general knowledge of the QGIS
    3.x layout item schema rather than lifted from a working file. Treat
    this as the piece most likely to need a small cosmetic touch-up in the
    QGIS GUI (see templates/README.md's manual QA note).
    """
    layouts = ET.Element('Layouts')
    layout = ET.SubElement(layouts, 'Layout', {'name': 'Recipe map', 'units': 'mm'})
    ET.SubElement(layout, 'PageCollection').append(
        ET.fromstring(
            '<LayoutItem type="65538" position="0,0,mm" size="297,210,mm" '
            'positionOnPage="0,0,mm" positionIndex="0" itemRotation="0" '
            'uuid="{page}" id="" visibility="1" zValue="0">'
            '<PageSize units="mm"><PageSize width="297" height="210"/></PageSize>'
            '</LayoutItem>'.replace('{page}', str(uuid.uuid4()))
        )
    )
    layout.append(
        ET.fromstring(
            '<LayoutItem type="65539" position="5,5,mm" size="287,160,mm" '
            'positionOnPage="5,5,mm" positionIndex="1" itemRotation="0" '
            f'uuid="{uuid.uuid4()}" id="map" visibility="1" zValue="1" '
            'referencePoint="0">'
            '<mapExtent><mapExtentXmin>0</mapExtentXmin>'
            '<mapExtentYmin>0</mapExtentYmin>'
            '<mapExtentXmax>1</mapExtentXmax>'
            '<mapExtentYmax>1</mapExtentYmax></mapExtent>'
            '</LayoutItem>'
        )
    )
    title_expression = (
        "[% 'Recipe: ' || @recipe_id || '   Admin: ' || @admin_id || "
        "'   Generated: ' || @generated_at %]"
    )
    label_item = ET.fromstring(
        '<LayoutItem type="65541" position="5,168,mm" size="287,12,mm" '
        'positionOnPage="5,168,mm" positionIndex="2" itemRotation="0" '
        f'uuid="{uuid.uuid4()}" id="title" visibility="1" zValue="2">'
        '</LayoutItem>'
    )
    _set_text(label_item, 'label', title_expression)
    ET.SubElement(label_item, 'labelFont').text = ''
    layout.append(label_item)
    layout.append(
        ET.fromstring(
            '<LayoutItem type="65542" position="230,5,mm" size="62,120,mm" '
            'positionOnPage="230,5,mm" positionIndex="3" itemRotation="0" '
            f'uuid="{uuid.uuid4()}" id="legend" visibility="1" zValue="3" '
            'linkedMap="map">'
            '</LayoutItem>'
        )
    )
    layout.append(
        ET.fromstring(
            '<LayoutItem type="65546" position="5,182,mm" size="80,10,mm" '
            'positionOnPage="5,182,mm" positionIndex="4" itemRotation="0" '
            f'uuid="{uuid.uuid4()}" id="scalebar" visibility="1" zValue="4" '
            'linkedMap="map" style="Single Box">'
            '</LayoutItem>'
        )
    )
    layout.append(
        ET.fromstring(
            '<LayoutItem type="65540" position="270,182,mm" size="15,15,mm" '
            'positionOnPage="270,182,mm" positionIndex="5" itemRotation="0" '
            f'uuid="{uuid.uuid4()}" id="northarrow" visibility="1" zValue="5" '
            'resizeMode="0" '
            'path=":/images/north_arrows/layout_default_north_arrow.svg">'
            '</LayoutItem>'
        )
    )
    return layouts


def build(source_qgz: Path, output_path: Path) -> None:
    with zipfile.ZipFile(source_qgz) as zf:
        [qgs_name] = [n for n in zf.namelist() if n.lower().endswith('.qgs')]
        source_bytes = zf.read(qgs_name)
        styles_name = next(
            (n for n in zf.namelist() if n.lower().endswith('_styles.db')), None
        )
        styles_bytes = zf.read(styles_name) if styles_name else None

    doctype = _extract_doctype(source_bytes)
    source_root = ET.fromstring(source_bytes)
    source_projectlayers = source_root.find('projectlayers')
    source_index = _index_by_layername(source_projectlayers)
    source_by_id = {
        ml.find('id').text: ml for ml in source_projectlayers.findall('maplayer')
    }
    source_mapcanvas = source_root.find('.//mapcanvas')

    root = ET.Element(
        'qgis',
        {
            'projectname': '',
            'version': source_root.get('version', '3.44'),
            'saveDateTime': datetime.now(UTC).isoformat(timespec='seconds'),
        },
    )
    _set_text(root, 'title', '')

    projectlayers = ET.SubElement(root, 'projectlayers')
    tree_root = ET.SubElement(root, 'layer-tree-group')
    legend_root = ET.SubElement(root, 'legend', {'updateDrawingOrder': 'true'})

    def _place(geo_id: str, name: str, group: str, checked: bool, source: str) -> None:
        tree_group = _get_or_create_group(tree_root, group)
        tree_group.append(
            _tree_layer(layer_id=geo_id, name=name, source=source, checked=checked)
        )
        legend_group = _get_or_create_legendgroup(legend_root, group)
        legend_group.append(_legend_layer(layer_id=geo_id, name=name, checked=checked))

    def _strip_joins(elem: ET.Element) -> None:
        vectorjoins = elem.find('vectorjoins')
        if vectorjoins is not None:
            for join in list(vectorjoins.findall('join')):
                vectorjoins.remove(join)

    real_data_srs: ET.Element | None = None
    for base in BASE_LAYERS:
        geo_source = source_index.get(base.source_geo_name)
        if geo_source is None:
            raise KeyError(
                f'Reference project is missing layer {base.source_geo_name!r}.'
            )
        geo = copy.deepcopy(geo_source)
        geo.find('layername').text = base.template_name
        _set_placeholder_datasource(geo, base.template_name)
        if real_data_srs is None and geo.find('srs') is not None:
            real_data_srs = geo.find('srs')

        if base.combined:
            # Attributes and geometry live in one file (recipe's own
            # save_to.combined: true) — no attr sibling, strip any join the
            # reference layer happens to carry (see footprint-output: the
            # reference project loaded this one both directly, with no join,
            # and redundantly re-joined to itself for the styled variants).
            _strip_joins(geo)
            projectlayers.append(geo)
        else:
            # Resolve the attr layer by the geo layer's *own join target id*,
            # not by name: the reference project has duplicate-named
            # (different-id) attr layers left over from repeated
            # Processing-toolbox loads (observed for
            # 'admin-openplaces-2026_admin2_attr'), so a name-only lookup can
            # silently grab the wrong instance and produce a dangling join.
            join = geo_source.find('vectorjoins/join')
            if join is None:
                raise ValueError(
                    f'{base.source_geo_name!r} has no join to resolve its attr layer.'
                )
            attr_source = source_by_id.get(join.get('joinLayerId'))
            if attr_source is None:
                raise KeyError(
                    f'Join target {join.get("joinLayerId")!r} of '
                    f'{base.source_geo_name!r} not found among source layers.'
                )
            if attr_source.find('layername').text != base.source_attr_name:
                raise ValueError(
                    f'{base.source_geo_name!r} joins a layer named '
                    f'{attr_source.find("layername").text!r}, expected '
                    f'{base.source_attr_name!r}.'
                )
            attr = copy.deepcopy(attr_source)
            attr_name = f'{base.template_name}_attr'
            attr.find('layername').text = attr_name
            _set_placeholder_datasource(attr, attr_name)
            projectlayers.append(attr)
            projectlayers.append(geo)

        _place(
            geo.find('id').text,
            base.template_name,
            base.group,
            base.checked,
            geo.find('datasource').text,
        )
        for variant in base.variants:
            vgeo = _renamed_clone(
                source_index,
                variant.source_name,
                variant.template_name,
                placeholder_datasource=True,
            )
            if base.combined:
                _strip_joins(vgeo)
            projectlayers.append(vgeo)
            _place(
                vgeo.find('id').text,
                variant.template_name,
                base.group,
                False,
                vgeo.find('datasource').text,
            )

    google_satellite_raw = source_index['Google Satellite']
    for template_name, source_name in RASTER_BASEMAPS:
        clone = _renamed_clone(source_index, source_name, template_name)
        projectlayers.append(clone)
        _place(
            clone.find('id').text,
            template_name,
            'Basemaps',
            False,
            clone.find('datasource').text,
        )

    osm = _osm_basemap(google_satellite_raw)
    projectlayers.append(osm)
    _place(
        osm.find('id').text,
        'template_basemap_open',
        'Basemaps',
        True,
        osm.find('datasource').text,
    )

    fallback_geo, fallback_attr = _fallback_layers(real_data_srs)
    projectlayers.append(fallback_attr)
    projectlayers.append(fallback_geo)
    _place(
        fallback_geo.find('id').text,
        'template_fallback',
        'Unstyled',
        False,
        fallback_geo.find('datasource').text,
    )

    properties = ET.SubElement(root, 'properties')
    ET.SubElement(properties, 'Variables')
    # Legacy flag QGIS still writes/reads for backward compatibility (3.x
    # always reprojects on the fly regardless), but a project missing it
    # entirely -- unlike every real QGIS-GUI-saved project, which always
    # includes it -- can show as having no properly recognized project CRS.
    spatial_ref_sys = ET.SubElement(properties, 'SpatialRefSys')
    projections_enabled = ET.SubElement(spatial_ref_sys, 'ProjectionsEnabled')
    projections_enabled.set('type', 'int')
    projections_enabled.text = '1'

    source_vertical_crs = source_root.find('verticalCrs')
    if source_vertical_crs is not None:
        root.append(copy.deepcopy(source_vertical_crs))

    mapcanvas = ET.SubElement(
        root, 'mapcanvas', {'annotationsVisible': '1', 'name': 'theMapCanvas'}
    )
    _set_text(mapcanvas, 'units', 'meters')
    extent = ET.SubElement(mapcanvas, 'extent')
    for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
        _set_text(extent, tag, '0')
    _set_text(mapcanvas, 'rotation', '0')
    if source_mapcanvas is not None:
        src_srs = source_mapcanvas.find('destinationsrs')
        if src_srs is not None:
            mapcanvas.append(copy.deepcopy(src_srs))
            src_spatialrefsys = src_srs.find('spatialrefsys')
            if src_spatialrefsys is not None:
                # <mapcanvas><destinationsrs> alone isn't enough: QGIS treats
                # the top-level <projectCrs> as the actual project CRS
                # setting (status bar, on-the-fly reprojection target). A
                # project missing it opens with no reliable working CRS, so
                # layers and basemap silently fail to line up.
                project_crs = ET.SubElement(root, 'projectCrs')
                project_crs.append(copy.deepcopy(src_spatialrefsys))

    root.append(_print_layout())

    body = ET.tostring(root, encoding='utf-8', xml_declaration=False)
    header = b"<?xml version='1.0' encoding='UTF-8'?>\n"
    if doctype:
        header += doctype + b'\n'
    output_bytes = header + body

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as out_zf:
        out_zf.writestr(f'{output_path.stem}.qgs', output_bytes)
        if styles_bytes is not None:
            out_zf.writestr(f'{output_path.stem}_styles.db', styles_bytes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'source_qgz', type=Path, help='Path to the reference US-NC-AR.qgz project.'
    )
    parser.add_argument('--output', type=Path, default=_OUTPUT_PATH)
    args = parser.parse_args()
    build(args.source_qgz, args.output)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
