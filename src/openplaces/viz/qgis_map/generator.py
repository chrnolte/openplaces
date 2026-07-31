"""Build a standardized QGIS project (.qgz) for a curate recipe + admin unit.

Clones the join-based layer pattern from a hand-authored template project
(built in the QGIS GUI, see ``qgis/templates/README.md``) for every resolved
layer, prunes template layers that are not relevant to this run, and
rewrites project variables and the map extent. Pure ``zipfile`` +
``xml.etree.ElementTree`` — no ``qgis.core`` import, so this works in any
conda environment, not just inside QGIS.
"""

from __future__ import annotations

import copy
import os
import random
import re
import uuid
import warnings
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

import pandas as pd
from pyproj import CRS

from openplaces.config import cfg
from openplaces.core.schema import AdminId, sanitize
from openplaces.io.readers import get_admin
from openplaces.path import path as build_path
from openplaces.recipe import get_recipe_by_id, get_recipe_id
from openplaces.viz.qgis_map.resolver import LayerSpec, resolve_layers
from openplaces.viz.qgis_map.style_registry import (
    get_fallback_style,
    get_static_styles,
    get_style,
    get_style_variants,
)


def _default_template_path() -> Path:
    return resources.files('openplaces.qgis') / 'templates' / 'openplaces_template.qgz'


def _default_output_path(recipe: dict, admin_id: AdminId) -> Path:
    return build_path(
        admin_id,
        recipe.get('entity'),
        filename='map',
        root=cfg.share_dir,
        default_extension='qgz',
    )


def _find_single_member(zf: zipfile.ZipFile, suffix: str) -> str:
    matches = [n for n in zf.namelist() if n.lower().endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(
            f'Expected exactly one {suffix!r} member in the template, found {matches}.'
        )
    return matches[0]


def _extract_doctype(raw: bytes) -> bytes | None:
    match = re.search(rb'<!DOCTYPE[^>]*>', raw[:1000])
    return match.group(0) if match else None


def _set_text(elem: ET.Element, tag: str, text: str) -> None:
    child = elem.find(tag)
    if child is None:
        child = ET.SubElement(elem, tag)
    child.text = text


def _index_maplayers_by_layername(
    projectlayers: ET.Element,
) -> dict[str, tuple[ET.Element, str]]:
    """Map literal ``<layername>`` text to (maplayer element, its `<id>` text)."""
    index: dict[str, tuple[ET.Element, str]] = {}
    for maplayer in projectlayers.findall('maplayer'):
        layername_el = maplayer.find('layername')
        id_el = maplayer.find('id')
        if layername_el is not None and layername_el.text and id_el is not None:
            index[layername_el.text] = (maplayer, id_el.text or '')
    return index


def _relative_datasource(target: Path, *, anchor: Path) -> str:
    """Return *target*'s path relative to *anchor*, `/`-separated.

    Falls back to an absolute path if *target* and *anchor* are on different
    drives on Windows (`os.path.relpath` cannot express that as a relative
    path) — a real possibility since individual `STANDARD_DIRS` buckets can
    be configured to an absolute path outside `data_root`.
    """
    try:
        rel = os.path.relpath(target.resolve(), start=anchor)
    except ValueError:
        return str(target.resolve())
    return rel.replace(os.sep, '/')


def _clone_maplayer(
    template_layer: ET.Element, *, new_id: str, new_layername: str, new_datasource: str
) -> ET.Element:
    clone = copy.deepcopy(template_layer)
    _set_text(clone, 'id', new_id)
    _set_text(clone, 'datasource', new_datasource)
    _set_text(clone, 'layername', new_layername)
    return clone


def _rewrite_join(geo_clone: ET.Element, *, old_attr_id: str, new_attr_id: str) -> None:
    vectorjoins = geo_clone.find('vectorjoins')
    if vectorjoins is None:
        return
    for join in vectorjoins.findall('join'):
        if join.get('joinLayerId') == old_attr_id:
            join.set('joinLayerId', new_attr_id)


def _set_join_target(clone: ET.Element, *, new_attr_id: str) -> None:
    """Point *clone*'s join(s) at *new_attr_id*, regardless of their old target.

    Used for style-variant clones: each variant's own template join targets a
    different (variant-specific) template attr layer, so there is no single
    ``old_attr_id`` to match against as :func:`_rewrite_join` does for the
    base clone — every variant's join is repointed at the same shared,
    freshly-cloned attr layer instead.
    """
    vectorjoins = clone.find('vectorjoins')
    if vectorjoins is None:
        return
    for join in vectorjoins.findall('join'):
        join.set('joinLayerId', new_attr_id)


def _random_fill_symbol(*, name: str, seed: str) -> ET.Element:
    """Build a `<symbol type="fill">` with a 50%-opacity color seeded by *seed*.

    Seeding `random.Random` with the category's own label (rather than
    leaving it unseeded) makes the color stable per label across re-exports
    and across admin units, instead of reshuffling every run.
    """
    rng = random.Random(seed)
    r, g, b = (rng.randint(0, 255) for _ in range(3))
    symbol = ET.Element(
        'symbol',
        {
            'type': 'fill',
            'name': name,
            'alpha': '1',
            'clip_to_extent': '1',
            'force_rhr': '0',
        },
    )
    layer = ET.SubElement(
        symbol,
        'layer',
        {
            'class': 'SimpleFill',
            'enabled': '1',
            'locked': '0',
            'pass': '0',
            'id': str(uuid.uuid4()),
        },
    )
    options = ET.SubElement(layer, 'Option', {'type': 'Map'})
    for opt_name, opt_value in (
        ('color', f'{r},{g},{b},127'),
        ('outline_color', '35,35,35,255'),
        ('outline_style', 'solid'),
        ('outline_width', '0.26'),
        ('outline_width_unit', 'MM'),
        ('style', 'solid'),
    ):
        ET.SubElement(
            options, 'Option', {'type': 'QString', 'name': opt_name, 'value': opt_value}
        )
    return symbol


def _regenerate_categories(
    clone: ET.Element, data_path: Path, attr_field: str, *, verbose: bool
) -> None:
    """Rebuild *clone*'s categorized-symbol renderer from *data_path*'s real data.

    Replaces the template-baked `<categories>`/`<symbols>` lists with one
    entry per distinct non-null value of *attr_field* actually present in
    this run's resolved data — for columns whose value set is data-dependent
    (evidence conflicts, per-source crosswalked classifications) rather than
    a fixed enum, the template author's reference county's category list is
    otherwise wrong or incomplete for any other admin unit. Fails soft (warns
    and leaves the template-baked renderer as-is) on any read/shape problem,
    matching :func:`_update_extent`'s posture toward missing/bad data.
    """
    renderer = clone.find('.//renderer-v2')
    categories_el = renderer.find('categories') if renderer is not None else None
    symbols_el = renderer.find('symbols') if renderer is not None else None
    if renderer is None or categories_el is None or symbols_el is None:
        if verbose:
            warnings.warn(
                f'{attr_field!r} is flagged dynamic_categorize_attr but the cloned '
                'layer has no categorized-symbol <renderer-v2>; leaving it untouched.'
            )
        return

    try:
        values = pd.read_parquet(data_path, columns=[attr_field])[attr_field]
    except Exception as exc:
        warnings.warn(
            f'Could not read {attr_field!r} from {data_path} to regenerate '
            f'categories: {exc}; leaving template-baked categories as-is.'
        )
        return
    unique_values = sorted(str(v) for v in values.dropna().unique().tolist())
    if not unique_values:
        warnings.warn(
            f'{attr_field!r} has no non-null values in {data_path}; leaving '
            'template-baked categories as-is.'
        )
        return

    for child in list(categories_el):
        categories_el.remove(child)
    for child in list(symbols_el):
        symbols_el.remove(child)
    renderer.set('attr', attr_field)

    for i, value in enumerate(unique_values):
        symbol_name = str(i)
        ET.SubElement(
            categories_el,
            'category',
            {
                'value': value,
                'label': value,
                'symbol': symbol_name,
                'render': 'true',
            },
        )
        symbols_el.append(_random_fill_symbol(name=symbol_name, seed=value))


def _find_or_create_group(
    root_group: ET.Element, group_path: str, *, verbose: bool
) -> ET.Element:
    current = root_group
    for segment in [s for s in group_path.split('/') if s]:
        found = None
        for child in current.findall('layer-tree-group'):
            if child.get('name') == segment:
                found = child
                break
        if found is None:
            if verbose:
                warnings.warn(
                    f'Template layer tree is missing group {segment!r}; creating it.'
                )
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


def _find_or_create_legendgroup(
    root_legend: ET.Element, group_path: str, *, verbose: bool
) -> ET.Element:
    current = root_legend
    for segment in [s for s in group_path.split('/') if s]:
        found = None
        for child in current.findall('legendgroup'):
            if child.get('name') == segment:
                found = child
                break
        if found is None:
            found = ET.SubElement(
                current,
                'legendgroup',
                {'name': segment, 'open': 'true', 'checked': 'Qt::Checked'},
            )
        current = found
    return current


def _insert_layer_tree_layer(
    group_elem: ET.Element, *, layer_id: str, name: str, source: str, checked: bool
) -> None:
    ET.SubElement(
        group_elem,
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


def _insert_legend_layer(
    group_elem: ET.Element, *, layer_id: str, name: str, checked: bool
) -> None:
    legendlayer = ET.SubElement(
        group_elem,
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


def _remove_layer_tree_entry(group: ET.Element, layer_id: str) -> bool:
    for child in list(group.findall('layer-tree-layer')):
        if child.get('id') == layer_id:
            group.remove(child)
            return True
    for child in list(group.findall('layer-tree-group')):
        if _remove_layer_tree_entry(child, layer_id):
            return True
    return False


def _remove_legend_entry(group: ET.Element, layer_id: str) -> bool:
    for legendlayer in list(group.findall('legendlayer')):
        filegroup = legendlayer.find('filegroup')
        legendlayerfile = (
            filegroup.find('legendlayerfile') if filegroup is not None else None
        )
        if legendlayerfile is not None and legendlayerfile.get('layerid') == layer_id:
            group.remove(legendlayer)
            return True
    for child in list(group.findall('legendgroup')):
        if _remove_legend_entry(child, layer_id):
            return True
    return False


def _set_layer_tree_checked(group: ET.Element, layer_id: str, checked: bool) -> bool:
    checked_str = 'Qt::Checked' if checked else 'Qt::Unchecked'
    for child in group.findall('layer-tree-layer'):
        if child.get('id') == layer_id:
            child.set('checked', checked_str)
            return True
    for child in group.findall('layer-tree-group'):
        if _set_layer_tree_checked(child, layer_id, checked):
            return True
    return False


def _set_legend_checked(group: ET.Element, layer_id: str, checked: bool) -> bool:
    checked_str = 'Qt::Checked' if checked else 'Qt::Unchecked'
    for legendlayer in group.findall('legendlayer'):
        filegroup = legendlayer.find('filegroup')
        legendlayerfile = (
            filegroup.find('legendlayerfile') if filegroup is not None else None
        )
        if legendlayerfile is not None and legendlayerfile.get('layerid') == layer_id:
            legendlayer.set('checked', checked_str)
            return True
    for child in group.findall('legendgroup'):
        if _set_legend_checked(child, layer_id, checked):
            return True
    return False


def _prune_empty_groups(group: ET.Element) -> None:
    for child in list(group.findall('layer-tree-group')):
        _prune_empty_groups(child)
        if not child.findall('layer-tree-layer') and not child.findall(
            'layer-tree-group'
        ):
            group.remove(child)


def _prune_empty_legend_groups(group: ET.Element) -> None:
    for child in list(group.findall('legendgroup')):
        _prune_empty_legend_groups(child)
        if not child.findall('legendlayer') and not child.findall('legendgroup'):
            group.remove(child)


def _collapse_layer_tree(layer_tree_root: ET.Element) -> None:
    """Collapse every group and layer entry so the Layers panel opens closed.

    Applies regardless of whether a group/layer entry came from the template
    (already-present groups like ``Buildings``) or was freshly inserted this
    run — both are hand-set to ``expanded="1"`` by the template author and by
    :func:`_find_or_create_group`/:func:`_insert_layer_tree_layer`
    respectively, so this is a single pass overriding both.
    """
    for el in layer_tree_root.iter():
        if el.tag in ('layer-tree-group', 'layer-tree-layer'):
            el.set('expanded', '0')


def _collapse_legend(legend_root: ET.Element) -> None:
    """Legacy-legend-tree counterpart to :func:`_collapse_layer_tree`."""
    for el in legend_root.iter():
        if el.tag in ('legendgroup', 'legendlayer'):
            el.set('open', 'false')


def _set_project_variables(
    root: ET.Element, *, recipe: dict, admin_id: AdminId
) -> None:
    properties = root.find('properties')
    if properties is None:
        properties = ET.SubElement(root, 'properties')
    variables = properties.find('Variables')
    if variables is None:
        variables = ET.SubElement(properties, 'Variables')
    names_el = variables.find('variableNames')
    if names_el is None:
        names_el = ET.SubElement(variables, 'variableNames', {'type': 'QStringList'})
    values_el = variables.find('variableValues')
    if values_el is None:
        values_el = ET.SubElement(variables, 'variableValues', {'type': 'QStringList'})

    existing_names = [v.text or '' for v in names_el.findall('value')]
    existing_values = [v.text or '' for v in values_el.findall('value')]
    pairs = dict(zip(existing_names, existing_values, strict=False))
    pairs.update(
        {
            'recipe_id': get_recipe_id(recipe),
            'admin_id': str(admin_id),
            'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
        }
    )

    for el in (names_el, values_el):
        for child in list(el):
            el.remove(child)
    for name, value in pairs.items():
        ET.SubElement(names_el, 'value').text = name
        ET.SubElement(values_el, 'value').text = value


def _update_extent(root: ET.Element, admin_id: AdminId, *, verbose: bool) -> None:
    mapcanvas = root.find('.//mapcanvas')
    if mapcanvas is None:
        if verbose:
            warnings.warn('Template has no <mapcanvas>; skipping extent update.')
        return
    extent = mapcanvas.find('extent')
    wkt_el = mapcanvas.find('destinationsrs/spatialrefsys/wkt')
    if extent is None or wkt_el is None or not wkt_el.text:
        if verbose:
            warnings.warn(
                'Template <mapcanvas> is missing extent/destinationsrs; skipping.'
            )
        return
    try:
        admin_gdf = get_admin(admin_id, geom=True)
    except Exception as exc:
        if verbose:
            warnings.warn(f'Could not resolve admin geometry for {admin_id}: {exc}')
        return
    target_crs = CRS.from_wkt(wkt_el.text)
    reprojected = admin_gdf.to_crs(target_crs)
    xmin, ymin, xmax, ymax = reprojected.total_bounds
    for tag, value in (('xmin', xmin), ('ymin', ymin), ('xmax', xmax), ('ymax', ymax)):
        _set_text(extent, tag, repr(float(value)))


def _ensure_project_crs(root: ET.Element) -> None:
    """Mirror `<mapcanvas><destinationsrs>` into a top-level `<projectCrs>`.

    QGIS treats `<projectCrs>` — not `<mapcanvas><destinationsrs>` — as the
    actual project CRS setting (status bar, on-the-fly reprojection target);
    a project missing it has no reliable working CRS and layers silently
    fail to line up with the basemap. The packaged template writes both (see
    `qgis/templates/build_template_from_reference.py`); this is a defensive
    backstop in case a template is ever hand-edited without it.
    """
    if root.find('projectCrs') is not None:
        return
    mapcanvas = root.find('.//mapcanvas')
    if mapcanvas is None:
        return
    src_spatialrefsys = mapcanvas.find('destinationsrs/spatialrefsys')
    if src_spatialrefsys is None:
        return
    project_crs = ET.SubElement(root, 'projectCrs')
    project_crs.append(copy.deepcopy(src_spatialrefsys))


def _ensure_projections_enabled(root: ET.Element) -> None:
    """Ensure `<properties><SpatialRefSys><ProjectionsEnabled>` is set to 1.

    Legacy flag QGIS still writes on every save for backward compatibility
    (3.x always reprojects on the fly regardless of its value), but every
    real QGIS-GUI-saved project includes it — a project missing it entirely
    can show as having no properly recognized project CRS. Defensive
    backstop alongside :func:`_ensure_project_crs`.
    """
    properties = root.find('properties')
    if properties is None:
        properties = ET.SubElement(root, 'properties')
    spatial_ref_sys = properties.find('SpatialRefSys')
    if spatial_ref_sys is None:
        spatial_ref_sys = ET.SubElement(properties, 'SpatialRefSys')
    projections_enabled = spatial_ref_sys.find('ProjectionsEnabled')
    if projections_enabled is None:
        projections_enabled = ET.SubElement(spatial_ref_sys, 'ProjectionsEnabled')
    projections_enabled.set('type', 'int')
    projections_enabled.text = '1'


def _serialize(root: ET.Element, doctype: bytes | None) -> bytes:
    body = ET.tostring(root, encoding='utf-8', xml_declaration=False)
    header = b"<?xml version='1.0' encoding='UTF-8'?>\n"
    if doctype:
        header += doctype + b'\n'
    return header + body


def generate_qgz(
    recipe: str | dict,
    admin_id: str | AdminId,
    *,
    layer_specs: list[LayerSpec] | None = None,
    template_path: str | Path | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
    verbose: bool = False,
) -> Path:
    """Generate a standardized QGIS project (.qgz) for *recipe* at *admin_id*.

    Clones the geometry+attribute layer pair from the packaged template for
    every layer :func:`~openplaces.viz.qgis_map.resolver.resolve_layers`
    finds (or the given *layer_specs*), prunes template layers not used by
    this run, updates project variables (``recipe_id``, ``admin_id``,
    ``generated_at``) and the map extent, and writes the result to
    *output_path* (default: inside `cfg.share_dir`).

    Parameters
    ----------
    recipe : str or dict
        Curate-stage recipe ID or loaded recipe dict.
    admin_id : str or `openplaces.core.schema.AdminId`
        Admin unit to generate the map for.
    layer_specs : list of `~openplaces.viz.qgis_map.resolver.LayerSpec`, optional
        Pre-resolved layers to use instead of calling
        :func:`~openplaces.viz.qgis_map.resolver.resolve_layers`.
    template_path : str or pathlib.Path, optional
        Template ``.qgz`` to clone from. Defaults to the packaged
        ``qgis/templates/openplaces_template.qgz``.
    output_path : str or pathlib.Path, optional
        Where to write the generated ``.qgz``. Defaults to a path under
        `cfg.share_dir`.
    title : str, optional
        Project title to set. When omitted, the template's own (typically
        dynamic, project-variable-driven) title is left as-is.
    verbose : bool, optional
        Warn about template gaps and resolver skips.

    Returns
    -------
    pathlib.Path
        Path to the generated ``.qgz``.
    """
    recipe = get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe
    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)
    template_path = Path(template_path) if template_path else _default_template_path()
    if not template_path.exists():
        raise FileNotFoundError(
            f'QGIS map template not found at {template_path}. Author the standardized '
            'template in the QGIS GUI (see qgis/templates/README.md) and save it there '
            'before calling export_qgis_map.'
        )
    if layer_specs is None:
        layer_specs = resolve_layers(recipe, admin_id, verbose=verbose)

    # Resolved up front (not just at write time) so cloned layers' datasources
    # can be written relative to it — the output almost always lands in the
    # same data tree as the layers it references (see module docstring).
    output_path = (
        Path(output_path) if output_path else _default_output_path(recipe, admin_id)
    ).resolve()

    with zipfile.ZipFile(template_path) as zf:
        qgs_name = _find_single_member(zf, '.qgs')
        qgs_bytes = zf.read(qgs_name)
        styles_name = None
        styles_bytes = None
        for name in zf.namelist():
            if name != qgs_name:
                styles_name = name
                styles_bytes = zf.read(name)

    doctype = _extract_doctype(qgs_bytes)
    root = ET.fromstring(qgs_bytes)

    projectlayers = root.find('projectlayers')
    layer_tree_root = root.find('layer-tree-group')
    legend_root = root.find('legend')
    if projectlayers is None or layer_tree_root is None or legend_root is None:
        raise ValueError(
            'Template .qgs is missing <projectlayers>, root <layer-tree-group>, or '
            '<legend>; is this a valid QGIS project file?'
        )

    original_index = _index_maplayers_by_layername(projectlayers)
    static_styles_by_layername = {s.template_layer_name: s for s in get_static_styles()}

    new_maplayers: list[ET.Element] = []

    for spec in layer_specs:
        style = get_style(spec.entity_type, spec.source, spec.role)
        unstyled = style is None
        if unstyled:
            style = get_fallback_style()
            warnings.warn(
                f'No style registered for entity_type={spec.entity_type!r}, '
                f'source={spec.source!r}; using fallback style. Add a row to '
                'qgis_map_style_registry.csv to style it properly.'
            )

        geo_name = style.template_layer_name
        if geo_name not in original_index:
            raise ValueError(
                f'Template is missing layer {geo_name!r} required by style_key '
                f'{style.style_key!r}.'
            )
        geo_template, geo_template_id = original_index[geo_name]
        new_geo_id = f'{sanitize(spec.display_name)}_{uuid.uuid4().hex}'
        geo_datasource = _relative_datasource(spec.geo_path, anchor=output_path.parent)
        geo_clone = _clone_maplayer(
            geo_template,
            new_id=new_geo_id,
            new_layername=spec.display_name,
            new_datasource=geo_datasource,
        )
        if style.dynamic_categorize_attr:
            _regenerate_categories(
                geo_clone,
                spec.attr_path,
                style.dynamic_categorize_attr,
                verbose=verbose,
            )

        # (layer_id, display_name, checked) for every clone inserted into the
        # tree/legend for this spec: the base clone plus any style variants.
        tree_entries: list[tuple[str, str, bool]] = [
            (new_geo_id, spec.display_name, style.default_visible)
        ]

        new_attr_id: str | None = None
        if spec.combined:
            # Attributes and geometry already live in one file (e.g. a
            # share-ready terminal deliverable); no attr sibling to join.
            vectorjoins = geo_clone.find('vectorjoins')
            if vectorjoins is not None:
                for join in list(vectorjoins.findall('join')):
                    vectorjoins.remove(join)
            new_maplayers.append(geo_clone)
        else:
            attr_name = f'{geo_name}_attr'
            if attr_name not in original_index:
                raise ValueError(
                    f'Template is missing the {attr_name!r} attribute layer required '
                    f'by style_key {style.style_key!r}.'
                )
            attr_template, attr_template_id = original_index[attr_name]
            new_attr_id = f'{sanitize(spec.display_name)}_attr_{uuid.uuid4().hex}'
            attr_clone = _clone_maplayer(
                attr_template,
                new_id=new_attr_id,
                new_layername=f'{spec.display_name}_attr',
                new_datasource=_relative_datasource(
                    spec.attr_path, anchor=output_path.parent
                ),
            )
            _rewrite_join(
                geo_clone, old_attr_id=attr_template_id, new_attr_id=new_attr_id
            )
            new_maplayers.extend((attr_clone, geo_clone))

        # Style variants apply regardless of whether the base spec is
        # combined: a combined variant clone just shares the base's
        # datasource with no join (attributes are already on the geometry);
        # a non-combined variant clone joins the freshly-cloned attr layer.
        for variant in get_style_variants(style.style_key):
            variant_geo_name = variant.template_layer_name
            if variant_geo_name not in original_index:
                raise ValueError(
                    f'Template is missing layer {variant_geo_name!r} required by '
                    f'variant {variant.style_key!r} of style_key {style.style_key!r}.'
                )
            variant_template, _ = original_index[variant_geo_name]
            new_variant_id = (
                f'{sanitize(spec.display_name)}_{sanitize(variant.style_key)}_'
                f'{uuid.uuid4().hex}'
            )
            variant_label = variant.variant_label or variant.style_key
            variant_display_name = f'{spec.display_name} — {variant_label}'
            variant_clone = _clone_maplayer(
                variant_template,
                new_id=new_variant_id,
                new_layername=variant_display_name,
                new_datasource=geo_datasource,
            )
            if variant.dynamic_categorize_attr:
                _regenerate_categories(
                    variant_clone,
                    spec.attr_path,
                    variant.dynamic_categorize_attr,
                    verbose=verbose,
                )
            if new_attr_id is not None:
                _set_join_target(variant_clone, new_attr_id=new_attr_id)
            else:
                vectorjoins = variant_clone.find('vectorjoins')
                if vectorjoins is not None:
                    for join in list(vectorjoins.findall('join')):
                        vectorjoins.remove(join)
            new_maplayers.append(variant_clone)
            tree_entries.append(
                (new_variant_id, variant_display_name, variant.default_visible)
            )

        group_path = f'{style.group_path}/Unstyled' if unstyled else style.group_path
        tree_group = _find_or_create_group(layer_tree_root, group_path, verbose=verbose)
        legend_group = _find_or_create_legendgroup(
            legend_root, group_path, verbose=verbose
        )
        for layer_id, name, checked in tree_entries:
            _insert_layer_tree_layer(
                tree_group,
                layer_id=layer_id,
                name=name,
                source=geo_datasource,
                checked=checked,
            )
            _insert_legend_layer(
                legend_group,
                layer_id=layer_id,
                name=name,
                checked=checked,
            )

    for maplayer in new_maplayers:
        projectlayers.append(maplayer)

    for layername, (elem, elem_id) in original_index.items():
        static_style = static_styles_by_layername.get(layername)
        if static_style is not None:
            # Basemap/static layers always survive pruning, but their
            # checked state is still registry-driven: sync it to
            # default_visible rather than leaving whatever the template
            # author happened to hand-set.
            _set_layer_tree_checked(
                layer_tree_root, elem_id, static_style.default_visible
            )
            _set_legend_checked(legend_root, elem_id, static_style.default_visible)
            continue
        projectlayers.remove(elem)
        _remove_layer_tree_entry(layer_tree_root, elem_id)
        _remove_legend_entry(legend_root, elem_id)

    _prune_empty_groups(layer_tree_root)
    _prune_empty_legend_groups(legend_root)
    _collapse_layer_tree(layer_tree_root)
    _collapse_legend(legend_root)

    _set_project_variables(root, recipe=recipe, admin_id=admin_id)
    if title:
        _set_text(root, 'title', title)
    _update_extent(root, admin_id, verbose=verbose)
    _ensure_project_crs(root)
    _ensure_projections_enabled(root)

    output_bytes = _serialize(root, doctype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as out_zf:
        out_zf.writestr(f'{output_path.stem}.qgs', output_bytes)
        if styles_bytes is not None:
            out_zf.writestr(styles_name, styles_bytes)

    return output_path
