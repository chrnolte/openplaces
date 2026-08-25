"""Standardized QGIS map export for curate-stage recipes.

Generates a `.qgz` project showing a curate recipe's essential ingested
inputs and its curated output, styled consistently across admin regions by
cloning layers from a hand-authored template (see
``qgis/templates/README.md``). Opt-in and on-demand: call
:func:`export_qgis_map` whenever a map deliverable is wanted; nothing here
runs automatically as part of :func:`openplaces.io.curator.curate`.
"""

from openplaces.core.schema import AdminId
from openplaces.viz.qgis_map.generator import generate_qgz
from openplaces.viz.qgis_map.resolver import LayerSpec, resolve_layers
from openplaces.viz.qgis_map.style_registry import get_style

__all__ = [
    'ensure_delivery_admin_outlines',
    'export_qgis_map',
    'resolve_layers',
    'LayerSpec',
    'get_style',
]


def ensure_delivery_admin_outlines(recipe, region_id, *, verbose=False):
    """Write county and county-subdivision outlines beside a bundle.

    Two aggregate sidecars per delivered region, so the shipped map can
    draw administrative context without referencing this machine's cache:
    `{stem}_admin3_geo.parquet` (the region's counties, from the
    harmonized admin layer) and `{stem}_admin4_geo.parquet` (their county
    subdivisions from the census admin4 layer, where any exist). Skips
    whatever already exists; returns the paths written or found.
    """
    import warnings

    import openplaces as op
    from openplaces.io.delivery import delivery_members, delivery_paths

    paths = delivery_paths(recipe, region=region_id)
    canonical = paths['canonical']
    members = delivery_members(recipe, region=region_id)
    state = '-'.join(members[0].split('-')[:2])
    out = []

    a3_path = canonical.with_name(f'{canonical.stem}_admin3_geo.parquet')
    if not a3_path.exists():
        adm = op.get_admin(state, level=3, geom=True)
        sub = adm.loc[[m for m in members if m in adm.index], ['name', 'geometry']]
        sub.to_parquet(a3_path)
        out.append(a3_path)
    a4_path = canonical.with_name(f'{canonical.stem}_admin4_geo.parquet')
    if not a4_path.exists():
        try:
            a4 = op.get_entities('US_admin-census-2025_admin4', 'US', geom=True)
            cols = [c for c in ('name', 'admin3_id') if c in a4.columns]
            sub4 = a4[a4['admin3_id'].isin(set(members))][cols + ['geometry']]
            if len(sub4):
                sub4.to_parquet(a4_path)
                out.append(a4_path)
            elif verbose:
                warnings.warn(f'No county subdivisions for {region_id}.')
        except Exception as exc:
            warnings.warn(f'Could not build admin4 outlines: {exc}')
    return out


def export_qgis_map(
    recipe: str | dict,
    admin_id: str | AdminId,
    *,
    output_path=None,
    title: str | None = None,
    template_path=None,
    filter_existing: bool = True,
    include_inputs: bool = True,
    verbose: bool = False,
):
    """Generate a standardized QGIS project (.qgz) for *recipe* at *admin_id*.

    See :func:`~openplaces.viz.qgis_map.generator.generate_qgz` for the full
    parameter reference; *filter_existing* is passed through to
    :func:`~openplaces.viz.qgis_map.resolver.resolve_layers`.

    Parameters
    ----------
    recipe : str or dict
        Curate-stage recipe ID or loaded recipe dict.
    admin_id : str or `openplaces.core.schema.AdminId`
        Admin unit to generate the map for.
    output_path : str or pathlib.Path, optional
        Where to write the generated ``.qgz``. Defaults to a path under
        `cfg.share_dir`.
    title : str, optional
        Project title to set.
    template_path : str or pathlib.Path, optional
        Template ``.qgz`` to clone from. Defaults to the packaged template.
    filter_existing : bool, optional
        Drop layers whose parquet files do not exist on disk for this admin
        unit (default True).
    include_inputs : bool, optional
        Include the ingest-stage source layers (default True). Pass False
        for a map of the delivered product alone: the curated output, admin
        context, and the template's basemaps.
    verbose : bool, optional
        Warn about template gaps and resolver skips.

    Returns
    -------
    pathlib.Path
        Path to the generated ``.qgz``.
    """
    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)
    # A delivery map should carry its own administrative context; write
    # the outline sidecars first so the resolver can pick them up.
    try:
        from openplaces.io.delivery import delivery_regions

        for region in delivery_regions(recipe):
            rid = region.get('region_id')
            from openplaces.io.delivery import delivery_admin_id

            bundle_admin = delivery_admin_id(recipe, region=rid)
            if (
                tuple(admin_id.levels)
                == tuple(bundle_admin.levels)[: len(admin_id.levels)]
            ):
                ensure_delivery_admin_outlines(recipe, rid, verbose=verbose)
    except Exception:
        pass
    layer_specs = resolve_layers(
        recipe,
        admin_id,
        filter_existing=filter_existing,
        include_inputs=include_inputs,
        verbose=verbose,
    )
    if not include_inputs:
        # A product map ships with the bundle; an 'admin' context layer
        # resolved from this machine's cache would arrive broken on any
        # other machine. The bundled outline sidecars replace it.
        layer_specs = [s for s in layer_specs if s.role != 'admin']
    return generate_qgz(
        recipe,
        admin_id,
        layer_specs=layer_specs,
        template_path=template_path,
        output_path=output_path,
        title=title,
        verbose=verbose,
    )
