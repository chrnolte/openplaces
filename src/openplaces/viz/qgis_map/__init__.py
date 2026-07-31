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

__all__ = ['export_qgis_map', 'resolve_layers', 'LayerSpec', 'get_style']


def export_qgis_map(
    recipe: str | dict,
    admin_id: str | AdminId,
    *,
    output_path=None,
    title: str | None = None,
    template_path=None,
    filter_existing: bool = True,
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
    verbose : bool, optional
        Warn about template gaps and resolver skips.

    Returns
    -------
    pathlib.Path
        Path to the generated ``.qgz``.
    """
    admin_id = admin_id if isinstance(admin_id, AdminId) else AdminId(admin_id)
    layer_specs = resolve_layers(
        recipe, admin_id, filter_existing=filter_existing, verbose=verbose
    )
    return generate_qgz(
        recipe,
        admin_id,
        layer_specs=layer_specs,
        template_path=template_path,
        output_path=output_path,
        title=title,
        verbose=verbose,
    )
