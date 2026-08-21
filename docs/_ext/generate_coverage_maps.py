"""Regenerate the recipe-coverage maps shown on the docs landing page.

Standalone script, not a notebook and not a live Sphinx extension: the
docs build itself has neither openplaces installed nor ingested boundary
data available (see ``catalog_data``'s module docstring), so these PNGs
are rendered once, locally, with a real environment and real ingested
GADM/admin data, and committed under ``docs/_static/images/``. Run in the
``openplaces`` conda environment after a material change in recipe
coverage::

    conda activate openplaces
    python docs/_ext/generate_coverage_maps.py

This deliberately reproduces rather than imports the drawing logic in
``openplaces.diagnostics.map_recipe_coverage``: that function serves ad
hoc, single-stage, whole-world exploration from a notebook, one entity
type at a time. These maps need several stages unioned into one entity
type's coverage, an admin-id prefix filter (the linked-entity map is US
only), and a bounding-box crop so the actual covered area is legible --
different enough, and small enough (~150 lines), that copying beats
parameterizing the shared function for a shape it otherwise doesn't need.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from openplaces.diagnostics import find_recipes

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_IMAGES = REPO_ROOT / 'docs' / '_static' / 'images'

#: One base hue blended with white at level-dependent weights (0 = pastel
#: global wash, higher = more specific admin scope), matching the scheme
#: `map_recipe_coverage` uses so a reader doesn't have to learn a second
#: color language between a notebook map and a docs map.
_BASE = (0.13, 0.47, 0.71, 1.0)  # tab10 blue
_UNCOVERED = (0.91, 0.91, 0.91, 1.0)
_LEVEL_WEIGHTS = {0: 0.22, 1: 0.48, 2: 0.72, 3: 0.92}
_LEVEL_EDGECOLORS = {0: '#d8d8d8', 1: '#c0c0c0', 2: '#888888', 3: '#505050'}
_LEVEL_LABELS = {0: 'global', 1: 'country', 2: 'state', 3: 'county'}

#: Fraction of a geography's own largest polygon that a disjoint part
#: must reach to count toward the crop's bounding box. Drops small
#: outlying islands (Hawaii, the Aleutian tail crossing the antimeridian)
#: that would otherwise pull the crop back out to world scale, while
#: still keeping Alaska's mainland (a large fraction of the US polygon,
#: not a sliver) in frame.
_MAINLAND_FRACTION = 0.02

#: (entity_type, stages, admin_prefix, title, out_name). admin_prefix
#: restricts to that admin id or one nested under it; None means every
#: geography. Order here is the stacking order on the docs page.
MAPS = [
    ('admin', ('ingest',), None, 'Admin boundaries', 'recipe_coverage_admin'),
    (
        'footprint',
        ('ingest',),
        None,
        'Building footprints',
        'recipe_coverage_footprint',
    ),
    ('parcel', ('ingest',), None, 'Parcels', 'recipe_coverage_parcel'),
    (
        'footprint',
        ('harmonize', 'curate'),
        'US',
        'Footprints linked to parcels · harmonize / curate · US only',
        'recipe_coverage_footprint_parcel_linked',
    ),
]


def _admin_level(admin_id: str) -> int:
    return len(admin_id.split('-')) if admin_id else 0


def _level_color(level: int) -> tuple:
    w = _LEVEL_WEIGHTS.get(level, min(0.22 + level * 0.24, 0.95))
    r, g, b, _ = _BASE
    return ((1 - w) + w * r, (1 - w) + w * g, (1 - w) + w * b, 1.0)


def _level_edgecolor(level: int) -> str:
    return _LEVEL_EDGECOLORS.get(level, '#303030')


def _select(entity_type: str, stages: tuple[str, ...], admin_prefix: str | None):
    df = find_recipes(entity_type)
    df = df[df['stage'].isin(stages)]
    if admin_prefix is not None:
        prefix = f'{admin_prefix}-'
        df = df[
            (df['admin_id'] == admin_prefix) | df['admin_id'].str.startswith(prefix)
        ]
    df = df.copy()
    df['_level'] = df['admin_id'].apply(_admin_level)
    return df


def _mainland_bounds(geom, min_area_fraction: float = _MAINLAND_FRACTION):
    """Bounds of one geometry's dominant part(s), by area within itself.

    Parameters
    ----------
    geom : shapely geometry
        A (Multi)Polygon.
    min_area_fraction : float
        Minimum area, as a fraction of this geometry's own largest part,
        for a disjoint part to count toward the bounds.

    Returns
    -------
    tuple of float or None
        (minx, miny, maxx, maxy), or None for an empty geometry.
    """
    if geom is None or geom.is_empty:
        return None
    parts = list(geom.geoms) if hasattr(geom, 'geoms') else [geom]
    max_area = max(p.area for p in parts)
    kept = [p for p in parts if p.area >= max_area * min_area_fraction]
    xs = [p.bounds[0] for p in kept] + [p.bounds[2] for p in kept]
    ys = [p.bounds[1] for p in kept] + [p.bounds[3] for p in kept]
    return min(xs), min(ys), max(xs), max(ys)


def _union_bounds(bounds_list):
    xs_min = [b[0] for b in bounds_list]
    ys_min = [b[1] for b in bounds_list]
    xs_max = [b[2] for b in bounds_list]
    ys_max = [b[3] for b in bounds_list]
    return min(xs_min), min(ys_min), max(xs_max), max(ys_max)


def draw_coverage_map(
    entity_type: str,
    stages: tuple[str, ...],
    admin_prefix: str | None,
    title: str,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw one entity type's coverage across one or more pipeline stages.

    Parameters
    ----------
    entity_type : str
        Entity type to map (e.g. 'parcel').
    stages : tuple of str
        Pipeline stages to union together (e.g. ``('harmonize', 'curate')``).
    admin_prefix : str or None
        Restrict to this admin id or one nested under it (e.g. 'US').
        None maps every geography with a matching recipe.
    title : str
        Axes title.

    Returns
    -------
    (fig, ax) : tuple[plt.Figure, plt.Axes]
        The figure is cropped to the full extent of every geography with
        *any* coverage. A global-scope recipe (e.g. GADM for admin
        boundaries) gives every country "some" coverage, so its presence
        turns off cropping rather than being the one thing excluded from
        the bounding box -- otherwise the crop would hide the very
        coverage a global recipe provides.
    """
    from openplaces.io.readers import get_admin

    df = _select(entity_type, stages, admin_prefix)
    if df.empty:
        raise ValueError(
            f'No recipes found for entity_type={entity_type!r}, '
            f'stages={stages!r}, admin_prefix={admin_prefix!r}.'
        )
    has_global = not df[df['_level'] == 0].empty

    fig, ax = plt.subplots(figsize=(12, 6))
    # None (rather than []) means "don't crop": a global-scope recipe's
    # true extent is the whole world, which is what an empty bounds list
    # would otherwise be mistaken for on the final `if covered_bounds:`.
    covered_bounds = None if has_global else []
    _admin_geo_recipe = 'admin-openplaces-2026'

    world = get_admin(level=1, geom='simplified', recipe=f'{_admin_geo_recipe}_admin1')
    color_map = {idx: _UNCOVERED for idx in world.index}
    if has_global:
        for idx in color_map:
            color_map[idx] = _level_color(level=0)
    for admin_id in df.loc[df['_level'] == 1, 'admin_id']:
        if admin_id in color_map:
            color_map[admin_id] = _level_color(level=1)
    world.plot(
        ax=ax,
        color=[color_map[idx] for idx in world.index],
        edgecolor=_level_edgecolor(1),
        linewidth=0.1,
    )
    if covered_bounds is not None:
        for admin_id in df.loc[df['_level'] == 1, 'admin_id']:
            if admin_id in world.index:
                bounds = _mainland_bounds(world.loc[admin_id].geometry)
                if bounds is not None:
                    covered_bounds.append(bounds)

    max_level = int(df['_level'].max())
    for level in range(2, max_level + 1):
        unique_ids = df.loc[df['_level'] == level, 'admin_id'].unique().tolist()
        if not unique_ids:
            continue
        gdf = get_admin(
            unique_ids, geom='simplified', recipe=f'{_admin_geo_recipe}_admin{level}'
        )
        gdf.plot(
            ax=ax,
            color=_level_color(level=level),
            edgecolor=_level_edgecolor(level),
            linewidth=0.1,
        )
        if covered_bounds is not None:
            for geom in gdf.geometry:
                bounds = _mainland_bounds(geom)
                if bounds is not None:
                    covered_bounds.append(bounds)

    handles = [
        mpatches.Patch(
            color=_level_color(level=level),
            label=_LEVEL_LABELS.get(level, f'level {level}'),
        )
        for level in sorted(df['_level'].unique())
    ]
    ax.legend(handles=handles, loc='lower left', framealpha=0.9, fontsize=9)
    ax.set_title(title, fontsize=11, pad=6)
    ax.set_axis_off()

    if covered_bounds:
        minx, miny, maxx, maxy = _union_bounds(covered_bounds)
        pad_x = max((maxx - minx) * 0.04, 1.0)
        pad_y = max((maxy - miny) * 0.06, 1.0)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

    return fig, ax


def main() -> None:
    """Regenerate every map in `MAPS` and save it to `DOCS_IMAGES`."""
    DOCS_IMAGES.mkdir(parents=True, exist_ok=True)
    for entity_type, stages, admin_prefix, title, out_name in MAPS:
        fig, _ax = draw_coverage_map(entity_type, stages, admin_prefix, title)
        out_path = DOCS_IMAGES / f'{out_name}.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        print(f'wrote {out_path.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
