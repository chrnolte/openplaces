"""
Registered enrich step: the share of each entity's area covered by a
reference polygon layer, and the attributes of the reference polygon
covering most of it.

Written for a classified area layer whose polygons do not overlap one
another (protected open space, zoning districts, flood zones): the
question per parcel is "how much of it lies inside, and inside what".
Source-agnostic: the reference is any polygon dataset recipe, and the
recipe names the columns to carry.

Why not the existing overlays. `enrich_parcels_from_reference_crosswalk`
builds a parcel-to-parcel crosswalk keyed on matching geometry
identifiers and calibrates offsets between two cadastres, which a
thematic layer does not need and cannot supply. The harmonize polygon
attribution (`io/harmonizer/attributes/polygon.py`) attaches evidence
during the spine build, so a new layer would cost a geometry rerun of
every spine. Here the overlay runs once per admin unit against the
persisted spine, and the result is evidence only: curate decides what
to merge.

The dominant polygon is the one with the largest intersection area;
the share is the summed intersection area over the entity's own area,
capped at 1. Summing is exact only where reference polygons do not
overlap; the recipe states what was measured on its layer. A layer
whose polygons overlap one another (PAD-US easements, where one
easement can be recorded by two sources) sets ``dissolve: true``, and
the share is then the area of the union of the entity's intersections,
so that land under two records counts once.

A recipe can also restrict the reference to rows with given codes
(``where``) and turn the share into a flag at a stated threshold
(``flag_column``, ``min_share``). The flag is a fixed cut on one
measured share: no score, no vote between layers.
"""

from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from openplaces.core.schema import AdminId
from openplaces.geo.polygon import subdivide_polygons
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_dataset
from openplaces.recipe import get_recipe_by_id, get_save_admin_level


@lru_cache(maxsize=2)
def _load_reference(recipe_id: str, admin_id: str) -> gpd.GeoDataFrame | None:
    """Reference polygons saved for *admin_id*, or None when absent.

    Cached because a state-wide layer is read once per county or town
    of the same run; two entries cover a run that alternates layers.
    """
    try:
        gdf = get_dataset(recipe_id, admin_id=admin_id, geom=True)
    except FileNotFoundError:
        return None
    if gdf is None or gdf.empty:
        return None
    return gdf


def _reference_admin_id(reference_recipe: dict, admin_id: AdminId) -> AdminId | None:
    """The unit the reference saves that holds *admin_id*, or None.

    None where the reference does not cover *admin_id* (a state layer
    asked about a county of another state) or saves at a finer level
    than the unit processed.
    """
    scope = reference_recipe['admin_id']
    if not scope.is_parent_or_equal_of(admin_id):
        return None
    level = get_save_admin_level(reference_recipe)
    if level is None or level > admin_id.get_level():
        return None
    return AdminId(*admin_id.levels[:level]) if level else scope


def _dissolved_cover(
    covered: pd.Series,
    pairs: pd.DataFrame,
    entity_ids: np.ndarray,
    shared: np.ndarray,
) -> pd.Series:
    """Covered area per entity with overlapping reference land counted once.

    *covered* is the summed area per entity; *pairs* the per (entity,
    reference) areas. An entity touched by one reference polygon keeps
    its sum: the pieces of one polygon do not overlap. An entity touched
    by two or more gets the area of the union of its intersection
    pieces (*shared*, aligned with *entity_ids*).
    """
    n_refs = pairs.groupby('entity')['ref'].nunique()
    several = n_refs.index[n_refs > 1]
    if len(several) == 0:
        return covered
    covered = covered.copy()
    keep = pd.Index(entity_ids).isin(several) & ~shapely.is_empty(shared)
    groups = pd.Series(np.flatnonzero(keep)).groupby(entity_ids[keep])
    for entity, rows in groups:
        covered.loc[entity] = shapely.area(
            shapely.union_all(shapely.make_valid(shared[rows.to_numpy()]))
        )
    return covered


def overlay_shares(
    entities: gpd.GeoSeries,
    reference: gpd.GeoDataFrame,
    max_vertices: int = 256,
    dissolve: bool = False,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Covered share and dominant reference polygon per entity.

    Parameters
    ----------
    entities : geopandas.GeoSeries
        Entity polygons in a projected CRS, indexed by entity id.
    reference : geopandas.GeoDataFrame
        Reference polygons in the same CRS, indexed by their own id.
    max_vertices : int
        Reference polygons are cut into pieces of at most this many
        coordinates (`openplaces.geo.polygon.subdivide_polygons`) before
        the overlay, so that a large site is not scanned once per
        entity it touches.
    dissolve : bool
        Measure the covered area as the union of the entity's
        intersections with all reference polygons, not their sum, for
        a reference whose polygons overlap one another.

    Returns
    -------
    share : pandas.Series
        Covered area (summed, or dissolved with *dissolve*) over the
        entity's area, capped at 1; 0.0 for an entity no reference
        polygon touches.
    dominant : pandas.Series
        Index label of the reference polygon with the largest
        intersection area; missing where none intersects.
    dominant_share : pandas.Series
        That polygon's intersection area over the entity's area.
    """
    share = pd.Series(0.0, index=entities.index)
    dominant = pd.Series(pd.NA, index=entities.index, dtype=object)
    dominant_share = pd.Series(np.nan, index=entities.index)
    if entities.empty or reference.empty:
        return share, dominant, dominant_share
    ref = gpd.GeoSeries(
        shapely.make_valid(reference.geometry.values),
        index=reference.index,
        crs=reference.crs,
    )
    pieces = subdivide_polygons(ref, max_vertices=max_vertices)
    pieces = pieces[~pieces.is_empty & pieces.notna()]
    geoms = shapely.make_valid(entities.values)
    i_e, i_p = shapely.STRtree(pieces.values).query(geoms, predicate='intersects')
    if len(i_e) == 0:
        return share, dominant, dominant_share
    shared = shapely.intersection(geoms[i_e], pieces.values[i_p])
    area = shapely.area(shared)
    pairs = (
        pd.DataFrame(
            {
                'entity': entities.index.to_numpy()[i_e],
                'ref': pieces.index.to_numpy()[i_p],
                'area': area,
            }
        )
        .groupby(['entity', 'ref'], sort=False)['area']
        .sum()
        .reset_index()
    )
    pairs = pairs[pairs['area'] > 0]
    own = pd.Series(shapely.area(geoms), index=entities.index)
    own = own.where(own > 0)
    covered = pairs.groupby('entity')['area'].sum()
    if dissolve:
        covered = _dissolved_cover(
            covered, pairs, entities.index.to_numpy()[i_e], shared
        )
    share.loc[covered.index] = (covered / own.reindex(covered.index)).clip(upper=1)
    # Largest overlap first; a tie goes to the smaller reference id so
    # that a rerun picks the same polygon.
    top = pairs.sort_values(['entity', 'area', 'ref'], ascending=[True, False, True])
    top = top.drop_duplicates('entity').set_index('entity')
    dominant.loc[top.index] = top['ref'].to_numpy()
    dominant_share.loc[top.index] = (top['area'] / own.reindex(top.index)).clip(upper=1)
    return share.fillna(0.0), dominant, dominant_share


@_register('overlay_reference_polygons')
def overlay_reference_polygons(
    state: EnrichState,
    reference_recipe_id: str,
    share_column: str,
    columns: dict[str, str] | None = None,
    id_column: str | None = None,
    suffix: str | None = None,
    withhold: dict | None = None,
    max_vertices: int = 256,
    where: dict | None = None,
    dissolve: bool = False,
    flag_column: str | None = None,
    min_share: float | None = None,
) -> EnrichState:
    """Write covered share and dominant-polygon attributes for each entity.

    Parameters
    ----------
    state : EnrichState
        The recipe must set ``spine_geom: true``.
    reference_recipe_id : str
        Polygon dataset recipe. Read at its own save level, the unit
        that holds the admin unit processed (a state layer for a town).
        An admin unit outside the reference's scope, or with no saved
        reference, gets the columns written all-missing (the share
        too), so curate's merge finds them and leaves its values empty.
    share_column : str
        Name of the covered-share column (before *suffix*).
    columns : dict, optional
        ``{reference column: evidence column}`` copied from the dominant
        reference polygon. Columns absent from the reference are
        written all-missing.
    id_column : str, optional
        Evidence column receiving the dominant polygon's own index
        label (its id in the source).
    suffix : str, optional
        Appended to every evidence column; defaults to ``_`` plus the
        reference recipe's source id (``_massgis``).
    withhold : dict, optional
        ``{column, values, columns}``: on reference polygons whose
        *column* is one of *values*, the listed evidence columns stay
        missing. For a label that may name a person on private land (a
        site named after its owner); the share and the other columns
        are still written.
    max_vertices : int
        Passed to :func:`overlay_shares`.
    where : dict, optional
        ``{column, values}``: overlay only the reference polygons whose
        *column* is one of *values* (fee land held by a public owner,
        say). The dominant polygon is chosen among those too.
    dissolve : bool
        Passed to :func:`overlay_shares`; set it for a reference whose
        polygons overlap one another.
    flag_column : str, optional
        Boolean evidence column, true where the share is at least
        *min_share*. Missing where the reference does not cover the
        admin unit, like the share.
    min_share : float, optional
        Threshold for *flag_column*, required with it. The recipe
        states why its value fits its layer.

    Returns
    -------
    EnrichState
        With the share, flag, id and attribute evidence columns added.
    """
    columns = dict(columns or {})
    if flag_column and min_share is None:
        raise ValueError("overlay_reference_polygons: 'flag_column' needs 'min_share'.")
    spine = state.spine
    if 'geometry' not in getattr(spine, 'columns', ()):
        raise ValueError(
            "overlay_reference_polygons needs spine geometry; set 'spine_geom: true'."
        )
    reference_recipe = get_recipe_by_id(reference_recipe_id)
    if suffix is None:
        dataset = reference_recipe.get('dataset') or reference_recipe.get('entity')
        suffix = f'_{dataset.source.source_id}'
    outputs = [
        share_column,
        *([flag_column] if flag_column else []),
        *([id_column] if id_column else []),
        *columns.values(),
    ]
    evidence = pd.DataFrame(index=spine.index, columns=outputs, dtype=object)

    def _finish():
        evidence[share_column] = pd.to_numeric(evidence[share_column], errors='coerce')
        if flag_column:
            share = evidence[share_column]
            evidence[flag_column] = share.ge(min_share).astype('boolean')
            evidence.loc[share.isna(), flag_column] = pd.NA
        for column in outputs:
            state.evidence[f'{column}{suffix}'] = evidence[column].reindex(
                state.evidence.index
            )
        return state

    ref_admin = _reference_admin_id(reference_recipe, AdminId(str(state.admin_id)))
    reference = (
        _load_reference(reference_recipe_id, str(ref_admin)) if ref_admin else None
    )
    if reference is None:
        if state.verbose:
            print(f'  no {reference_recipe_id} for {state.admin_id}; all missing.')
        return _finish()

    entities = gpd.GeoSeries(spine.geometry, crs=spine.crs)
    entities = entities[~entities.is_empty & entities.notna()]
    crs = entities.estimate_utm_crs()
    entities = entities.to_crs(crs)
    bounds = gpd.GeoSeries([shapely.box(*entities.total_bounds)], crs=crs)
    window = bounds.to_crs(reference.crs).iloc[0]
    nearby = reference.iloc[
        reference.sindex.query(window, predicate='intersects')
    ].to_crs(crs)
    if where:
        if where['column'] not in nearby.columns:
            raise ValueError(
                f"overlay_reference_polygons: 'where' column {where['column']!r} "
                f'is not in {reference_recipe_id}.'
            )
        nearby = nearby[nearby[where['column']].isin(list(where['values']))]
    if state.timer:
        state.timer.mark('load reference polygons')

    share, dominant, _ = overlay_shares(
        entities, nearby, max_vertices=max_vertices, dissolve=dissolve
    )
    evidence[share_column] = share.reindex(evidence.index)
    has = dominant.notna()
    hit = dominant[has]
    if id_column:
        # As text, as the registry types an id: an integer row id
        # would otherwise turn float wherever an entity has none.
        evidence.loc[hit.index, id_column] = hit.astype(str).to_numpy()
    withheld = pd.Series(False, index=hit.index)
    if withhold:
        gate = withhold['column']
        if gate in nearby.columns:
            flagged = nearby[gate].isin(list(withhold['values']))
            withheld = hit.map(flagged).fillna(False).astype(bool)
    for ref_col, out_col in columns.items():
        if ref_col not in nearby.columns:
            continue
        values = hit.map(nearby[ref_col])
        if withhold and out_col in (withhold.get('columns') or []):
            values = values.mask(withheld)
        evidence.loc[hit.index, out_col] = values.to_numpy()
    if state.timer:
        state.timer.mark('overlay reference polygons')
    if state.verbose:
        print(
            f'  {int(share.gt(0).sum()):,d} of {len(evidence):,d} entities '
            f'overlap {reference_recipe_id}.'
        )
    return _finish()
