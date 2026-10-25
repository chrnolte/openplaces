"""
Enrichment step locating entities in core-based land-value regions.

The PLACES lab regions (US_land-regions-placeslab-fmvc0003) partition the
contiguous U.S. by the high-value core that dominates the local land value
gradient. An entity's region type (metro, micro, city, resort) and its
distance to that core describe its position in a land market in terms that
transfer between regions: a house 30 km from a mid-size core resembles one
30 km from another, which a county or coordinate does not express. Building
imputation reads these as inputs (io/building_imputation.py).
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd

from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_dataset

# Equal-area CONUS projection: distances in meters, as the regions were
# built (places, b6y_regions.ipynb).
ALBERS = 'EPSG:5070'


@_register('assign_land_value_region')
def assign_land_value_region(
    state: EnrichState,
    regions_recipe_id: str,
    cores_recipe_id: str,
) -> EnrichState:
    """Write each entity's land-value region, region type and core distance.

    Parameters
    ----------
    regions_recipe_id : str
        Dataset recipe of the region polygons (columns
        `land_value_region_id`, `land_value_region_type`).
    cores_recipe_id : str
        Dataset recipe of the cores, one point per region, keyed by the
        same `land_value_region_id`.

    Notes
    -----
    Entities are located by the centroid of their geometry; an entity
    outside every region (offshore) gets no region and no distance.
    Evidence columns: `land_value_region_id`, `land_value_region_type`,
    `distance_to_land_value_core_km`.
    """
    spine = state.spine
    if not isinstance(spine, gpd.GeoDataFrame):
        raise ValueError(
            "This step requires spine geometry; set 'spine_geom: true' on the "
            'enrich recipe.'
        )
    regions = get_dataset(regions_recipe_id, 'US', geom=True).to_crs(ALBERS)
    cores = get_dataset(cores_recipe_id, 'US', geom=True).to_crs(ALBERS)
    points = gpd.GeoDataFrame(
        geometry=spine.to_crs(ALBERS).geometry.centroid, crs=ALBERS
    )
    hit = gpd.sjoin(
        points,
        regions[['land_value_region_id', 'land_value_region_type', 'geometry']],
        how='left',
        predicate='within',
    )
    # A centroid on a shared border falls in both regions; keep one.
    hit = hit[~hit.index.duplicated()]
    region_id = hit['land_value_region_id'].reindex(points.index)
    core = (
        cores.drop_duplicates('land_value_region_id')
        .set_index('land_value_region_id')
        .geometry
    )
    core_x = region_id.map(core.x)
    core_y = region_id.map(core.y)
    distance = np.hypot(points.geometry.x - core_x, points.geometry.y - core_y)
    state.evidence['land_value_region_id'] = region_id.reindex(state.evidence.index)
    state.evidence['land_value_region_type'] = hit['land_value_region_type'].reindex(
        state.evidence.index
    )
    state.evidence['distance_to_land_value_core_km'] = pd.Series(
        distance / 1000.0, index=points.index
    ).reindex(state.evidence.index)
    if state.verbose:
        located = int(state.evidence['land_value_region_id'].notna().sum())
        print(f'  Located {located:,d} of {len(points):,d} entities in a region.')
    return state
