import geopandas as gpd
import pandas as pd
from pyproj import CRS


def add_tile_utm_derivatives(
    df: gpd.GeoDataFrame, tile_id_col: str, tile_type: str
) -> gpd.GeoDataFrame:
    """Derive native UTM CRS and bounds for Sentinel-2 or Landsat tiles.

    Parameters
    ----------
    df : GeoDataFrame
        Input tile grid geodataframe
    tile_id_col : str
        Name of the column containing the tile identifier
    tile_type : str
        Either 'sentinel' or 'landsat'
    """
    df = df.copy()

    if tile_id_col not in df.columns:
        raise ValueError(
            f"Tile ID column '{tile_id_col}' not found in "
            f'GeoDataFrame columns: {df.columns.tolist()}'
        )

    # 1. Derive CRS
    crs_list = []
    if tile_type == 'sentinel':
        for x in df[tile_id_col]:
            # e.g., '18SUU' -> '18S'
            zone = x[:2]
            band = x[2]
            south = band < 'N'
            epsg = CRS.from_dict(
                {'proj': 'utm', 'zone': zone, 'south': south}
            ).to_authority()[1]
            crs_list.append(f'epsg:{epsg}')
    elif tile_type == 'landsat':
        # Derive zone from centroid longitude
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            centroids = df.geometry.centroid
        zone = ((centroids.x + 180) / 6).astype(int) + 1
        crs_list = ('epsg:326' + zone.astype(str).str.zfill(2)).tolist()
    else:
        raise ValueError(
            f"Unsupported tile_type '{tile_type}'. Must be 'sentinel' or 'landsat'."
        )

    df['crs'] = crs_list

    # 2. Compute native UTM bounds (xmin_utm, ymin_utm, xmax_utm, ymax_utm)
    bounds_dfs = []
    for crs_name in df['crs'].unique():
        mask = df['crs'] == crs_name
        gdf_subset = df[mask].to_crs(crs_name)
        bounds = gdf_subset.geometry.bounds.rename(
            columns={
                'minx': 'xmin_utm',
                'miny': 'ymin_utm',
                'maxx': 'xmax_utm',
                'maxy': 'ymax_utm',
            }
        )
        bounds = bounds.round().astype(int)
        bounds_dfs.append(bounds)

    bounds_df = pd.concat(bounds_dfs)
    df = df.join(bounds_df)
    return df
