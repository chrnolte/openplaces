"""
overlay.py

Spatial overlay operations on polygon datasets.
Depends on recipes, admin lookups, and DuckDB for fast parquet-based overlays.
"""

import json
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely

from openplaces.api import get_admin
from openplaces.core.constants import STRING_SEPARATOR_BETWEEN_IDS
from openplaces.geo.polygon import get_lat_long_centroids
from openplaces.recipe import get_recipe
from openplaces.timing import get_timer


def overlay_admin_ids(
    gdf,
    admin_geometries=None,
    admin_level=2,
    admin_id=None,
    admin_recipe=None,
    include_overlays=False,
    timer=None,
):
    """Add administrative unit IDs to GeoDataFrame using spatial joins

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame to join admin IDs to
    admin_geometries : gpd.GeoSeries
        GeoSeries of admin geometries with AdminID index.
        Pass this or `admin_level`
    admin_level : int
        Administrative level for which administrative IDs are to be
        joined. Typically a lower level (larger number) than the level
        of `admin_id`.
    admin_id : str
        Administrative unit of the GeoDataFrame.
        Determines which administrative units to consider.
    admin_recipe : dict or str
        Recipe of the administrative unit dataset to be used.
        String identifier or resolved recipe (dictionary).
        If None, the default recipe for administrations is used.
    include_overlays : bool
        If True, attempt a spatial polygon overlay for polygons for
        which the spatial join (centroids) returned no results.
    timer : openplaces.timing.Timer or None
        Timer
    """
    if timer is None:
        timer = get_timer('overlay_admin_ids', verbose=True)

    if isinstance(admin_recipe, str):
        # Assume recipe contained the complete filename, infer parts
        # {admin_id}_{entity}_{filename.ext}
        recipe_admin_id, entity, filename = admin_recipe.split(
            STRING_SEPARATOR_BETWEEN_IDS
        )
        admin_recipe = get_recipe(recipe_admin_id, entity, filename=filename)

    if admin_geometries is None:
        admin = get_admin(
            admin_id, admin_level, recipe=admin_recipe, columns=[], geom=True
        )
    else:
        admin = admin_geometries.to_frame()

    # Cast admin index to pd.Categorical to later save space in the joined column
    admin.index = pd.Index(pd.Categorical(admin.index), name=admin.index.name)
    timer.mark('Admin overlay: get admin layer')

    # Get centroid points with range index (for quicker processing)
    gdf_centroids = get_lat_long_centroids(gdf.reset_index()[['geometry']], geom=True)[
        ['geometry']
    ]
    timer.mark('Admin overlay: get centroids')

    gdf_sjoin = gpd.sjoin(gdf_centroids, admin[['geometry']], how='left')
    gdf[admin.index.name] = gdf_sjoin[admin.index.name].values
    del gdf_sjoin
    timer.mark('Admin overlay: spatial join')

    if include_overlays:
        mask = gdf[admin.index.name].isnull()
        if mask.any():
            gdf_overlay = (
                gpd.overlay(
                    gdf[mask][['geometry']].reset_index(),
                    admin[['geometry']].reset_index(),
                )
                .drop(columns='geometry')
                .set_index(gdf.index.name)
            )
            gdf.loc[mask, admin.index.name] = gdf_overlay[admin.index.name]
            del gdf_overlay
            timer.mark('Admin overlay: spatial overlay')

    return gdf


def _detect(attr_path: Path, geo_path: Path) -> tuple[str, str, list[str], str]:
    """Return (index_name, join_key, data_columns, crs) from parquet metadata."""
    schema = pq.read_schema(attr_path)
    index_name = schema.pandas_metadata['index_columns'][0]
    data_cols = [
        c['name'] for c in schema.pandas_metadata['columns'] if c['name'] != index_name
    ]
    geo_schema = pq.read_schema(geo_path)
    join_key = 'geo_id' if 'geo_id' in geo_schema.names else '_join_id'
    geo_meta = json.loads(geo_schema.metadata[b'geo'])
    crs = geo_meta['columns']['geometry']['crs']['id']['code']
    return index_name, join_key, data_cols, f'EPSG:{crs}'


def overlay_polygons(
    path1: Path,
    path2: Path,
    columns: list[str] | None = None,
    geom: bool = False,
    iou: bool = False,
    suffixes: tuple[str, str] | None = None,
    how: str = 'intersection',
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Intersect two polygon datasets stored as GeoParquet.

    Each path should point to an attribute parquet file saved with
    `save_parquet()`. The corresponding `_geo.parquet` file is imputed
    automatically.

    Parameters
    ----------
    path1 :
        Path to the first attribute parquet file.
    path2 :
        Path to the second attribute parquet file.
    columns :
        Columns to return from the attribute tables. Columns are auto-detected
        from both tables. If a column exists in both tables, `suffixes` must
        be provided.
    geom :
        If True, return the clipped intersection geometry as a GeoDataFrame.
    iou :
        If True, compute intersection area, union area, and
        intersection-over-union ratio. Areas are computed in EPSG:6933 (m²).
        Only meaningful for matched pairs; unmatched rows get NaN.
    suffixes :
        Required when any requested column exists in both attribute tables,
        or when both tables share the same index name. Tuple of two strings,
        e.g. ('_tiles', '_admin'), appended to disambiguate column names,
        analogous to gpd.sjoin suffixes.
    how : {'intersection', 'union', 'identity'}
        Type of overlay operation:
        - 'intersection': only overlapping pairs (inner join)
        - 'union': all polygons from both sides; unmatched rows retain
          original geometry and get NaN for the missing index level
        - 'identity': all polygons from path1; unmatched path1 polygons
          retain original geometry and get NaN for the path2 index level

    Returns
    -------
    pd.DataFrame or gpd.GeoDataFrame
        MultiIndex of (index1, index2) detected from parquet metadata.
        Columns: those requested via `columns`, plus iou columns if iou=True,
        plus geometry if geom=True.

    Raises
    ------
    FileNotFoundError
        If any of the expected parquet files are missing.
    ValueError
        If any requested column exists in both tables and suffixes is None,
        or if both tables share the same index name and suffixes is None,
        or if how is not one of the supported operations.
    """
    _SUPPORTED_HOW = {'intersection', 'union', 'identity'}
    if how not in _SUPPORTED_HOW:
        raise ValueError(
            f'how={how!r} is not supported. Choose from {sorted(_SUPPORTED_HOW)}.'
        )

    path1 = Path(path1)
    path2 = Path(path2)
    geo_path1 = path1.with_stem(path1.stem + '_geo')
    geo_path2 = path2.with_stem(path2.stem + '_geo')

    for p in [path1, path2, geo_path1, geo_path2]:
        if not p.exists():
            raise FileNotFoundError(p)

    # Detect index names, join keys, and data columns
    index1, join_key1, data_cols1, crs1 = _detect(path1, geo_path1)
    index2, join_key2, data_cols2, crs2 = _detect(path2, geo_path2)

    # Resolve MultiIndex name conflict

    if index1 == index2:
        if suffixes is None:
            raise ValueError(
                f'Both tables have index name {index1!r}. '
                f'Pass suffixes to disambiguate.'
            )
        alias1 = f'{index1}{suffixes[0]}'
        alias2 = f'{index2}{suffixes[1]}'
    else:
        alias1 = index1
        alias2 = index2

    # Resolve requested columns

    col_select: list[str] = []

    if columns:
        set1 = set(data_cols1)
        set2 = set(data_cols2)

        ambiguous = [c for c in columns if c in set1 and c in set2]
        if ambiguous and suffixes is None:
            raise ValueError(
                f'Columns {ambiguous} exist in both tables. '
                f'Pass suffixes to disambiguate.'
            )

        missing = [c for c in columns if c not in set1 and c not in set2]
        if missing:
            raise ValueError(f'Columns {missing} not found in either attribute table.')

        for col in columns:
            in1 = col in set1
            in2 = col in set2
            if in1 and in2:
                col_select.append(f'a1.{col} AS {col}{suffixes[0]}')
                col_select.append(f'a2.{col} AS {col}{suffixes[1]}')
            elif in1:
                col_select.append(f'a1.{col}')
            else:
                col_select.append(f'a2.{col}')

    # Build SELECT clause

    def _area_m2(geom_expr: str, src_crs: str) -> str:
        """Reproject to EPSG:6933 and compute area in m²."""
        return f"ST_Area(ST_Transform({geom_expr}, '{src_crs}', 'EPSG:6933', true))"

    # SQL JOIN type and geometry expression depend on how
    if how == 'intersection':
        join_type = 'JOIN'
        geom_expr = 'ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY)'
    elif how == 'identity':
        join_type = 'LEFT JOIN'
        geom_expr = (
            f'CASE WHEN g2.{join_key2} IS NULL '
            'THEN g1.geometry::GEOMETRY '
            'ELSE ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY) '
            'END'
        )
    elif how == 'union':
        join_type = 'FULL OUTER JOIN'
        geom_expr = (
            f'CASE WHEN g2.{join_key2} IS NULL '
            'THEN g1.geometry::GEOMETRY '
            f'WHEN g1.{join_key1} IS NULL '
            'THEN g2.geometry::GEOMETRY '
            'ELSE ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY) '
            'END'
        )

    select_parts = [
        f'a1.{index1} AS {alias1}',
        f'a2.{index2} AS {alias2}',
    ]

    if iou:
        intersection_expr = (
            'ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY)'
        )
        select_parts += [
            f'{_area_m2("g1.geometry::GEOMETRY", crs1)} AS area1_m2',
            f'{_area_m2("g2.geometry::GEOMETRY", crs2)} AS area2_m2',
            f'{_area_m2(intersection_expr, crs1)} AS area_intersection_m2',
        ]

    select_parts += col_select

    if geom:
        select_parts.append(f'ST_AsWKB({geom_expr})::BLOB AS geometry')

    select_clause = ',\n            '.join(select_parts)

    # Execute query
    con = duckdb.connect()
    con.execute('INSTALL spatial; LOAD spatial;')
    con.execute('SET enable_progress_bar = false;')

    overlaps = con.execute(
        f"""
        SELECT
            {select_clause}
        FROM read_parquet('{geo_path1}') AS g1
        JOIN read_parquet('{path1}') AS a1
            ON g1.{join_key1} = a1.{join_key1}
        {join_type} read_parquet('{geo_path2}') AS g2
            ON ST_Intersects(
                g1.geometry::GEOMETRY,
                g2.geometry::GEOMETRY
            )
        LEFT JOIN read_parquet('{path2}') AS a2
            ON g2.{join_key2} = a2.{join_key2}
    """
    ).df()

    con.close()

    # Post-process
    if iou:
        area_union = (
            overlaps['area1_m2']
            + overlaps['area2_m2']
            - overlaps['area_intersection_m2']
        )
        overlaps['iou'] = overlaps['area_intersection_m2'] / area_union.replace(
            0, float('nan')
        )

    overlaps = overlaps.set_index([alias1, alias2])

    if geom:
        overlaps['geometry'] = shapely.from_wkb(overlaps['geometry'].apply(bytes))
        overlaps = overlaps[[v for v in overlaps if v != 'geometry'] + ['geometry']]
        return gpd.GeoDataFrame(overlaps, geometry='geometry', crs=crs1)

    return overlaps
