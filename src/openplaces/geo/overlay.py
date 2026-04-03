"""
overlay.py

Spatial overlay operations on polygon datasets.
Depends on recipes, admin lookups, and DuckDB for fast parquet-based overlays.
"""

import contextlib
import json
import tempfile
import time
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
import shapely

from openplaces.core.constants import STRING_SEPARATOR_BETWEEN_IDS
from openplaces.geo.polygon import get_lat_long_centroids, overlay_polygons
from openplaces.io import save_parquet
from openplaces.io.readers import get_admin
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


@contextlib.contextmanager
def _as_parquet(layer):
    """Yield a Path to a parquet file, saving a GeoDataFrame to temp if needed."""
    if isinstance(layer, gpd.GeoDataFrame):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / 'data.parquet'
            save_parquet(layer, p)
            yield p
    else:
        yield Path(layer)


def overlay_polygons_with_duckdb(
    layer1: Path | gpd.GeoDataFrame,
    layer2: Path | gpd.GeoDataFrame,
    columns: list[str] | None = None,
    geom: bool = False,
    iou: bool = False,
    suffixes: tuple[str, str] | None = None,
    how: str = 'intersection',
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Intersect two polygon datasets using DuckDB (parquet-streaming path).

    Each layer may be a GeoDataFrame or a path to an attribute parquet file
    saved with `save_parquet()`. When a GeoDataFrame is passed it is written
    to a temporary parquet file automatically. The corresponding `_geo.parquet`
    sidecar is imputed automatically for file-based layers.

    Parameters
    ----------
    layer1 :
        First polygon layer — GeoDataFrame or path to attribute parquet file.
    layer2 :
        Second polygon layer — GeoDataFrame or path to attribute parquet file.
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
        - 'identity': all polygons from layer1; unmatched layer1 polygons
          retain original geometry and get NaN for the layer2 index level

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

    with _as_parquet(layer1) as path1, _as_parquet(layer2) as path2:
        return _overlay_polygons_paths(
            path1,
            path2,
            columns=columns,
            geom=geom,
            iou=iou,
            suffixes=suffixes,
            how=how,
        )


def _overlay_polygons_paths(
    path1: Path,
    path2: Path,
    columns: list[str] | None = None,
    geom: bool = False,
    iou: bool = False,
    suffixes: tuple[str, str] | None = None,
    how: str = 'intersection',
) -> pd.DataFrame | gpd.GeoDataFrame:
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
    col_select_null: list[str] = []  # NULL-substituted, for unmatched/leftover
    group_by_extras: list[str] = []  # a1-sourced cols needed in GROUP BY

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
                col_select_null.append(f'a1.{col} AS {col}{suffixes[0]}')
                col_select_null.append(f'NULL AS {col}{suffixes[1]}')
                group_by_extras.append(f'a1.{col}')
            elif in1:
                col_select.append(f'a1.{col}')
                col_select_null.append(f'a1.{col}')
                group_by_extras.append(f'a1.{col}')
            else:
                col_select.append(f'a2.{col}')
                col_select_null.append(f'NULL AS {col}')

    # Build SELECT clause

    def _area_m2(geom_expr: str, src_crs: str) -> str:
        """Reproject to EPSG:6933 and compute area in m²."""
        return f"ST_Area(ST_Transform({geom_expr}, '{src_crs}', 'EPSG:6933', true))"

    # Execute query
    con = duckdb.connect()
    con.execute('INSTALL spatial; LOAD spatial;')
    con.execute('SET enable_progress_bar = false;')

    if how == 'identity':
        # --- identity fast path: single CTE query, spatial join runs once ---
        intersection_expr = (
            'ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY)'
        )

        # inner_rows columns. When iou=True, reuse area cols for coverage;
        # when iou=False, add dedicated _area1_m2/_aint_m2 tracking cols.
        if iou:
            cov_num = 'SUM(area_intersection_m2)'
            cov_den = 'NULLIF(FIRST(area1_m2), 0)'
            inner_extra = [
                f'{_area_m2("g1.geometry::GEOMETRY", crs1)} AS area1_m2',
                f'{_area_m2("g2.geometry::GEOMETRY", crs2)} AS area2_m2',
                f'{_area_m2(intersection_expr, crs1)} AS area_intersection_m2',
            ]
            tracking_cols: list[str] = []
        else:
            cov_num = 'SUM(_aint_m2)'
            cov_den = 'NULLIF(FIRST(_area1_m2), 0)'
            inner_extra = []
            tracking_cols = [
                f'{_area_m2("g1.geometry::GEOMETRY", crs1)} AS _area1_m2',
                f'{_area_m2(intersection_expr, crs1)} AS _aint_m2',
            ]

        inner_row_parts = (
            [f'a1.{index1} AS {alias1}', f'a2.{index2} AS {alias2}']
            + tracking_cols
            + inner_extra
            + col_select
            + ([f'ST_AsWKB({intersection_expr})::BLOB AS geometry'] if geom else [])
        )
        inner_rows_sql = ',\n                '.join(inner_row_parts)

        # clean_inner: exclude tracking cols, alias2 comes through
        def _col_name(expr: str) -> str:
            return (
                expr.split(' AS ')[-1] if ' AS ' in expr else expr.split('.')[-1]
            ).strip()

        clean_parts = [alias1, alias2]
        if iou:
            clean_parts += ['area1_m2', 'area2_m2', 'area_intersection_m2']
        clean_parts += [_col_name(e) for e in col_select]
        if geom:
            clean_parts.append('geometry')
        clean_inner_sql = ', '.join(clean_parts)

        # unmatched / leftover column lists
        null_iou = (
            [
                f'{_area_m2("g1.geometry::GEOMETRY", crs1)} AS area1_m2',
                'NULL::DOUBLE AS area2_m2',
                'NULL::DOUBLE AS area_intersection_m2',
            ]
            if iou
            else []
        )
        unmatched_parts = (
            [f'a1.{index1} AS {alias1}', f'NULL AS {alias2}']
            + null_iou
            + col_select_null
            + (['ST_AsWKB(g1.geometry::GEOMETRY)::BLOB AS geometry'] if geom else [])
        )
        unmatched_sql = ',\n                '.join(unmatched_parts)

        leftover_geom_sql = (
            'ST_Difference(g1.geometry::GEOMETRY, ST_Union_Agg(g2.geometry::GEOMETRY))'
        )
        leftover_parts = (
            [f'a1.{index1} AS {alias1}', f'NULL AS {alias2}']
            + null_iou
            + col_select_null
            + ([f'ST_AsWKB({leftover_geom_sql})::BLOB AS geometry'] if geom else [])
        )
        leftover_sql = ',\n                '.join(leftover_parts)

        group_by_sql = ', '.join(
            [f'g1.{join_key1}', 'g1.geometry', f'a1.{index1}'] + group_by_extras
        )
        _TOL = 1e-6

        overlaps = con.execute(f"""
            WITH
            inner_rows AS MATERIALIZED (
                SELECT
                    {inner_rows_sql}
                FROM read_parquet('{geo_path1}') AS g1
                JOIN read_parquet('{path1}') AS a1
                    ON g1.{join_key1} = a1.{join_key1}
                JOIN read_parquet('{geo_path2}') AS g2
                    ON ST_Intersects(
                        g1.geometry::GEOMETRY,
                        g2.geometry::GEOMETRY
                    )
                    AND NOT ST_Touches(
                        g1.geometry::GEOMETRY,
                        g2.geometry::GEOMETRY
                    )
                JOIN read_parquet('{path2}') AS a2
                    ON g2.{join_key2} = a2.{join_key2}
            ),
            coverage AS (
                SELECT {alias1},
                       {cov_num} / {cov_den} AS _frac
                FROM inner_rows
                GROUP BY {alias1}
            )
            SELECT {clean_inner_sql} FROM inner_rows
            UNION ALL
            SELECT
                {unmatched_sql}
            FROM read_parquet('{geo_path1}') AS g1
            JOIN read_parquet('{path1}') AS a1
                ON g1.{join_key1} = a1.{join_key1}
            WHERE a1.{index1} NOT IN (SELECT {alias1} FROM inner_rows)
            UNION ALL
            SELECT
                {leftover_sql}
            FROM read_parquet('{geo_path1}') AS g1
            JOIN read_parquet('{path1}') AS a1
                ON g1.{join_key1} = a1.{join_key1}
            JOIN read_parquet('{geo_path2}') AS g2
                ON ST_Intersects(
                    g1.geometry::GEOMETRY,
                    g2.geometry::GEOMETRY
                )
            WHERE a1.{index1} IN (
                SELECT {alias1} FROM coverage WHERE _frac < {1 - _TOL}
            )
            GROUP BY {group_by_sql}
        """).df()

    else:
        # intersection / union single-query path
        if how == 'intersection':
            join_type = 'JOIN'
            geom_expr = 'ST_Intersection(g1.geometry::GEOMETRY, g2.geometry::GEOMETRY)'
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
                AND NOT ST_Touches(
                    g1.geometry::GEOMETRY,
                    g2.geometry::GEOMETRY
                )
            LEFT JOIN read_parquet('{path2}') AS a2
                ON g2.{join_key2} = a2.{join_key2}
        """
        ).df()

    con.close()

    # DuckDB 1.5+ returns string columns as StringDtype; normalize to object
    for _col in overlaps.select_dtypes(include='string').columns:
        overlaps[_col] = overlaps[_col].astype(object)

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
        _sfx1 = suffixes[0] if suffixes is not None else '_left'
        _sfx2 = suffixes[1] if suffixes is not None else '_right'
        overlaps = overlaps.rename(
            columns={
                'area1_m2': f'area{_sfx1}_m2',
                'area2_m2': f'area{_sfx2}_m2',
                'area_intersection_m2': 'area_intersection_m2',
            }
        )

    overlaps = overlaps.set_index([alias1, alias2])

    if geom:
        overlaps['geometry'] = shapely.from_wkb(overlaps['geometry'].apply(bytes))
        overlaps = overlaps[[v for v in overlaps if v != 'geometry'] + ['geometry']]
        return gpd.GeoDataFrame(overlaps, geometry='geometry', crs=crs1)

    return overlaps


def benchmark_iou(
    gdf_left,
    gdf_right,
    suffixes=None,
    how='intersection',
):
    """Compare overlay_polygons (geopandas) vs overlay_polygons_with_duckdb for IOU.

    Runs both approaches and prints wall-clock time for each.

    Parameters
    ----------
    gdf_left : GeoDataFrame
    gdf_right : GeoDataFrame
    suffixes : tuple[str, str] or None
        Required when both GeoDataFrames share the same index name.
        Passed to both functions unchanged.
    how : str
        Overlay type — 'intersection', 'union', or 'identity'.

    Returns
    -------
    result_mem : pd.DataFrame
        Result of overlay_polygons (geopandas).
    result_disk : pd.DataFrame
        Result of overlay_polygons_with_duckdb.
    """
    t0 = time.perf_counter()
    result_mem = overlay_polygons(
        gdf_left, gdf_right, iou=True, suffixes=suffixes, how=how
    )
    t_mem = time.perf_counter() - t0

    t1 = time.perf_counter()
    result_disk = overlay_polygons_with_duckdb(
        gdf_left, gdf_right, iou=True, suffixes=suffixes, how=how
    )
    t_disk = time.perf_counter() - t1

    print(f'overlay_polygons            : {t_mem:.3f}s')
    print(f'overlay_polygons_with_duckdb: {t_disk:.3f}s')

    # Compare iou values — normalize index level names before aligning
    iou_mem = result_mem[['iou']].sort_index()
    iou_disk = result_disk[['iou']].copy()
    iou_disk.index.names = iou_mem.index.names
    iou_disk = iou_disk.sort_index()

    matched_mem = iou_mem[iou_mem.index.get_level_values(1).notna()]
    matched_disk = iou_disk[iou_disk.index.get_level_values(1).notna()]
    unmatched_mem = iou_mem[iou_mem.index.get_level_values(1).isna()]
    unmatched_disk = iou_disk[iou_disk.index.get_level_values(1).isna()]

    print(f'Matched rows   — mem: {len(matched_mem)}, disk: {len(matched_disk)}')
    print(f'Unmatched rows — mem: {len(unmatched_mem)}, disk: {len(unmatched_disk)}')

    pd.testing.assert_frame_equal(
        matched_mem.sort_index(),
        matched_disk.sort_index(),
        rtol=1e-5,
        check_index_type=False,
    )
    print('Matched rows: iou values agree.')

    unmatched_ids_mem = set(unmatched_mem.index.get_level_values(0))
    unmatched_ids_disk = set(unmatched_disk.index.get_level_values(0))
    assert unmatched_ids_mem == unmatched_ids_disk, (
        f'Unmatched left IDs differ: '
        f'mem-only={unmatched_ids_mem - unmatched_ids_disk}, '
        f'disk-only={unmatched_ids_disk - unmatched_ids_mem}'
    )
    print(f'Unmatched rows: left IDs agree ({len(unmatched_mem)} rows).')

    return result_mem, result_disk


def benchmark_overlay(
    gdf_left: gpd.GeoDataFrame,
    gdf_right: gpd.GeoDataFrame,
    suffixes: tuple[str, str] | None = None,
    how: str = 'intersection',
    n: int = 1,
):
    """Benchmark overlay_polygons vs overlay_polygons_with_duckdb across all
    combinations of input_type × iou × geom.

    Parameters
    ----------
    gdf_left, gdf_right : GeoDataFrame
        Representative input data (already in memory).
    suffixes :
        Passed through to both functions.
    how :
        Overlay type.
    n :
        Number of repetitions per cell; median is reported.
    """
    import statistics
    import tempfile
    from pathlib import Path as _Path

    with (
        tempfile.TemporaryDirectory() as tmp1,
        tempfile.TemporaryDirectory() as tmp2,
    ):
        p1 = _Path(tmp1) / 'layer1.parquet'
        p2 = _Path(tmp2) / 'layer2.parquet'
        save_parquet(gdf_left, p1)
        save_parquet(gdf_right, p2)

        rows = []
        for input_label, l1, l2 in [
            ('GeoDataFrame', gdf_left, gdf_right),
            ('Path', p1, p2),
        ]:
            for iou_flag in (False, True):
                for geom_flag in (False, True):
                    kw = dict(
                        suffixes=suffixes,
                        how=how,
                        iou=iou_flag,
                        geom=geom_flag,
                    )
                    for fn, label in [
                        (overlay_polygons, 'geopandas'),
                        (overlay_polygons_with_duckdb, 'duckdb   '),
                    ]:
                        times = []
                        for _ in range(n):
                            t0 = time.perf_counter()
                            fn(l1, l2, **kw)
                            times.append(time.perf_counter() - t0)
                        rows.append(
                            dict(
                                input=input_label,
                                iou=iou_flag,
                                geom=geom_flag,
                                fn=label,
                                t=statistics.median(times),
                            )
                        )

    header = (
        f'{"input":>12}  {"iou":<5}  {"geom":<5}'
        f'  {"geopandas":>10}  {"duckdb":>10}  winner'
    )
    print(header)
    print('-' * len(header))
    for i in range(0, len(rows), 2):
        gp, dk = rows[i], rows[i + 1]
        winner = 'geopandas' if gp['t'] < dk['t'] else 'duckdb   '
        print(
            f'{gp["input"]:>12}  {str(gp["iou"]):<5}  {str(gp["geom"]):<5}'
            f'  {gp["t"]:>9.3f}s  {dk["t"]:>9.3f}s  {winner}'
        )
