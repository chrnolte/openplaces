"""Tests for the generic table-profiling diagnostics (`profile_columns`,
`summarize_categoricals`, `check_geometry`).

These are the reusable functions behind `notebooks/diagnostics/
inspect_entity_statistics.ipynb`: unlike `find_recipes`/`map_recipe_coverage`
(which report on the recipe *registry* -- what exists), these profile the
*data* an already-loaded recipe output actually contains.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, box

from openplaces.diagnostics import (
    check_geometry,
    profile_columns,
    summarize_categoricals,
)


def test_profile_columns_completeness_fractions():
    gdf = pd.DataFrame({'owner_name': ['Smith', 'Jones', None, 'Lee']})
    profile = profile_columns(gdf)
    row = profile.loc['owner_name']
    assert row['n_values'] == 3
    assert row['frac_values'] == 0.75


def test_profile_columns_distinguishes_placeholder_from_real_null():
    # NC OneMap's year_built: 0 is a placeholder ("never recorded"), not a
    # real construction year -- notna() alone can't tell the difference,
    # but frac_meaningful must.
    gdf = pd.DataFrame({'year_built': [0.0, 0.0, 1964.0]})
    profile = profile_columns(gdf)
    row = profile.loc['year_built']
    assert row['n_values'] == 3
    assert row['frac_values'] == 1.0
    assert row['n_meaningful'] == 1
    assert row['frac_meaningful'] == pytest.approx(1 / 3)


def test_profile_columns_no_entity_type_skips_registry_check():
    gdf = pd.DataFrame({'use_group': ['RESIDENTIAL', 'COMMERCIAL']})
    profile = profile_columns(gdf)
    row = profile.loc['use_group']
    assert row['expected_data_type'] is None
    assert bool(row['dtype_mismatch']) is False


def test_profile_columns_flags_real_dtype_mismatch():
    # land_value is registered as float; a string column is a genuine mismatch.
    gdf = pd.DataFrame({'land_value': ['not-a-number', 'also-not-a-number']})
    profile = profile_columns(gdf, entity_type='parcel')
    row = profile.loc['land_value']
    assert row['expected_data_type'] == 'float'
    assert bool(row['dtype_mismatch']) is True


def test_profile_columns_string_dtype_not_flagged_for_categorical():
    # use_group is registered categorical; plain string/object dtype (before
    # any explicit `columns_to_categorical` cast) must NOT be flagged as a
    # mismatch -- this is the normal, expected state for a freshly-ingested
    # column.
    gdf = pd.DataFrame({'use_group': ['RESIDENTIAL', 'COMMERCIAL', None]})
    profile = profile_columns(gdf, entity_type='parcel')
    row = profile.loc['use_group']
    assert row['expected_data_type'] == 'categorical'
    assert bool(row['dtype_mismatch']) is False

    # Already cast to pandas Categorical must also not be flagged.
    gdf2 = pd.DataFrame({'use_group': pd.Categorical(['RESIDENTIAL', 'COMMERCIAL'])})
    profile2 = profile_columns(gdf2, entity_type='parcel')
    assert bool(profile2.loc['use_group', 'dtype_mismatch']) is False


def test_profile_columns_resolves_provenance_suffixed_column():
    # improvement_value_parcel should resolve to the registered
    # improvement_value attribute (float) for the mismatch check.
    gdf = pd.DataFrame({'improvement_value_parcel': ['bad', 'bad']})
    profile = profile_columns(gdf, entity_type='parcel')
    row = profile.loc['improvement_value_parcel']
    assert row['expected_data_type'] == 'float'
    assert bool(row['dtype_mismatch']) is True


def test_profile_columns_skips_geometry_column():
    gdf = gpd.GeoDataFrame(
        {'value': [1.0, 2.0]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)]
    )
    profile = profile_columns(gdf)
    assert 'geometry' not in profile.index
    assert 'value' in profile.index


def test_summarize_categoricals_detects_near_categorical_column():
    gdf = pd.DataFrame({'use_group': ['RESIDENTIAL'] * 8 + ['COMMERCIAL'] * 2})
    result = summarize_categoricals(gdf)
    assert 'use_group' in result
    assert result['use_group']['RESIDENTIAL'] == 8


def test_summarize_categoricals_excludes_id_like_column():
    # parcel_id_local: every value is unique in real data, but even a
    # high-duplication id-like column name should never be reported as
    # categorical.
    gdf = pd.DataFrame({'parcel_id_local': ['a', 'a', 'a', 'b', 'b', 'b']})
    result = summarize_categoricals(gdf)
    assert 'parcel_id_local' not in result


def test_summarize_categoricals_force_includes_registered_attribute():
    # use_group is registered categorical (attribute_registry.csv); even
    # with low duplication (every value distinct), it should still be
    # reported, since sparse-but-genuinely-categorical data shouldn't be
    # dropped by the duplication heuristic alone.
    gdf = pd.DataFrame({'use_group': ['A', 'B', 'C', 'D', 'E']})
    result = summarize_categoricals(gdf)
    assert 'use_group' in result


def test_summarize_categoricals_skips_high_cardinality_non_registered_column():
    gdf = pd.DataFrame({'legal_description': [f'lot {i}' for i in range(10)]})
    result = summarize_categoricals(gdf)
    assert 'legal_description' not in result


def test_check_geometry_flags_martin_county_shaped_defects():
    # Mirrors the real Martin County, NC bug: 16,609 clean Polygons, one
    # stray degenerate LineString, one null geometry.
    gdf = gpd.GeoDataFrame(
        geometry=[
            box(0, 0, 1, 1),
            box(1, 1, 2, 2),
            box(2, 2, 3, 3),
            LineString([(0, 0), (0.001, 0.001)]),
            None,
        ]
    )
    report = check_geometry(gdf)
    assert report['n_total'] == 5
    assert report['n_null'] == 1
    assert report['n_invalid'] == 0
    assert report['geom_type_counts']['Polygon'] == 3
    assert report['geom_type_counts']['LineString'] == 1


def test_check_geometry_clean_polygon_layer_reports_no_defects():
    gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)])
    report = check_geometry(gdf)
    assert report['n_null'] == 0
    assert report['n_empty'] == 0
    assert report['n_invalid'] == 0
    assert set(report['geom_type_counts'].index) == {'Polygon'}
