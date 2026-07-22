"""Tests for the curate-stage `derive_metrics` step."""

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_metrics


def _state(n_rows=1, **columns):
    # 10m x 10m squares in an equal-area CRS: area = 100 m2 each.
    geometry = [box(20 * i, 0, 20 * i + 10, 10) for i in range(n_rows)]
    curated = gpd.GeoDataFrame({**columns, 'geometry': geometry}, crs='epsg:6933')
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_derive_metrics_builds_value_per_area():
    state = _state(value=[500.0])
    state = derive_metrics(state)
    assert state.curated['area_m2'].iloc[0] == 100.0
    assert state.curated['value_per_area'].iloc[0] == 5.0


def test_derive_metrics_does_not_touch_value_source():
    # 'value_source' is a provenance sidecar reconcile_values writes; it must
    # not be mistaken for the 'value' column via a prefix match.
    state = _state(value=[500.0], value_source=['improvement_value_parcel'])
    state = derive_metrics(state)
    assert 'value_source_per_area' not in state.curated.columns


def test_derive_metrics_does_not_touch_improvement_value_source():
    # improvement_value_source (a reconcile_values provenance sidecar for a
    # bare 'improvement_value' column) starts with 'improvement_value' and
    # would otherwise be swept up by the prefix match and divided as if it
    # were numeric, raising a TypeError (str / float).
    state = _state(improvement_value=[500.0], improvement_value_source=['imputed'])
    state = derive_metrics(state)
    assert state.curated['improvement_value_per_area'].iloc[0] == 5.0
    assert 'improvement_value_source_per_area' not in state.curated.columns


def test_derive_metrics_does_not_touch_structure_value_source():
    state = _state(
        structure_value_building_nsi=[300.0],
        structure_value_building_nsi_source=['building_nsi'],
    )
    state = derive_metrics(state)
    assert 'structure_value_building_nsi_source_per_area' not in state.curated.columns


def test_derive_metrics_still_builds_evidence_per_area_columns():
    state = _state(
        improvement_value_parcel=[200.0], structure_value_building_nsi=[300.0]
    )
    state = derive_metrics(state)
    assert state.curated['improvement_value_parcel_per_area'].iloc[0] == 2.0
    assert state.curated['structure_value_building_nsi_per_area'].iloc[0] == 3.0


def test_m2_missing_on_synthetic_reference_derived_rows():
    # A synthetic fallback row (geometry_source '{entity}.{source}', added by
    # infer_spine_additions) carries the parcel boundary, not a building
    # outline: its m2 must be missing so it isn't mistaken for a footprint
    # area, and its _per_area ratio must inherit the missing denominator.
    state = _state(
        n_rows=2,
        geometry_source=['obm', 'parcel.spine'],
        improvement_value_parcel=[200.0, 200.0],
    )
    state = derive_metrics(state)
    assert state.curated['area_m2'].iloc[0] == 100.0
    assert pd.isna(state.curated['area_m2'].iloc[1])
    assert state.curated['improvement_value_parcel_per_area'].iloc[0] == 2.0
    assert pd.isna(state.curated['improvement_value_parcel_per_area'].iloc[1])


def test_m2_kept_when_geometry_source_absent():
    # Entities without a geometry_source column (e.g. a parcel curate lane)
    # keep their computed area everywhere.
    state = _state(value=[500.0])
    state = derive_metrics(state)
    assert state.curated['area_m2'].iloc[0] == 100.0
