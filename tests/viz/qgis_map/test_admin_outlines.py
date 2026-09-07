"""Tests for the admin outline sidecars shipped beside a delivery bundle."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.viz import qgis_map


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A delivery bundle of two New England towns, in a temporary folder."""
    canonical = tmp_path / 'US-MA_footprint-openplaces-2026.parquet'
    members = ['US-MA-BOS', 'US-MA-CAM']
    admin = gpd.GeoDataFrame(
        {'name': ['Boston', 'Cambridge'], 'type': ['City', 'Town']},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        index=pd.Index(members, name='admin3_id'),
        crs='EPSG:4326',
    )

    import openplaces as op

    monkeypatch.setattr(op, 'get_admin', lambda *a, **k: admin)
    monkeypatch.setattr(
        'openplaces.io.delivery.delivery_paths',
        lambda *a, **k: {'canonical': canonical},
    )
    monkeypatch.setattr(
        'openplaces.io.delivery.delivery_members', lambda *a, **k: members
    )
    # The admin4 half needs the real entity store; make it a no-op.
    monkeypatch.setattr(
        op, 'get_entities', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError())
    )
    return canonical


def test_admin3_sidecar_carries_the_type_column(bundle):
    """The resolver reads `type` to label New England's towns.

    Written without it, every shipped sidecar labeled its towns
    "Counties".
    """
    with pytest.warns(UserWarning):
        qgis_map.ensure_delivery_admin_outlines('a-recipe', 'a-region')
    side = bundle.with_name(f'{bundle.stem}_admin3_geo.parquet')
    written = gpd.read_parquet(side)
    assert 'type' in written.columns
    assert set(written['type']) == {'City', 'Town'}


def test_a_sidecar_without_the_type_column_is_refreshed(bundle):
    """The exists-guard alone meant shipped sidecars never refreshed."""
    side = bundle.with_name(f'{bundle.stem}_admin3_geo.parquet')
    stale = gpd.GeoDataFrame(
        {'name': ['Boston']},
        geometry=[box(0, 0, 1, 1)],
        index=pd.Index(['US-MA-BOS'], name='admin3_id'),
        crs='EPSG:4326',
    )
    stale.to_parquet(side)

    with pytest.warns(UserWarning):
        qgis_map.ensure_delivery_admin_outlines('a-recipe', 'a-region')
    written = gpd.read_parquet(side)
    assert 'type' in written.columns
    assert len(written) == 2
