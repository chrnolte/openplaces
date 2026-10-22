"""`link_transactions_by_overlap` links the parcels lying mostly inside a
transaction's own polygon, independently of the key passes, and merges
into the same link table without removing what they wrote.

Fabricated geometry: a 10 x 10 purchase polygon over a grid of parcels.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from openplaces.core.schema import AdminId, Entity
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import transactions as spine_steps

CRS = 'EPSG:3116'


def _parcels():
    # P1 and P2 lie wholly inside the purchase; P3 is half inside; P4 is
    # outside. area_ha apportions shares.
    geoms = [box(0, 0, 5, 10), box(5, 0, 10, 10), box(8, 0, 12, 10), box(20, 0, 25, 5)]
    frame = gpd.GeoDataFrame(
        {'parcel_id_local': ['K1', 'K2', 'K3', 'K4']},
        geometry=geoms,
        crs=CRS,
        index=pd.Index(['P1', 'P2', 'P3', 'P4'], name='parcel_id'),
    )
    frame['area_ha'] = frame.area / 1e4
    return frame


def _patch(monkeypatch, tmp_path, existing=None):
    parcels = _parcels()
    monkeypatch.setattr(spine_steps, 'get_entities', lambda rid, admin_id, **k: parcels)
    monkeypatch.setattr(spine_steps, 'get_recipe_id', lambda recipe: 'tx-spine')
    monkeypatch.setattr(
        spine_steps, 'get_link_owner_recipe_id', lambda recipe: 'tx-spine'
    )
    path = tmp_path / 'links.parquet'
    monkeypatch.setattr(
        spine_steps, 'get_entity_link_path', lambda own, other, admin_id: path
    )
    monkeypatch.setattr(
        spine_steps, 'write_entity_link', lambda p, links, **k: links.to_parquet(p)
    )
    if existing is not None:
        existing.to_parquet(path)
    return path


def _state():
    spine = gpd.GeoDataFrame(
        {'source': ['reboredo']},
        geometry=[box(0, 0, 10, 10)],
        crs=CRS,
        index=pd.Index(['CO-AN-CV_1'], name='transaction_id'),
    )
    return HarmonizeState(
        recipe={'entity': Entity('transaction', 'spine', '2026')},
        admin_id=AdminId('CO-AN-CV'),
        verbose=False,
        timer=None,
        spine=spine,
    )


def test_links_parcels_mostly_inside_the_polygon(monkeypatch, tmp_path):
    path = _patch(monkeypatch, tmp_path)
    state = spine_steps.link_transactions_by_overlap(
        _state(), parcel_recipe_id='CO_parcel-spine-2026'
    )
    links = pd.read_parquet(path).set_index('parcel_id')
    # P3 is half inside (below 0.8) and P4 outside: neither links.
    assert sorted(links.index) == ['P1', 'P2']
    assert set(links['link_method']) == {'polygon_overlap'}
    assert links['share'].sum() == 1.0
    assert state.spine['parcel_link_method'].iloc[0] == 'polygon_overlap'


def test_a_pair_both_rules_found_is_one_row_with_both_labels(monkeypatch, tmp_path):
    existing = pd.DataFrame(
        {
            'transaction_id': ['CO-AN-CV_1'],
            'parcel_id': ['P1'],
            'link_method': ['parcel_id_local'],
            'link_source': ['reboredo'],
            'share': pd.array([pd.NA], dtype='Float64'),
            'share_basis': pd.array([pd.NA], dtype='string'),
        }
    )
    path = _patch(monkeypatch, tmp_path, existing=existing)
    state = spine_steps.link_transactions_by_overlap(
        _state(), parcel_recipe_id='CO_parcel-spine-2026'
    )
    links = pd.read_parquet(path).set_index('parcel_id')
    assert links.loc['P1', 'link_method'] == 'parcel_id_local+polygon_overlap'
    assert links.loc['P2', 'link_method'] == 'polygon_overlap'
    assert len(links) == 2
    assert state.spine['parcel_link_method'].iloc[0] == (
        'parcel_id_local+polygon_overlap'
    )


def test_an_existing_link_outside_the_polygon_is_kept(monkeypatch, tmp_path):
    """A key link the overlap would not find stays: no link is removed."""
    existing = pd.DataFrame(
        {
            'transaction_id': ['CO-AN-CV_1'],
            'parcel_id': ['P4'],
            'link_method': ['parcel_id_local'],
            'link_source': ['reboredo'],
            'share': pd.array([pd.NA], dtype='Float64'),
            'share_basis': pd.array([pd.NA], dtype='string'),
        }
    )
    path = _patch(monkeypatch, tmp_path, existing=existing)
    spine_steps.link_transactions_by_overlap(
        _state(), parcel_recipe_id='CO_parcel-spine-2026'
    )
    links = pd.read_parquet(path).set_index('parcel_id')
    assert sorted(links.index) == ['P1', 'P2', 'P4']
    assert links.loc['P4', 'link_method'] == 'parcel_id_local'
