"""A hub parcel in several clusters is reported, not silently trimmed.

`_cluster_condo_parcels` can attach one hub to more than one qualifying
cluster, which gives it a repeated entry in the component Series. A
scalar column can hold only one, so the first is kept; the count is now
printed under verbose rather than dropped without a word.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.attributes as attributes
from openplaces.io.harmonizer import HarmonizeState


def _state(verbose):
    spine = gpd.GeoDataFrame(
        geometry=[box(i, 0, i + 1, 1) for i in range(3)],
        crs='epsg:6933',
        index=pd.Index(['hub', 'u1', 'u2'], name='parcel_id'),
    )
    return HarmonizeState(
        recipe={}, admin_id='US-NC-WAK', verbose=verbose, timer=None, spine=spine
    )


def _component(labels, values):
    return pd.Series(values, index=pd.Index(labels, name='parcel_id'))


def _run(monkeypatch, component, verbose=True):
    monkeypatch.setattr(
        attributes, '_cluster_condo_parcels', lambda *a, **k: (component, {'hub'})
    )
    return attributes.detect_condo_building_clusters(_state(verbose))


def test_hub_in_two_clusters_is_reported(monkeypatch, capsys):
    component = _component(['hub', 'hub', 'u1', 'u2'], ['c1', 'c2', 'c1', 'c2'])
    state = _run(monkeypatch, component)
    out = capsys.readouterr().out
    assert 'building_cluster_id: 1 hub parcel(s) belong to more than one' in out
    assert state.spine.loc['hub', 'building_cluster_id'] == 'c1'


def test_one_cluster_per_parcel_prints_nothing(monkeypatch, capsys):
    component = _component(['hub', 'u1', 'u2'], ['c1', 'c1', 'c1'])
    state = _run(monkeypatch, component)
    assert 'belong to more than one' not in capsys.readouterr().out
    assert state.spine['building_cluster_id'].tolist() == ['c1', 'c1', 'c1']


def test_nothing_is_printed_without_verbose(monkeypatch, capsys):
    component = _component(['hub', 'hub', 'u1', 'u2'], ['c1', 'c2', 'c1', 'c2'])
    _run(monkeypatch, component, verbose=False)
    assert 'belong to more than one' not in capsys.readouterr().out
