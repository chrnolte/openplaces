"""A point reference built with no rows in the unit still writes its
link file; one never built for the unit writes none.

The file is a declared output of the geospine; without it the workflow
read nine Wisconsin counties with no Overture point as failed jobs
(2026-10-19). Fabricated frames throughout.
"""

import geopandas as gpd
import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer.links import points


def _state():
    spine = gpd.GeoDataFrame(
        {'geometry': []},
        geometry='geometry',
        crs='EPSG:4326',
        index=pd.Index([], name='footprint_id'),
    )
    return HarmonizeState(
        recipe={}, admin_id=AdminId('US-XX-YY'), verbose=False, timer=None, spine=spine
    )


def _patch(monkeypatch, tmp_path, built=True):
    empty = gpd.GeoDataFrame({'geometry': []}, geometry='geometry', crs='EPSG:4326')
    monkeypatch.setattr(points, 'get_entities', lambda *a, **k: empty)
    # Built for the unit and empty, or never built: get_entities with
    # missing='warn' returns the two alike.
    monkeypatch.setattr(points, 'entity_outputs_exist', lambda *a, **k: built)
    monkeypatch.setattr(points, 'raise_if_coverage_complete', lambda *a, **k: None)
    monkeypatch.setattr(points, 'get_recipe_id', lambda recipe: 'geospine')
    monkeypatch.setattr(
        points,
        'get_entity_link_path',
        lambda own, other, admin_id: tmp_path / f'{other}_{own}.parquet',
    )
    monkeypatch.setattr(points, '_link_fingerprint', lambda *a, **k: {'step': 'x'})
    return tmp_path / 'points-ref_geospine.parquet'


def test_an_empty_reference_writes_an_empty_link(monkeypatch, tmp_path):
    path = _patch(monkeypatch, tmp_path)
    state = points._link_spatial_point(
        _state(), 'points-ref', 'dwelling', None, {}, save_link=True
    )
    assert path.exists()
    assert pd.read_parquet(path).empty
    # Nothing to restore from it: no crosswalk, the type still recorded.
    assert 'points-ref' not in state.crosswalks
    assert state.reference_types['points-ref'] == 'dwelling'


def test_an_unbuilt_reference_writes_no_link(monkeypatch, tmp_path):
    path = _patch(monkeypatch, tmp_path, built=False)
    state = points._link_spatial_point(
        _state(), 'points-ref', 'dwelling', None, {}, save_link=True
    )
    # No file: the curate readers refuse the missing link until it is
    # computed against a built reference.
    assert not path.exists()
    assert state.reference_types['points-ref'] == 'dwelling'


def test_without_save_link_nothing_is_written(monkeypatch, tmp_path):
    path = _patch(monkeypatch, tmp_path)
    points._link_spatial_point(_state(), 'points-ref', 'dwelling', None, {})
    assert not path.exists()
