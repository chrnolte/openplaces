"""Keep-first drops in the harmonizer say how many rows they discarded.

None of these change what is kept; each was silent before, so a run gave
no sign that an overlapping reference or a repeated match had cost it
rows.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import openplaces.io.harmonizer.spine as spine_module
from openplaces.io.harmonizer import HarmonizeState

# Two reference polygons overlapping between x=1 and x=2, so a point in
# the overlap matches both and a polygon spanning it overlaps both.
_OVERLAPPING_REF = gpd.GeoDataFrame(
    geometry=[box(0, 0, 2, 1), box(1, 0, 3, 1)],
    crs='epsg:4326',
    index=pd.Index(['A', 'B'], name='the_id'),
)

# Two disjoint references with a gap between x=1 and x=2, so a polygon
# spanning the gap has its centroid outside both.
_GAPPED_REF = gpd.GeoDataFrame(
    geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1)],
    crs='epsg:4326',
    index=pd.Index(['A', 'B'], name='the_id'),
)


def _spine(geometry, lat, long):
    return gpd.GeoDataFrame(
        {'lat': [lat], 'long': [long]},
        geometry=[geometry],
        crs='epsg:4326',
        index=pd.Index(['P1'], name='parcel_id'),
    )


def _state(spine, verbose):
    return HarmonizeState(
        recipe={'entity': {'entity_type': 'parcel'}},
        admin_id='US-NC-WAK',
        verbose=verbose,
        timer=None,
        spine=spine,
    )


def _links():
    return [{'recipe_id': 'US_tile-census-2025_tract', 'output_column': 'the_id'}]


def _run(monkeypatch, reference, spine, verbose):
    monkeypatch.setattr(spine_module, 'get_entities', lambda *a, **k: reference)
    return spine_module.link_geographic_ids(_state(spine, verbose), links=_links())


def test_centroid_in_two_reference_polygons_is_reported(monkeypatch, capsys):
    spine = _spine(box(1.2, 0.2, 1.4, 0.4), lat=0.3, long=1.3)
    state = _run(monkeypatch, _OVERLAPPING_REF, spine, verbose=True)
    out = capsys.readouterr().out
    assert '1 centroid(s) fell in more than one reference polygon' in out
    assert state.spine.loc['P1', 'the_id'] in {'A', 'B'}


def test_largest_overlap_fallback_reports_the_discarded_match(monkeypatch, capsys):
    # Centroid at x=1.5 sits in the gap, so pass 2 runs; the polygon
    # overlaps A over 0.5 units and B over 0.3, so B is discarded.
    spine = _spine(box(0.5, 0.2, 2.3, 0.4), lat=0.3, long=1.5)
    state = _run(monkeypatch, _GAPPED_REF, spine, verbose=True)
    out = capsys.readouterr().out
    assert '1 smaller overlap(s) discarded' in out
    assert state.spine.loc['P1', 'the_id'] == 'A'


def test_nothing_is_printed_without_verbose(monkeypatch, capsys):
    spine = _spine(box(1.2, 0.2, 1.4, 0.4), lat=0.3, long=1.3)
    _run(monkeypatch, _OVERLAPPING_REF, spine, verbose=False)
    assert 'more than one reference polygon' not in capsys.readouterr().out


def test_a_clean_match_reports_no_drop(monkeypatch, capsys):
    spine = _spine(box(0.2, 0.2, 0.4, 0.4), lat=0.3, long=0.3)
    state = _run(monkeypatch, _GAPPED_REF, spine, verbose=True)
    out = capsys.readouterr().out
    assert 'more than one reference polygon' not in out
    assert 'smaller overlap' not in out
    assert state.spine.loc['P1', 'the_id'] == 'A'
