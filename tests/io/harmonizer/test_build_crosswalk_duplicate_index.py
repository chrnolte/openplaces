"""The spine-reference crosswalk refuses a spine label it saw twice.

A real multi-overlap pairs one spine id with several distinct reference
ids; the same pair twice means the spine carried the label twice. Left
alone, the two copies of one entity are routed into the multi branch,
compete on intersection area, and the smaller is trimmed away as a
neighbor, so it is refused instead.
"""

import pandas as pd
import pytest

from openplaces.io.harmonizer.links import _build_crosswalk


def _overlay(pairs, areas):
    index = pd.MultiIndex.from_tuples(pairs, names=['footprint_id', 'parcel_id'])
    return pd.DataFrame({'area_intersection_m2': areas}, index=index)


def test_repeated_spine_reference_pair_raises():
    overlay = _overlay([('f1', 'p1'), ('f1', 'p1'), ('f2', 'p2')], [90.0, 10.0, 50.0])
    with pytest.raises(ValueError, match='link_to_reference crosswalk on footprint_id'):
        _build_crosswalk(overlay, 'footprint_id', min_fraction=0.2, area_min_m2=5.0)


def test_genuine_multi_overlap_still_splits():
    overlay = _overlay([('f1', 'p1'), ('f1', 'p2'), ('f2', 'p3')], [90.0, 80.0, 50.0])
    crosswalk = _build_crosswalk(
        overlay, 'footprint_id', min_fraction=0.2, area_min_m2=5.0
    )
    links = crosswalk['link'].to_dict()
    assert links[('f1', 'p1')] == 'multi-parcel footprint'
    assert links[('f1', 'p2')] == 'multi-parcel footprint'
    assert links[('f2', 'p3')] == 'unique parcel'


def test_small_neighbor_is_still_trimmed_on_distinct_references():
    overlay = _overlay([('f1', 'p1'), ('f1', 'p2')], [90.0, 1.0])
    crosswalk = _build_crosswalk(
        overlay, 'footprint_id', min_fraction=0.2, area_min_m2=5.0
    )
    assert crosswalk['link'].tolist() == ['unique parcel (dropping small neighbor)']
    assert crosswalk.index.tolist() == [('f1', 'p1')]
