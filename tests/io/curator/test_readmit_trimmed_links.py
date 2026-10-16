"""Which trimmed footprint-parcel overlaps rejoin a parcel's value split.

The harmonize crosswalk trims a footprint's link to a parcel below 1/6
of its largest overlap. `_readmittable_trimmed_links` brings a trimmed
overlap back only where it is larger than everything still linked to
the parcel, the case of a large building standing on several lots.
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.curator.evidence import _readmittable_trimmed_links


def _links(rows):
    return pd.DataFrame(
        rows,
        columns=['fid', 'parcel_id', 'area_intersection_m2', 'link', 'link_chain'],
    )


def test_trimmed_overlap_larger_than_the_parcels_linked_footprints_returns():
    links = _links(
        [
            ('BIG', 'P1', 740.0, None, None),  # 8% of BIG, trimmed
            ('BIG', 'P2', 8000.0, 'multi-parcel footprint', None),
            ('KIOSK', 'P1', 14.0, 'unique parcel', None),
        ]
    )
    out = _readmittable_trimmed_links(links, 'fid', 'parcel_id', 10.0)
    assert list(zip(out['fid'], out['parcel_id'])) == [('BIG', 'P1')]


def test_trimmed_edge_overlap_beside_the_parcels_own_building_stays_out():
    links = _links(
        [
            ('HOUSE_A', 'P2', 25.0, None, None),  # edge onto the neighbor
            ('HOUSE_A', 'P1', 180.0, 'unique parcel', None),
            ('HOUSE_B', 'P2', 160.0, 'unique parcel', None),
        ]
    )
    assert _readmittable_trimmed_links(links, 'fid', 'parcel_id', 10.0).empty


def test_parcel_without_a_linked_footprint_gets_no_readmission():
    # Such a parcel holds the harmonize stand-in when it is improved;
    # a neighbor's edge must not take the stand-in's value.
    links = _links([('HOUSE_A', 'P2', 25.0, None, None)])
    assert _readmittable_trimmed_links(links, 'fid', 'parcel_id', 10.0).empty


def test_snapped_minor_and_small_overlaps_stay_out():
    links = _links(
        [
            ('A', 'P1', 400.0, None, 'snapped minor'),
            ('B', 'P2', 8.0, None, None),  # below the minimum area
        ]
    )
    assert _readmittable_trimmed_links(links, 'fid', 'parcel_id', 10.0).empty
