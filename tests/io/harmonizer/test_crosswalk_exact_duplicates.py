"""Exact duplicate crosswalk rows are dropped; disagreeing ones still raise.

The identity overlay can emit the same (spine id, reference id) row
several times over, identical in every column. Measured across the
2026-09-08 rebuild of the two CHEER regions, every repeated pair was an
exact copy, 2 to 32 of them (Wake NC 386 pairs over 1,886 rows, Harris
TX 3 over 6), and 28 of 86 counties failed the uniqueness guard on rows
that carried no information at all.

Dropping exact copies is lossless, and it also restores the right
*label*: left in place, the copies go down the multi-overlap branch and
a footprint on one parcel is called a multi-parcel footprint. A pair
whose rows genuinely disagree is a different thing and must still be
refused, because the crosswalk feeds an index-aligned operation that
cannot choose between them.
"""

import pandas as pd
import pytest

from openplaces.io.harmonizer.links import _build_crosswalk

KEYS = ['footprint_id', 'parcel_id']


def _overlay(rows):
    frame = pd.DataFrame(
        rows, columns=[*KEYS, 'area_intersection_m2', 'area_spine_m2', 'area_ref_m2']
    )
    return frame.set_index(KEYS)


def test_exact_duplicate_rows_collapse_to_one_unique_parcel():
    overlay = _overlay(
        [
            ('8762PGXG+W8Q', 'parcel-a', 3254.589579, None, None),
            ('8762PGXG+W8Q', 'parcel-a', 3254.589579, None, None),
            ('8762PGXG+W8Q', 'parcel-a', 3254.589579, None, None),
            ('8762QG2G+22G', 'parcel-b', 4462.537421, None, None),
        ]
    )

    crosswalk = _build_crosswalk(overlay, 'footprint_id', 0.0, 0.0)

    assert len(crosswalk) == 2
    assert crosswalk.index.get_level_values('footprint_id').tolist() == [
        '8762PGXG+W8Q',
        '8762QG2G+22G',
    ]
    # The point of dropping the copies: one footprint on one parcel is a
    # unique parcel, not a multi-parcel footprint.
    assert crosswalk['link'].tolist() == ['unique parcel', 'unique parcel']
    assert crosswalk.loc[('8762PGXG+W8Q', 'parcel-a'), 'area_intersection_m2'] == (
        pytest.approx(3254.589579)
    )


def test_a_pair_whose_rows_disagree_still_raises():
    """Two rows for one pair that differ cannot be silently collapsed."""
    overlay = _overlay(
        [
            ('8762PGXG+W8Q', 'parcel-a', 3254.5, None, None),
            ('8762PGXG+W8Q', 'parcel-a', 9999.9, None, None),
        ]
    )

    with pytest.raises(ValueError, match='repeated label'):
        _build_crosswalk(overlay, 'footprint_id', 0.0, 0.0)


def test_a_genuine_multi_overlap_is_untouched():
    """One footprint on two distinct parcels keeps both of its links."""
    overlay = _overlay(
        [
            ('8762PGXG+W8Q', 'parcel-a', 300.0, None, None),
            ('8762PGXG+W8Q', 'parcel-b', 700.0, None, None),
        ]
    )

    crosswalk = _build_crosswalk(overlay, 'footprint_id', 0.0, 0.0)

    assert len(crosswalk) == 2
    assert set(crosswalk['link']) == {'multi-parcel footprint'}
    assert crosswalk['fraction_of_largest'].max() == pytest.approx(1.0)


def test_an_all_null_companion_row_is_dropped():
    """A populated row plus a null twin is not a conflict.

    Seven counties produce the same pair twice, once with every measure
    populated and once with all of them NaN. The null row says nothing
    about the overlap, so it loses to its informative sibling.
    """
    overlay = _overlay(
        [
            ('8764CFH4+C5H', 'parcel-a', 778.948158, 7.941888, 0.010196),
            ('8764CFH4+C5H', 'parcel-a', None, None, None),
        ]
    )

    crosswalk = _build_crosswalk(overlay, 'footprint_id', 0.0, 0.0)

    assert len(crosswalk) == 1
    assert crosswalk['area_intersection_m2'].iloc[0] == pytest.approx(778.948158)


def test_two_null_rows_for_one_pair_still_raise():
    """With no informative sibling there is nothing to prefer."""
    overlay = _overlay(
        [
            ('8764CFH4+C5H', 'parcel-a', None, None, 1.0),
            ('8764CFH4+C5H', 'parcel-a', None, None, 2.0),
        ]
    )

    with pytest.raises(ValueError, match='repeated label'):
        _build_crosswalk(overlay, 'footprint_id', 0.0, 0.0)
