"""Tests for the quality ranking that picks one representative point.

Several reference points can link to one spine entity. The pick is ordered
by source label (ascending, the ranking baked into the label) and then by
structure value (descending, the larger structure wins). Each column keeps
its own direction even when the other is absent.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import SourceGeometryType


def test_quality_sort_keeps_value_descending_without_a_source_column():
    frame = pd.DataFrame({'structure_value': [1.0, 2.0]})
    assert links._point_quality_sort(frame) == (['structure_value'], [False])


def test_quality_sort_pairs_each_present_column_with_its_direction():
    frame = pd.DataFrame({'source': ['a'], 'structure_value': [1.0]})
    assert links._point_quality_sort(frame) == (
        ['source', 'structure_value'],
        [True, False],
    )


def test_quality_sort_is_empty_without_either_column():
    assert links._point_quality_sort(pd.DataFrame({'other': [1]})) == ([], [])


def test_multipoint_representative_is_the_highest_value_point():
    # No 'source' column: the surviving 'structure_value' must still
    # sort descending, so the representative is the larger one. Sliced
    # positionally, the direction paired with it was ascending and the
    # smaller structure represented the footprint.
    linked = pd.DataFrame(
        {
            'footprint_id': ['f1', 'f1'],
            'structure_value': [10.0, 900.0],
            'building_id': ['small', 'large'],
        },
        index=pd.Index(['p1', 'p2'], name='point_id'),
    )

    out = links._aggregate_multipoint(
        linked, 'footprint_id', SourceGeometryType.single_building_point
    )

    assert len(out) == 1
    assert out['building_id'].iloc[0] == 'large'
    assert out['structure_value'].iloc[0] == 900.0
