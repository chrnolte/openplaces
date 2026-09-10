"""A fill-only link never overwrites what a precise key already set.

`_write_prioritized` overwrites a column outright once the incoming
source covers half the spine, and gap-fills below that. A second pass
on a lossier key relies on gap-filling for its safety, but its coverage
is highest exactly where the key is least trustworthy. Carteret County
NC: 97% of parcels share their punctuation-free key with others, in
groups of up to 335, so the pass matched 98.6% of the spine, tripped the
overwrite, and replaced every parcel's own improvement value with one
arbitrary row's. The county's total came out 31 times its source.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from openplaces.io.harmonizer.links import _write_prioritized


def _spine():
    return gpd.GeoDataFrame(
        {
            'improvement_value': [100.0, 200.0, None, 400.0],
            'geometry': [Point(i, i) for i in range(4)],
        },
        index=pd.Index(['a', 'b', 'c', 'd'], name='parcel_id'),
        crs='epsg:4326',
    )


def _incoming():
    # Covers every row: one arbitrary group value broadcast to all.
    return pd.Series([999.0, 999.0, 999.0, 999.0], index=['a', 'b', 'c', 'd'])


def test_the_default_overwrites_once_coverage_reaches_half():
    """The rule this test exists to pin, so the fix is not mistaken for it."""
    spine = _spine()

    _write_prioritized(spine, 'improvement_value', _incoming())

    assert spine['improvement_value'].tolist() == [999.0, 999.0, 999.0, 999.0]


def test_fill_only_keeps_every_existing_value_and_fills_the_gap():
    spine = _spine()

    _write_prioritized(
        spine, 'improvement_value', _incoming(), majority_coverage=float('inf')
    )

    assert spine['improvement_value'].tolist() == [100.0, 200.0, 999.0, 400.0]


def test_fill_only_records_provenance_only_for_the_filled_cell():
    spine = _spine()
    spine['improvement_value_source'] = pd.Series(
        ['county', 'county', None, 'county'], index=spine.index, dtype=object
    )

    _write_prioritized(
        spine,
        'improvement_value',
        _incoming(),
        majority_coverage=float('inf'),
        provenance_token='nconemap',
    )

    assert spine['improvement_value_source'].tolist() == [
        'county',
        'county',
        'nconemap',
        'county',
    ]


@pytest.mark.parametrize('coverage', [0.0, 0.5, 1.0])
def test_fill_only_is_independent_of_coverage(coverage):
    spine = _spine()
    n = int(round(coverage * 4))
    incoming = pd.Series([999.0] * n + [None] * (4 - n), index=['a', 'b', 'c', 'd'])

    _write_prioritized(
        spine, 'improvement_value', incoming, majority_coverage=float('inf')
    )

    kept = spine['improvement_value']
    assert kept['a'] == 100.0 and kept['b'] == 200.0 and kept['d'] == 400.0
