"""split_curated_values over the cut overlay (parcel pieces).

plans/buildings-from-parcel-pieces.md, step 3: each piece takes its own
parcel's values, and what a parcel's buildings receive in all is what
the uncut outline received. The sidecar is the one the split writes
beside the building geospine. Every geometry, id and value is
fabricated (docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.io.curator.evidence import split_curated_values
from tests.io.curator.test_apportion_curated_values import _patch, _state
from tests.io.harmonizer.test_parcel_pieces import _row_of_four, _split

LOT_VALUES = pd.DataFrame(
    {
        'parcel_id': ['P0', 'P1', 'P2', 'P3'],
        'improvement_value': [110000.0, 120000.0, 130000.0, 140000.0],
        'land_value': [21000.0, 22000.0, 23000.0, 24000.0],
    }
)
COLUMNS = {
    'improvement_value': 'improvement_value_parcel',
    'land_value': 'land_value_parcel',
}


def _curated(spine):
    return pd.DataFrame(
        {
            'footprint_id': spine['footprint_id'].to_numpy(),
            'priority_on_parcel': 'primary',
            'n_dwellings_overture': 1.0,
            'geometry_source': spine['geometry_source'].to_numpy(),
        },
        index=pd.Index(spine.index, name='building_id'),
    )


def _apportion(monkeypatch, tmp_path, cut):
    spine, lots, points, values = _row_of_four()
    # No point refuses every piece: the outline stays whole.
    spine, flat, stats = _split(spine, lots, points if cut else [], values)
    assert stats['outlines_cut'] == (1 if cut else 0)
    _patch(monkeypatch, tmp_path, flat, LOT_VALUES)
    out = split_curated_values(
        _state(_curated(spine)),
        recipe_id='US_parcel-openplaces-2026',
        columns=COLUMNS,
    ).curated
    return out, flat


def test_each_piece_takes_its_own_parcels_values(monkeypatch, tmp_path):
    out, flat = _apportion(monkeypatch, tmp_path, cut=True)
    lot_of = flat[flat['link'].notna()].set_index('building_id')['parcel_id']
    assert len(out) == 4
    expected = LOT_VALUES.set_index('parcel_id')
    for building, parcel in lot_of.items():
        for ref_col, entity_col in COLUMNS.items():
            assert out.loc[building, entity_col] == pytest.approx(
                expected.loc[parcel, ref_col]
            )


def test_the_parcels_totals_reach_the_buildings_as_before_the_cut(
    monkeypatch, tmp_path
):
    uncut, _ = _apportion(monkeypatch, tmp_path, cut=False)
    cut, _ = _apportion(monkeypatch, tmp_path, cut=True)
    assert len(uncut) == 1
    for ref_col, entity_col in COLUMNS.items():
        total = LOT_VALUES[ref_col].sum()
        assert uncut[entity_col].sum() == pytest.approx(total)
        assert cut[entity_col].sum() == pytest.approx(total)
