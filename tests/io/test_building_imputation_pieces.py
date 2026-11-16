"""Building imputation over buildings cut at parcel lines.

plans/buildings-from-parcel-pieces.md, step 3. The imputation runs in
the CHEER footprint delivery (US_footprint-cheer-2026), whose rows are
outlines with their pieces folded back; a building-level delivery would
hand it the pieces themselves. Both shapes of row go through the
per-row parts every model reads: the inputs, the value labels, the
rows a target applies to and the lot-value hold. Every id and value is
fabricated (docs/5_contribute/no-personal-data.rst).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from openplaces.io import building_imputation as bi


def _frame(rows):
    """Rows with every input column, missing unless *rows* states it."""
    base = {
        'area_m2': 100.0,
        'x': 0.0,
        'y': 0.0,
        'land_value_2010': 9.0,
        'distance_to_core_km': 5.0,
        'region_type': 'metro',
        'occupancy_type': 'Single-Family',
        'geometry_source': 'obm',
        'structure_value': np.nan,
        'structure_value_source': None,
        'has_homestead_exemption_parcel': np.nan,
    }
    columns = dict.fromkeys(bi.EVIDENCE_COLUMNS, np.nan) | base
    frame = pd.DataFrame([columns | row for row in rows.values()], index=list(rows))
    frame['geometry_source'] = frame['geometry_source'].astype(object)
    return frame


def _pieces():
    """A row of four cut on lots P0 to P3; P3's roll states no value."""
    return _frame(
        {
            f'c{i}': {
                'parcel_id': f'P{i}',
                'structure_value': value,
                'structure_value_source': None if np.isnan(value) else 'parcel',
                'n_parcels_per_footprint': 1.0,
                'n_footprints_per_parcel': 1.0,
            }
            for i, value in enumerate([110000.0, 110000.0, 120000.0, np.nan])
        }
    )


def test_piece_rows_build_inputs_labels_and_targets():
    frame = _pieces()
    features = bi.build_features(frame)
    assert len(features) == 4
    assert features['geometry_kind'].tolist() == ['obm'] * 4
    labels = bi.value_labels(frame)
    assert labels.tolist()[:3] == [110000.0, 110000.0, 120000.0]
    assert np.isnan(labels['c3'])
    spec = next(s for s in bi.TARGETS if s.column == 'structure_value')
    assert bi.applies_to(spec, frame).all()


def test_a_piece_on_an_unvalued_lot_is_filled_not_held():
    """Its lot's value sits on no other building: the hold, meant for a
    lot whose value another building on it already carries, does not
    apply across the pieces' different lots."""
    held = bi.lot_value_elsewhere(_pieces())
    assert not held.any()


def test_a_folded_outline_row_reads_as_one_building():
    """The footprint delivery's row for the cut outline: values summed,
    lots joined, n_buildings_per_footprint 4."""
    frame = _frame(
        {
            'f1': {
                'area_m2': 400.0,
                'parcel_id': 'P0 | P1 | P2 | P3',
                'structure_value': 340000.0,
                'structure_value_source': 'parcel',
                'n_buildings_per_footprint': 4.0,
                'n_parcels_per_footprint': 1.0,
            },
            'f2': {'parcel_id': 'P4'},
        }
    )
    features = bi.build_features(frame)
    assert features.loc['f1', 'log_area_m2'] == pytest.approx(np.log1p(400.0))
    assert bi.value_labels(frame)['f1'] == 340000.0
    assert not bi.lot_value_elsewhere(frame).any()
