"""impute_n_dwellings must pick its input by vocabulary, not column order.

Several occupancy-shaped columns coexist on the footprint spine and only
the NSI strings key the units lookup, so a first-prefix-match scan could
silently impute nothing depending on incidental DataFrame order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.curator import CurateState
from openplaces.io.curator.imputers import impute_n_dwellings


def _state(curated, admin_id=None):
    return CurateState(
        recipe={'entity': {'entity_type': 'footprint'}},
        entity_recipe={},
        admin_id=admin_id,
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_dwelling_class_column_is_chosen_by_vocabulary_not_column_order():
    """The NSI column must win however the frame happens to be ordered.

    Only the NSI occupancy strings key the units lookup. With a
    first-prefix-match scan, a frame whose Overture-derived class or raw
    subgroup code came first imputed nothing at all.
    """
    columns = {
        'occupancy_type_dwelling_overture': ['Single-Family', 'Single-Family'],
        'use_subgroup_code': ['R1', 'R2'],
        'occupancy_type_building_nsi': [
            'Multi-Family, 3-4 units',
            'Single Family',
        ],
        'n_dwellings': [np.nan, np.nan],
    }
    for order in ([*columns], ['occupancy_type_building_nsi', *columns][:-1]):
        curated = pd.DataFrame({k: columns[k] for k in dict.fromkeys(order)})
        result = impute_n_dwellings(_state(curated)).curated
        assert result['n_dwellings'].tolist() == [3.5, 1.0], order


def test_a_frame_without_any_class_column_imputes_nothing():
    curated = pd.DataFrame({'n_dwellings': [np.nan, 2.0]})
    result = impute_n_dwellings(_state(curated)).curated
    assert result['n_dwellings'].isna().tolist() == [True, False]


def test_an_explicit_column_overrides_the_default_order():
    curated = pd.DataFrame(
        {
            'occupancy_type_building_nsi': ['Single Family', 'Single Family'],
            'occupancy_type': ['Multi-Family, 2 units', 'Manufactured Home'],
            'n_dwellings': [np.nan, np.nan],
        }
    )
    result = impute_n_dwellings(_state(curated), column='occupancy_type').curated
    assert result['n_dwellings'].tolist() == [2.0, 1.0]


def test_imputed_dwellings_keep_the_provenance_marker():
    """The provenance invariant: a cell openplaces filled must say so."""
    from openplaces.io.curator.provenance import is_imputed, source_column

    curated = pd.DataFrame(
        {
            'occupancy_type_building_nsi': ['Single Family', 'Single Family'],
            'n_dwellings': [np.nan, 4.0],
        }
    )
    result = impute_n_dwellings(_state(curated)).curated
    assert is_imputed(result[source_column('n_dwellings')]).tolist() == [True, False]
