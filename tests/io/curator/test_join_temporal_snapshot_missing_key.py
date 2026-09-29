"""A source without the join key is "no panel here", not a crash.

Massachusetts registry deeds carry no parcel_id_assessor, and the
national transaction recipe runs join_temporal_snapshot everywhere.
Until 2026-09-29 the step raised KeyError on the column selection before
its panel check, and Somerville's transaction curate died. Fabricated
rows only.
"""

from __future__ import annotations

import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io.curator import CurateState
from openplaces.io.curator.transactions import join_temporal_snapshot


def _state(df: pd.DataFrame) -> CurateState:
    return CurateState(
        recipe={},
        entity_recipe={},
        admin_id=AdminId('US-MA-SOM'),
        verbose=False,
        timer=None,
        curated=df,
    )


def test_a_missing_join_key_reads_as_no_panel_under_require_panel():
    tx = pd.DataFrame({'book': ['1', '2'], 'sale_year': [2020, 2021]})
    out = join_temporal_snapshot(
        _state(tx),
        recipe_id='US-FL_property-fldor-2026',
        require_panel=True,
        join_key='parcel_id_assessor',
        date_column='sale_year',
        vintage_column='tax_year',
        direction='backward',
        match_type_column='property_match_type',
        columns=['tax_year', 'year_built'],
    ).curated
    assert out['property_match_type'].tolist() == ['no_panel', 'no_panel']
    assert 'tax_year' not in out.columns


def test_a_missing_join_key_is_a_recipe_error_without_require_panel():
    tx = pd.DataFrame({'book': ['1'], 'sale_year': [2020]})
    with pytest.raises(KeyError, match='parcel_id_assessor'):
        join_temporal_snapshot(
            _state(tx),
            recipe_id='US-FL_property-fldor-2026',
            join_key='parcel_id_assessor',
            date_column='sale_year',
            vintage_column='tax_year',
            direction='backward',
            match_type_column='property_match_type',
            columns=['tax_year'],
        )
