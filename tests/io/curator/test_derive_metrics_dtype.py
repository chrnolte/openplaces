"""derive_metrics must survive a value column an ingest left str-typed.

Seen in production: the TX txgio parcel ingest ships improvement_value as
text ('0'), and the _per_area ratio crashed the whole county's curate with
a str/float TypeError. The registry declares the column float, so the
ingest recipe owes the cast -- but curate coerces defensively and warns
rather than failing the county.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces.io.curator import CurateState
from openplaces.io.curator.inferers import derive_metrics


def _state(improvement_value):
    curated = gpd.GeoDataFrame(
        {
            'improvement_value': improvement_value,
            'area_ha': [1.0, 2.0],
            'geometry': [box(0, 0, 1, 1), box(1, 0, 2, 1)],
        },
        crs='epsg:4326',
    )
    return CurateState(
        recipe={'entity': {'entity_type': 'parcel'}},
        entity_recipe={},
        admin_id=None,
        verbose=False,
        timer=None,
        curated=curated,
    )


def test_str_typed_value_column_is_coerced_with_warning():
    state = _state(pd.array(['100', '0'], dtype='str'))
    with pytest.warns(UserWarning, match='not numeric'):
        state = derive_metrics(state)
    ratios = state.curated['improvement_value_per_area']
    assert ratios.tolist() == [100.0, 0.0]


def test_numeric_value_column_stays_silent():
    import warnings

    state = _state([100.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        state = derive_metrics(state)
    assert state.curated['improvement_value_per_area'].tolist() == [100.0, 0.0]
