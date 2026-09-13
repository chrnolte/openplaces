"""Aggregate mode must keep an all-missing group unknown, not zero.

`link_by_id` in aggregate mode sums a column over the reference rows
that share a key. pandas' plain group sum returns 0 for a group whose
values are all missing, so a property whose every bath row failed to
parse read as having zero baths. The sum now uses min_count=1, the same
rule `transform._aggregate_cols` applies row-wise. Values are fabricated.
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer.links import link_by_id


class _State:
    def __init__(self, spine):
        self.spine = spine
        self.admin_id = AdminId('US-TX-LAV')
        self.recipe = {
            'recipe_id': 'US_property-spine-2026',
            'admin_id': AdminId('US'),
        }
        self.references = {}
        self.metadata = {}
        self.verbose = False


def test_an_all_missing_group_sums_to_missing(monkeypatch):
    spine = gpd.GeoDataFrame(
        {
            'parcel_id_local': ['a', 'b'],
            'n_bathrooms': [None, None],
            'geometry': [Point(0, 0), Point(1, 1)],
        },
        index=pd.Index(['s1', 's2'], name='property_id'),
        crs='epsg:4326',
    )
    reference = pd.DataFrame(
        {
            'parcel_id_local': ['a', 'a', 'b', 'b'],
            'n_bathrooms': [None, None, 1.0, 1.5],
        },
        index=pd.Index(['r1', 'r2', 'r3', 'r4'], name='property_id'),
    )
    monkeypatch.setattr(
        'openplaces.io.harmonizer.links.get_entities', lambda *a, **k: reference
    )
    out = link_by_id(
        _State(spine),
        recipe_id='US-TX-LAV_property-lavacacad-2026_bathrooms',
        mode='aggregate',
        spine_key='parcel_id_local',
        ref_key='parcel_id_local',
        columns=['n_bathrooms'],
        aggregation_function={'n_bathrooms': 'sum'},
    ).spine['n_bathrooms']

    assert pd.isna(out['s1'])
    assert out['s2'] == 2.5
