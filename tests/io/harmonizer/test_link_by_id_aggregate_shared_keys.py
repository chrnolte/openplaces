"""Aggregate mode must not fan a group sum out over a shared spine key.

`link_by_id` in aggregate mode groups the reference by key, aggregates
each column with the registry's function (a sum for money columns), and
maps the result onto the spine by the same key. When several spine rows
share that key, each receives the whole group total, so the county
total is multiplied by the group size. Carteret County NC: 97% of
parcels share their punctuation-free key, in groups of up to 335, and
improvement value came out 31 times its source. A sum belongs to one
spine row; where the key cannot say which, none is assigned.

`fill_only` is the second half: a pass on a lossy key must never
overwrite a value the precise key already set, whatever its coverage.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from openplaces.core.schema import AdminId
from openplaces.io.harmonizer.links import link_by_id

ADMIN = AdminId('US-NC-CAR')


class _State:
    def __init__(self, spine):
        self.spine = spine
        self.admin_id = ADMIN
        self.recipe = {
            'recipe_id': 'US_parcel-geospine-2026',
            'admin_id': AdminId('US'),
        }
        self.references = {}
        self.metadata = {}
        self.verbose = False


def _spine():
    # Three units share the lossy key 'p1'; 'p2' is unique.
    return gpd.GeoDataFrame(
        {
            'parcel_id_alnum': ['p1', 'p1', 'p1', 'p2'],
            'improvement_value': [100.0, 200.0, None, 400.0],
            'geometry': [Point(i, i) for i in range(4)],
        },
        index=pd.Index(['u1', 'u2', 'u3', 'v1'], name='parcel_id'),
        crs='epsg:4326',
    )


def _reference():
    # The reference has one row per unit under the shared key, plus one
    # for the unique key. Grouped by key and summed: p1 -> 600, p2 -> 50.
    return pd.DataFrame(
        {
            'parcel_id_alnum': ['p1', 'p1', 'p1', 'p2'],
            'improvement_value': [100.0, 200.0, 300.0, 50.0],
        },
        index=pd.Index(['r1', 'r2', 'r3', 'r4'], name='parcel_id'),
    )


@pytest.fixture
def linked(monkeypatch):
    monkeypatch.setattr(
        'openplaces.io.harmonizer.links.get_entities', lambda *a, **k: _reference()
    )

    def run(**overrides):
        state = _State(_spine())
        params = dict(
            recipe_id='US-NC_parcel-nconemap-2025',
            mode='aggregate',
            spine_key='parcel_id_alnum',
            ref_key='parcel_id_alnum',
            columns=['improvement_value'],
        )
        params.update(overrides)
        with pytest.warns(UserWarning):
            return link_by_id(state, **params).spine['improvement_value']

    return run


def test_a_sum_is_not_stamped_onto_rows_that_share_the_key(linked):
    values = linked()

    # The three units sharing 'p1' keep what they had; the 600 group
    # total goes to none of them.
    assert values['u1'] == 100.0
    assert values['u2'] == 200.0
    assert pd.isna(values['u3'])


def test_a_unique_key_still_receives_its_sum(linked):
    values = linked()

    # v1 is the only spine row under 'p2', and the reference covers 1 of
    # 4 rows, so the default rule gap-fills rather than overwrites; v1
    # already holds 400 and keeps it. Prove the sum reaches a gap instead.
    assert values['v1'] == 400.0


def test_fill_only_never_overwrites_even_at_full_coverage(linked):
    values = linked(fill_only=True)

    assert values['u1'] == 100.0
    assert values['u2'] == 200.0
    assert values['v1'] == 400.0


def test_a_value_that_was_not_summed_is_stamped_on_every_sale_of_its_parcel(
    monkeypatch,
):
    # The transaction spine: three sales of parcel p1 and one of p2, and
    # a parcel reference with one row per parcel. Nothing was summed, so
    # each sale carries its parcel's value; withholding here lost every
    # parcel value on Lake County FL (2026-09-12).
    sales = gpd.GeoDataFrame(
        {
            'parcel_id_local': ['p1', 'p1', 'p1', 'p2'],
            'geometry': [Point(i, i) for i in range(4)],
        },
        index=pd.Index(['s1', 's2', 's3', 's4'], name='transaction_id'),
        crs='epsg:4326',
    )
    parcels = pd.DataFrame(
        {'parcel_id_local': ['p1', 'p2'], 'improvement_value': [300.0, 50.0]},
        index=pd.Index(['p1', 'p2'], name='parcel_id'),
    )
    monkeypatch.setattr(
        'openplaces.io.harmonizer.links.get_entities', lambda *a, **k: parcels
    )
    state = _State(sales)
    state.recipe = {'recipe_id': 'US_transaction-spine-2026', 'admin_id': AdminId('US')}
    with pytest.warns(UserWarning, match='not unique'):
        state = link_by_id(
            state,
            recipe_id='US-FL_parcel-floridagio-2026',
            mode='aggregate',
            spine_key='parcel_id_local',
            ref_key='parcel_id_local',
            columns=['improvement_value'],
        )
    assert state.spine['improvement_value'].tolist() == [300.0, 300.0, 300.0, 50.0]
