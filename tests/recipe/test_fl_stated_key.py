"""Florida keys deed counts and the sale-time join on STATE_PAR_ID first.

A county that renumbered its parcels lists one sale in two years' files
under two county numbers but one DOR STATE_PAR_ID (`parcel_id_admin2`).
Values are fabricated.
"""

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.io.sale_records import count_parcels_per_document
from openplaces.recipe import apply_recipe_patches, get_recipe_by_id


def _step(recipe_id, admin, name):
    recipe, _ = apply_recipe_patches(get_recipe_by_id(recipe_id), AdminId(admin))
    return [s for s in recipe['pipeline'] if s['step'] == name]


def test_florida_counts_parcels_by_state_parcel_id_first():
    (step,) = _step(
        'US_transaction-spine-2026', 'US-FL-DU', 'count_parcels_per_document'
    )
    assert step['parcel_column'] == [
        'parcel_id_admin2',
        'parcel_id_local',
        'parcel_id_assessor',
    ]


def test_other_states_keep_the_national_count():
    (step,) = _step(
        'US_transaction-spine-2026', 'US-WI-DA', 'count_parcels_per_document'
    )
    assert step['parcel_column'] == ['parcel_id_local', 'parcel_id_assessor']


def test_florida_join_tries_state_parcel_id_then_county_key():
    steps = _step(
        'US_transaction-openplaces-2026', 'US-FL-DU', 'join_temporal_snapshot'
    )
    assert len(steps) == 2
    assert all(s['join_key'] == ['parcel_id_admin2', 'parcel_id_local'] for s in steps)
    assert all(s['normalize_key'] is False for s in steps)


def test_one_parcel_under_two_county_numbers_counts_once():
    frame = pd.DataFrame(
        {
            'sale_document_id': ['D1', 'D1', 'D2', 'D2'],
            'parcel_id_admin2': ['A01-001', 'A01-001', 'A01-002', 'A01-003'],
            'parcel_id_local': ['OLD9', 'NEW7', 'NEW8', 'NEW9'],
            'parcel_id_assessor': ['OLD-9', 'NEW-7', 'NEW-8', 'NEW-9'],
        }
    )
    out = count_parcels_per_document(
        frame, ['parcel_id_admin2', 'parcel_id_local', 'parcel_id_assessor']
    )
    assert out['n_parcels_per_sale'].tolist() == [1.0, 1.0, 2.0, 2.0]
