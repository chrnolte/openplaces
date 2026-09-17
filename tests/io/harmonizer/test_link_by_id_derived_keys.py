"""`link_by_id` derived keys: address keys on either side, and ids kept.

A permit export is no entity of any spine, so no harmonized table ever
keys its addresses; `ref_address_key` derives the key on the reference
with the spine's own normalization, and `spine_address_key` derives a
join-only key on the spine without persisting it. Separately, an
auto-discovered link must never copy the matched source's parcel key
over the spine's: Pender County NC lost its key on 99.8% of parcels to a
statewide layer's placeholder that way. All values are fabricated.
"""

import pandas as pd

import openplaces.io.harmonizer.links as links
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import HarmonizeState


def _state(spine):
    return HarmonizeState(
        recipe={},
        admin_id=AdminId('US-XX-AAA'),
        verbose=False,
        timer=None,
        spine=spine,
    )


def _patch_reference(monkeypatch, ref):
    monkeypatch.setattr(links, 'get_entities', lambda *a, **k: ref.copy())
    monkeypatch.setattr(links, 'restrict_to_admin_by_name', lambda df, *a: df)


def _spine():
    return pd.DataFrame(
        {
            'address_street': ['Sample Street', 'Example Avenue', 'Sample Street'],
            'address_number': ['10', '22', '99'],
            'admin4_id': ['US-XX-AAA-T1', 'US-XX-AAA-T2', 'US-XX-AAA-T1'],
        },
        index=pd.Index(['f1', 'f2', 'f3'], name='footprint_id'),
    )


def _permits():
    return pd.DataFrame(
        {
            'street': ['SAMPLE ST', 'SAMPLE ST', 'EXAMPLE AVE', 'OTHER RD'],
            'street_no': ['10', '10', '22', '5'],
            'occupancy_type_raw': ['Class A', 'Class A', 'Class B', 'Class C'],
        },
        index=pd.Index(['p1', 'p2', 'p3', 'p4'], name='permit_id'),
    )


def test_address_keys_derived_on_both_sides_count_and_label(monkeypatch):
    _patch_reference(monkeypatch, _permits())
    key_spec = {
        'admin4_column': None,
        'city_column': None,
        'output_column': 'address_key_county',
        'output_column_city': None,
    }

    state = links.link_by_id(
        _state(_spine()),
        recipe_id='XX_property-permits-2026',
        mode='aggregate',
        spine_key='address_key_county',
        ref_key='address_key_county',
        columns={'occupancy_type_raw': 'occupancy_type_permit'},
        count_as='n_permits_per_footprint_address',
        ref_address_key={
            **key_spec,
            'street_column': 'street',
            'number_column': 'street_no',
        },
        spine_address_key=key_spec,
    )

    spine = state.spine
    assert spine['n_permits_per_footprint_address'].tolist() == [2, 1, 0]
    assert spine['occupancy_type_permit'].tolist()[:2] == ['Class A', 'Class B']
    assert pd.isna(spine['occupancy_type_permit'].iloc[2])
    assert 'address_key_county' not in spine.columns


def test_auto_discovered_link_does_not_copy_the_parcel_key(monkeypatch):
    spine = pd.DataFrame(
        {
            'parcel_id_local': [None, 'K2'],
            'parcel_id_assessor': ['K1', 'K2'],
            'land_value': [None, None],
        },
        index=pd.Index(['a', 'b'], name='parcel_id'),
    )
    # A statewide layer matched on another key whose own
    # standardized key is a placeholder on every row.
    ref = pd.DataFrame(
        {
            'parcel_id_local': ['0', '0'],
            'parcel_id_assessor': ['K1', 'K2'],
            'land_value': [5.0, 7.0],
        }
    )
    _patch_reference(monkeypatch, ref)
    monkeypatch.setattr(
        links,
        '_discover_link_sources',
        lambda state, entity_type: [
            {
                'recipe_id': 'XX_parcel-statewide-2026',
                'layer': None,
                'key': 'parcel_id_local',
                'aggregation_function': None,
                'supplements': None,
                'supplements_key': None,
            }
        ],
    )
    monkeypatch.setattr(links, '_apply_remap_csvs', lambda state, recipe_id: state)

    state = links.link_by_id(
        _state(spine),
        auto_discover=True,
        entity_type='parcel',
        spine_key='parcel_id_assessor',
        ref_key='parcel_id_assessor',
        fill_only=True,
    )

    # The spine's own missing key stays missing rather than taking '0'.
    assert pd.isna(state.spine['parcel_id_local'].iloc[0])
    assert state.spine['parcel_id_local'].iloc[1] == 'K2'
    assert state.spine['land_value'].tolist() == [5.0, 7.0]
