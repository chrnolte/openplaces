"""Id-based n:m link tables between entities (io.harmonizer.entity_links).

Every id and value below is fabricated.
"""

import pandas as pd

from openplaces.io.harmonizer.entity_links import (
    LINK_COLUMNS_TAIL,
    build_id_links,
)


def _parcels():
    # Lot B is drawn as two parcel rows carrying one key.
    return pd.DataFrame(
        {'parcel_id_local': ['A', 'B', 'B', 'C', None]},
        index=pd.Index(['pa', 'pb1', 'pb2', 'pc', 'pd'], name='parcel_id'),
    )


def _properties():
    return pd.DataFrame(
        {
            'parcel_id_local': ['A', 'A', 'B', 'Z', None, '000'],
            'source': ['roll'] * 6,
            'owner': ['not', 'a', 'column', 'that', 'may', 'leave'],
        },
        index=pd.RangeIndex(6),
    )


def _links(properties=None, parcels=None):
    return build_id_links(
        _properties() if properties is None else properties,
        _parcels() if parcels is None else parcels,
        'parcel_id_local',
        'parcel_id_local',
        'property_id',
        'parcel_id',
        'parcel_id_local',
    )


def test_a_link_table_holds_two_ids_and_three_labels_and_nothing_else():
    links = _links()
    assert list(links.columns) == ['property_id', 'parcel_id', *LINK_COLUMNS_TAIL]
    assert LINK_COLUMNS_TAIL == ('link_method', 'link_source', 'share')


def test_several_properties_on_one_parcel_each_get_a_row():
    links = _links()
    on_a = links[links['parcel_id'] == 'pa']
    assert on_a['property_id'].tolist() == [0, 1]


def test_a_property_whose_key_sits_on_two_parcels_gets_two_rows():
    links = _links()
    of_two = links[links['property_id'] == 2]
    assert of_two['parcel_id'].tolist() == ['pb1', 'pb2']


def test_unmatched_missing_and_placeholder_keys_get_no_row():
    links = _links()
    # 'Z' names no parcel, one key is missing, '000' is a placeholder.
    assert set(links['property_id']) == {0, 1, 2}
    assert 'pc' not in set(links['parcel_id'])
    assert 'pd' not in set(links['parcel_id'])


def test_labels_are_fixed_and_share_is_empty():
    links = _links()
    assert links['link_method'].unique().tolist() == ['parcel_id_local']
    assert links['link_source'].unique().tolist() == ['roll']
    assert links['share'].isna().all()


def test_a_key_on_very_many_rows_and_several_parcels_is_kept_and_labeled():
    # 1,200 rows of one source share lot B's key, which two parcel rows
    # carry: many to many, the case a sum must not touch. The pairs are
    # kept under their own method. The same value on another source's
    # single row is an ordinary identifier.
    stack = pd.DataFrame({'parcel_id_local': ['B'] * 1200, 'source': 'stack'})
    roll = pd.DataFrame({'parcel_id_local': ['B'], 'source': 'roll'})
    properties = pd.concat([stack, roll], ignore_index=True)
    links = _links(properties=properties)
    methods = links.groupby('link_source')['link_method'].unique()
    assert methods['stack'].tolist() == ['parcel_id_local_shared_key']
    assert methods['roll'].tolist() == ['parcel_id_local']
    assert len(links) == 2402


def test_a_large_stack_on_exactly_one_parcel_is_an_ordinary_link():
    stack = pd.DataFrame({'parcel_id_local': ['A'] * 1200, 'source': 'stack'})
    links = _links(properties=stack)
    assert links['link_method'].unique().tolist() == ['parcel_id_local']
    assert len(links) == 1200


def test_a_key_on_very_many_parcel_rows_is_labeled_too():
    parcels = pd.DataFrame(
        {'parcel_id_local': ['Q'] * 150 + ['A']},
        index=pd.Index([f'p{i}' for i in range(151)], name='parcel_id'),
    )
    properties = pd.DataFrame({'parcel_id_local': ['Q', 'A'], 'source': 'roll'})
    links = _links(properties=properties, parcels=parcels)
    by_property = links.groupby('property_id')['link_method'].unique()
    assert by_property[0].tolist() == ['parcel_id_local_shared_key']
    assert by_property[1].tolist() == ['parcel_id_local']


def test_a_missing_key_column_gives_an_empty_table_with_the_columns():
    links = _links(properties=pd.DataFrame({'other': [1]}))
    assert links.empty
    assert list(links.columns) == ['property_id', 'parcel_id', *LINK_COLUMNS_TAIL]


def test_a_table_without_a_source_column_still_links():
    links = _links(properties=_properties().drop(columns='source'))
    assert len(links) == 4
    assert links['link_source'].isna().all()
