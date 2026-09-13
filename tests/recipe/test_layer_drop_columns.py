"""`drop_columns` belongs to the table that names a scratch column.

An additional layer's own `drop_columns` used to be ignored: the layer
inherited the host's list and a scratch column the layer's own
transformation produced reached its output, while a host-level list
dropped columns from every layer. It is a per-table key now, like
`columns`, `query` and `transformations`.
"""

from openplaces.recipe import build_table_recipe


def _host(**extra):
    host = {
        'admin_id': 'US-XX-YY',
        'entity': {'entity_type': 'parcel', 'source': {'source_id': 's'}},
        'columns': {'parcel_id_assessor': 'PIN'},
        'save_to': {'data_dir': 'core'},
    }
    host.update(extra)
    return host


def _layer(**extra):
    layer = {
        'entity': {'entity_type': 'property', 'source': {'source_id': 's'}},
        'columns': {'parcel_id_assessor': 'PIN', 'n_bathrooms': 'BATHS'},
    }
    layer.update(extra)
    return layer


def test_a_layer_keeps_its_own_drop_columns():
    table = build_table_recipe(_host(), _layer(drop_columns=['scratch']))
    assert table['drop_columns'] == ['scratch']


def test_a_layer_does_not_inherit_the_hosts_drop_columns():
    table = build_table_recipe(_host(drop_columns=['host_only']), _layer())
    assert 'drop_columns' not in table


def test_the_layers_list_wins_over_the_hosts():
    table = build_table_recipe(
        _host(drop_columns=['host_only']), _layer(drop_columns=['scratch'])
    )
    assert table['drop_columns'] == ['scratch']
