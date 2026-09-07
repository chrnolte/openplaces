"""The legacy-column upgrade applied by partition roll-ups.

`_legacy_upgrader` exists to bring a file written before a recipe rename up
to the current schema. A partition written after the rename is already
current, so the recipe's transformations must not run over it again.
"""

import pandas as pd

from openplaces.io.aggregate import _legacy_upgrader

RECIPE = {
    'legacy_columns': {'sale_amount': 'sale_price'},
    'transformations': [
        {
            'type': 'string',
            'operation': 'add_prefix',
            'input': 'deed_id',
            'output': 'deed_id',
            'args': {'prefix': 'P-'},
        }
    ],
}


def test_a_current_partition_is_not_transformed_again():
    upgrade = _legacy_upgrader(RECIPE)
    current = pd.DataFrame({'sale_price': [100], 'deed_id': ['P-1']})

    result = upgrade(current)

    assert list(result['deed_id']) == ['P-1']


def test_a_legacy_file_is_renamed_and_transformed_once():
    upgrade = _legacy_upgrader(RECIPE)
    legacy = pd.DataFrame({'sale_amount': [100], 'deed_id': ['1']})

    result = upgrade(legacy)

    assert 'sale_amount' not in result.columns
    assert list(result['sale_price']) == [100]
    assert list(result['deed_id']) == ['P-1']
