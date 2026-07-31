"""`_load_parcel_id_overrides` reads the recipe-external, git-tracked
``{country}_{entity_type}_id-overrides.csv`` tables so a per-county
`parcel_id_local` conversion can be shared across every recipe that ingests
that county's data, instead of being duplicated (or hardcoded via an inline
`instruction:` block) in each individual recipe. These tests exercise the
real bundled WI rows added to
``recipes/US/_all/transaction/_all/US_transaction_id-overrides.csv`` and
``recipes/US/_all/parcel/_all/US_parcel_id-overrides.csv``, and the
per-row ``admin_id_column`` path that lets one multi-county ingest chunk
(the `widor` transaction recipe processes the whole state at once) apply
the right county's conversion to each row.
"""

from pathlib import Path

import pandas as pd

from openplaces.core.schema import AdminId, Entity
from openplaces.io.ingester.table_ingester import TableIngester


def _ingester(admin_id, entity):
    return TableIngester(
        table_recipe={'admin_id': AdminId(admin_id), 'entity': entity},
        download_partition={},
        processing_chunk={'admin_id_to_process': admin_id},
        recipe_heap_dir=Path('.'),
        timer=None,
    )


def test_transaction_overrides_table_has_wi_tax_rows():
    ti = _ingester('US-WI', Entity('transaction', 'widor', '2026'))
    overrides = ti._load_parcel_id_overrides('tax')

    assert overrides['US-WI-DA'] == {
        'pattern': 'Sx/Dx',
        'conv': 'drop_cols: 0 & skip_empty: 1',
    }
    assert overrides['US-WI-ON']['pattern'] == 'Sx-[S.]x(-Sx)(-Sx)(-Sx)'
    assert overrides['US-WI-VI']['pattern'] == 'Sx-[S.]x(-Sx)(-Sx)(-Sx)'
    # Vilas standardizes away inconsistently zero-padded municipal prefixes
    # (e.g. '010-1044' and '10-1044' are the same parcel filed two ways
    # across repeat sales) at a rate above the 0.5% default guard tolerance
    # -- verified against real cached WIDOR/wiscedu data -- so it carries an
    # explicit, wider tolerance.
    assert overrides['US-WI-VI']['tolerance'] == '0.03'


def test_parcel_overrides_table_has_wi_on_vi_rows():
    ti = _ingester('US-WI', Entity('parcel', 'wiscedu', 'v11'))
    overrides = ti._load_parcel_id_overrides('parcel')

    assert overrides['US-WI-ON']['pattern'] == 'Sx-[S.]x(-Sx)(-Sx)(-Sx)'
    assert overrides['US-WI-VI']['pattern'] == 'Sx-[S.]x(-Sx)(-Sx)(-Sx)'
    assert overrides['US-WI-VI']['tolerance'] == '0.03'
    # Dane's parcel side is intentionally left on the bundled default.
    assert 'US-WI-DA' not in overrides


def test_multi_county_transaction_chunk_applies_each_countys_pattern():
    # widor ingests the whole state as one chunk; admin_id_column is what
    # lets a single DataFrame apply Dane/Oneida/Vilas's different tax-PIN
    # conventions row by row.
    recipe = {
        'admin_id': AdminId('US-WI'),
        'entity': Entity('transaction', 'widor', '2026'),
        'parcel_id_local': {
            'source': 'parcel_id_assessor',
            'kind': 'tax',
            'admin_id_column': 'admin3_id',
        },
    }
    ti = TableIngester(recipe, {}, {'admin_id_to_process': 'US-WI'}, Path('.'), None)
    df = pd.DataFrame(
        {
            'parcel_id_assessor': [
                '282/081104231077',  # Dane
                '\tSC-612-10',  # Oneida
                '\t024-2503',  # Vilas
            ],
            'admin3_id': ['US-WI-DA', 'US-WI-ON', 'US-WI-VI'],
        }
    )

    result = ti._add_parcel_id_local(df)

    assert result['parcel_id_local'].tolist() == [
        '81104231077',
        'SC|612|10',
        '24|2503',
    ]


def test_tax_and_parcel_side_conversions_agree_for_oneida_and_vilas():
    # The whole point of A1/A2: the same raw PIN structure on the parcel
    # side and the transaction side must standardize to the same key.
    tax_ti = TableIngester(
        {
            'admin_id': AdminId('US-WI'),
            'entity': Entity('transaction', 'widor', '2026'),
            'parcel_id_local': {'source': 'parcel_id_assessor', 'kind': 'tax'},
        },
        {},
        {'admin_id_to_process': 'US-WI-ON'},
        Path('.'),
        None,
    )
    parcel_ti = TableIngester(
        {
            'admin_id': AdminId('US-WI'),
            'entity': Entity('parcel', 'wiscedu', 'v11'),
            'parcel_id_local': {'source': 'parcel_id_assessor', 'kind': 'parcel'},
        },
        {},
        {'admin_id_to_process': 'US-WI-ON'},
        Path('.'),
        None,
    )

    tax_key = tax_ti._add_parcel_id_local(
        pd.DataFrame({'parcel_id_assessor': ['\tSC-612-10']})
    )['parcel_id_local'].iloc[0]
    parcel_key = parcel_ti._add_parcel_id_local(
        pd.DataFrame({'parcel_id_assessor': ['SC-612-10']})
    )['parcel_id_local'].iloc[0]

    assert tax_key == parcel_key == 'SC|612|10'
