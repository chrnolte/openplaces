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
import pytest

from openplaces.core.schema import AdminId, Entity
from openplaces.io.ingester.table_ids import partition_in_range
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


def test_orange_county_sales_ids_drop_the_default_switch():
    """Orange County FL writes its 15-digit DOR parcel id in two orders.

    The parcel layer writes section, township, range (`012433...`) and
    the default conversion switches the first and third groups, so the
    parcel spine's key reads `33|24|1|4`. The DOR sales file writes the
    same id range-first (`332401...`), so the same conversion put the
    sales key at `1|24|33|4` and 12.6% of Orange's sales found their
    parcel. The sales side needs the same split without the switch:
    88.2% then match (2026-09-12).
    """
    from openplaces.geo.ids import convert_parcel_id

    ti = _ingester('US-FL', Entity('transaction', 'fldor', '2026'))
    overrides = ti._load_parcel_id_overrides('parcel')
    assert overrides['US-FL-OR'] == {
        'pattern': 'Dx',
        'conv': 'string_lengths: 2 2 2 4 2 3 & skip_empty: 1',
    }
    parcel_side = convert_parcel_id(
        pd.Series(['012027000000005']),
        pattern='Dx',
        conv_code='string_lengths: 2 2 2 4 2 3 & switch: 0|2 & skip_empty: 1',
    )
    sales_side = convert_parcel_id(
        pd.Series(['272001000000005']),
        pattern=overrides['US-FL-OR']['pattern'],
        conv_code=overrides['US-FL-OR']['conv'],
    )
    assert parcel_side.iloc[0] == sales_side.iloc[0] == '27|20|1|5'


def test_partition_ranges():
    assert partition_in_range('2010', '')
    assert partition_in_range(None, '')
    assert not partition_in_range(None, '2024-')
    assert partition_in_range('2024', '2024-') and partition_in_range(2031, '2024-')
    assert not partition_in_range('2023', '2024-')
    assert partition_in_range('2023', '-2023') and not partition_in_range(2024, '-2023')
    assert partition_in_range('2020', '2019-2021')
    assert partition_in_range('2025', '2025') and not partition_in_range('2026', '2025')
    assert not partition_in_range('tile-a', '2024-')


def test_orange_county_roll_years_from_2024_take_the_parcel_layers_order():
    """Orange's DOR files switched to section-first ids with the 2024 roll.

    Through 2023 the roll and sales files write range-first and need
    the split without the switch; from 2024 they write the id as the
    parcel layer does and need the switch. Measured 2026-10-22 on
    US-FL-OR: 0.890 of the 2025 roll's ids are floridagio's as written,
    0.031 of the 2023 roll's. Each roll year then reaches the same key.
    Measured 2026-10-22 on US-FL-OR: 2025 keys reach 0.986 of floridagio's, was 0.033.
    """
    raw = {'2023': '272001000000005', '2025': '012027000000005'}
    keys = {}
    for entity in (
        Entity('property', 'fldor', '2026'),
        Entity('transaction', 'fldor', '2026'),
    ):
        for year, parcel_id in raw.items():
            ti = TableIngester(
                {
                    'admin_id': AdminId('US-FL'),
                    'entity': entity,
                    'parcel_id_local': {
                        'source': 'parcel_id_assessor',
                        'kind': 'parcel',
                    },
                },
                {'partition_id_to_download': year},
                {'admin_id_to_process': 'US-FL-OR'},
                Path('.'),
                None,
            )
            frame = pd.DataFrame({'parcel_id_assessor': [parcel_id]})
            keys[(str(entity.entity_type), year)] = ti._add_parcel_id_local(frame)[
                'parcel_id_local'
            ].iloc[0]
    assert set(keys.values()) == {'27|20|1|5'}


# One fabricated parcel per county, written as each era of the DOR's
# files writes it, and the roll years those eras cover. Every spelling
# must mint the key of the newest era, which is the one the parcel layer
# is keyed on. Measured 2026-10-24 on US-FL: for each of these counties
# the old era's real ids, read by the ranged row, gave the new era's
# key on every account present in both years (82,603 to 426,014 each).
ERA_SPELLINGS = {
    'US-FL-AL': {'2014': '01234 056 078', '2015': '01234-056-078'},
    'US-FL-BK': {
        '2013': '123A04000500060007',
        '2016': '12-3A-04-0005-0006-0007',
        '2019': '123A04000500060007',
    },
    'US-FL-OS': {'2011': '01-23-45-0006-0070-0800', '2012': '012345000600700800'},
    'US-FL-PI': {'2007': '012345000060070008', '2008': '01 23 45 00006 007 0008'},
    'US-FL-PU': {'2008': '012345000600700800', '2009': '01-23-45-0006-0070-0800'},
    'US-FL-SA': {'2012': '0123-45-0678', '2013': '0123450678'},
    'US-FL-SJ': {'2006': '012345 0678', '2007': '0123450678'},
}


@pytest.mark.parametrize('admin_id', sorted(ERA_SPELLINGS))
def test_each_era_of_a_respelled_county_mints_todays_key(admin_id):
    spellings = ERA_SPELLINGS[admin_id]
    keys = {}
    for entity in (
        Entity('property', 'fldor', '2026'),
        Entity('transaction', 'fldor', '2026'),
    ):
        for year, parcel_id in spellings.items():
            ti = TableIngester(
                {
                    'admin_id': AdminId('US-FL'),
                    'entity': entity,
                    'parcel_id_local': {
                        'source': 'parcel_id_assessor',
                        'kind': 'parcel',
                    },
                },
                {'partition_id_to_download': year},
                {'admin_id_to_process': admin_id},
                Path('.'),
                None,
            )
            frame = pd.DataFrame({'parcel_id_assessor': [parcel_id]})
            keys[(str(entity.entity_type), year)] = ti._add_parcel_id_local(frame)[
                'parcel_id_local'
            ].iloc[0]
    assert len(set(keys.values())) == 1, keys
    assert pd.notna(next(iter(keys.values())))


def test_strip_separators_reads_an_old_punctuated_spelling():
    from openplaces.geo.ids import convert_parcel_id

    bare = convert_parcel_id(
        pd.Series(['012345000600700800']),
        pattern='Sx',
        conv_code='string_lengths: 1 2 2 2 3 4 4 & skip_empty: 1',
    )
    punctuated = convert_parcel_id(
        pd.Series(['01-23-45-0006-0070-0800']),
        pattern='Sx',
        conv_code='strip_separators: 1 & string_lengths: 1 2 2 2 3 4 4 & skip_empty: 1',
    )
    assert bare.iloc[0] == punctuated.iloc[0]
