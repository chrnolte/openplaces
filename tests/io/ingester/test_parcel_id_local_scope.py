"""`_add_parcel_id_local` must prefix with the ingest chunk's own admin id
whenever the recipe processes at a finer admin level than it saves at (e.g.
MassGIS: per-town `process_by`, per-county `save_to`) -- otherwise two
different chunks' identical raw ids (each valid, since `compute_parcel_id_local`'s
duplicate guard only checks uniqueness within one chunk) collide once the
chunks are merged for the coarser save level. Confirmed on real Middlesex
County data: 29% of parcels gained a "duplicate" this way before this fix.
"""

from pathlib import Path

import pandas as pd

from openplaces.core.schema import AdminId, Entity
from openplaces.io.ingester.table_ingester import TableIngester


def _ingester(recipe, admin_id_to_process):
    return TableIngester(
        table_recipe=recipe,
        download_partition={},
        processing_chunk={'admin_id_to_process': admin_id_to_process},
        recipe_heap_dir=Path('.'),
        timer=None,
    )


def _recipe(process_level, save_level):
    return {
        'admin_id': AdminId('US-MA'),
        'entity': Entity('parcel', 'massgis', '2025'),
        'process_by': {'admin_level': process_level},
        'save_to': {'admin_level': save_level},
        'parcel_id_local': {'source': 'parcel_id_assessor', 'kind': 'parcel'},
    }


def test_prefixes_when_process_level_finer_than_save_level():
    # Real collision from Middlesex County: Watertown's raw id "1005 0 0"
    # and Concord's raw id "1005" both convert to the bare key "1005".
    recipe = _recipe(process_level=4, save_level=3)

    town_a = TableIngester(
        recipe, {}, {'admin_id_to_process': 'US-MA-WAT'}, Path('.'), None
    )._add_parcel_id_local(pd.DataFrame({'parcel_id_assessor': ['1005 0 0']}))
    town_b = TableIngester(
        recipe, {}, {'admin_id_to_process': 'US-MA-CON'}, Path('.'), None
    )._add_parcel_id_local(pd.DataFrame({'parcel_id_assessor': ['1005']}))

    # Same standardized key, different towns -- must no longer collide once merged.
    assert town_a['parcel_id_local'].iloc[0] != town_b['parcel_id_local'].iloc[0]
    assert town_a['parcel_id_local'].iloc[0] == 'US-MA-WAT|1005'
    assert town_b['parcel_id_local'].iloc[0] == 'US-MA-CON|1005'


def test_no_prefix_when_process_level_equals_save_level():
    # Mirrors WI/NC/FL/TX: a single chunk already matches the save scope.
    recipe = _recipe(process_level=3, save_level=3)
    df = pd.DataFrame({'parcel_id_assessor': ['1005']})

    result = _ingester(recipe, 'US-WI-DA')._add_parcel_id_local(df)

    assert result['parcel_id_local'].iloc[0] == '1005'


def test_null_raw_value_stays_null_after_prefix():
    recipe = _recipe(process_level=4, save_level=3)
    df = pd.DataFrame({'parcel_id_assessor': [None]})

    result = _ingester(recipe, 'US-MA-WAT')._add_parcel_id_local(df)

    assert pd.isna(result['parcel_id_local'].iloc[0])


def test_admin_id_column_groupby_path_also_prefixed():
    recipe = _recipe(process_level=4, save_level=3)
    recipe['parcel_id_local']['admin_id_column'] = 'town'
    df = pd.DataFrame(
        {
            'parcel_id_assessor': ['1005 0 0', '1005'],
            'town': ['US-MA-WAT', 'US-MA-CON'],
        }
    )

    result = _ingester(recipe, None)._add_parcel_id_local(df)

    values = result['parcel_id_local'].tolist()
    assert values[0] != values[1]
    assert set(values) == {'US-MA-WAT|1005', 'US-MA-CON|1005'}
