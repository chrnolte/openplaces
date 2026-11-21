"""An admin-column crosswalk that matches no row of a download raises.

FEMA's USA Structures carry a county FIPS code, and the national
recipe chunks them by it at admin level 3. In New England level 3 is
the town, so no code matched, and every town's file was saved without
rows; the gap surfaced only in a later harmonize. The ingest now stops
at the mismatch.
"""

import pandas as pd
import pytest

from openplaces.io.ingester.table_ingester import TableIngester


def _ingester(codes, crosswalk_codes):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {
        'entity': 'footprint-test-2023',
        'process_by': {'admin_level': 3, 'admin_id_column': 'FIPS'},
    }
    ingester.download_partition = {
        'admin_id_to_download': 'US-XX',
        'admin_id_crosswalk': pd.DataFrame(
            {'admin3_id': [f'US-XX-T{i}' for i in range(len(crosswalk_codes))]},
            index=pd.Index(crosswalk_codes, name='admin3_id_admin1'),
        ),
        'table_fids': {},
    }
    rows = pd.DataFrame({'FIPS': codes}, index=pd.RangeIndex(len(codes), name='fid'))
    ingester._read_recipe_data = lambda **kwargs: rows
    return ingester


def test_codes_of_another_level_raise():
    # Fabricated codes: county-style rows against town-style codes.
    ingester = _ingester(['99001', '99001', '99003'], ['9900100010', '9900300020'])

    with pytest.raises(ValueError, match='None of the 3 rows of US-XX match'):
        ingester._prepare_table_fid_filter()


def test_matching_codes_assign_units():
    ingester = _ingester(['99001', '99003', '99005'], ['99001', '99003'])

    ingester._prepare_table_fid_filter()

    fids = ingester.download_partition['table_fids'][ingester.table_name]
    assert fids.tolist()[:2] == ['US-XX-T0', 'US-XX-T1']
    assert pd.isna(fids.iloc[2])


def test_an_empty_download_does_not_raise():
    ingester = _ingester([], ['99001'])

    ingester._prepare_table_fid_filter()

    assert ingester.download_partition['table_fids'][ingester.table_name].empty
