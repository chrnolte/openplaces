"""The registry crawl keeps only rows of the town, dates and type it asked for.

A search whose criteria did not apply returns other rows; saving them
put a town's deeds in two partitions (Nantucket, 2026-10-19). Fabricated
rows throughout.
"""

import pytest

from openplaces.io.ingester.registry_ingester import RegistryIngester


def _ingester():
    ingester = object.__new__(RegistryIngester)
    ingester.recipe = {'columns': {}}
    return ingester


def _row(doc_type, town='EXAMPLE', date='06/15/2021'):
    return {'town': town, 'recorded_date': date, 'doc_type': doc_type, 'book': '1'}


def test_rows_of_another_type_are_dropped():
    rows = [_row('DEED'), _row('MORTGAGE'), _row('DEED')]
    with pytest.warns(UserWarning, match='document types'):
        kept = _ingester()._drop_mismatched_records(
            rows, 'Example', '2021-06-01', '2021-06-30', '2021-06_DEED', 'DEED'
        )
    assert [r['doc_type'] for r in kept] == ['DEED', 'DEED']


def test_a_search_whose_type_filter_failed_is_not_saved():
    rows = [_row('MORTGAGE'), _row('DISCHARGE'), _row('DEED')]
    with pytest.warns(UserWarning, match='not saved'):
        kept = _ingester()._drop_mismatched_records(
            rows,
            'Example',
            '2021-06-01',
            '2021-06-30',
            '2021-06_FORECLOSURE_DEED',
            'FORECLOSURE DEED',
        )
    assert kept is None


def test_spacing_of_the_type_label_does_not_matter():
    rows = [_row('UNIT DEED'), _row('UNITDEED')]
    kept = _ingester()._drop_mismatched_records(
        rows, 'Example', '2021-06-01', '2021-06-30', '2021-06_UNITDEED', 'UNITDEED'
    )
    assert len(kept) == 2


def test_without_a_type_only_town_and_dates_are_checked():
    rows = [_row('MORTGAGE'), _row('DEED', town='OTHER')]
    with pytest.warns(UserWarning):
        kept = _ingester()._drop_mismatched_records(
            rows, 'Example', '2021-06-01', '2021-06-30', '2021-06_DEED'
        )
    assert [r['doc_type'] for r in kept] == ['MORTGAGE']
