"""Tests for resolving an unzipped file whose inner name varies from expected.

Some county archives ship a doubled extension (e.g. `sales_2019_YTD.xlsx.xlsx`).
`_match_extracted_file` must resolve such an unambiguous variant while staying
strict when the match is ambiguous.
"""

from pathlib import Path

from openplaces.io.ingester import _match_extracted_file


def test_exact_name_matches(tmp_path):
    (tmp_path / 'sales_2018_YTD.xlsx').touch()
    found = _match_extracted_file(tmp_path, Path('sales_2018_YTD.xlsx'))
    assert found == tmp_path / 'sales_2018_YTD.xlsx'


def test_doubled_extension_resolves_via_stem(tmp_path):
    (tmp_path / 'sales_2019_YTD.xlsx.xlsx').touch()
    found = _match_extracted_file(tmp_path, Path('sales_2019_YTD.xlsx'))
    assert found == tmp_path / 'sales_2019_YTD.xlsx.xlsx'


def test_nested_extraction_resolves(tmp_path):
    nested = tmp_path / 'inner'
    nested.mkdir()
    (nested / 'sales_2019_YTD.xlsx.xlsx').touch()
    found = _match_extracted_file(tmp_path, Path('sales_2019_YTD.xlsx'))
    assert found == nested / 'sales_2019_YTD.xlsx.xlsx'


def test_ambiguous_stem_returns_none(tmp_path):
    (tmp_path / 'sales_2019_YTD.xls').touch()
    (tmp_path / 'sales_2019_YTD.xlsx.xlsx').touch()
    found = _match_extracted_file(tmp_path, Path('sales_2019_YTD.csv'))
    assert found is None


def test_no_match_returns_none(tmp_path):
    (tmp_path / 'something_else.xlsx').touch()
    found = _match_extracted_file(tmp_path, Path('sales_2019_YTD.xlsx'))
    assert found is None
