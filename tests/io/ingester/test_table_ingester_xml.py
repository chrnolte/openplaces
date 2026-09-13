"""The flat-table reader reads a flat XML table, one element per row.

ISSG auditor extracts (Ohio) ship each table as one XML file whose root
holds one element per row, with every field as an attribute, and a
UTF-8 byte-order mark. Values are fabricated.
"""

from __future__ import annotations

from openplaces.core.constants import PANDAS_EXTENSIONS
from openplaces.io.ingester.table_ingester import TableIngester

XML = (
    '﻿<?xml version="1.0" encoding="utf-8"?>'
    '<Parcel_Dwellings>'
    '<Parcel_Dwelling Parcel_Number_U="000000000001" Card="1" Bed_Rooms="3"'
    ' Full_Baths="2" Half_Baths="1" Story_Height="1.50" Notes="" />'
    '<Parcel_Dwelling Parcel_Number_U="000000000002" Card="1" Bed_Rooms="2"'
    ' Full_Baths="1" Half_Baths="0" Story_Height="1.00" Notes="" />'
    '</Parcel_Dwellings>'
)


def _reader(recipe):
    ti = TableIngester.__new__(TableIngester)
    ti.recipe = recipe
    return ti


def test_xml_is_a_flat_table_extension():
    assert '.xml' in PANDAS_EXTENSIONS


def test_attributes_become_columns_and_ids_keep_leading_zeros(tmp_path):
    path = tmp_path / 'Parcel Dwelling.xml'
    path.write_text(XML, encoding='utf-8')
    df = _reader({'csv_dtype': 'str'})._read_flat_table(path, None, None)
    assert len(df) == 2
    assert df['Parcel_Number_U'].tolist() == ['000000000001', '000000000002']
    assert df['Story_Height'].tolist() == ['1.50', '1.00']


def test_text_is_the_default_so_leading_zeros_survive(tmp_path):
    """Type inference read the ids as integers and dropped the zeros."""
    path = tmp_path / 'Parcel Dwelling.xml'
    path.write_text(XML, encoding='utf-8')
    df = _reader({})._read_flat_table(path, None, None)
    assert df['Parcel_Number_U'].tolist() == ['000000000001', '000000000002']


def test_columns_selects_a_subset(tmp_path):
    path = tmp_path / 'Parcel Dwelling.xml'
    path.write_text(XML, encoding='utf-8')
    df = _reader({'csv_dtype': 'str'})._read_flat_table(
        path, ['Parcel_Number_U', 'Bed_Rooms'], None
    )
    assert list(df.columns) == ['Parcel_Number_U', 'Bed_Rooms']
