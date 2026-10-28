"""Tests for `table.summarize_conflicts` with missing values on object columns."""

import pandas as pd

from openplaces.table import summarize_conflicts


def test_pd_na_in_object_column_does_not_raise():
    index = pd.RangeIndex(3)
    parcel = pd.Series(['Retail', pd.NA, 'Retail'], index=index, dtype=object)
    nsi = pd.Series(
        ['Retail', 'Single Family', 'Single Family'], index=index, dtype=object
    )
    conflict = summarize_conflicts([('parcel', parcel), ('nsi', nsi)], index)
    assert conflict.isna().tolist() == [True, True, False]
    assert conflict[2] == 'parcel: Retail | nsi: Single Family'


def test_agreeing_values_leave_no_conflict():
    index = pd.RangeIndex(2)
    a = pd.Series(['Retail', pd.NA], index=index, dtype=object)
    b = pd.Series(['Retail', pd.NA], index=index, dtype=object)
    assert summarize_conflicts([('a', a), ('b', b)], index).isna().all()
