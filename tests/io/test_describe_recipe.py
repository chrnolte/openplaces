"""describe_recipe reports a recipe's columns from one file's parquet
footer, joined to the attribute registry, without reading the table."""

import pandas as pd
import pytest

import openplaces.io.readers as readers
from openplaces.core.schema import AdminId


@pytest.fixture
def output_file(tmp_path, monkeypatch):
    path = tmp_path / 'US-NC-WAK_parcel-test-2026.parquet'
    pd.DataFrame(
        {
            'land_value': [1.0, 2.0],
            'land_value_parcel': [10.0, 12.0],
            'not_registered_anywhere': ['a', 'b'],
        },
        index=pd.Index(['p1', 'p2'], name='parcel_id'),
    ).to_parquet(path)
    recipe = {'recipe_id': 'US-NC_parcel-test-2026', 'admin_id': AdminId('US-NC')}
    monkeypatch.setattr(readers, 'get_recipe_by_id', lambda rid: recipe)
    monkeypatch.setattr(
        readers,
        '_get_output_admin_ids',
        lambda recipe, admin_id: ([AdminId('US-NC-WAK')], [], []),
    )
    monkeypatch.setattr(readers, 'get_output_path', lambda *a, **k: path)
    return path


def test_columns_join_to_the_registry_without_reading_rows(output_file):
    described = readers.describe_recipe('US-NC_parcel-test-2026', 'US-NC-WAK')

    assert list(described.index) == [
        'land_value',
        'land_value_parcel',
        'not_registered_anywhere',
        'parcel_id',
    ]
    assert described.loc['land_value', 'attribute'] == 'land_value'
    assert described.loc['land_value', 'aggregation'] == 'sum'
    # A provenance-suffixed evidence column resolves to its base.
    assert described.loc['land_value_parcel', 'attribute'] == 'land_value'
    assert described.loc['not_registered_anywhere', 'attribute'] is None
    assert described.attrs['n_rows'] == 2
    assert described.attrs['admin_id'] == 'US-NC-WAK'


def test_a_missing_file_is_an_error(output_file):
    output_file.unlink()
    with pytest.raises(FileNotFoundError):
        readers.describe_recipe('US-NC_parcel-test-2026', 'US-NC-WAK')
