import pandas as pd
import pytest

from openplaces.core.schema import AdminId
from openplaces.io import aggregate, readers
from openplaces.utils import inspect_table


def test_get_entities_reads_single_file_aggregate(tmp_path, monkeypatch):
    recipe = {
        'admin_id': AdminId('US', 'WI'),
        'save_to': {'admin_level': 2},
        'aggregate_by': {'single_file': True},
    }
    path = tmp_path / 'US-WI_transaction_all.parquet'
    path.touch()
    calls = []

    def get_output_path(recipe, admin_id, partition_id=None):
        calls.append((str(admin_id), partition_id))
        return path

    monkeypatch.setattr(readers, 'get_output_path', get_output_path)
    monkeypatch.setattr(
        readers,
        'read_parquet',
        lambda path, **kwargs: pd.DataFrame({'price': [100.0]}),
    )
    monkeypatch.setattr(
        aggregate,
        'read_partition_coverage',
        lambda path: {'202601', '202602'},
    )

    data = readers.get_entities(recipe)

    assert calls == [('US-WI', 'all')]
    assert data['price'].tolist() == [100.0]
    assert data.attrs['openplaces_output_paths'] == [str(path)]
    assert data.attrs['openplaces_partition_ids'] == ['202601', '202602']


def test_get_entities_expands_parent_and_warns_for_missing(
    tmp_path,
    monkeypatch,
):
    recipe = {
        'admin_id': AdminId('US', 'MA'),
        'save_to': {'admin_level': 4},
        'aggregate_by': {'single_file': True},
    }
    existing = tmp_path / 'US-MA-SOE_transaction_all.parquet'
    missing = tmp_path / 'US-MA-SU-CA_transaction_all.parquet'
    existing.touch()

    monkeypatch.setattr(
        readers,
        'get_admin',
        lambda *args, **kwargs: pd.DataFrame(index=['US-MA-SOE', 'US-MA-SU-CA']),
    )

    def get_output_path(recipe, admin_id, partition_id=None):
        assert partition_id == 'all'
        return existing if str(admin_id) == 'US-MA-SOE' else missing

    monkeypatch.setattr(readers, 'get_output_path', get_output_path)
    monkeypatch.setattr(
        readers,
        'read_parquet',
        lambda path, **kwargs: pd.DataFrame({'price': [200.0]}),
    )
    monkeypatch.setattr(
        aggregate,
        'read_partition_coverage',
        lambda path: {'2020'},
    )

    with pytest.warns(UserWarning, match='1 recipe output file'):
        data = readers.get_entities(recipe, admin_id='US-MA', missing='warn')

    assert data['price'].tolist() == [200.0]
    assert data.attrs['openplaces_missing_paths'] == [str(missing)]


def test_get_entities_stamps_admin_id_when_combining_multiple(tmp_path, monkeypatch):
    recipe = {'admin_id': AdminId('US'), 'save_to': {'admin_level': 3}}
    path_mi = tmp_path / 'US-MA-MI_footprint.parquet'
    path_su = tmp_path / 'US-MA-SU_footprint.parquet'
    path_mi.touch()
    path_su.touch()

    def get_output_path(recipe, admin_id, partition_id=None):
        return path_mi if str(admin_id) == 'US-MA-MI' else path_su

    def read_parquet(path, **kwargs):
        if path == path_mi:
            return pd.DataFrame({'value': [1, 2]})
        return pd.DataFrame({'value': [3]})

    monkeypatch.setattr(readers, 'get_output_path', get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', read_parquet)

    data = readers.get_entities(recipe, admin_id=['US-MA-MI', 'US-MA-SU'])

    assert data['admin3_id'].tolist() == ['US-MA-MI', 'US-MA-MI', 'US-MA-SU']
    assert data['value'].tolist() == [1, 2, 3]


def test_get_entities_does_not_overwrite_existing_admin_id_column(
    tmp_path, monkeypatch
):
    recipe = {'admin_id': AdminId('US'), 'save_to': {'admin_level': 3}}
    path_mi = tmp_path / 'US-MA-MI_footprint.parquet'
    path_su = tmp_path / 'US-MA-SU_footprint.parquet'
    path_mi.touch()
    path_su.touch()

    def get_output_path(recipe, admin_id, partition_id=None):
        return path_mi if str(admin_id) == 'US-MA-MI' else path_su

    def read_parquet(path, **kwargs):
        if path == path_mi:
            return pd.DataFrame({'value': [1], 'admin3_id': ['custom']})
        return pd.DataFrame({'value': [2], 'admin3_id': ['also-custom']})

    monkeypatch.setattr(readers, 'get_output_path', get_output_path)
    monkeypatch.setattr(readers, 'read_parquet', read_parquet)

    data = readers.get_entities(recipe, admin_id=['US-MA-MI', 'US-MA-SU'])

    assert data['admin3_id'].tolist() == ['custom', 'also-custom']


def test_inspect_table_summarizes_and_returns_selected_sample(capsys):
    data = pd.DataFrame(
        {
            'price_raw': ['$10.00', '$20.00'],
            'price': [10.0, 20.0],
            'geometry': ['a', 'b'],
        }
    )
    data.attrs['openplaces_output_paths'] = ['one.parquet', 'two.parquet']
    data.attrs['openplaces_partition_ids'] = ['202601', '202602']

    sample = inspect_table(
        data,
        n=5,
        columns=['price_raw', 'price', 'not_present'],
        random_state=0,
    )

    assert sample.index.tolist() == ['price_raw', 'price']
    assert sample.shape == (2, 2)
    assert (
        capsys.readouterr().out == '2 rows x 3 columns from 2 file(s), 2 partition(s)\n'
    )


def test_get_entities_adds_admin_column(tmp_path, monkeypatch):
    recipe = {
        'admin_id': AdminId('US', 'MA'),
        'save_to': {'admin_level': 3},
        'aggregate_by': {'single_file': False},
    }

    path1 = tmp_path / 'US-MA-BR.parquet'
    path2 = tmp_path / 'US-MA-ES.parquet'
    path1.touch()
    path2.touch()

    monkeypatch.setattr(
        readers,
        'get_admin',
        lambda *args, **kwargs: pd.DataFrame(index=['US-MA-BR', 'US-MA-ES']),
    )

    def get_output_path(recipe, admin_id, partition_id=None):
        return path1 if str(admin_id) == 'US-MA-BR' else path2

    monkeypatch.setattr(readers, 'get_output_path', get_output_path)

    def mock_read_parquet(path, columns=None, **kwargs):
        df = pd.DataFrame({'value': [1.0, 2.0]})
        if columns is not None:
            assert 'admin3_id' not in columns
            df = df[[c for c in columns if c in df.columns]]
        return df

    monkeypatch.setattr(readers, 'read_parquet', mock_read_parquet)

    # 1. Columns=None (default): should add admin3_id and cast to category
    data = readers.get_entities(recipe, admin_id=['US-MA-BR', 'US-MA-ES'])
    assert 'admin3_id' in data.columns
    assert data['admin3_id'].dtype.name == 'category'
    assert data['admin3_id'].tolist() == [
        'US-MA-BR',
        'US-MA-BR',
        'US-MA-ES',
        'US-MA-ES',
    ]
    assert data['value'].tolist() == [1.0, 2.0, 1.0, 2.0]

    # 2. Columns containing admin3_id: should request 'value' from disk,
    # add admin3_id, and cast to category.
    data2 = readers.get_entities(
        recipe,
        admin_id=['US-MA-BR', 'US-MA-ES'],
        columns=['value', 'admin3_id'],
    )
    assert 'admin3_id' in data2.columns
    assert data2['admin3_id'].dtype.name == 'category'
    assert data2['admin3_id'].tolist() == [
        'US-MA-BR',
        'US-MA-BR',
        'US-MA-ES',
        'US-MA-ES',
    ]
    assert data2['value'].tolist() == [1.0, 2.0, 1.0, 2.0]

    # 3. Columns NOT containing admin3_id: should request 'value'
    # and NOT add admin3_id.
    data3 = readers.get_entities(
        recipe,
        admin_id=['US-MA-BR', 'US-MA-ES'],
        columns=['value'],
    )
    assert 'admin3_id' not in data3.columns
