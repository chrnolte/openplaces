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
    existing = tmp_path / 'US-MA-MI-SO_transaction_all.parquet'
    missing = tmp_path / 'US-MA-SU-CA_transaction_all.parquet'
    existing.touch()

    monkeypatch.setattr(
        readers,
        'get_admin',
        lambda *args, **kwargs: pd.DataFrame(index=['US-MA-MI-SO', 'US-MA-SU-CA']),
    )

    def get_output_path(recipe, admin_id, partition_id=None):
        assert partition_id == 'all'
        return existing if str(admin_id) == 'US-MA-MI-SO' else missing

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
