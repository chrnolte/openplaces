"""A locked `.gdb` asks whether to delete it, and an unattended run has
no one to answer.

Under the Snakemake orchestrator `input()` reads end-of-file and the job
dies with a bare EOFError that names neither the folder nor the cause.
The ingester now checks `can_prompt` first, the same test the
terms-consent gate uses, and raises with the path and the remedy.
"""

from pathlib import Path

import pytest
from pyogrio.errors import DataSourceError

from openplaces.io.ingester import table_ingester as ti_module
from openplaces.io.ingester.table_ingester import TableIngester
from openplaces.timing import Timer


def _ingester(data_path):
    ingester = TableIngester.__new__(TableIngester)
    ingester.recipe = {}
    ingester.download_partition = {'data_path': data_path}
    ingester.processing_chunk = {}
    ingester.timer = Timer('test')
    ingester.verbose = False
    return ingester


@pytest.fixture
def locked_gdb(tmp_path, monkeypatch):
    gdb = tmp_path / 'locked.gdb'
    gdb.mkdir()

    def denied(*args, **kwargs):
        raise DataSourceError(f'{gdb}: Permission denied')

    monkeypatch.setattr(ti_module, 'read_gdb_with_domains', denied)
    return gdb


def test_unattended_run_raises_with_the_path_instead_of_prompting(
    locked_gdb, monkeypatch
):
    monkeypatch.setattr(ti_module, 'can_prompt', lambda: False)

    def no_prompt(*args, **kwargs):
        raise AssertionError('input() must not be called unattended')

    monkeypatch.setattr('builtins.input', no_prompt)
    with pytest.raises(RuntimeError, match='cannot ask') as excinfo:
        _ingester(locked_gdb)._read_recipe_data()
    assert str(locked_gdb) in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, DataSourceError)


def test_attended_run_still_asks(locked_gdb, monkeypatch):
    monkeypatch.setattr(ti_module, 'can_prompt', lambda: True)
    monkeypatch.setattr('builtins.input', lambda *a, **k: 'n')
    with pytest.raises(RuntimeError, match='manually'):
        _ingester(locked_gdb)._read_recipe_data()
    assert Path(locked_gdb).exists()
