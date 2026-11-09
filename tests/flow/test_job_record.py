"""A job skips only when nothing it read and none of its code changed."""

import pandas as pd
import pytest

from openplaces.flow import code_staleness, job_record
from openplaces.io import tables

RULE = 'curate_X_US-NC-CUR'


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(job_record, '_logs_dir', lambda: tmp_path / 'logs')
    changes = {'value': ()}
    monkeypatch.setattr(code_staleness, 'job_changes', lambda *args: changes['value'])
    return changes


@pytest.fixture
def job(tmp_path):
    """A job that reads column a of `spine` and writes `out`."""
    spine = tmp_path / 'spine.parquet'
    other = tmp_path / 'other.csv'
    out = tmp_path / 'out.parquet'
    tables.to_parquet(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), spine, index=False)
    other.write_text('x\n1\n')
    job_io = {
        'inputs': [str(spine), str(other)],
        'outputs': [str(out)],
        'forced': False,
    }

    def run():
        log = tables.start_read_log()
        try:
            frame = tables.read_parquet(spine, columns=['a'])
        finally:
            tables.stop_read_log()
        tables.to_parquet(frame, out, index=False)
        job_record.write_record(
            'curate', 'X', 'US-NC-CUR', RULE, job_io, log, 'feedfacecafe'
        )

    run()
    return {'spine': spine, 'other': other, 'out': out, 'io': job_io}


def skip(job_io):
    return job_record.can_skip('curate', 'X', 'US-NC-CUR', RULE, job_io)


def test_an_unchanged_job_skips(logs, job):
    assert skip(job['io']) == (True, 'inputs and code unchanged since feedfacecafe')


def test_a_rewrite_of_a_column_it_does_not_read_still_skips(logs, job):
    tables.to_parquet(
        pd.DataFrame({'a': [1, 2], 'b': [9, 9]}), job['spine'], index=False
    )
    assert skip(job['io'])[0]


def test_a_change_in_a_column_it_reads_runs_it(logs, job):
    tables.to_parquet(
        pd.DataFrame({'a': [1, 5], 'b': [3, 4]}), job['spine'], index=False
    )
    assert skip(job['io']) == (False, 'input changed: spine.parquet')


def test_a_declared_input_it_never_read_counts_whole(logs, job):
    job['other'].write_text('x\n2\n')
    assert skip(job['io']) == (False, 'input changed: other.csv')


def test_a_changed_output_runs_it(logs, job):
    tables.to_parquet(pd.DataFrame({'a': [0]}), job['out'], index=False)
    assert not skip(job['io'])[0]


def test_a_missing_output_runs_it(logs, job):
    job['out'].unlink()
    assert not skip(job['io'])[0]


def test_a_forced_job_runs(logs, job):
    assert skip({**job['io'], 'forced': True}) == (False, 'forced')


def test_a_code_change_runs_it(logs, job):
    logs['value'] = ('src/openplaces/io/curator/__init__.py',)
    assert skip(job['io']) == (
        False,
        'code changed since feedfacecafe: src/openplaces/io/curator/__init__.py',
    )


def test_a_new_input_runs_it(logs, job, tmp_path):
    extra = tmp_path / 'new.parquet'
    tables.to_parquet(pd.DataFrame({'a': [1]}), extra, index=False)
    job_io = {**job['io'], 'inputs': [*job['io']['inputs'], str(extra)]}
    assert skip(job_io) == (False, 'new input: new.parquet')


def test_without_a_record_it_runs(logs, job):
    job_record.discard_record(RULE)
    assert skip(job['io']) == (False, 'no record of a previous run')


def test_ingests_always_run(logs, job):
    assert not job_record.can_skip('ingest', 'X', 'US-NC-CUR', RULE, job['io'])[0]


def test_run_stage_records_then_skips(logs, job, monkeypatch):
    import types

    from openplaces.flow import run_stage

    calls = []

    def curate(recipe_id, **kwargs):
        calls.append(recipe_id)
        frame = tables.read_parquet(job['spine'], columns=['a'])
        tables.to_parquet(frame, job['out'], index=False)

    monkeypatch.setattr(
        run_stage, 'import_module', lambda name: types.SimpleNamespace(curate=curate)
    )
    job_record.discard_record(RULE)
    (path,) = job_record.write_job_io(
        [{**job['io'], 'admin_id': 'US-NC-CUR', 'rule': RULE}]
    )
    argv = ['curate', 'X', 'US-NC-CUR', '--job-io', str(path), '--rule', RULE]
    run_stage.main(argv)
    assert calls == ['X']
    assert job_record.record_path(RULE).exists()
    run_stage.main(argv)
    assert calls == ['X']
    run_stage.main([*argv, '--no-skip'])
    assert calls == ['X', 'X']


def _job_io_argv(job, stage='curate'):
    (path,) = job_record.write_job_io(
        [{**job['io'], 'admin_id': 'US-NC-CUR', 'rule': RULE}]
    )
    return [stage, 'X', 'US-NC-CUR', '--job-io', str(path), '--rule', RULE]


def test_a_job_the_record_runs_ignores_its_existing_output(logs, job, monkeypatch):
    """The record is the only skip: the real Curator, whose own skip on
    an existing output made every rerun since update() outputs a no-op
    recorded as current, must run and rewrite the file."""
    import types

    from openplaces.flow import run_stage
    from openplaces.io import curator as curator_module

    curated = []

    def curate_one(self, admin_id):
        curated.append(str(admin_id))
        tables.to_parquet(pd.DataFrame({'a': [7]}), job['out'], index=False)

    monkeypatch.setattr(curator_module.Curator, '__init__', _fake_curator_init)
    monkeypatch.setattr(curator_module.Curator, '_curate_one', curate_one)
    monkeypatch.setattr(curator_module, 'get_output_path', lambda *a: job['out'])
    monkeypatch.setattr(
        curator_module,
        'get_timer',
        lambda *a, **k: types.SimpleNamespace(finish=lambda: None),
    )
    job_record.discard_record(RULE)
    assert job['out'].exists()
    run_stage.main(_job_io_argv(job))
    assert curated == ['US-NC-CUR']
    assert tables.read_parquet(job['out'])['a'].tolist() == [7]
    assert job_record.record_path(RULE).exists()


def _fake_curator_init(self, recipe, admin_ids=None, **kwargs):
    self.admin_ids = admin_ids
    self.recipe = {}
    self.verbose = False


@pytest.mark.parametrize(
    ('stage', 'flag'), [('curate', 'reprocess'), ('enrich', 'rerun_covered')]
)
def test_run_stage_tells_the_stage_to_rerun_an_existing_output(
    logs, job, monkeypatch, stage, flag
):
    import types

    from openplaces.flow import run_stage

    seen = {}

    def stage_fn(recipe_id, **kwargs):
        seen.update(kwargs)
        tables.to_parquet(pd.DataFrame({'a': [1]}), job['out'], index=False)

    monkeypatch.setattr(
        run_stage,
        'import_module',
        lambda name: types.SimpleNamespace(**{stage: stage_fn}),
    )
    job_record.discard_record(RULE)
    run_stage.main(_job_io_argv(job, stage))
    assert seen[flag] is True


def test_a_job_that_rewrites_no_output_fails_unrecorded(logs, job, monkeypatch):
    import types

    from openplaces.flow import run_stage

    monkeypatch.setattr(
        run_stage,
        'import_module',
        lambda name: types.SimpleNamespace(curate=lambda recipe_id, **kwargs: None),
    )
    job_record.discard_record(RULE)
    with pytest.raises(run_stage.OutputsNotWrittenError, match='out.parquet'):
        run_stage.main(_job_io_argv(job))
    assert not job_record.record_path(RULE).exists()


def test_a_job_whose_output_is_new_passes_the_write_check(tmp_path):
    from openplaces.flow import run_stage

    out = tmp_path / 'new.parquet'
    before = {str(out): run_stage._file_stamp(out)}
    out.write_bytes(b'x')
    run_stage.require_outputs_written(RULE, before)


def test_job_io_files_merge_by_rule(logs):
    first = {'admin_id': 'US-NC-CUR', 'inputs': [], 'outputs': ['o'], 'forced': False}
    job_record.write_job_io([{**first, 'rule': 'r1'}])
    (path,) = job_record.write_job_io([{**first, 'rule': 'r2', 'forced': True}])
    assert job_record.load_job_io(path, 'r1')['forced'] is False
    assert job_record.load_job_io(path, 'r2')['forced'] is True
    assert job_record.load_job_io(path, 'r3') is None
