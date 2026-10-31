"""Tests for flow.code_staleness: which kept outputs older code built."""

import os
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from openplaces.flow import code_staleness as cs
from openplaces.flow.dag import StageNode, node_key
from openplaces.io.tables import to_parquet
from openplaces.path import git_commit, recipe_root_records

SHA = '0123456789abcdef0123456789abcdef01234567'


@pytest.fixture(autouse=True)
def _ledger(tmp_path, monkeypatch):
    """Every test reads and writes its own, initially absent, ledger."""
    path = tmp_path / 'ledger' / cs.LEDGER_NAME
    monkeypatch.setattr(cs, 'ledger_path', lambda: path)
    cs._clearances.cache_clear()
    yield path
    cs._clearances.cache_clear()


def _git_dir(root, ref='ref: refs/heads/main'):
    git = root / '.git'
    (git / 'refs' / 'heads').mkdir(parents=True)
    (git / 'HEAD').write_text(ref + '\n', encoding='utf-8')
    return git


def test_git_commit_reads_a_loose_ref(tmp_path):
    git = _git_dir(tmp_path)
    (git / 'refs' / 'heads' / 'main').write_text(SHA + '\n', encoding='utf-8')
    assert git_commit(tmp_path) == SHA[:12]


def test_git_commit_reads_a_packed_ref(tmp_path):
    git = _git_dir(tmp_path)
    (git / 'packed-refs').write_text(
        f'# pack-refs with: peeled\n{SHA} refs/heads/main\n', encoding='utf-8'
    )
    assert git_commit(tmp_path) == SHA[:12]


def test_git_commit_reads_a_detached_head(tmp_path):
    _git_dir(tmp_path, ref=SHA)
    assert git_commit(tmp_path) == SHA[:12]


def test_git_commit_follows_a_worktree_to_the_common_refs(tmp_path):
    main = tmp_path / 'main'
    common = _git_dir(main)
    (common / 'refs' / 'heads' / 'side').write_text(SHA + '\n', encoding='utf-8')
    own = common / 'worktrees' / 'side'
    own.mkdir(parents=True)
    (own / 'HEAD').write_text('ref: refs/heads/side\n', encoding='utf-8')
    (own / 'commondir').write_text('../..\n', encoding='utf-8')
    tree = tmp_path / 'side'
    tree.mkdir()
    (tree / '.git').write_text(f'gitdir: {own}\n', encoding='utf-8')
    assert git_commit(tree) == SHA[:12]


def test_git_commit_is_none_outside_a_checkout(tmp_path):
    assert git_commit(tmp_path) is None


def test_a_written_table_records_the_commit(tmp_path):
    path = tmp_path / 'table.parquet'
    to_parquet(pd.DataFrame({'a': [1]}), path)
    assert cs.built_commit(path) == recipe_root_records()[0]['commit']


def test_built_commit_is_none_without_a_footer(tmp_path):
    path = tmp_path / 'plain.parquet'
    pd.DataFrame({'a': [1]}).to_parquet(path)
    assert cs.built_commit(path) is None


def test_a_stage_closure_reaches_its_step_modules_and_packages():
    closure = cs.import_closure(frozenset({'openplaces.io.harmonizer'}))
    names = {p.relative_to(cs.PACKAGE_DIR).as_posix() for p in closure}
    # The package files run on import, and the step modules are loaded
    # by name (io.steps.make_loader), so no import statement names them.
    assert {'__init__.py', 'io/__init__.py', 'io/harmonizer/__init__.py'} <= names
    assert 'io/harmonizer/spine.py' in names
    assert 'core/schema.py' in names


def test_job_dependencies_include_data_beside_modules_and_the_recipe():
    deps = cs.job_dependencies('harmonize', 'US_parcel-spine-2026', 'US-NC-CUR')
    assert 'src/openplaces/core/attribute_registry.csv' in deps
    assert any(d.endswith('/US_parcel-spine-2026.yaml') for d in deps)
    assert not any(d.endswith('README.md') for d in deps)


class FakeDag:
    """Three jobs in a chain: ingest -> harmonize -> curate."""

    def __init__(self, root):
        self.root = root
        self._nodes = [
            StageNode('ingest', 'src-a', 'XX-AA'),
            StageNode('harmonize', 'spine-a', 'XX-AA'),
            StageNode('curate', 'cur-a', 'XX-AA'),
        ]
        self._edges = [
            (node_key(self._nodes[0]), node_key(self._nodes[1])),
            (node_key(self._nodes[1]), node_key(self._nodes[2])),
        ]

    def nodes(self):
        return list(self._nodes)

    def output_path(self, stage, recipe_id, admin_id=None, region=None):
        return self.root / f'{recipe_id}.parquet'

    def target_paths(self):
        return [self.output_path('curate', 'cur-a')]

    def input_paths(self, stage, recipe_id, admin_id=None, region=None):
        upstream = {'spine-a': 'src-a', 'cur-a': 'spine-a'}.get(recipe_id)
        return [self.output_path('', upstream)] if upstream else []

    def extra_outputs(self, stage, recipe_id, admin_id=None, region=None):
        return []

    def _recipe(self, recipe_id):
        return {'recipe_id': recipe_id}


@pytest.fixture
def fake(tmp_path, monkeypatch):
    # Written oldest first, so every output is newer than its input.
    for age, name in ((300, 'src-a'), (200, 'spine-a'), (100, 'cur-a')):
        path = tmp_path / f'{name}.parquet'
        to_parquet(pd.DataFrame({'a': [1]}), path)
        stamp = time.time() - age
        os.utime(path, (stamp, stamp))
    monkeypatch.setattr(cs, 'built_commit', lambda path: 'abc123def456')
    monkeypatch.setattr(
        cs, 'changed_since', lambda commit: frozenset({'src/openplaces/io/x.py'})
    )
    deps = {
        'ingest': frozenset({'src/openplaces/io/y.py'}),
        'harmonize': frozenset({'src/openplaces/io/x.py'}),
        'curate': frozenset({'src/openplaces/io/x.py'}),
    }
    monkeypatch.setattr(
        cs, '_dependencies', lambda stage, r, a: (deps[stage], frozenset())
    )
    return tmp_path


def test_a_kept_job_whose_code_changed_is_reported(fake):
    rows = cs.report(FakeDag(fake))
    assert [(r['recipe_id'], r['status']) for r in rows] == [
        ('spine-a', 'code changed'),
        ('cur-a', 'code changed'),
    ]
    assert rows[0]['changed'] == ('src/openplaces/io/x.py',)


def test_a_forced_job_and_everything_downstream_is_not_reported(fake):
    rows = cs.report(FakeDag(fake), forcerun=['harmonize_spine-a_XX-AA'])
    assert rows == []


def test_a_job_the_mtime_trigger_reruns_is_not_reported(fake):
    stamp = time.time() - 1000
    os.utime(fake / 'cur-a.parquet', (stamp, stamp))
    rows = cs.report(FakeDag(fake))
    assert [r['recipe_id'] for r in rows] == ['spine-a']


def test_inputs_passed_in_are_used_instead_of_derived(fake):
    dag = FakeDag(fake)
    newer = fake / 'newer.parquet'
    to_parquet(pd.DataFrame({'a': [1]}), newer)
    inputs = {node_key(dag.nodes()[1]): [newer]}
    rows = cs.report(dag, inputs=inputs)
    # The spine reruns on the passed input, and so does its consumer.
    assert rows == []


def test_force_targets_covers_only_the_target(fake):
    rows = cs.report(FakeDag(fake), forcetargets=True)
    assert [r['recipe_id'] for r in rows] == ['spine-a']


def test_an_output_without_a_commit_is_counted_apart(fake, monkeypatch):
    monkeypatch.setattr(cs, 'built_commit', lambda path: None)
    rows = cs.report(FakeDag(fake))
    assert {r['status'] for r in rows} == {'no commit recorded'}
    text = cs.format_report(rows)
    assert text == (
        'Kept outputs with no commit recorded: '
        'ingest src-a (1), harmonize spine-a (1), curate cur-a (1)'
    )
    assert 'Code changed since' not in text


@pytest.fixture
def registry_only(fake, monkeypatch):
    """Every job depends on the registry, and only it changed."""
    monkeypatch.setattr(cs, 'changed_since', lambda commit: frozenset({cs.REGISTRY}))
    monkeypatch.setattr(
        cs, '_dependencies', lambda stage, r, a: (frozenset({cs.REGISTRY}), frozenset())
    )
    return fake


def test_a_registry_row_no_table_carries_reports_only_ingests(
    registry_only, monkeypatch
):
    monkeypatch.setattr(cs, 'registry_changes', lambda commit: frozenset({'b'}))
    rows = cs.report(FakeDag(registry_only))
    # An ingest reads a raw source, whose columns no table records.
    assert [r['recipe_id'] for r in rows] == ['src-a']


def test_a_registry_row_naming_a_column_reports_its_jobs(registry_only, monkeypatch):
    monkeypatch.setattr(cs, 'registry_changes', lambda commit: frozenset({'a'}))
    rows = cs.report(FakeDag(registry_only))
    assert [r['recipe_id'] for r in rows] == ['src-a', 'spine-a', 'cur-a']
    assert rows[1]['changed'] == (cs.REGISTRY,)


def test_a_changed_registry_header_counts_for_every_job(registry_only, monkeypatch):
    monkeypatch.setattr(cs, 'registry_changes', lambda commit: None)
    rows = cs.report(FakeDag(registry_only))
    assert len(rows) == 3


def test_a_suffixed_column_carries_its_attribute():
    names = frozenset({'floor_area'})
    assert cs._touches({'floor_area_building_nsi'}, names)
    assert cs._touches({'floor_area_source'}, names)
    assert not cs._touches({'floor', 'area_floor'}, names)


def test_registry_rows_compare_whole_rows():
    old = cs._registry_rows('name,unit\na,m\nb,m\n')
    new = cs._registry_rows('name,unit\na,ft\nc,m\n')
    assert old[0] == new[0]
    changed = {
        n for n in old[1].keys() | new[1].keys() if old[1].get(n) != new[1].get(n)
    }
    assert changed == {'a', 'b', 'c'}


def test_registry_changes_since_head_reads_git():
    head = git_commit(cs.repo_root())
    assert isinstance(cs.registry_changes(head), frozenset)


HEAD = 'feedfacecafe'


@pytest.fixture
def verifiable(fake, monkeypatch):
    """The chain's code changed since abc123def456 and is committed."""
    monkeypatch.setattr('openplaces.path.git_commit', lambda root: HEAD)
    monkeypatch.setattr(
        cs,
        'changed_since',
        lambda commit: (
            frozenset() if commit == HEAD else frozenset({'src/openplaces/io/x.py'})
        ),
    )
    return fake


def _runner(touch=None, returncode=0):
    calls = []

    def run(cmd, check=False):
        calls.append(cmd)
        if touch is not None and cmd[4] in touch:
            path = touch[cmd[4]]
            to_parquet(pd.DataFrame({'a': [2]}), path)
        return SimpleNamespace(returncode=returncode)

    run.calls = calls
    return run


def test_a_reproduced_sample_clears_its_group(verifiable):
    dag = FakeDag(verifiable)
    run = _runner()
    results = cs.verify(dag, cs.report(dag), n=3, run=run)
    assert [(r['recipe_id'], r['verdict']) for r in results] == [
        ('spine-a', 'reproduced'),
        ('cur-a', 'reproduced'),
    ]
    assert run.calls[0][3:] == [
        'harmonize',
        'spine-a',
        'XX-AA',
        '--reprocess',
        '--keep-if-equal',
    ]
    assert cs.report(dag) == []


def test_a_sample_that_rewrites_its_output_clears_nothing(verifiable, _ledger):
    dag = FakeDag(verifiable)
    run = _runner(touch={'cur-a': verifiable / 'cur-a.parquet'})
    results = cs.verify(dag, cs.report(dag), run=run)
    assert [r['verdict'] for r in results] == ['reproduced', 'changed']
    assert _ledger.read_text(encoding='utf-8').count('\n') == 3
    assert not cs.cleared(
        'curate', 'cur-a', 'abc123def456', ('src/openplaces/io/x.py',)
    )
    assert cs.cleared(
        'harmonize', 'spine-a', 'abc123def456', ('src/openplaces/io/x.py',)
    )


def test_a_failed_sample_clears_nothing(verifiable):
    dag = FakeDag(verifiable)
    results = cs.verify(dag, cs.report(dag), run=_runner(returncode=1))
    assert {r['verdict'] for r in results} == {'failed'}
    assert len(cs.report(dag)) == 2


def test_uncommitted_changes_are_not_sampled(verifiable, monkeypatch):
    monkeypatch.setattr(
        cs, 'changed_since', lambda commit: frozenset({'src/openplaces/io/x.py'})
    )
    dag = FakeDag(verifiable)
    run = _runner()
    results = cs.verify(dag, cs.report(dag), run=run)
    assert {r['verdict'] for r in results} == {'not sampled (uncommitted changes)'}
    assert run.calls == []


def test_a_job_built_by_other_code_than_the_samples_is_not_cleared(
    verifiable, monkeypatch
):
    dag = FakeDag(verifiable)
    cs.verify(dag, cs.report(dag), run=_runner())
    monkeypatch.setattr(cs, '_diff', lambda a, b: frozenset({'src/openplaces/io/x.py'}))
    assert not cs.cleared(
        'harmonize', 'spine-a', '0ldc0mmit000', ('src/openplaces/io/x.py',)
    )
    assert cs.cleared(
        'harmonize', 'spine-a', 'abc123def456', ('src/openplaces/io/x.py',)
    )


def test_a_change_after_the_verified_commit_is_not_cleared(verifiable, monkeypatch):
    dag = FakeDag(verifiable)
    cs.verify(dag, cs.report(dag), run=_runner())
    monkeypatch.setattr(
        cs, 'changed_since', lambda commit: frozenset({'src/openplaces/io/x.py'})
    )
    assert not cs.cleared(
        'harmonize', 'spine-a', 'abc123def456', ('src/openplaces/io/x.py',)
    )


def test_samples_spread_over_output_size(tmp_path):
    rows = []
    for i, size in enumerate((5, 1, 4, 2, 3)):
        path = tmp_path / f'r{i}.parquet'
        path.write_bytes(b'x' * size)
        rows.append(
            {'stage': 'curate', 'recipe_id': f'r{i}', 'admin_id': None, 'size': size}
        )
    dag = SimpleNamespace(output_path=lambda s, r, a=None: tmp_path / f'{r}.parquet')
    assert [r['size'] for r in cs._sample(rows, dag, 3)] == [1, 3, 5]
    assert [r['size'] for r in cs._sample(rows, dag, 1)] == [3]


def test_format_report_groups_units_and_names_a_rule():
    rows = [
        {
            'stage': 'harmonize',
            'recipe_id': 'spine-a',
            'admin_id': unit,
            'rule': f'harmonize_spine-a_{unit}',
            'commit': 'abc123def456',
            'status': 'code changed',
            'changed': ('src/openplaces/io/x.py',),
        }
        for unit in ('XX-AA', 'XX-BB')
    ]
    text = cs.format_report(rows)
    assert 'harmonize spine-a: 2 units built at abc123def456; changed: io/x.py' in text
    assert '--forcerun <rule>' in text
    assert 'e.g. harmonize_spine-a_XX-AA' in text
    assert cs.format_report([]) == ''


def test_format_summary_counts_stages_and_files():
    rows = [
        {
            'stage': stage,
            'recipe_id': 'r',
            'admin_id': 'XX-AA',
            'rule': f'{stage}_r_XX-AA',
            'commit': 'abc',
            'status': 'code changed',
            'changed': changed,
        }
        for stage, changed in (
            ('curate', ('src/openplaces/io/x.py',)),
            ('harmonize', ('src/openplaces/io/x.py', 'src/openplaces/io/y.py')),
        )
    ]
    rows.append({**rows[0], 'status': 'no commit recorded', 'changed': ()})
    text = cs.format_summary(rows, 'US_footprint-cheer-2026')
    assert '2 jobs this run keeps' in text
    assert 'by stage: harmonize 1, curate 1' in text
    assert 'io/x.py (2), io/y.py (1)' in text
    assert 'code_staleness US_footprint-cheer-2026' in text
    assert '1 kept outputs record no commit' in text
    assert cs.format_summary([], 'r') == ''


def test_the_git_diff_names_files_since_a_commit():
    head = recipe_root_records()[0]['commit']
    if head is None:
        pytest.skip('not a git checkout')
    changed = cs.changed_since(head)
    assert changed is not None
    assert all(f.startswith('src/') for f in changed)
    assert cs.changed_since('0' * 12) is None


def test_main_prints_a_line_when_nothing_is_kept(monkeypatch, capsys):
    rows = []
    monkeypatch.setattr(cs, 'report', lambda dag, forcerun=(): rows)
    monkeypatch.setattr(
        'openplaces.flow.dag.RecipeDAG', lambda *a, **k: SimpleNamespace()
    )
    assert cs.main(['US_footprint-cheer-2026', 'XX-AA']) == 0
    assert 'No kept output' in capsys.readouterr().out
    rows.append({'rule': 'curate_x_XX-AA', 'status': 'code changed', 'commit': 'abc'})
    cs.main(['US_footprint-cheer-2026', 'XX-AA', '--jobs'])
    assert capsys.readouterr().out == 'curate_x_XX-AA\tcode changed\tabc\n'
