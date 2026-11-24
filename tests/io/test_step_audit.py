"""Audit mode: column hashes before and after each step (io.steps).

All tables are fabricated: made-up ids, values and squares.
"""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from openplaces import path as path_module
from openplaces.core.schema import AdminId
from openplaces.io import curator as curator_module
from openplaces.io import enricher as enricher_module
from openplaces.io import harmonizer as harmonizer_module
from openplaces.io import steps as steps_module
from openplaces.io.curator import Curator
from openplaces.io.enricher import Enricher
from openplaces.io.harmonizer import Harmonizer
from openplaces.io.steps import DropRule, StepAudit, make_register

ADMIN_ID = AdminId('US-XX-AAA')
_COLUMN_HASHES = steps_module.column_hashes


def _register():
    return make_register({})


def _frame():
    return gpd.GeoDataFrame(
        {
            'height': [3.0, 6.0, 9.0],
            'label': ['x', 'y', 'z'],
            'tags': [['a'], ['b'], []],
            'geometry': [box(i, 0, i + 1, 1) for i in range(3)],
        },
        index=pd.Index(['e1', 'e2', 'e3'], name='footprint_id'),
        crs='epsg:6933',
    )


def _audit_one(fn, params, mutate):
    frame = _frame()
    audit = StepAudit('Curator', 'XX_footprint-test-1', ADMIN_ID, {'t': frame})
    audit.after_step(0, 'step', fn, params, {'t': mutate(frame.copy())})
    return audit.report()


def test_declared_change_passes_silently():
    @_register()('derive', writes=['{column}'])
    def derive(state, column='n_stories'):
        return state

    def mutate(frame):
        frame['n_stories'] = frame['height'] / 3
        return frame

    report = _audit_one(derive, {}, mutate)
    assert report[['column', 'change', 'declared']].values.tolist() == [
        ['n_stories', 'added', 'yes']
    ]
    assert not (report['declared'] == 'no').any()


def test_undeclared_change_is_reported():
    @_register()('derive', writes=['{column}'])
    def derive(state, column='n_stories'):
        return state

    def mutate(frame):
        frame['n_stories'] = 1
        frame.loc['e2', 'height'] = 7.5
        frame['tags'] = [['a'], ['b'], ['c']]
        frame['geometry'] = frame.geometry.translate(1, 0)
        return frame.drop(columns='label')

    report = _audit_one(derive, {}, mutate).set_index('column')
    assert report.loc['n_stories', 'declared'] == 'yes'
    assert report.loc['height', ['change', 'n_rows', 'declared']].tolist() == [
        'changed',
        1,
        'no',
    ]
    assert report.loc['tags', 'n_rows'] == 1
    assert report.loc['geometry', 'n_rows'] == 3
    assert report.loc['label', 'change'] == 'removed'
    assert report['step_declares'].all()


def test_a_step_without_a_declaration_reports_every_change_undeclared():
    def plain(state):
        return state

    def mutate(frame):
        frame['label'] = frame['label'].str.upper()
        return frame

    report = _audit_one(plain, {}, mutate)
    assert report[['column', 'declared', 'step_declares']].values.tolist() == [
        ['label', 'no', False]
    ]


def test_rows_are_aligned_on_the_index():
    def plain(state):
        return state

    reordered = _audit_one(plain, {}, lambda frame: frame.iloc[::-1])
    assert reordered.empty

    filtered = _audit_one(plain, {}, lambda frame: frame.drop(index='e2'))
    assert filtered[['column', 'change', 'n_rows']].values.tolist() == [['', 'rows', 1]]


def test_a_removal_named_by_drops_is_declared():
    @_register()('trim', drops=['{drop}'])
    def trim(state, drop=()):
        return state

    def mutate(frame):
        frame['height'] = frame['height'] + 1
        return frame.drop(columns=['label', 'tags'])

    report = _audit_one(trim, {'drop': ['label']}, mutate).set_index('column')
    assert report.loc['label', ['change', 'declared']].tolist() == ['removed', 'yes']
    assert report.loc['tags', ['change', 'declared']].tolist() == ['removed', 'no']
    # drops covers removal only: a changed column it names is no write.
    assert report.loc['height', 'declared'] == 'no'


def test_a_removal_a_drop_rule_holds_for_is_declared():
    keep_listed = DropRule('not kept', lambda column, args: column not in args['keep'])

    @_register()('allow', drops=keep_listed)
    def allow(state, keep=()):
        return state

    def mutate(frame):
        return frame.drop(columns=['label', 'tags'])

    report = _audit_one(allow, {'keep': ['tags', 'height']}, mutate)
    report = report.set_index('column')
    assert report.loc['label', 'declared'] == 'yes'
    assert report.loc['tags', 'declared'] == 'no'


def test_a_cast_without_a_value_change_is_a_dtype_change():
    def plain(state):
        return state

    def mutate(frame):
        frame['label'] = frame['label'].astype('category')
        return frame

    report = _audit_one(plain, {}, mutate)
    assert report[['column', 'change']].values.tolist() == [['label', 'dtype']]


def _no_hashing(monkeypatch):
    def refuse(series):
        raise AssertionError('a column was hashed with audit mode off')

    monkeypatch.setattr(steps_module, 'column_hashes', refuse)


def _count_hashing(monkeypatch):
    calls = []

    def counted(series):
        calls.append(series.name)
        return _COLUMN_HASHES(series)

    monkeypatch.setattr(steps_module, 'column_hashes', counted)
    return calls


def _instance(cls, **attributes):
    instance = object.__new__(cls)
    instance.__dict__.update(attributes)
    return instance


@_register()('write_height', writes=['height'])
def _write_height(state):
    frame = getattr(state, 'curated', None)
    if frame is None:
        frame = state.evidence
    frame['height'] = frame['height'] * 2
    frame['stray'] = 1
    return state


def _curator(monkeypatch, tmp_path, audit_steps):
    monkeypatch.setattr(curator_module, 'get_entities', lambda *a, **k: _frame())
    monkeypatch.setattr(curator_module, 'save_parquet', lambda *a, **k: None)
    monkeypatch.setattr(curator_module, 'get_output_path', lambda *a, **k: 'out')
    monkeypatch.setattr(
        path_module, 'logs_path', lambda *a, filename, **k: tmp_path / filename
    )
    monkeypatch.setitem(curator_module._STEP_REGISTRY, 'write_height', _write_height)
    return _instance(
        Curator,
        recipe={
            'recipe_id': 'XX_footprint-test-1',
            'pipeline': [{'step': 'write_height'}],
        },
        entity_recipe={},
        verbose=False,
        save_statistics=False,
        skip_steps=set(),
        _timer=None,
        audit_steps=audit_steps,
    )


def test_curate_with_audit_off_hashes_nothing(monkeypatch, tmp_path):
    _no_hashing(monkeypatch)
    curator = _curator(monkeypatch, tmp_path, audit_steps=False)
    with pytest.warns(steps_module.UndeclaredColumnWarning):
        curator._curate_one(ADMIN_ID)
    assert curator.step_audit is None
    assert not list(tmp_path.iterdir())


def test_curate_with_audit_on_reports_and_writes(monkeypatch, tmp_path):
    calls = _count_hashing(monkeypatch)
    curator = _curator(monkeypatch, tmp_path, audit_steps=True)
    with pytest.warns(steps_module.UndeclaredColumnWarning):
        curator._curate_one(ADMIN_ID)
    assert calls
    report = curator.step_audit.set_index('column')
    assert report.loc['height', 'declared'] == 'yes'
    assert report.loc['stray', 'declared'] == 'no'
    written = list(tmp_path.glob('Curator_audit_*.csv'))
    assert len(written) == 1
    assert len(pd.read_csv(written[0])) == len(report)


def _harmonizer(monkeypatch, tmp_path, audit_steps):
    def build_spine(state):
        state.spine = _frame()
        return state

    monkeypatch.setitem(harmonizer_module._STEP_REGISTRY, 'build_spine', build_spine)
    monkeypatch.setattr(harmonizer_module, 'save_parquet', lambda *a, **k: None)
    monkeypatch.setattr(harmonizer_module, 'get_output_path', lambda *a, **k: 'out')
    monkeypatch.setattr(harmonizer_module, 'saves_geometry', lambda recipe: True)
    monkeypatch.setattr(
        path_module, 'logs_path', lambda *a, filename, **k: tmp_path / filename
    )
    return _instance(
        Harmonizer,
        recipe={
            'recipe_id': 'XX_footprint-spine-1',
            'pipeline': [{'step': 'build_spine'}],
        },
        verbose=False,
        save_statistics=False,
        _timer=None,
        audit_steps=audit_steps,
    )


def test_harmonize_audit_off_and_on(monkeypatch, tmp_path):
    _no_hashing(monkeypatch)
    harmonizer = _harmonizer(monkeypatch, tmp_path, audit_steps=False)
    harmonizer._harmonize_one(ADMIN_ID)
    assert harmonizer.step_audit is None

    _count_hashing(monkeypatch)
    harmonizer = _harmonizer(monkeypatch, tmp_path, audit_steps=True)
    harmonizer._harmonize_one(ADMIN_ID)
    report = harmonizer.step_audit
    # The spine appears in this step: its rows and every column are
    # new, and the undeclared step names none of the columns.
    rows = report[report['change'] == 'rows']
    assert rows['n_rows'].tolist() == [3]
    columns = report[report['change'] != 'rows']
    assert set(columns['column']) == set(_frame().columns)
    assert set(columns['change']) == {'added'}
    assert set(columns['declared']) == {'no'}


def _enricher(monkeypatch, tmp_path, audit_steps):
    monkeypatch.setattr(enricher_module, 'get_entities', lambda *a, **k: _frame())
    monkeypatch.setattr(enricher_module, 'save_parquet', lambda *a, **k: None)
    monkeypatch.setattr(enricher_module, 'get_output_path', lambda *a, **k: 'out')
    monkeypatch.setattr(
        path_module, 'logs_path', lambda *a, filename, **k: tmp_path / filename
    )

    def add_evidence(state):
        state.evidence['height'] = state.spine['height']
        return _write_height(state)

    _register()('add_evidence', writes=['height'])(add_evidence)
    monkeypatch.setitem(enricher_module._STEP_REGISTRY, 'add_evidence', add_evidence)
    return _instance(
        Enricher,
        recipe={
            'recipe_id': 'XX_footprint-evidence-1',
            'pipeline': [{'step': 'add_evidence'}],
        },
        entity_recipe={},
        verbose=False,
        _timer=None,
        audit_steps=audit_steps,
    )


def test_enrich_audit_off_and_on(monkeypatch, tmp_path):
    _no_hashing(monkeypatch)
    enricher = _enricher(monkeypatch, tmp_path, audit_steps=False)
    enricher._enrich_one(ADMIN_ID)
    assert enricher.step_audit is None

    _count_hashing(monkeypatch)
    enricher = _enricher(monkeypatch, tmp_path, audit_steps=True)
    enricher._enrich_one(ADMIN_ID)
    report = enricher.step_audit
    rows = report[['table', 'column', 'declared']].values.tolist()
    assert ['evidence', 'height', 'yes'] in rows
    assert ['evidence', 'stray', 'no'] in rows
    assert not (report['table'] == 'spine').any()
