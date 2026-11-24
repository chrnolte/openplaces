"""Column declarations on registered steps (io.steps)."""

import warnings

import pytest

from openplaces.io.steps import (
    DropRule,
    UndeclaredColumnWarning,
    make_register,
    resolve_step_columns,
    step_columns,
    undeclared_new_columns,
    warn_undeclared_columns,
)


def _registry():
    registry = {}
    return registry, make_register(registry)


def test_step_without_declaration_resolves_to_none():
    registry, register = _registry()

    @register('plain')
    def plain(state):
        return state

    assert step_columns(plain) is None
    assert resolve_step_columns(plain, {}) is None
    assert undeclared_new_columns(plain, {}, ['a'], ['a', 'b']) == []


def test_templates_fill_from_arguments_then_defaults():
    registry, register = _registry()

    @register(
        'derive',
        reads=['{height_column}'],
        writes=['{column}', '{column}_source'],
        decides=['{column}'],
    )
    def derive(state, column='n_stories', height_column='height'):
        return state

    resolved = resolve_step_columns(derive, {'height_column': 'height_fema'})
    assert resolved.reads == {'height_fema'}
    assert resolved.writes == {'n_stories', 'n_stories_source'}
    assert resolved.decides == {'n_stories'}


def test_list_mapping_none_and_callable_entries():
    registry, register = _registry()

    @register(
        'gather',
        reads=['{sources}', '{rename:keys}', '{optional}'],
        writes=['{rename:values}', lambda args: [s + '_flag' for s in args['sources']]],
    )
    def gather(state, sources=(), rename=None, optional=None):
        return state

    resolved = resolve_step_columns(
        gather, {'sources': ['a', 'b'], 'rename': {'old': 'new'}}
    )
    assert resolved.reads == {'a', 'b', 'old'}
    assert resolved.writes == {'new', 'a_flag', 'b_flag'}


def test_reads_from_names_the_upstream_recipe():
    registry, register = _registry()

    @register('link', reads_from={'recipe_id': ['{columns}']}, writes=['{columns}'])
    def link(state, recipe_id=None, columns=()):
        return state

    resolved = resolve_step_columns(
        link, {'recipe_id': 'XX_parcel-example-1', 'columns': ['land_use']}
    )
    assert resolved.reads_from == {'XX_parcel-example-1': frozenset({'land_use'})}


def test_template_naming_no_parameter_is_refused():
    registry, register = _registry()

    @register('typo', writes=['{colum}'])
    def typo(state, column='x'):
        return state

    with pytest.raises(ValueError, match='names no parameter'):
        resolve_step_columns(typo, {})


def test_mapping_without_part_is_refused():
    registry, register = _registry()

    @register('mapped', reads=['{rename}'])
    def mapped(state, rename=None):
        return state

    with pytest.raises(ValueError, match='keys'):
        resolve_step_columns(mapped, {'rename': {'a': 'b'}})


def test_decides_must_be_written():
    registry, register = _registry()
    with pytest.raises(ValueError, match='decides'):

        @register('bad', writes=['a'], decides=['b'])
        def bad(state):
            return state


def test_new_columns_outside_writes_are_reported():
    registry, register = _registry()

    @register('adds', writes=['{column}', 'score_*'])
    def adds(state, column='x'):
        return state

    before, after = ['id'], ['id', 'x', 'score_roll', 'stray']
    assert undeclared_new_columns(adds, {}, before, after) == ['stray']
    with pytest.warns(UndeclaredColumnWarning, match='stray'):
        warn_undeclared_columns('adds', adds, {}, before, after)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warn_undeclared_columns('adds', adds, {}, before, ['id', 'x'])


def test_drops_resolve_names_templates_callables_and_rules():
    registry, register = _registry()
    rule = DropRule('kept list', lambda column, args: column not in args['keep'])

    @register(
        'trim',
        drops=['tmp', '{drop}', lambda args: [f'{c}_x' for c in args['drop']], rule],
    )
    def trim(state, drop=(), keep=()):
        return state

    resolved = resolve_step_columns(trim, {'drop': ['a', 'b'], 'keep': ['id']})
    assert resolved.drops == {'tmp', 'a', 'b', 'a_x', 'b_x'}
    assert resolved.writes == frozenset()
    assert [name for name, _ in resolved.drop_rules] == ['kept list']
    assert resolved.drops_column('a_x')
    assert resolved.drops_column('anything')
    assert not resolved.drops_column('id')


def test_a_drops_only_declaration_declares_the_step():
    registry, register = _registry()
    rule = DropRule('none', lambda column, args: False)

    @register('only_rule', drops=rule)
    def only_rule(state):
        return state

    declared = step_columns(only_rule)
    assert declared is not None and declared.drops == (rule,)
    resolved = resolve_step_columns(only_rule, {})
    assert resolved.drops == frozenset() and not resolved.drops_column('x')


def test_a_drop_rule_outside_drops_is_refused():
    registry, register = _registry()
    with pytest.raises(TypeError, match='drops only'):

        @register('misplaced', writes=[DropRule('r', lambda column, args: True)])
        def misplaced(state):
            return state


def test_wildcard_writes_reports_nothing():
    registry, register = _registry()

    @register('anything', writes='*')
    def anything(state):
        return state

    assert undeclared_new_columns(anything, {}, [], ['a', 'b']) == []
