"""The usage gate warns and asks; only an unattended mismatch stops a run.

A source may condition access on who is asking (non-commercial use only,
a restricted environment, a jurisdictional interest). The contract
pinned here mirrors the terms-consent one: the requirement is recorded
on the source by a person who read the terms, the declared profile is
checked before any download, and a mismatch is resolved by the operator
at the prompt -- for this run, or recorded as a standing decision. An
unattended run with an unresolved mismatch raises rather than silently
downloading or silently skipping.

Unlike consent, there is no forbidden recipe-side lever: recording
`usage_requirement` states a fact about the source, it does not consent
to anything on the user's behalf.
"""

from pathlib import Path

import pytest
import yaml

from openplaces.core.schema import Source, UsageRequirement
from openplaces.io.usage_profile import (
    UsageProfileMismatchError,
    forget_usage_answers,
    require_usage_compatible,
)

RECIPES = Path(__file__).resolve().parents[2] / 'src' / 'openplaces' / 'recipes'

UNDECLARED = {
    'commercial': None,
    'environment': {
        'licensed': False,
        'restricted': False,
        'encrypted_at_rest': False,
        'offline_only': False,
    },
    'admin_interests': [],
}


def _profile(**overrides):
    profile = {**UNDECLARED, 'environment': dict(UNDECLARED['environment'])}
    for key, value in overrides.items():
        if key in profile['environment']:
            profile['environment'][key] = value
        else:
            profile[key] = value
    return profile


@pytest.fixture(autouse=True)
def _no_standing_override(monkeypatch):
    """Isolate from whatever this developer recorded in their own config."""
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_override', lambda source: None
    )


@pytest.fixture(autouse=True)
def _undeclared_profile(monkeypatch):
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_profile', lambda: _profile()
    )


@pytest.fixture(autouse=True)
def _forget():
    forget_usage_answers()
    yield
    forget_usage_answers()


def _attended(monkeypatch, attended=True):
    """Declare whether someone is able to answer the prompt.

    Patches `config.can_prompt`, not `sys.stdin.isatty`: a Jupyter
    kernel answers `input` without being a terminal, and a test session
    is unattended even when it has one, so the tty is not the question.
    The terms gate asks the same helper.
    """
    monkeypatch.setattr('openplaces.io.usage_profile.can_prompt', lambda: attended)


def _source(**requirement):
    # No '-' in the id: it is the id separator, and Source maps it to a
    # tilde so a source id stays one part of an entity id.
    return Source(
        source_id='a_source',
        terms_url='https://example.org/terms',
        usage_requirement=UsageRequirement(**requirement),
    )


def test_no_requirement_is_a_complete_no_op(monkeypatch):
    """The ~140 existing recipes short-circuit before config is read."""
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_profile',
        lambda: pytest.fail('profile read for a source without a requirement'),
    )
    assert require_usage_compatible(Source('massgis')) is True
    assert require_usage_compatible(_source()) is True  # empty requirement


def test_a_compatible_profile_passes_without_a_prompt(monkeypatch):
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_profile',
        lambda: _profile(commercial=False),
    )
    monkeypatch.setattr(
        'builtins.input', lambda *a: pytest.fail('prompted despite a match')
    )
    assert require_usage_compatible(_source(non_commercial=True)) is True


def test_a_standing_override_is_honored(monkeypatch):
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_override', lambda source: True
    )
    assert require_usage_compatible(_source(non_commercial=True)) is True

    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_override', lambda source: False
    )
    forget_usage_answers()
    assert require_usage_compatible(_source(non_commercial=True)) is False


def test_an_unattended_mismatch_raises_with_directions(monkeypatch):
    _attended(monkeypatch, False)
    monkeypatch.setattr(
        'builtins.input', lambda *a: pytest.fail('prompted an unattended run')
    )
    with pytest.raises(UsageProfileMismatchError, match='set-usage-profile'):
        require_usage_compatible(_source(non_commercial=True))


def test_declining_at_the_prompt_returns_false(monkeypatch):
    _attended(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a: 'n')
    assert require_usage_compatible(_source(non_commercial=True)) is False


@pytest.mark.parametrize('answer', ['', 'no', 'never', 'q'])
def test_anything_but_yes_is_a_refusal(monkeypatch, answer):
    _attended(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a: answer)
    assert require_usage_compatible(_source(non_commercial=True)) is False


def test_proceeding_at_the_prompt_returns_true(monkeypatch):
    _attended(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a: 'y')
    assert require_usage_compatible(_source(non_commercial=True)) is True


def test_yes_does_not_record_a_standing_decision(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        'openplaces.io.usage_profile.set_usage_override',
        lambda *a, **k: recorded.setdefault('called', (a, k)),
    )
    _attended(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a: 'y')
    require_usage_compatible(_source(non_commercial=True))
    assert not recorded


def test_always_records_a_standing_decision(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        'openplaces.io.usage_profile.set_usage_override',
        lambda source, compatible, reason=None: recorded.update({source: compatible}),
    )
    _attended(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a: 'a')
    assert require_usage_compatible(_source(non_commercial=True)) is True
    assert recorded == {'a_source': True}


def test_an_answer_is_remembered_for_the_rest_of_the_process(monkeypatch):
    _attended(monkeypatch)
    asked = []
    monkeypatch.setattr('builtins.input', lambda *a: asked.append(1) or 'y')
    require_usage_compatible(_source(non_commercial=True))
    require_usage_compatible(_source(non_commercial=True))
    assert len(asked) == 1


def test_a_remembered_refusal_is_not_re_asked(monkeypatch):
    _attended(monkeypatch)
    asked = []
    monkeypatch.setattr('builtins.input', lambda *a: asked.append(1) or 'n')
    assert require_usage_compatible(_source(non_commercial=True)) is False
    assert require_usage_compatible(_source(non_commercial=True)) is False
    assert len(asked) == 1


@pytest.mark.parametrize(
    'declared, expected',
    [
        ({'restricted': True, 'encrypted_at_rest': True}, True),
        ({'restricted': True, 'offline_only': True}, True),
        ({'restricted': True}, False),
        ({'encrypted_at_rest': True}, False),
    ],
)
def test_and_of_or_environment_evaluation(monkeypatch, declared, expected):
    """ZTRAX's shape: restricted AND (encrypted-at-rest OR offline-only)."""
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_profile',
        lambda: _profile(**declared),
    )
    _attended(monkeypatch, False)
    source = _source(environment=['restricted', ['encrypted_at_rest', 'offline_only']])
    if expected:
        assert require_usage_compatible(source) is True
    else:
        with pytest.raises(UsageProfileMismatchError):
            require_usage_compatible(source)


@pytest.mark.parametrize(
    'interest, admin_id, expected',
    [
        ('US-MA', 'US-MA-MI', True),  # ancestor interest covers the county
        ('US-MA-MI', 'US-MA', True),  # statewide file serves the county
        ('US-MA', 'US-NC', False),
    ],
)
def test_admin_interest_jurisdiction_matching(
    monkeypatch, interest, admin_id, expected
):
    monkeypatch.setattr(
        'openplaces.io.usage_profile.get_usage_profile',
        lambda: _profile(admin_interests=[interest]),
    )
    _attended(monkeypatch, False)
    source = _source(admin_interest=True)
    if expected:
        assert require_usage_compatible(source, admin_id=admin_id) is True
    else:
        with pytest.raises(UsageProfileMismatchError):
            require_usage_compatible(source, admin_id=admin_id)


def test_every_recipe_with_a_usage_requirement_names_its_source():
    """A requirement keyed on a missing source_id would collide across
    recipes in the remembered-answer and override stores."""
    for path in RECIPES.rglob('*.yaml'):
        text = path.read_text(encoding='utf-8')
        if 'usage_requirement' not in text:
            continue
        recipe = yaml.safe_load(text)
        block = recipe.get('entity') or recipe.get('dataset') or {}
        source = block.get('source') or {}
        assert 'usage_requirement' in source, path.name
        assert source.get('source_id'), path.name
        # And the requirement itself must construct cleanly
        requirement = UsageRequirement(**source['usage_requirement'])
        assert not requirement.is_empty(), path.name


def test_a_terminal_alone_does_not_make_a_test_run_attended(monkeypatch):
    """The gate asks whether anyone can answer, not whether a tty exists.

    `can_prompt` is False inside a test session (and in CI) whatever
    stdin looks like, so a suite that happens to run in a terminal is
    still refused rather than blocking on `input`.
    """
    monkeypatch.setattr(
        'builtins.input', lambda *a: pytest.fail('prompted a test session')
    )
    with pytest.raises(UsageProfileMismatchError):
        require_usage_compatible(_source(non_commercial=True))


def test_end_of_input_at_the_prompt_is_a_refusal(monkeypatch):
    """Ctrl-D closes the prompt without answering it.

    Not an answer, so not a confirmation: the download is skipped, the
    same as any other non-answer.
    """
    _attended(monkeypatch)

    def _eof(*args):
        raise EOFError

    monkeypatch.setattr('builtins.input', _eof)
    assert require_usage_compatible(_source(non_commercial=True)) is False


def test_a_keyboard_interrupt_still_stops_the_run(monkeypatch):
    """Ctrl-C means stop, not skip this download."""
    _attended(monkeypatch)

    def _interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr('builtins.input', _interrupt)
    with pytest.raises(KeyboardInterrupt):
        require_usage_compatible(_source(non_commercial=True))


def test_a_failure_to_record_the_answer_does_not_lose_it(monkeypatch):
    """The person answered; failing to write it down holds for this run."""
    _attended(monkeypatch)
    monkeypatch.setattr('builtins.input', lambda *a: 'a')

    def _fails(*args, **kwargs):
        raise OSError('config is read-only')

    monkeypatch.setattr('openplaces.io.usage_profile.set_usage_override', _fails)
    with pytest.warns(UserWarning, match='holds for this run only'):
        assert require_usage_compatible(_source(non_commercial=True)) is True


def test_no_recipe_key_can_declare_a_profile_match():
    """The gate reads the user's config, never the recipe.

    A recipe records `usage_requirement` (a fact about the source) and
    nothing else: there is no recipe-side field that satisfies, waives,
    or pre-answers the requirement, so no committed file can decide for
    the person running it.
    """
    import inspect

    signature = inspect.signature(require_usage_compatible)
    assert set(signature.parameters) == {
        'source',
        'recipe_id',
        'admin_id',
        'verbose',
    }
