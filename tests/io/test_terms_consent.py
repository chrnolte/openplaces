"""No scraper, and no recipe, accepts a source's terms of use.

Clicking "I Agree" on a portal forms a contract, and the party to it is
whoever runs the download, not openplaces. The Wisconsin RETR scraper used
to default ``accept_terms=True`` and click through the gate unattended.

The replacement contract, pinned here: a person accepts or nothing is
accepted. A recipe may decline (worst case, a skipped download) but cannot
consent, because a committed recipe would bind everyone who runs it. A
standing "always" is available, but only as a choice made at the prompt and
recorded in that user's own config.
"""

import inspect
from pathlib import Path

import pytest

from openplaces.io.consent import (
    ConsentNotDelegableError,
    TermsNotAcceptedError,
    forget_terms_consent,
    require_terms_consent,
)

SRC = Path(__file__).resolve().parents[2] / 'src' / 'openplaces'
SCRAPERS = SRC / 'io' / 'scrapers'


@pytest.fixture(autouse=True)
def _no_standing_consent(monkeypatch):
    """Isolate from whatever this developer recorded in their own config."""
    monkeypatch.setattr('openplaces.io.consent.get_terms_consent', lambda source: None)


@pytest.fixture(autouse=True)
def _forget():
    forget_terms_consent()
    yield
    forget_terms_consent()


class _NoTty:
    """stdin of an unattended run: readable, but nobody is watching."""

    @staticmethod
    def isatty():
        return False


class _Tty:
    @staticmethod
    def isatty():
        return True


def test_a_recipe_may_decline_but_never_accept():
    """The asymmetry is the whole design: declining costs a download.

    Accepting binds the operator to someone else's contract, so a
    committed recipe cannot do it: `accept_terms: true` is refused
    outright rather than downgraded to a prompt, so a recipe that reads
    as though consent were handled cannot ship.
    """
    assert require_terms_consent('a source', accept_terms=False) is False

    with pytest.raises(ConsentNotDelegableError, match='not allowed'):
        require_terms_consent('a source', accept_terms=True)


def test_a_standing_decision_in_the_user_config_is_honored(monkeypatch):
    monkeypatch.setattr('openplaces.io.consent.get_terms_consent', lambda source: True)

    assert require_terms_consent('a portal') is True

    monkeypatch.setattr('openplaces.io.consent.get_terms_consent', lambda source: False)
    forget_terms_consent()

    assert require_terms_consent('a portal') is False


def test_always_records_a_standing_decision(monkeypatch):
    recorded = []

    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: 'a')
    monkeypatch.setattr(
        'openplaces.io.consent.set_terms_consent',
        lambda source, accepted: recorded.append((source, accepted)),
    )

    assert require_terms_consent('a portal') is True
    assert recorded == [('a portal', True)]


def test_yes_does_not_record_a_standing_decision(monkeypatch):
    """[y] is for this run; only [a] is a commitment worth persisting."""
    recorded = []

    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: 'y')
    monkeypatch.setattr(
        'openplaces.io.consent.set_terms_consent',
        lambda source, accepted: recorded.append((source, accepted)),
    )

    assert require_terms_consent('a portal') is True
    assert recorded == []


def test_an_unattended_run_raises_rather_than_accepting(monkeypatch):
    monkeypatch.setattr('sys.stdin', _NoTty())

    with pytest.raises(TermsNotAcceptedError) as excinfo:
        require_terms_consent('a portal', terms_url='https://example.invalid/terms')

    message = str(excinfo.value)
    assert 'interactively' in message, 'must say how consent is created'
    assert 'no recipe setting' in message.lower(), 'must not point at a recipe'
    assert 'https://example.invalid/terms' in message


def test_declining_at_the_prompt_returns_false(monkeypatch, capsys):
    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: 'n')

    assert require_terms_consent('a portal') is False
    assert 'skipping' in capsys.readouterr().out


@pytest.mark.parametrize('answer', ['', ' ', 'no', 'maybe', 'sure', 'always?'])
def test_anything_but_yes_is_a_refusal(monkeypatch, answer):
    """Ambiguity resolves against agreeing, which is the safe direction."""
    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: answer)

    assert require_terms_consent('a portal') is False


def test_accepting_at_the_prompt_returns_true(monkeypatch):
    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: 'y')

    assert require_terms_consent('a portal') is True


def test_an_answer_is_remembered_for_the_rest_of_the_process(monkeypatch):
    """A recipe downloading twelve months asks once, not twelve times."""
    asked = []

    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: asked.append(1) or 'y')

    assert require_terms_consent('a portal') is True
    assert require_terms_consent('a portal') is True
    assert len(asked) == 1

    # A different source is a different contract, so it asks again.
    assert require_terms_consent('another portal') is True
    assert len(asked) == 2


def test_a_remembered_refusal_is_not_re_asked(monkeypatch):
    answers = iter(['n', 'y'])
    monkeypatch.setattr('sys.stdin', _Tty())
    monkeypatch.setattr('builtins.input', lambda _: next(answers))

    assert require_terms_consent('a portal') is False
    assert require_terms_consent('a portal') is False


def test_no_scraper_defaults_to_accepting_terms():
    """The regression this whole module exists to prevent."""
    offenders = []
    for path in SCRAPERS.glob('*.py'):
        source = path.read_text(encoding='utf-8')
        if 'accept_terms: bool = True' in source:
            offenders.append(path.name)

    assert offenders == [], f'scrapers accepting terms by default: {offenders}'


def test_the_wisconsin_scraper_asks_before_opening_a_browser():
    """Consent is settled in `fetch`, so a refusal costs no browser launch."""
    import importlib.util

    path = SCRAPERS / 'US-WI_transaction-widor-2026_scraper.py'
    spec = importlib.util.spec_from_file_location('_widor_scraper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    default = inspect.signature(module.fetch).parameters['accept_terms'].default
    assert default is None, 'None means "nobody has decided yet", not "yes"'

    # The browser worker no longer carries the flag: by the time it runs,
    # the question has already been answered.
    assert 'accept_terms' not in inspect.signature(module._fetch_in_browser).parameters


def test_no_recipe_ships_a_pre_accepted_terms_gate():
    """A committed recipe must not accept terms for every future user.

    `accept_terms: true` now raises `ConsentNotDelegableError`, so a
    recipe setting it is broken rather than merely misleading. This scan
    catches it at test time instead of at download time. The Wisconsin
    recipe carried it only because it mirrored the scraper's old default.
    """
    import yaml

    recipes = SRC / 'recipes'
    offenders = []
    for path in recipes.rglob('*.yaml'):
        recipe = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        options = recipe.get('scraper_options') or {}
        if options.get('accept_terms'):
            offenders.append(path.name)

    assert offenders == [], (
        f'recipes setting `accept_terms: true`: {offenders}. The key raises; '
        'record consent per user by answering [a] at the prompt.'
    )
