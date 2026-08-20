"""openplaces identifies itself to the servers it downloads from.

The download path used to send ``User-Agent: Mozilla/5.0`` -- an automated
client presenting as a browser. The remedy is not "send nothing" (some
servers reject the bare python-requests agent with a 403) but a named agent
carrying the project, its URL, and an operator handle a provider can ask
about, plus the driving AI agent when there is one.

These tests pin that contract: no browser impersonation anywhere, a
well-formed identity when one is configured, and an honest 'unidentified'
when none is.
"""

from pathlib import Path

import pytest
import yaml

from openplaces.config import (
    PROJECT_URL,
    build_user_agent,
    detect_agent,
    write_identity,
)
from openplaces.core.constants import VERSION

SRC = Path(__file__).resolve().parents[2] / 'src' / 'openplaces'


def test_no_browser_impersonation_remains_in_the_source():
    """No request path may present openplaces as a browser."""
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob('*.py')
        if 'Mozilla/' in path.read_text(encoding='utf-8')
    ]

    assert offenders == [], f'browser User-Agent spoofing in: {offenders}'


def test_user_agent_names_the_project_its_url_and_the_operator():
    agent = build_user_agent('ada', 'some-university')

    assert agent == (f'openplaces/{VERSION} (+{PROJECT_URL}; ada@some-university)')


@pytest.mark.parametrize(
    ('nickname', 'place'),
    [(None, None), ('', ''), ('   ', None)],
)
def test_unset_identity_is_reported_honestly(nickname, place):
    """Absent is 'unidentified', never a guess and never a browser."""
    assert build_user_agent(nickname, place).endswith('; unidentified)')


def test_half_an_identity_still_identifies():
    """A nickname with no place is worth more than nothing."""
    assert build_user_agent('ada', None).endswith('; ada)')
    assert build_user_agent(None, 'some-university').endswith('; some-university)')


def test_an_agent_driving_the_run_is_disclosed():
    agent = build_user_agent('ada', 'some-university', 'claude-code')

    assert agent.endswith('; ada@some-university; agent: claude-code)')


def test_detect_agent_reads_the_environment(monkeypatch):
    monkeypatch.delenv('CLAUDECODE', raising=False)
    monkeypatch.delenv('CLAUDE_CODE', raising=False)
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    monkeypatch.setenv('OPENPLACES_AGENT', 'some-other-agent')

    assert detect_agent() == 'some-other-agent'


def test_detect_agent_is_none_when_a_person_is_driving(monkeypatch):
    from openplaces.config import AGENT_ENV_VARS

    for var in AGENT_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    assert detect_agent() is None


def test_write_identity_preserves_the_rest_of_the_config(tmp_path):
    """dev.py writes an identity before directories exist; both survive."""
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(
        yaml.dump({'directories': {'data_root': '/somewhere'}}), encoding='utf-8'
    )

    write_identity(config_path, 'ada', 'some-university')
    written = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    assert written['directories'] == {'data_root': '/somewhere'}
    assert written['identity'] == {'nickname': 'ada', 'place': 'some-university'}


def test_write_identity_creates_a_file_that_does_not_suppress_directory_setup(
    tmp_path,
):
    """The first-use prompt keys on 'directories', not on the file existing.

    `dev.py setup` records an identity before anyone has chosen data
    directories, so a file already present must not be read as "already
    configured".
    """
    from openplaces.config import OpenPlacesConfig

    config_path = tmp_path / 'config.yaml'
    write_identity(config_path, 'ada', 'some-university')

    config = OpenPlacesConfig.__new__(OpenPlacesConfig)
    config.user_config_path = config_path

    assert config._user_config_has('identity')
    assert not config._user_config_has('directories')


def test_blank_answers_clear_the_identity(tmp_path):
    config_path = tmp_path / 'config.yaml'

    write_identity(config_path, '  ', '')
    written = yaml.safe_load(config_path.read_text(encoding='utf-8'))

    assert written['identity'] == {'nickname': None, 'place': None}
