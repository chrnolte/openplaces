"""The user's own config file and the prompts that fill it: merging a
key into it, the identity and usage-profile writers, whether a
prompt can be answered at all, agent detection.
"""

import os
import sys
from pathlib import Path

import yaml

from openplaces.config.identity import (
    AGENT_ENV_VARS,
)


class ConfigFileUnreadableError(OSError):
    """Raised when an existing user config file cannot be parsed.

    Its own class because "your config file is broken" is a different
    situation from "something went wrong writing it": the caller must
    stop rather than rewrite, since rewriting would discard every other
    setting the file holds.
    """


def merge_user_config(config_path, key: str, value) -> None:
    """Set one top-level key in a user config file, creating it if needed.

    Reads and rewrites the whole file rather than appending, so a key can
    be recorded before the directory setup has run (dev.py) or changed
    long afterwards without disturbing anything else in it.

    Parameters
    ----------
    config_path : str or pathlib.Path
        User config file to update.
    key : str
        Top-level key to set.
    value : Any
        Value to store under *key*, replacing whatever is there.

    Raises
    ------
    ConfigFileUnreadableError
        When the file exists but cannot be parsed as a YAML mapping.
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if config_path.exists():
        # A file that cannot be read is not an empty file.
        # Treating it as one rewrote the whole config with just this
        # key, so a stray tab in config.yaml plus one recorded
        # decision used to erase the user's directories and
        # identity.
        try:
            with open(config_path, encoding='utf-8') as f:
                existing = yaml.safe_load(f)
        except (OSError, yaml.YAMLError) as error:
            raise ConfigFileUnreadableError(
                f'Cannot update {config_path}: it exists but could not be '
                f'read ({error}). Fix or move the file by hand; rewriting '
                'it here would discard everything else it holds.'
            ) from error
        if existing is None:
            existing = {}
        elif not isinstance(existing, dict):
            raise ConfigFileUnreadableError(
                f'Cannot update {config_path}: its top level is a '
                f'{type(existing).__name__}, not a mapping. Fix or move '
                'the file by hand.'
            )

    existing[key] = value
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(
            existing, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def write_identity(config_path, nickname: str | None, place: str | None) -> None:
    """Merge an identity into a user config file, creating it if needed."""
    merge_user_config(
        config_path,
        'identity',
        {
            'nickname': (nickname or '').strip() or None,
            'place': (place or '').strip() or None,
        },
    )


class DataRootNotSetError(RuntimeError):
    """Raised when no data directory has been configured.

    Its own class because "you have not finished setting up" is a
    different situation from "something went wrong", and a caller
    (a notebook, `dev.py`) may want to offer setup rather than a
    traceback.
    """


def can_prompt() -> bool:
    """Return True when something is able to answer a prompt.

    Not the same question as "is stdin a terminal", which is what this
    used to ask. A Jupyter kernel is interactive and is not a tty:
    `ipykernel` implements `input` over its stdin channel and routes it
    to the frontend, so a notebook can answer perfectly well. Asking
    `isatty` shut out the one surface the project designates for
    interactive configuration.

    What must stay shut out is a run with nobody watching, where a
    blocking read is a hang rather than an error: CI, a container, a
    cluster job, a test session.

    Returns
    -------
    bool
        True for a terminal or a notebook kernel, False for an
        unattended run.
    """
    # An unattended runner says so in the environment. Checked first, so
    # a CI job that happens to allocate a tty is still refused.
    if os.environ.get('CI') or os.environ.get('PYTEST_CURRENT_TEST'):
        return False

    shell = sys.modules.get('IPython')
    if shell is not None:
        try:
            active = shell.get_ipython()
        except Exception:  # noqa: BLE001 - IPython present but not running
            active = None
        # ZMQInteractiveShell is the kernel behind Jupyter and qtconsole;
        # TerminalInteractiveShell is plain ipython, covered by isatty.
        if active is not None and type(active).__name__ == 'ZMQInteractiveShell':
            return True

    stdin = sys.stdin
    if stdin is None or getattr(stdin, 'closed', False):
        return False
    try:
        return bool(stdin.isatty())
    except (ValueError, OSError):
        # A detached or replaced stdin raises rather than answering.
        return False


_UNSET = object()

# Environment axes a usage profile can declare. Mirrored by
# `core.schema.UsageRequirement.ENV_FLAGS`; kept as a literal here so
# config stays importable without the schema module.
USAGE_ENV_FLAGS = ('licensed', 'restricted', 'encrypted_at_rest', 'offline_only')

USAGE_PROFILE_NOTICE = """\
What is this installation's usage context?

Some sources place conditions on who may download their data (for
example, non-commercial use only, or a restricted computing
environment). Declaring your context once lets openplaces check those
conditions before an automatic download, instead of asking every time.

Nothing is verified or sent anywhere; this is a self-declaration stored
in your own config, and every answer can be changed later with
`python -m openplaces.config --set-usage-profile`. Leave anything blank
to keep it undeclared.\
"""


def prompt_usage_profile() -> dict:
    """Show the usage-context notice and ask for a profile interactively.

    Returns
    -------
    dict
        A profile shaped like the `usage_profile` config default. Blank
        answers leave `commercial` undeclared (None) and the other axes
        at their defaults.
    """
    print('\n' + '-' * 70)
    print(USAGE_PROFILE_NOTICE)
    print()

    answer = input('Used commercially? [y/n, Enter to leave undeclared]: ')
    answer = answer.strip().lower()
    commercial = {'y': True, 'yes': True, 'n': False, 'no': False}.get(answer)

    print(
        '\nEnvironment properties, if any apply (used by sources whose '
        'terms\nrequire one): [l]icensed  [r]estricted  [e]ncrypted-at-rest  '
        '[o]ffline-only'
    )
    letters = input('Letters that apply (e.g. "re", Enter for none): ')
    letters = letters.strip().lower()
    environment = {
        'licensed': 'l' in letters,
        'restricted': 'r' in letters,
        'encrypted_at_rest': 'e' in letters,
        'offline_only': 'o' in letters,
    }

    interests = input(
        'Admin ids you declare an interest in (comma-separated, e.g. '
        'US-MA, Enter for none): '
    ).strip()
    admin_interests = [part.strip() for part in interests.split(',') if part.strip()]

    return {
        'commercial': commercial,
        'environment': environment,
        'admin_interests': admin_interests,
    }


def write_usage_profile(config_path, profile: dict) -> None:
    """Merge a usage profile into a user config file, creating it if needed."""
    merge_user_config(config_path, 'usage_profile', profile)


def detect_agent() -> str | None:
    """Return the name of the agent driving this run, or None.

    openplaces downloads from other people's servers, and a provider
    reading its logs is entitled to know whether a person or a piece of
    autonomous software is on the other end. Set ``OPENPLACES_AGENT`` to
    name an agent this function does not recognize.
    """
    for var, name in AGENT_ENV_VARS.items():
        value = os.environ.get(var)
        if not value:
            continue
        if name is None:
            return str(value).strip() or None
        return name
    return None


def _merge_nested(base: dict, override: dict) -> dict:
    """Recursively merge override into base in place (dicts merge, else replace)."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_nested(base[key], value)
        else:
            base[key] = value
    return base
