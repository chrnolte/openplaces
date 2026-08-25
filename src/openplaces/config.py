"""
openplaces configuration management

Hierarchical configuration system with interactive first-use setup.

Priority (highest to lowest):
1. User config (~/.config/openplaces/<username>.yaml) - User-specific overrides
2. Project config (./openplaces.yaml) - Project defaults
3. Built-in defaults - Fallback values

On first use, users are prompted to customize directory paths or accept defaults.
"""

import argparse
import copy
import getpass
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_config_dir

from openplaces.core.constants import (
    CRS,
    GEO_MIN_AREA_M2,
    NEVER_DELETE,
    RETENTION_CLASSES,
    STANDARD_DIRS,
    VERSION,
)

# Application name and author to locate user configuration files
APPNAME = 'openplaces'
APPAUTHOR = 'placeslab'

# Project page named in the User-Agent, so a data provider reading its
# logs can find out what openplaces is without contacting anyone.
PROJECT_URL = 'https://openplaces.io'

# Environment variables meaning "an AI coding agent is driving this
# run", mapped to the name reported in the User-Agent. Checked in
# order; the first match wins. OPENPLACES_AGENT is the explicit escape
# hatch for an agent this list does not know.
AGENT_ENV_VARS = {
    'OPENPLACES_AGENT': None,  # value is the agent name itself
    'CLAUDECODE': 'claude-code',
    'CLAUDE_CODE': 'claude-code',
    'CURSOR_AGENT': 'cursor',
    'AIDER_MODEL': 'aider',
    'CODEX_SANDBOX': 'codex',
    'GITHUB_ACTIONS': 'github-actions',
}

__all__ = [
    'cfg',
    'get_config',
    'show_config',
    'show_credentials',
    'reset_config',
    'edit_config',
    'reload_config',
    'set_identity',
    'get_terms_consent',
    'set_terms_consent',
    'get_usage_profile',
    'set_usage_profile',
    'get_usage_override',
    'set_usage_override',
    'merge_user_config',
    'detect_agent',
    'OpenPlacesConfig',
    'DataRootNotSetError',
]


IDENTITY_NOTICE = f"""\
How should openplaces identify itself?

openplaces downloads from public servers run by other people -- county GIS
portals, state agencies, national statistical offices. Every request it
makes carries a User-Agent naming the project, so an operator seeing
unexpected load knows what the traffic is and has someone to ask.

You are not registering anything and nothing is sent to this project. Pick
any nickname and place you are willing to have appear in a server log; a
work handle and an institution or city is the usual choice. Leave it blank
to stay unidentified.

  openplaces/{VERSION} (+{PROJECT_URL}; ada@some-university)\
"""


def build_user_agent(
    nickname: str | None,
    place: str | None,
    agent: str | None = None,
) -> str:
    """Assemble the User-Agent string for a given identity.

    Shaped like a conventional crawler identity -- product token, then a
    parenthesized comment holding the project URL and a contact handle --
    because that is the form a server operator's log tooling already knows
    how to read.

    Parameters
    ----------
    nickname : str or None
        Self-chosen handle. Never verified and never an email address.
    place : str or None
        Institution, city, or organization the nickname belongs to.
    agent : str or None
        Name of the AI coding agent driving the run, appended so a
        provider can tell autonomous traffic from a person at a keyboard.

    Returns
    -------
    str
        e.g. ``openplaces/0.1.0 (+https://openplaces.io; ada@some-university)``
    """
    nickname = (nickname or '').strip()
    place = (place or '').strip()
    if nickname and place:
        who = f'{nickname}@{place}'
    else:
        # An installation that set neither reports 'unidentified',
        # honest and still better than impersonating a browser.
        who = nickname or place or 'unidentified'

    parts = [f'+{PROJECT_URL}', who]
    if agent:
        parts.append(f'agent: {agent}')
    return f'openplaces/{VERSION} ({"; ".join(parts)})'


def prompt_identity() -> tuple[str, str]:
    """Show the identity notice and ask for a nickname and place.

    Returns
    -------
    tuple of str
        (nickname, place), either possibly empty. An empty nickname skips
        the place question: half an identity is not worth a second prompt.
    """
    print('\n' + '-' * 70)
    print(IDENTITY_NOTICE)
    print()

    nickname = input('Nickname (Enter to stay unidentified): ').strip()
    place = input('Place (university, city, or org): ').strip() if nickname else ''
    print(f'\nRequests will be sent as:\n  {build_user_agent(nickname, place)}')
    return nickname, place


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
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if config_path.exists():
        try:
            with open(config_path, encoding='utf-8') as f:
                existing = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            existing = {}

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


def get_terms_consent(source: str) -> bool | None:
    """Return the standing decision for *source*, or None if never asked.

    Parameters
    ----------
    source : str
        Key the decision was recorded under (a recipe id or portal name).

    Returns
    -------
    bool or None
        True when this user chose to always accept that source's terms,
        False when they chose to always decline, None when no standing
        decision exists and they should be asked.
    """
    recorded = (cfg.get('consent') or {}).get('terms') or {}
    entry = recorded.get(source)
    if isinstance(entry, dict):
        return entry.get('accepted')
    return entry if isinstance(entry, bool) else None


def set_terms_consent(source: str, accepted: bool) -> None:
    """Record a standing decision about one source's terms of use.

    Stored in this user's own config, never in a recipe: accepting terms
    is a commitment by the person running the download, and a committed
    recipe would extend it to everyone who ever runs that recipe.

    Parameters
    ----------
    source : str
        Key to record the decision under (a recipe id or portal name).
    accepted : bool
        True to accept that source's terms from now on, False to decline
        from now on.
    """
    from datetime import date

    recorded = dict((cfg.get('consent') or {}).get('terms') or {})
    recorded[source] = {
        'accepted': bool(accepted),
        'recorded': date.today().isoformat(),
    }
    merge_user_config(cfg.user_config_path, 'consent', {'terms': recorded})
    reload_config()


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


def set_identity(nickname: str | None, place: str | None) -> str:
    """Set the nickname and place openplaces identifies itself by.

    Parameters
    ----------
    nickname : str or None
        Self-chosen handle; None or empty clears the identity.
    place : str or None
        Institution, city, or organization.

    Returns
    -------
    str
        The User-Agent that will now be sent.
    """
    write_identity(cfg.user_config_path, nickname, place)
    reload_config()
    return cfg.user_agent


# Sentinel for set_usage_profile: distinguishes "not passed" from an
# explicit None, so a partial CLI call never clobbers the other fields.
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


def get_usage_profile() -> dict:
    """Return the declared usage profile, complete over the defaults."""
    return cfg.usage_profile


def set_usage_profile(
    commercial=_UNSET,
    licensed=_UNSET,
    restricted=_UNSET,
    encrypted_at_rest=_UNSET,
    offline_only=_UNSET,
    admin_interests=_UNSET,
) -> dict:
    """Update the declared usage profile, leaving unpassed fields alone.

    Partial-update semantics, a deliberate departure from
    `set_identity`'s full replace: the profile has six independent axes,
    and incremental setup across separate calls (e.g. cluster job
    scripts) must not reset the axes it does not mention.

    Parameters
    ----------
    commercial : bool or None, optional
        Whether this installation is used commercially. None returns it
        to undeclared. Not passing it leaves the current value.
    licensed, restricted, encrypted_at_rest, offline_only : bool, optional
        Environment axes. Not passing one leaves its current value.
    admin_interests : list of str, optional
        Replaces the whole declared-interest list; an empty list clears
        it. Not passing it leaves the current list.

    Returns
    -------
    dict
        The resulting complete profile.
    """
    profile = cfg.usage_profile
    if commercial is not _UNSET:
        profile['commercial'] = commercial
    env_updates = {
        'licensed': licensed,
        'restricted': restricted,
        'encrypted_at_rest': encrypted_at_rest,
        'offline_only': offline_only,
    }
    for flag, value in env_updates.items():
        if value is not _UNSET:
            profile['environment'][flag] = bool(value)
    if admin_interests is not _UNSET:
        profile['admin_interests'] = list(admin_interests or [])
    write_usage_profile(cfg.user_config_path, profile)
    reload_config()
    return cfg.usage_profile


def get_usage_override(source_id: str) -> bool | None:
    """Return the standing usage decision for *source_id*, or None.

    Returns
    -------
    bool or None
        True when this user chose to proceed with that source despite a
        profile mismatch, False when they chose to always skip it, None
        when no standing decision exists and they should be asked.
    """
    recorded = cfg.get('usage_overrides') or {}
    entry = recorded.get(source_id)
    if isinstance(entry, dict):
        return entry.get('compatible')
    return entry if isinstance(entry, bool) else None


def set_usage_override(source_id: str, compatible: bool, reason: str | None = None):
    """Record a standing decision about one source's usage requirement.

    Stored in this user's own config, never in a recipe: overriding a
    usage condition is a judgment by the person running the download,
    and a committed recipe would extend it to everyone who runs it.

    Parameters
    ----------
    source_id : str
        Key to record the decision under (a source id or recipe id).
    compatible : bool
        True to proceed with this source from now on, False to always
        skip it.
    reason : str, optional
        Optional note stored alongside, e.g. which unmet condition the
        person judged inapplicable.
    """
    from datetime import date

    recorded = dict(cfg.get('usage_overrides') or {})
    entry = {
        'compatible': bool(compatible),
        'recorded': date.today().isoformat(),
    }
    if reason:
        entry['reason'] = reason
    recorded[source_id] = entry
    merge_user_config(cfg.user_config_path, 'usage_overrides', recorded)
    reload_config()


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


class OpenPlacesConfig:
    """Configuration manager with interactive first-use setup."""

    DEFAULTS = {
        'crs': CRS,
        'geo_min_area_m2': GEO_MIN_AREA_M2,
        # Who to say you are when downloading from someone else's
        # server. Both unset means the User-Agent says 'unidentified'
        # rather than guessing: a wrong identity is worse than none.
        'identity': {'nickname': None, 'place': None},
        # Standing decisions about third-party terms of use, by source.
        # Empty by default: consent is something a person gives, so it
        # can only ever arrive from this user's own config.
        'consent': {'terms': {}},
        # Self-declared usage context, checked against a source's own
        # recorded `usage_requirement` before an automatic download.
        # `commercial` is tri-state: None means undeclared, which is not
        # the same as False -- a non-commercial-only source stays gated
        # until someone actually declares.
        'usage_profile': {
            'commercial': None,
            'environment': {
                'licensed': False,
                'restricted': False,
                'encrypted_at_rest': False,
                'offline_only': False,
            },
            # Admin ids this installation declares an interest in
            # (e.g. ['US-MA']) -- self-declared, not verified.
            'admin_interests': [],
        },
        # Per-source standing overrides of a usage mismatch, set only by
        # a person answering [a] at the prompt -- never by a recipe. A
        # separate key from usage_profile itself, the same separation
        # identity/consent already draw.
        'usage_overrides': {},
        # Data lifecycle policy. Bucket-level overrides live directly under
        # 'retention' (e.g. retention: {cache: keep}); per-recipe overrides
        # under 'recipes'; cleanup behavior switches under 'cleanup'.
        'retention': {
            'cleanup': {
                'enabled': True,
                'honor_receipts': True,
                'include_images': False,
                'exclude_patterns': [],
            },
            'recipes': {},
        },
    }

    def __init__(
        self, project_config_path: Path | None = None, interactive: bool = True
    ):
        """
        Initialize configuration system.

        Parameters
        ----------
        project_config_path : Path, optional
            Path to project config file. If None, searches current directory.
        interactive : bool, default True
            Whether to prompt user for configuration on first use.
        """
        self.username = getpass.getuser()
        # Use the actual user folder name (matches Windows user directory)
        self.username_dir = Path.home().name
        self._set_platform_defaults()
        self.user_config_path = self._get_user_config_path()
        self.project_config_path = project_config_path or self._find_project_config()

        # Handle first-use setup. Skipped without an interactive terminal:
        # importing openplaces must never block on a prompt, or a headless
        # run (CI, a container, a cluster job, a first `pytest`) hangs or
        # dies reading stdin. Those environments get the defaults instead.
        #
        # The trigger is a missing 'directories' block rather than a
        # missing file, because `dev.py setup` may already have written
        # the file to record an identity before anyone chose directories.
        if interactive and not self._user_config_has('directories') and can_prompt():
            self._interactive_setup()

        self.config = self._load_hierarchical_config()
        self._validate_config()
        self._resolve_directories()

    def _set_platform_defaults(self):
        """Set platform-appropriate default directory paths."""
        # These defaults assume single-user mode
        # Will be updated during interactive setup if multi-user is chosen
        self.code_root = self._get_code_root()
        self.multi_user = False
        self.user_data_dir = None  # Will be set if multi-user chosen

        # Extract defaults from STANDARD_DIRS
        self.default_dirs = {
            key: info['default'] for key, info in STANDARD_DIRS.items()
        }

    def _get_code_root(self) -> Path:
        """Get package root directory (2 levels up from config.py)."""
        # config.py is in src/openplaces/
        return Path(__file__).resolve().parent.parent.parent

    def _get_user_config_path(self) -> Path:
        """Get path to user-specific configuration file."""
        return Path(user_config_dir(APPNAME, APPAUTHOR)) / 'config.yaml'

    def _find_project_config(self) -> Path | None:
        """Search for project configuration file in current directory."""
        for filename in ['openplaces.yaml', '.openplaces.yaml']:
            config_file = self.code_root / filename
            if config_file.exists():
                return config_file
        return None

    def _user_config_has(self, key: str) -> bool:
        """True when the user config file exists and defines *key*."""
        if not self.user_config_path.exists():
            return False
        return key in self._load_yaml_config(self.user_config_path)

    def setup(self, force: bool = False):
        """Choose data directories interactively, and show the result.

        The import-time prompt is deliberately quiet: `from openplaces
        import cfg` blocking on a dialog is a strange first experience,
        and it fires from any cell that imports. Calling this is the
        visible, re-runnable version, which is what a setup notebook
        wants: run the cell, answer, see the paths change.

        Parameters
        ----------
        force : bool, optional
            Re-run even when directories are already configured. Default
            False, which reports the current settings and changes
            nothing.

        Returns
        -------
        OpenPlacesConfig
            This object, reloaded, so a cell can end on `cfg.setup()`
            and render the result.
        """
        if not force and self._user_config_has('directories'):
            print(f'Directories are already configured in {self.user_config_path}')
            print('Call cfg.setup(force=True) to choose again.\n')
            self.show_paths()
            return self

        if not can_prompt():
            raise RuntimeError(
                'cfg.setup() needs somewhere to read an answer from, and '
                'this run has nowhere: no terminal, no notebook kernel, or '
                'CI/PYTEST_CURRENT_TEST is set. Use '
                'python -m openplaces.config --set-identity, or write the '
                'directories block into '
                f'{self.user_config_path} directly.'
            )

        self._interactive_setup()
        reload_config()
        return get_config()

    def show_paths(self):
        """Print the directories this configuration resolves to."""
        print(f'code_root: {self.code_root}')
        print(f'data_root: {self.data_root}')
        for key in sorted(self.config.get('directories', {})):
            print(f'  {key}: {self.config["directories"][key]}')

    def _interactive_setup(self):
        """Interactive first-use configuration setup."""
        print('\n' + '=' * 70)
        print('Welcome to openplaces!')
        print('=' * 70)
        print(f'\nUser: {self.username}')
        print('\nThis appears to be your first time using openplaces.')
        print("Let's set up your data directories.\n\n")
        print('Root directory for data, models, and reports:')
        print('Change to separate code and data directories.')
        print(f'Default (code directory): {self.code_root}')
        print()

        custom_root = input('Press Enter to accept, or specify custom path: ').strip()

        if custom_root:
            self.default_dirs['data_root'] = str(Path(custom_root).resolve())

        print('Choose your configuration mode:')
        print('-' * 70)
        print(
            '(a) Single-user: Individual installation on a personal machine\n'
            '                 No user-specific subfolders for processed data & models.'
            '\n(b) Multi-user:  For teams sharing data folders and infrastructure.\n'
            '                 User-specific subfolders for processed data & models.'
        )
        print()

        mode_response = input('Choose mode [a/b] (default: a): ').strip().lower()

        if mode_response == 'b':
            # Multi-user mode
            self.multi_user = True

            # Step 2: Choose user folder name
            print('\n' + '-' * 70)
            print('User folder name:')
            print(f'Default: {self.username_dir} (from your system user folder)')
            print()

            custom_name = input('Press Enter to accept, or type custom name: ').strip()
            self.user_data_dir = custom_name if custom_name else self.username_dir

            # Update directories with user subfolder
            user_subdir = f'{self.user_data_dir}'
            for key, dir_info in STANDARD_DIRS.items():
                if key == 'data_root':
                    continue  # Don't modify root
                if dir_info['shared']:
                    # Keep shared directories as-is
                    self.default_dirs[key] = dir_info['default']
                else:
                    # Add user subfolder to user-specific directories
                    default_path = dir_info['default']
                    if default_path.startswith('data/'):
                        # Replace 'data/' with 'data/{user}/'
                        self.default_dirs[key] = default_path.replace(
                            'data/', f'data/{user_subdir}/', 1
                        )
                    elif default_path in ['models', 'reports']:
                        self.default_dirs[key] = f'{default_path}/{self.user_data_dir}'
                    else:
                        self.default_dirs[key] = default_path
        else:
            # Single-user mode (already set in defaults)
            self.multi_user = False
            self.user_data_dir = None

        # Step 3: Offer custom directory paths
        print('\n' + '-' * 70)
        print('Directory paths:')
        print('(a) Use defaults (recommended)')
        print('(b) Customize each directory path')
        print()

        customize = input('Choose option [a/b] (default: a): ').strip().lower()

        if customize == 'b':
            self._custom_directory_setup()
            self._interactive_identity()
            self._interactive_usage_profile()
            return  # _custom_directory_setup already creates the config

        # Show final directory structure
        print('\n' + '-' * 70)
        if self.multi_user:
            print(f'Directory structure (multi-user, folder: {self.user_data_dir}):')
        else:
            print('Directory structure (single-user):')
        print('-' * 70)

        for key, path in self.default_dirs.items():
            info = STANDARD_DIRS[key]
            is_user_specific = not info['shared']
            marker = '👤' if is_user_specific else '🌍'
            print(f'  {marker} {key:8s}: {path}')
            print(f'       {info["description"]}')

        # Final confirmation
        print('\n' + '-' * 70)
        response = input('\nAccept this configuration? [Y/n]: ').strip().lower()

        if response in ['n', 'no']:
            print('\nConfiguration cancelled. Please run again to reconfigure.')
            print('Or manually create: ~/.config/openplaces/config.yaml')
            sys.exit(0)

        self._create_user_config(self.default_dirs)
        self._interactive_identity()
        self._interactive_usage_profile()
        print(f'\nConfiguration saved to:\n\n  {self.user_config_path}\n')
        print('You can edit this file anytime to change your settings.')

        print('\n' + '=' * 70 + '\n')

    def _interactive_identity(self):
        """Ask how this installation should identify itself to providers."""
        nickname, place = prompt_identity()
        write_identity(self.user_config_path, nickname, place)

    def _interactive_usage_profile(self):
        """Ask for the usage context sources with conditions are checked against."""
        write_usage_profile(self.user_config_path, prompt_usage_profile())

    def _custom_directory_setup(self):
        """Allow user to customize directory paths."""
        print('\n' + '=' * 70)
        print('Custom Directory Setup')
        print('=' * 70)
        print('Press Enter to accept the default for each directory.\n')

        custom_dirs = {}
        for key, info in STANDARD_DIRS.items():
            default_path = self.default_dirs[key]
            desc = info['description']

            print(f'\n{key}: {desc}')
            print(f'  Default: {default_path}')
            custom = input('  Custom path (or Enter for default): ').strip()

            if custom:
                custom_dirs[key] = custom
            else:
                custom_dirs[key] = default_path

        self._create_user_config(custom_dirs)

    def _create_user_config(self, directories: dict[str, str]):
        """Create user configuration file with specified directories."""
        self.user_config_path.parent.mkdir(parents=True, exist_ok=True)

        config_content = {
            '_comment': f'openplaces user configuration for {self.username}',
            '_note': 'This file has highest priority and overrides project defaults '
            '(openplaces.yaml).',
            'directories': directories,
        }

        with open(self.user_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_content,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def _load_yaml_config(self, path: Path) -> dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(path, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                # Remove comment fields
                return {k: v for k, v in config.items() if not k.startswith('_')}
        except Exception as e:
            print(f'Warning: Could not load config from {path}: {e}')
            return {}

    def _load_hierarchical_config(self) -> dict[str, Any]:
        """Load configuration from all sources with correct priority."""
        # Start with built-in defaults (deep copy: 'retention' is nested and
        # must not leak per-instance mutations back into class-level DEFAULTS)
        config = copy.deepcopy(self.DEFAULTS)
        config['directories'] = self.default_dirs.copy()

        # Load project config (lower priority)
        if self.project_config_path and self.project_config_path.exists():
            project_config = self._load_yaml_config(self.project_config_path)
            if 'directories' in project_config:
                config['directories'].update(project_config.pop('directories'))
            if 'retention' in project_config:
                _merge_nested(config['retention'], project_config.pop('retention'))
            config.update(project_config)

        # Load user config (highest priority)
        if self.user_config_path.exists():
            user_config = self._load_yaml_config(self.user_config_path)
            if 'directories' in user_config:
                config['directories'].update(user_config.pop('directories'))
            if 'retention' in user_config:
                _merge_nested(config['retention'], user_config.pop('retention'))
            config.update(user_config)

        return config

    def _validate_config(self):
        """Validate configuration values."""
        retention = self.config.get('retention') or {}
        for key, value in retention.items():
            if key in ('cleanup', 'recipes'):
                continue
            if key not in STANDARD_DIRS:
                raise ValueError(
                    f"Unknown bucket '{key}' in retention config. "
                    f'Valid buckets: {sorted(STANDARD_DIRS)}'
                )
            if value not in RETENTION_CLASSES:
                raise ValueError(
                    f"Invalid retention class '{value}' for bucket '{key}'. "
                    f'Valid classes: {RETENTION_CLASSES}'
                )
            if key in NEVER_DELETE and value != 'keep':
                raise ValueError(
                    f"Bucket '{key}' is protected (NEVER_DELETE) and cannot "
                    f"be marked '{value}'."
                )
        for recipe_id, value in (retention.get('recipes') or {}).items():
            if value not in RETENTION_CLASSES:
                raise ValueError(
                    f"Invalid retention class '{value}' for recipe "
                    f"'{recipe_id}'. Valid classes: {RETENTION_CLASSES}"
                )

    def retention_for(
        self,
        data_dir: str,
        recipe_id: str | None = None,
        recipe_retention: str | None = None,
    ) -> str:
        """Resolve the retention class for an output in a data directory.

        Resolution order (later wins): the bucket default in STANDARD_DIRS,
        the bucket override in the config's retention block, the recipe's
        own save_to.retention, and the per-recipe override in
        retention.recipes. NEVER_DELETE buckets always resolve to 'keep'.

        Parameters
        ----------
        data_dir : str
            Bucket name from STANDARD_DIRS (e.g. 'cache', 'core').
        recipe_id : str, optional
            Recipe ID, for per-recipe overrides in retention.recipes.
        recipe_retention : str, optional
            The recipe's own save_to.retention value, if any.

        Returns
        -------
        str
            One of RETENTION_CLASSES.
        """
        if data_dir in NEVER_DELETE:
            return 'keep'
        retention = self.config.get('retention') or {}
        value = STANDARD_DIRS.get(data_dir, {}).get('retention', 'keep')
        value = retention.get(data_dir, value)
        if recipe_retention is not None:
            value = recipe_retention
        if recipe_id is not None:
            value = (retention.get('recipes') or {}).get(recipe_id, value)
        return value

    def _resolve_directories(self):
        """Resolve all configured directory paths.

        A missing ``data_root`` is refused rather than guessed. It used
        to fall back to the code directory, which does not fail: a fresh
        install quietly builds a second data store inside the checkout,
        which on a synced or version-controlled directory is worse than
        an error. Where somebody can answer, the setup runs; where
        nobody can, this raises and says how to set it.
        """
        if 'directories' not in self.config:
            return

        configured = self.config['directories'].get('data_root')
        if not configured:
            if can_prompt():
                self._interactive_setup()
                configured = (
                    self._load_yaml_config(self.user_config_path)
                    .get('directories', {})
                    .get('data_root')
                )
            if not configured:
                raise DataRootNotSetError(
                    '\n'.join(
                        [
                            'openplaces has no data_root, and will not invent one.',
                            '',
                            'Where should downloads, caches and outputs '
                            'live? It must be outside the code directory.',
                            '',
                            '  From a notebook or terminal:  cfg.setup()',
                            '  Non-interactively, write a directories block into',
                            f'  {self.user_config_path}',
                            '',
                            '      directories:',
                            '        data_root: /path/to/your/data',
                        ]
                    )
                )
            self.config['directories']['data_root'] = configured
        root = Path(configured)

        for dir_key, dir_value in self.config['directories'].items():
            if dir_key == 'data_root':
                # Shortcut to ensure the data root is a `Path` object
                dir_path = root
            else:
                dir_path = Path(dir_value)

            if not dir_path.is_absolute():
                # Relative paths are relative to root
                dir_path = root / dir_path
            self.config['directories'][dir_key] = dir_path.resolve()

    def get_dir(self, name: str) -> Path:
        """
        Get directory path by name.

        Parameters
        ----------
        name : str
            Directory name (e.g., 'raw', 'core', 'out')

        Returns
        -------
        Path
            Resolved directory path

        Raises
        ------
        KeyError
            If directory name not found
        """
        if name not in self.config.get('directories', {}):
            raise KeyError(f"Directory '{name}' not found in configuration")
        return self.config['directories'][name]

    def list_directories(self) -> dict[str, Path]:
        """Return dictionary of all configured directories."""
        return self.config.get('directories', {}).copy()

    def add_custom_directory(self, name: str, path: str, description: str = ''):
        """
        Add a custom directory to configuration.

        Parameters
        ----------
        name : str
            Directory key name
        path : str
            Directory path (relative or absolute)
        description : str, optional
            Description of directory purpose
        """
        dir_path = Path(path)
        if not dir_path.is_absolute():
            # Determine base directory for relative paths
            if 'data_root' in self.config['directories']:
                root = self.config['directories']['data_root']
            else:
                root = self.code_root
            dir_path = root / dir_path

        self.config['directories'][name] = dir_path.resolve()

        if description:
            STANDARD_DIRS[name] = {
                'default': path,
                'description': description,
                'shared': False,  # Custom dirs default to user-specific
            }

    @property
    def data_root(self) -> Path:
        """Data root directory."""
        return self.get_dir('data_root')

    @property
    def core_dir(self) -> Path:
        """Core processed data directory."""
        return self.get_dir('core')

    @property
    def external_dir(self) -> Path:
        """External data sources directory."""
        return self.get_dir('external')

    @property
    def rasters_dir(self) -> Path:
        """Root for rasters that recipes reference by relative path."""
        return self.get_dir('rasters')

    @property
    def raw_dir(self) -> Path:
        """Raw downloaded data directory."""
        return self.get_dir('raw')

    @property
    def cache_dir(self) -> Path:
        """Cache directory for intermediate files."""
        return self.get_dir('cache')

    @property
    def heap_dir(self) -> Path:
        """Heap directory for freshly unzipped data."""
        return self.get_dir('heap')

    @property
    def logs_dir(self) -> Path:
        """Heap directory for freshly unzipped data."""
        return self.get_dir('logs')

    @property
    def out_dir(self) -> Path:
        """Output data directory."""
        return self.get_dir('out')

    @property
    def share_dir(self) -> Path:
        """Shared data directory."""
        return self.get_dir('share')

    @property
    def models_dir(self) -> Path:
        """Models directory."""
        return self.get_dir('models')

    @property
    def reports_dir(self) -> Path:
        """Reports directory."""
        return self.get_dir('reports')

    def __getattr__(self, name: str) -> Any:
        """Enable attribute-style access to config values."""
        if name.startswith('dir_'):
            return self.get_dir(name[4:])
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"No configuration attribute '{name}'")

    @property
    def usage_profile(self) -> dict:
        """The declared usage profile, deep-merged over the defaults.

        The hierarchical loader only deep-merges 'directories' and
        'retention'; every other key is replaced wholesale, so a user
        config declaring just `commercial` would otherwise lose the
        `environment` and `admin_interests` keys callers rely on.
        """
        merged = copy.deepcopy(self.DEFAULTS['usage_profile'])
        return _merge_nested(merged, self.config.get('usage_profile') or {})

    @property
    def identity(self) -> dict:
        """Nickname and place this installation identifies itself by."""
        return dict(self.config.get('identity') or {})

    @property
    def user_agent(self) -> str:
        """User-Agent string sent with every request openplaces makes.

        Built from the configured nickname and place (see
        :func:`build_user_agent`), with the driving AI agent appended when
        one is detected. Recomputed per access rather than cached, because
        the agent is read from the environment.
        """
        identity = self.identity
        return build_user_agent(
            identity.get('nickname'), identity.get('place'), detect_agent()
        )

    @property
    def credentials_path(self) -> Path:
        """Path to the credentials file."""
        return Path(user_config_dir(APPNAME, APPAUTHOR)) / 'credentials.yaml'

    def get_credentials(self, service_id: str) -> dict:
        """Return credential dict for *service_id* from credentials.yaml.

        Parameters
        ----------
        service_id
            Key used in credentials.yaml (e.g. ``'google_streetview'``).

        Returns
        -------
        dict
            Credential fields for the service (e.g. ``{'api_key': '...'}``)

        Raises
        ------
        ValueError
            If *service_id* is absent from the credentials file, with a
            ready-to-paste YAML snippet showing what to add.
        """
        if not hasattr(self, '_credentials_cache'):
            if self.credentials_path.exists():
                with open(self.credentials_path, encoding='utf-8') as f:
                    self._credentials_cache = yaml.safe_load(f) or {}
            else:
                self._credentials_cache = {}
        creds = self._credentials_cache
        if service_id not in creds:
            raise ValueError(
                f"No credentials found for '{service_id}'.\n"
                f'Add an entry to {self.credentials_path}:\n\n'
                f'  {service_id}:\n'
                f'    api_key: YOUR_KEY_HERE\n'
            )
        return creds[service_id]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with optional default."""
        return self.config.get(key, default)

    def __repr__(self) -> str:
        return (
            f'OpenPlacesConfig(user={self.username}, '
            f'dirs={len(self.list_directories())})'
        )


_cfg = None


def get_config(interactive: bool = False) -> OpenPlacesConfig:
    """
    Get or create the global configuration instance.

    Parameters
    ----------
    interactive : bool, default False
        Whether to prompt user for configuration on first use.
        Should be False for normal imports, True for CLI setup commands.

    Returns
    -------
    OpenPlacesConfig
        The global configuration instance
    """
    global _cfg
    if _cfg is None:
        _cfg = OpenPlacesConfig(interactive=interactive)
    return _cfg


cfg = get_config()


def reset_config():
    """
    Delete user config file and reset to defaults.

    This will trigger the interactive setup on next import.
    """
    config_path = Path(user_config_dir(APPNAME, APPAUTHOR)) / 'config.yaml'

    if config_path.exists():
        print(f'Deleting user config: {config_path}')
        config_path.unlink()
        print('✓ Config file deleted')
        print(
            '\nNext time you import openplaces, you will be prompted to set up again.'
        )
    else:
        print(f'No user config file found at: {config_path}')
        print('Nothing to delete.')


def show_config():
    """Display current configuration file location and contents."""
    config_path = Path(user_config_dir(APPNAME, APPAUTHOR)) / 'config.yaml'

    print(f'User config location: {config_path}')
    print(f'Exists: {config_path.exists()}')

    if config_path.exists():
        print('\nContents:')
        print('-' * 70)
        with open(config_path, encoding='utf-8') as f:
            print(f.read())
        print('-' * 70)
    else:
        print('\nNo user config file exists (using project defaults).')


def show_credentials():
    """Show credentials file path and registered services (no secrets printed)."""
    path = Path(user_config_dir(APPNAME, APPAUTHOR)) / 'credentials.yaml'
    print(f'Credentials file: {path}')
    print(f'Exists: {path.exists()}')
    if path.exists():
        with open(path, encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        print(f'Services: {", ".join(data.keys()) or "(none)"}')


def edit_config():
    """Open user config file in default editor."""
    config_path = Path(user_config_dir(APPNAME, APPAUTHOR)) / 'config.yaml'

    if not config_path.exists():
        print(f'No config file exists at: {config_path}')
        response = input('Create a template config file? [y/N]: ').strip().lower()
        if response == 'y':
            config_path.parent.mkdir(parents=True, exist_ok=True)
            # Create a temporary config instance to get default directories
            temp_cfg = OpenPlacesConfig.__new__(OpenPlacesConfig)
            temp_cfg._set_platform_defaults()
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(temp_cfg._generate_user_config_template())
            print(f'✓ Created template config at: {config_path}')
        else:
            print('Cancelled.')
            return

    # Try to open with default editor
    system = platform.system()
    try:
        if system == 'Windows':
            subprocess.run(['notepad', str(config_path)])
        elif system == 'Darwin':  # macOS
            subprocess.run(['open', '-e', str(config_path)])
        else:  # Linux and others
            editor = os.environ.get('EDITOR', 'nano')
            subprocess.run([editor, str(config_path)])

        print(
            '\nAfter editing, reload the configuration:\n\n'
            '  from openplaces.config import reload_config\n'
            '  reload_config()'
        )
    except Exception as e:
        print(f'Could not open editor: {e}')
        print(f'Please manually edit: {config_path}')


def reload_config(interactive: bool = False) -> OpenPlacesConfig:
    """
    Reload configuration from disk.

    Parameters
    ----------
    interactive : bool, default False
        Whether to prompt if config doesn't exist.

    Returns
    -------
    OpenPlacesConfig
        Fresh configuration instance
    """
    global _cfg
    _cfg = OpenPlacesConfig(interactive=interactive)
    return _cfg


def main():
    """Command-line interface for config management."""
    parser = argparse.ArgumentParser(
        description='Manage openplaces configuration',
        epilog='Run without arguments to show current configuration.',
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Delete user config file and reset to defaults',
    )
    parser.add_argument(
        '--edit', action='store_true', help='Open config file in default editor'
    )
    parser.add_argument(
        '--show', action='store_true', help='Show current configuration'
    )
    parser.add_argument(
        '--reconfigure',
        action='store_true',
        help='Run interactive setup again (deletes existing config first)',
    )
    parser.add_argument(
        '--set-identity',
        nargs=2,
        metavar=('NICKNAME', 'PLACE'),
        help=(
            'Set the nickname and place sent in the User-Agent. Pass empty '
            'strings to stay unidentified. Used by dev.py setup, which asks '
            'for both before this environment exists to be imported from.'
        ),
    )
    parser.add_argument(
        '--user-agent',
        action='store_true',
        help='Print the User-Agent this installation sends, and exit',
    )
    profile_group = parser.add_argument_group(
        'usage profile',
        'Declare the usage context that sources with access conditions '
        'are checked against. Pass --set-usage-profile with any subset '
        'of the axis flags; unmentioned axes keep their current value.',
    )
    profile_group.add_argument(
        '--set-usage-profile',
        action='store_true',
        help='Apply the usage-profile flags below and print the result',
    )
    # choices=['true','false'] uniformly rather than store_true, so a
    # value can be corrected back off, not just set.
    for flag in ('commercial', *USAGE_ENV_FLAGS):
        profile_group.add_argument(
            f'--{flag.replace("_", "-")}',
            choices=['true', 'false'],
            default=None,
            help=f"Declare the '{flag}' axis",
        )
    profile_group.add_argument(
        '--admin-interest',
        action='append',
        metavar='ADMIN_ID',
        help=(
            'Declare an admin unit of interest (repeatable; the given '
            'list replaces the stored one, and a single empty string '
            'clears it)'
        ),
    )

    args = parser.parse_args()

    if args.set_usage_profile:
        updates = {}
        for flag in ('commercial', *USAGE_ENV_FLAGS):
            value = getattr(args, flag)
            if value is not None:
                updates[flag] = value == 'true'
        if args.admin_interest is not None:
            updates['admin_interests'] = [a for a in args.admin_interest if a]
        profile = set_usage_profile(**updates)
        print(yaml.dump({'usage_profile': profile}, sort_keys=False), end='')
    elif args.set_identity is not None:
        print(set_identity(*args.set_identity))
    elif args.user_agent:
        print(cfg.user_agent)
    elif args.reset:
        reset_config()
    elif args.edit:
        edit_config()
    elif args.reconfigure:
        print('Reconfiguring openplaces...\n')
        reset_config()
        print('\nStarting interactive setup:')
        print('-' * 70)
        # Trigger interactive setup
        get_config(interactive=True)
        print('\n✓ Configuration complete!')
    elif args.show:
        show_config()
    else:
        # Default: show config
        show_config()


if __name__ == '__main__':
    main()
