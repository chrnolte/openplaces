"""
openplaces configuration management

Hierarchical configuration system with interactive first-use setup.

Priority (highest to lowest):
1. User config (~/.config/openplaces/<username>.yaml) - User-specific overrides
2. Project config (./openplaces.yaml) - Project defaults
3. Built-in defaults - Fallback values

On first use, users are prompted to customize directory paths or accept defaults.

One package since 2026-09-30 (until then a single 1,589-line module):
`identity` (User-Agent and identity prompt), `user_config` (the user's
own file and the prompts that fill it), `settings` (OpenPlacesConfig).
This file keeps every function that reads or rebinds the process-wide
config (`cfg`, `_cfg`, `get_config`, `reload_config`, the consent,
personal-column, identity and usage-profile setters), because the
tests patch those names here and their callers look them up here.
"""

import argparse
import os
import platform
import subprocess
from pathlib import Path

import yaml
from platformdirs import user_config_dir

from openplaces.config.identity import (  # noqa: F401
    AGENT_ENV_VARS,
    APPAUTHOR,
    APPNAME,
    IDENTITY_NOTICE,
    PROJECT_URL,
    build_user_agent,
    prompt_identity,
)
from openplaces.config.settings import (  # noqa: F401
    OpenPlacesConfig,
)
from openplaces.config.user_config import (  # noqa: F401
    _UNSET,
    USAGE_ENV_FLAGS,
    USAGE_PROFILE_NOTICE,
    ConfigFileUnreadableError,
    DataRootNotSetError,
    _merge_nested,
    can_prompt,
    detect_agent,
    merge_user_config,
    prompt_usage_profile,
    write_identity,
    write_usage_profile,
)

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
    # `get_config()` rather than the module-level `cfg`:
    # `reload_config` rebinds `_cfg` only, so `cfg` still holds the
    # snapshot taken at import, and a decision recorded during this
    # process is invisible to the next read (and, worse, to the next
    # write, which rebuilds the whole mapping from it).
    recorded = (get_config().get('consent') or {}).get('terms') or {}
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

    # `get_config()` rather than the module-level `cfg`:
    # `reload_config` rebinds `_cfg` only, so `cfg` still holds the
    # snapshot taken at import, and a decision recorded during this
    # process is invisible to the next read (and, worse, to the next
    # write, which rebuilds the whole mapping from it).
    recorded = dict((get_config().get('consent') or {}).get('terms') or {})
    recorded[source] = {
        'accepted': bool(accepted),
        'recorded': date.today().isoformat(),
    }
    merge_user_config(get_config().user_config_path, 'consent', {'terms': recorded})
    reload_config()


def get_keep_personal_columns() -> bool:
    """Return whether this installation keeps personal columns in curate.

    Personal columns are the ones
    `openplaces.core.attribute_registry.is_personal_attribute` names: the
    parties to a deed and a property's owner. A curate recipe's
    `keep_registered_columns` step drops them by default. An
    installation used for research may need them (telling a sale within
    a family from one at arm's length, where the source states no
    relationship), and records that here. Default False.

    The choice covers this installation's own curated tables only. A
    delivery withholds personal columns whatever it says, so keeping
    them can never publish them.
    """
    privacy = get_config().get('privacy') or {}
    return bool(privacy.get('keep_personal_columns', False))


def set_keep_personal_columns(keep: bool) -> None:
    """Record whether this installation keeps personal columns in curate.

    Stored in this user's own config, never in a recipe, for the same
    reason a terms decision is: a committed recipe would make the choice
    for everyone who runs it.

    Parameters
    ----------
    keep : bool
        True to keep them in curated tables, False to drop them.
    """
    merge_user_config(
        get_config().user_config_path,
        'privacy',
        {'keep_personal_columns': bool(keep)},
    )
    reload_config()


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
    # `get_config()` rather than the module-level `cfg`:
    # `reload_config` rebinds `_cfg` only, so `cfg` still holds the
    # snapshot taken at import, and a decision recorded during this
    # process is invisible to the next read (and, worse, to the next
    # write, which rebuilds the whole mapping from it).
    recorded = get_config().get('usage_overrides') or {}
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

    # `get_config()` rather than the module-level `cfg`:
    # `reload_config` rebinds `_cfg` only, so `cfg` still holds the
    # snapshot taken at import, and a decision recorded during this
    # process is invisible to the next read (and, worse, to the next
    # write, which rebuilds the whole mapping from it).
    recorded = dict(get_config().get('usage_overrides') or {})
    entry = {
        'compatible': bool(compatible),
        'recorded': date.today().isoformat(),
    }
    if reason:
        entry['reason'] = reason
    recorded[source_id] = entry
    merge_user_config(get_config().user_config_path, 'usage_overrides', recorded)
    reload_config()


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
    parser.add_argument(
        '--keep-personal-columns',
        choices=['true', 'false'],
        default=None,
        help=(
            'Keep owner, grantor and grantee columns in this '
            "installation's curated tables (default false). Deliveries "
            'withhold them either way.'
        ),
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
    elif args.keep_personal_columns is not None:
        set_keep_personal_columns(args.keep_personal_columns == 'true')
        print(f'keep_personal_columns: {get_keep_personal_columns()}')
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
