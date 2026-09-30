"""OpenPlacesConfig: the merged configuration and its accessors."""

import copy
import getpass
import sys
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_config_dir

from openplaces.config.identity import (
    APPAUTHOR,
    APPNAME,
    build_user_agent,
    prompt_identity,
)
from openplaces.config.user_config import (
    DataRootNotSetError,
    _merge_nested,
    can_prompt,
    detect_agent,
    prompt_usage_profile,
    write_identity,
    write_usage_profile,
)
from openplaces.core.constants import (
    CRS,
    GEO_MIN_AREA_M2,
    NEVER_DELETE,
    RETENTION_CLASSES,
    STANDARD_DIRS,
)


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
        """Get the repository root (the parent of `src/`)."""
        # This module is src/openplaces/config/settings.py; one level
        # deeper than the single config.py it replaced on 2026-09-30.
        return Path(__file__).resolve().parents[3]

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
        from openplaces import config as _config

        _config.reload_config()
        return _config.get_config()

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

        The bucket is registered with retention 'keep' and marked custom,
        which is what keeps cleanup off it: files a user puts in their own
        directory belong to no recipe, and cleanup classes a file with no
        recipe as an orphan once it is old enough.

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

        # Registered whether or not a description was given: an
        # unregistered bucket cannot be named in a retention override
        # (validation rejects it as unknown) and cleanup has nothing to
        # read its class from.
        STANDARD_DIRS[name] = {
            'default': path,
            'description': description,
            'shared': False,  # Custom dirs default to user-specific
            'retention': 'keep',
            'custom': True,
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
