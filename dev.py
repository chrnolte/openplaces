#!/usr/bin/env python3
"""Development environment management for openplaces.

This script helps developers set up and manage their local development
environment. End users installing via 'pip install openplaces' don't need this.
"""

import importlib.resources
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ENV_NAME = 'openplaces'


def get_package_manager():
    """Detect if mamba is available, otherwise use conda."""
    # Get the full path to ensure we use the executable, not shell functions
    mamba_path = shutil.which('mamba')
    conda_path = shutil.which('conda')

    if mamba_path:
        return mamba_path
    elif conda_path:
        return conda_path
    else:
        print('✗ Error: Neither mamba nor conda found in PATH')
        print('\nPlease install one of the following:')
        print(
            '  - Miniforge (includes mamba): https://github.com/conda-forge/miniforge'
        )
        print('  - Miniconda (conda): https://docs.conda.io/en/latest/miniconda.html')
        sys.exit(1)


PKG_MGR = get_package_manager()


def get_env_name():
    """Prompt user for environment name."""
    env_name = input(
        f'Environment name (press Enter for "{DEFAULT_ENV_NAME}"): '
    ).strip()
    return env_name if env_name else DEFAULT_ENV_NAME


def run(cmd, check=True):
    """Run command and handle errors."""
    print(f'→ {cmd}')
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        print(f'✗ Command failed with code {result.returncode}')
        sys.exit(1)
    return result.returncode == 0


def ensure_7zip():
    """Ensure 7z is available, installing it if necessary."""
    if shutil.which('7z'):
        print('✓ 7z already installed')
        return True

    print('7z not found, attempting to install...')
    system = sys.platform

    try:
        if system == 'darwin':
            if not shutil.which('brew'):
                print('Homebrew not found. Install 7z manually: brew install sevenzip')
                return False
            run('brew install sevenzip', check=False)

        elif system == 'win32':
            if shutil.which('winget'):
                run(
                    'winget install -e --id 7zip.7zip -h --accept-source-agreements',
                    check=False,
                )
            elif shutil.which('choco'):
                run('choco install 7zip -y', check=False)
            else:
                print(
                    'No package manager found. '
                    'Download 7-Zip from https://www.7-zip.org/'
                )
                return False

        elif system.startswith('linux'):
            distro_id = ''
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release') as f:
                    for line in f:
                        if line.startswith('ID='):
                            distro_id = line.strip().split('=', 1)[1].strip('"').lower()
                            break

            if distro_id in ('ubuntu', 'debian', 'linuxmint', 'pop'):
                run('sudo apt-get install -y 7zip', check=False)
            elif distro_id in (
                'fedora',
                'rhel',
                'centos',
                'rocky',
                'almalinux',
            ):
                run('sudo dnf install -y p7zip p7zip-plugins', check=False)
            elif distro_id in ('arch', 'manjaro', 'endeavouros'):
                run('sudo pacman -S --noconfirm p7zip', check=False)
            elif distro_id in ('opensuse-leap', 'opensuse-tumbleweed', 'sles'):
                run('sudo zypper install -y p7zip', check=False)
            else:
                print(f'Unknown distro: {distro_id}. Install 7z manually.')
                return False
        else:
            print(f'Unsupported platform: {system}. Install 7z manually.')
            return False

    except Exception as e:
        print(f'Failed to install 7z: {e}')
        return False

    if shutil.which('7z'):
        print('✓ 7z installed successfully')
        return True
    elif sys.platform == 'win32':
        default_path = r'C:\Program Files\7-Zip\7z.exe'
        if os.path.exists(default_path):
            print('✓ 7z installed successfully')
            print('  Note: restart your terminal for 7z to be available in PATH.')
            return True

    print('7z installation may have failed — verify with: 7z i')
    return False


def install_qgis():
    """Copy openplaces QGIS processing scripts to the user's QGIS profile."""

    # Prefer the local source tree (works when running dev.py from the repo).
    # Fall back to importlib.resources for installed-package scenarios.
    local_qgis_dir = Path(__file__).parent / 'src' / 'openplaces' / 'qgis'

    if local_qgis_dir.is_dir():
        scripts = [
            p
            for p in local_qgis_dir.iterdir()
            if p.name.endswith('.py') and not p.name.startswith('__')
        ]
        as_file = None  # plain Path objects, no context manager needed
    else:
        try:
            qgis_pkg = importlib.resources.files('openplaces.qgis')
        except ModuleNotFoundError:
            print('✗ openplaces not found. Run from the repo root or install first.')
            return False
        scripts = [
            p
            for p in qgis_pkg.iterdir()
            if p.name.endswith('.py') and not p.name.startswith('__')
        ]
        as_file = importlib.resources.as_file

    if not scripts:
        print('No QGIS scripts found in openplaces.qgis.')
        return False

    # Resolve QGIS processing scripts folder for each OS
    system = sys.platform
    if system == 'win32':
        base = Path(os.environ.get('APPDATA', '~')).expanduser()
    elif system == 'darwin':
        base = Path('~/Library/Application Support').expanduser()
    else:
        base = Path('~/.local/share').expanduser()

    scripts_dir = (
        base / 'QGIS' / 'QGIS3' / 'profiles' / 'default' / 'processing' / 'scripts'
    )

    try:
        scripts_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f'✗ Could not create QGIS scripts directory: {e}')
        return False

    for script in scripts:
        dest = scripts_dir / script.name
        try:
            if as_file is None:
                shutil.copy2(script, dest)
            else:
                with as_file(script) as src_path:
                    shutil.copy2(src_path, dest)
        except OSError as e:
            print(f'  ✗ Could not copy {script.name}: {e}')
            continue
        print(f'  ✓ {script.name} → {dest}')

    print(f'\n✓ QGIS scripts installed to {scripts_dir}')
    print('  Restart QGIS and look for "openplaces" in the Processing Toolbox.')
    return True


def install_launcher(env_name):
    """Install a launcher to start openplaces from the command line."""
    notebooks_dir = Path(__file__).parent.resolve() / 'notebooks'
    sentinel = f'# openplaces-launcher:{env_name}'

    if sys.platform == 'win32':
        bat_path = Path.home() / f'{env_name}.bat'
        content = (
            f'@echo off\n'
            f'call conda activate {env_name}\n'
            f'cd /d "{notebooks_dir}"\n'
            f'jupyter notebook\n'
        )
        try:
            bat_path.write_text(content)
            print(f'✓ Launcher created: {bat_path}')
            print(
                '  Run it from an Anaconda Prompt or any shell with conda initialized.'
            )
        except OSError as e:
            print(f'✗ Could not write launcher: {e}')

    else:
        rc_file = (
            Path('~/.zshrc').expanduser()
            if sys.platform == 'darwin'
            else Path('~/.bashrc').expanduser()
        )

        # Check if launcher already installed for this env
        if rc_file.exists() and sentinel in rc_file.read_text():
            print(f'✓ Launcher already present in {rc_file} (sentinel found).')
            return

        func = (
            f'\n{sentinel}\n'
            f'{env_name}() {{\n'
            f'  cd "{notebooks_dir}" || return\n'
            f'  conda activate {env_name}\n'
            f'  jupyter notebook\n'
            f'}}\n'
        )
        try:
            with rc_file.open('a') as f:
                f.write(func)
            print(f'✓ Shell function `{env_name}()` added to {rc_file}')
            print(
                f'  Run `source {rc_file}` or open a new terminal, then type '
                f'`{env_name}`.'
            )
        except OSError as e:
            print(f'✗ Could not write to {rc_file}: {e}')


def setup():
    """Create conda environment and install package in editable mode."""

    pkg_mgr = PKG_MGR.split(os.sep)[-1]

    print('This will create a development environment for `openplaces`.\n')
    print(f'Found package manager: {PKG_MGR}\n')

    env_name = get_env_name()
    zip_response = (
        input(
            '\nEnsure `7z` is installed to unzip more ZIP formats?\n'
            '(needed for Windows `deflate64`, e.g. recipe `US-VA_parcel-vgin-2025`) '
            '[Y/n] '
        )
        .strip()
        .lower()
    )
    qgis_response = (
        input(
            '\nInstall `openplaces` data import tools for QGIS (Processing Toolbox)? '
            '[y/N] '
        )
        .strip()
        .lower()
    )
    launcher_response = (
        input(
            f'\nInstall `{env_name}` command to launch Jupyter from the terminal? '
            '[Y/n] '
        )
        .strip()
        .lower()
    )

    print(f'\nUsing package manager: {pkg_mgr}')
    print(f'Creating environment: {env_name}')

    # Create environment with specified name
    print('\nCreating conda environment from environment.yml...')
    if not run(f'{PKG_MGR} env create -f environment.yml -n {env_name} -y'):
        print('✗ Failed to create environment')
        return

    if zip_response in ('', 'y'):
        ensure_7zip()
    else:
        print('Skipping 7z installation.')
        if not shutil.which('7z'):
            print('Note: Deflate64 ZIP files will not be extractable.')

    print('\nInstalling openplaces in editable mode...')
    run(f'{PKG_MGR} run -n {env_name} pip install -e . --no-deps')

    print('\nSetting up nbstripout for automatic notebook cleaning...')
    run(f'{PKG_MGR} run -n {env_name} nbstripout --install')

    print('\nInstalling pre-commit hooks...')
    run(f'{PKG_MGR} run -n {env_name} pre-commit install')

    if qgis_response == 'y':
        install_qgis()

    if launcher_response in ('', 'y'):
        install_launcher(env_name)

    print('\n✓ Development environment ready!')
    print('\nNext steps:')

    if launcher_response in ('', 'y'):
        if sys.platform == 'win32':
            print(
                f'  1. Open an Anaconda Prompt and type `{env_name}` to launch Jupyter.'
            )
        else:
            print(f'  1. Open a new terminal and type `{env_name}` to launch Jupyter.')
    else:
        print(f'  1. {pkg_mgr} activate {env_name}')
        print('  2. cd notebooks')
        print('  3. jupyter notebook')
    print('  4. Start coding!')


def update():
    """Update existing environment with latest dependencies."""
    env_name = get_env_name()

    print(f'\nUsing package manager: {PKG_MGR}')
    print(f'Updating environment: {env_name}')

    print('\nUpdating conda environment...')
    run(f'{PKG_MGR} env update -f environment.yml -n {env_name} --prune')

    if not shutil.which('7z'):
        print(
            '7z not found. Deflate64 ZIP extraction unavailable. Run setup to install.'
        )

    print('\nReinstalling openplaces...')
    run(f'{PKG_MGR} run -n {env_name} pip install -e . --no-deps')

    print('\nEnsuring nbstripout is configured...')
    run(f'{PKG_MGR} run -n {env_name} nbstripout --install')

    print('\nEnsuring pre-commit hooks are installed...')
    run(f'{PKG_MGR} run -n {env_name} pre-commit install')

    qgis_response = input('\nReinstall QGIS processing scripts? [y/N] ').strip().lower()
    if qgis_response == 'y':
        install_qgis()

    print('\n✓ Environment updated!')


def clean():
    """Remove the development environment."""
    env_name = get_env_name()

    response = input(f'Remove {env_name} environment? [y/N] ')
    if response.lower() != 'y':
        print('Cancelled.')
        return

    print(f'\nRemoving environment: {env_name}')
    print('(This may take a minute...)')

    # Capture output to filter the mamba_trash.txt error
    result = subprocess.run(
        f'{PKG_MGR} env remove -n {env_name} -y',
        shell=True,
        capture_output=True,
        text=True,
    )

    # Print stdout (the actual removal progress)
    if result.stdout:
        print(result.stdout)

    # Print stderr but filter out the mamba_trash.txt error
    if result.stderr:
        # Filter out the known harmless error
        lines = result.stderr.split('\n')
        filtered_lines = [
            line
            for line in lines
            if 'mamba_trash.txt' not in line.lower()
            and 'error opening for writing' not in line.lower()
        ]
        if filtered_lines:
            print('\n'.join(filtered_lines))

    # Check if removal was successful
    if (
        result.returncode == 0
        or 'Environment removed' in result.stdout
        or 'Environment removed' in result.stderr
    ):
        print('\n✓ Environment removed!')
    else:
        print(f'\n✗ Failed to remove environment (exit code: {result.returncode})')


def test():
    """Run the test suite."""
    env_name = get_env_name()

    print(f'\nRunning tests in environment: {env_name}')
    run(f'{PKG_MGR} run -n {env_name} pytest')


def format_code():
    """Format code with ruff."""
    env_name = get_env_name()

    print(f'\nFormatting code in environment: {env_name}')
    print('Formatting Python files...')
    run(f'{PKG_MGR} run -n {env_name} ruff format openplaces/')

    print('Formatting notebooks (if any)...')
    run(f'{PKG_MGR} run -n {env_name} ruff format notebooks/', check=False)

    print('Fixing linting issues...')
    run(f'{PKG_MGR} run -n {env_name} ruff check --fix openplaces/')
    print('\n✓ Code formatted!')


def lint():
    """Check code with ruff."""
    env_name = get_env_name()

    print(f'\nLinting code in environment: {env_name}')
    run(f'{PKG_MGR} run -n {env_name} ruff check openplaces/')


def build():
    """Build package for distribution."""
    env_name = get_env_name()

    print(f'\nBuilding package in environment: {env_name}')
    run(f'{PKG_MGR} run -n {env_name} python -m build')
    print('\n✓ Package built! Check dist/ directory')


def list_envs():
    """List all conda environments."""
    print(f'\nListing all {PKG_MGR} environments:')
    run(f'{PKG_MGR} env list')


def main():
    """Main entry point."""
    commands = {
        'setup': ('Create development environment', setup),
        'update': ('Update development environment', update),
        'clean': ('Remove development environment', clean),
        'test': ('Run tests', test),
        'format': ('Format code with ruff', format_code),
        'lint': ('Check code with ruff', lint),
        'build': ('Build package for distribution', build),
        'list': ('List all conda/mamba environments', list_envs),
        'qgis': ('Install QGIS processing scripts', install_qgis),
        'launcher': (
            'Install terminal launcher command',
            lambda: install_launcher(get_env_name()),
        ),
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f'Development environment manager (using {PKG_MGR})')
        print('\nUsage: python dev.py <command>')
        print('\nDevelopment commands:')
        for cmd, (desc, _) in commands.items():
            print(f'  {cmd:8} - {desc}')
        sys.exit(1)

    commands[sys.argv[1]][1]()


if __name__ == '__main__':
    main()
