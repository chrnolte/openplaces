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
GPU_ENV_NAME = 'openplaces-amd'
GPU_MIN_DRIVER = '26.2.2'  # Adrenalin release required by the AMD GPU wheels

# Shown before asking for the identity that goes in openplaces' User-Agent.
# Duplicated from openplaces.config.IDENTITY_NOTICE rather than imported:
# dev.py runs before the environment exists, so it cannot import the
# package it is about to install. openplaces.config is the authority; keep
# the two in step when either changes.
IDENTITY_NOTICE = """\
openplaces downloads from public servers run by other people -- county GIS
portals, state agencies, national statistical offices. Every request it
makes carries a User-Agent naming the project, so an operator seeing
unexpected load knows what the traffic is and has someone to ask.

You are not registering anything and nothing is sent to this project. Pick
any nickname and place you are willing to have appear in a server log; a
work handle and an institution or city is the usual choice. Leave it blank
to stay unidentified.

  openplaces/0.1.0 (+https://openplaces.io; ada@some-university)\
"""

_SITECUSTOMIZE = """\
import ctypes
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

if sys.platform == 'win32':
    # Pre-load key torch DLLs from torch/lib before any other package loads
    # the older conda-forge copies from Library/bin (installed by libtorch).
    # Specifically:
    #   libiomp5md.dll  Library/bin has a ~150 KB stub (proxy to libomp);
    #                   torch_cpu.dll was compiled against the real 1.5 MB
    #                   runtime — loading the stub first causes WinError 127.
    #   c10.dll         Library/bin has an older build; torchvision/_C.pyd
    #                   needs the one from torch/lib or its entry points are
    #                   missing.
    # Once a DLL is in the process by base name, subsequent name-based loads
    # reuse it rather than searching again.
    _th_lib = os.path.join(sys.prefix, 'Lib', 'site-packages', 'torch', 'lib')
    for _dll in ('libiomp5md.dll', 'c10.dll'):
        _path = os.path.join(_th_lib, _dll)
        if os.path.exists(_path):
            try:
                ctypes.CDLL(_path)
            except OSError:
                pass

# Automatic Git Worktree / local checkout path resolver:
# Traverses upwards from current working directory or script directory to find
# the project root. If a different project checkout (like a git worktree) is
# detected, prepends its 'src' folder to sys.path.
def _setup_worktree():
    start_paths = []
    # 1. Directory of the script/notebook being run
    if sys.path and sys.path[0]:
        try:
            start_paths.append(Path(sys.path[0]).resolve())
        except Exception:
            pass
    # 2. Current working directory
    start_paths.append(Path.cwd().resolve())

    for start_path in start_paths:
        for parent in [start_path] + list(start_path.parents):
            if (
                (parent / 'pyproject.toml').exists()
                and (parent / 'src' / 'openplaces').exists()
            ):
                worktree_src = parent / 'src'
                src_str = str(worktree_src)
                # If it's already the primary path, nothing to do
                if sys.path and sys.path[0] == src_str:
                    return
                # Insert at index 0 to override editable install path
                if src_str in sys.path:
                    sys.path.remove(src_str)
                sys.path.insert(0, src_str)
                return

try:
    _setup_worktree()
except Exception:
    pass
"""


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
        if bat_path.exists() and sentinel in bat_path.read_text():
            print(f'✓ Launcher already installed: {bat_path}')
            return
        content = (
            f'@echo off\n'
            f'REM {sentinel}\n'
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
        func = (
            f'\n{sentinel}\n'
            f'{env_name}() {{\n'
            f'  cd "{notebooks_dir}" || return\n'
            f'  conda activate {env_name}\n'
            f'  jupyter notebook\n'
            f'}}\n'
        )

        if sys.platform == 'darwin':
            rc_file = Path('~/.zshrc').expanduser()
        else:
            shell = os.path.basename(os.environ.get('SHELL', 'bash'))
            if shell == 'zsh':
                rc_file = Path('~/.zshrc').expanduser()
            elif shell == 'bash':
                rc_file = Path('~/.bashrc').expanduser()
            else:
                print(
                    f'✗ Shell {shell!r} is not supported for automatic launcher '
                    f'setup.\n'
                    f'  Add this function to your shell rc file manually:\n'
                    f'{func}'
                )
                return

        # Check if launcher already installed for this env
        if rc_file.exists() and sentinel in rc_file.read_text():
            print(f'✓ Launcher already installed in {rc_file}')
            return

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


def install_sitecustomize_hook(env_name):
    """Write sitecustomize.py into the env to:
    1. Pre-load key torch DLLs on Windows to avoid WinError 127.
    2. Dynamically add the current git worktree/checkout src/ directory to sys.path,
       so pytest, scripts, and Jupyter notebooks automatically import from the worktree
       rather than the main repository's editable install path.
    """
    try:
        site_pkgs = subprocess.check_output(
            f'{PKG_MGR} run -n {env_name} python -c '
            '"import site; print(next(p for p in site.getsitepackages() '
            "if p.endswith('site-packages')))\"",
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        print('✗ Could not determine site-packages path — skipping sitecustomize.py')
        return

    dest = Path(site_pkgs) / 'sitecustomize.py'
    dest.write_text(_SITECUSTOMIZE)
    print(f'✓ sitecustomize.py → {dest}')


def ask_identity():
    """Ask how this installation should identify itself to data providers.

    Returns
    -------
    tuple of str
        (nickname, place), either possibly empty. An empty nickname skips
        the place question: half an identity is not worth a second prompt.
    """
    print('\n' + '-' * 70)
    print('How should openplaces identify itself when downloading data?\n')
    print(IDENTITY_NOTICE)
    print()

    nickname = input('Nickname (Enter to stay unidentified): ').strip()
    place = input('Place (university, city, or org): ').strip() if nickname else ''
    return nickname, place


def save_identity(env_name, nickname, place):
    """Write the identity into the installed package's user config.

    Delegated to the package rather than written here, so dev.py never has
    to know where openplaces keeps its config or how the file is shaped.
    """
    print('\nRecording how openplaces identifies itself...')
    # -W silences runpy's "found in sys.modules" note: openplaces/__init__
    # imports config eagerly (to trigger first-use setup), which makes
    # `-m openplaces.config` warn. Scoped to that one warning from that
    # one module, so nothing else is hidden.
    # check=False: failing to record a nickname must not abort an otherwise
    # finished install, and the message below says how to set it later.
    if not run(
        f'{PKG_MGR} run -n {env_name} python -W ignore::RuntimeWarning:runpy '
        f'-m openplaces.config --set-identity "{nickname}" "{place}"',
        check=False,
    ):
        print('✗ Could not record the identity; set it later with:')
        print('    python -m openplaces.config --set-identity NICKNAME PLACE')
    elif not nickname:
        print('Staying unidentified. Set one later with:')
        print('    python -m openplaces.config --set-identity NICKNAME PLACE')


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
    launcher_response = (
        input(
            f'\nInstall `{env_name}` command to launch Jupyter from the terminal? '
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

    # Asked here, with the other setup questions, but written after the
    # environment exists (the package owns where its config lives).
    nickname, place = ask_identity()

    print(f'\nUsing package manager: {pkg_mgr}')
    print(f'Creating environment: {env_name}')

    # Create environment with specified name
    print('\nCreating conda environment from environment.yml...')
    if not run(f'{PKG_MGR} env create -f environment.yml -n {env_name} -y'):
        print('✗ Failed to create environment')
        return

    print('\nConfiguring environment hooks (sitecustomize.py)...')
    install_sitecustomize_hook(env_name)

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

    save_identity(env_name, nickname, place)

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
        print('  2. Start coding!')
    else:
        print(f'  1. {pkg_mgr} activate {env_name}')
        print('  2. cd notebooks')
        print('  3. jupyter notebook')
        print('  4. Start coding!')


def _detect_windows_amd_gpu():
    """Return (name, driver_date) of the first AMD Radeon GPU, or None."""
    command = (
        'powershell -NoProfile -Command "'
        'Get-CimInstance Win32_VideoController | '
        'Where-Object Name -match Radeon | '
        "ForEach-Object { $_.Name + '|' + $_.DriverDate.ToString('yyyy-MM-dd') }"
        '"'
    )
    try:
        output = subprocess.check_output(command, shell=True, text=True).strip()
    except subprocess.CalledProcessError:
        return None
    if not output:
        return None
    name, _, driver_date = output.splitlines()[0].partition('|')
    return name, driver_date


def _add_kernel_env_vars(env_name):
    """Write conda env variables into the registered kernelspec.

    Jupyter launches kernels via the env's python.exe without conda
    activation, so GDAL/PROJ data paths and the OpenMP workaround must be
    set in kernel.json explicitly.
    """
    import json

    try:
        prefix = subprocess.check_output(
            f'{PKG_MGR} run -n {env_name} python -c "import sys; print(sys.prefix)"',
            shell=True,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        print('✗ Could not determine env prefix — skipping kernel env vars')
        return

    spec_path = (
        Path(os.environ.get('APPDATA', '~')).expanduser()
        / 'jupyter'
        / 'kernels'
        / env_name
        / 'kernel.json'
    )
    if not spec_path.exists():
        print(f'✗ Kernelspec not found at {spec_path} — skipping kernel env vars')
        return

    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    spec['env'] = {
        'GDAL_DATA': str(Path(prefix) / 'Library' / 'share' / 'gdal'),
        'PROJ_LIB': str(Path(prefix) / 'Library' / 'share' / 'proj'),
        'PYTHONUTF8': '1',
        'KMP_DUPLICATE_LIB_OK': 'TRUE',
    }
    spec_path.write_text(json.dumps(spec, indent=1), encoding='utf-8')
    print(f'✓ Kernel env vars (GDAL_DATA, PROJ_LIB, ...) → {spec_path}')


def setup_gpu():
    """Create a GPU-enabled environment for ML enrichment detectors."""
    if sys.platform.startswith('linux'):
        print('On Linux, the standard environment receives the CUDA torch')
        print('build automatically when an NVIDIA driver is visible at env')
        print('creation time (conda-forge __cuda virtual package). Verify:')
        print(
            f'  {PKG_MGR} run -n {DEFAULT_ENV_NAME} python -c '
            '"import torch; print(torch.cuda.is_available())"'
        )
        print('If it prints False on a GPU node, re-create the environment')
        print("on that node or pin 'pytorch=*=cuda*' in environment.yml.")
        return
    if sys.platform != 'win32':
        print('✗ GPU environment setup is automated for Windows and Linux only.')
        return

    gpu = _detect_windows_amd_gpu()
    if gpu is None:
        print('✗ No AMD Radeon GPU detected.')
        print('  (NVIDIA GPUs on Windows: install the CUDA torch build instead.)')
        return
    name, driver_date = gpu
    print(f'Found GPU: {name} (driver date {driver_date})')
    print(f'The AMD GPU wheels require AMD Adrenalin >= {GPU_MIN_DRIVER};')
    print('if verification fails below, update the driver via AMD Software.\n')

    env_name = (
        input(f'Environment name (press Enter for "{GPU_ENV_NAME}"): ').strip()
        or GPU_ENV_NAME
    )

    print('\nCreating GPU environment from environment-amd.yml')
    print('(downloads several GB of AMD GPU wheels; this takes a while) ...')
    created = run(
        f'{PKG_MGR} env create -f environment-amd.yml -n {env_name} -y',
        check=False,
    )
    if not created:
        print('✗ Environment creation failed. If it already exists, update it:')
        print(f'  {PKG_MGR} env update -f environment-amd.yml -n {env_name} --prune')
        return

    print('\nConfiguring environment hooks (sitecustomize.py)...')
    install_sitecustomize_hook(env_name)

    print('\nInstalling openplaces in editable mode...')
    run(f'{PKG_MGR} run -n {env_name} pip install -e . --no-deps')

    print('\nRegistering Jupyter kernel...')
    run(
        f'{PKG_MGR} run -n {env_name} python -m ipykernel install --user '
        f'--name {env_name} --display-name "Python ({env_name}, GPU)"'
    )
    _add_kernel_env_vars(env_name)

    print('\nVerifying GPU availability...')
    verified = run(
        f'{PKG_MGR} run -n {env_name} python -c '
        '"import torch; '
        "assert torch.cuda.is_available(), 'torch.cuda unavailable'; "
        'print(torch.__version__, torch.cuda.get_device_name(0))"',
        check=False,
    )
    if verified:
        print('\n✓ GPU environment ready!')
        print('\nNext steps:')
        print(f'  1. In Jupyter, select the kernel "Python ({env_name}, GPU)"')
        print('     to run enrichment notebooks on the GPU.')
        print(f'  2. Or run scripts with: conda run -n {env_name} python <script>')
    else:
        print('\n✗ torch does not see the GPU.')
        print(f'  Check that the AMD Adrenalin driver is >= {GPU_MIN_DRIVER}')
        print('  (AMD Software → System), update it, reboot, then re-run:')
        print('  python dev.py gpu')


def update():
    """Update existing environment with latest dependencies."""
    env_name = get_env_name()

    print(f'\nUsing package manager: {PKG_MGR}')
    print(f'Updating environment: {env_name}')

    print('\nUpdating conda environment...')
    run(f'{PKG_MGR} env update -f environment.yml -n {env_name} --prune')

    qgis_response = input('\nReinstall QGIS processing scripts? [y/N] ').strip().lower()

    if not shutil.which('7z'):
        print(
            '7z not found. Deflate64 ZIP extraction unavailable. Run setup to install.'
        )

    print('\nConfiguring environment hooks (sitecustomize.py)...')
    install_sitecustomize_hook(env_name)

    print('\nReinstalling openplaces...')
    run(f'{PKG_MGR} run -n {env_name} pip install -e . --no-deps')

    print('\nEnsuring nbstripout is configured...')
    run(f'{PKG_MGR} run -n {env_name} nbstripout --install')

    print('\nEnsuring pre-commit hooks are installed...')
    run(f'{PKG_MGR} run -n {env_name} pre-commit install')

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


def catalog_state():
    """Print an overview of the committed recipe catalog."""
    # The summary lives with the docs extensions that render it, and
    # deliberately reads recipes as plain YAML, so it runs without the
    # openplaces package (or its dependencies) being importable.
    sys.path.insert(0, str(Path(__file__).parent / 'docs' / '_ext'))
    import catalog_data

    print()
    print(catalog_data.format_overview())


def list_envs():
    """List all conda environments."""
    print(f'\nListing all {PKG_MGR} environments:')
    run(f'{PKG_MGR} env list')


def main():
    """Main entry point."""
    commands = {
        'setup': ('Create development environment', setup),
        'gpu': ('Create GPU (AMD/CUDA) environment for ML enrichment', setup_gpu),
        'update': ('Update development environment', update),
        'clean': ('Remove development environment', clean),
        'test': ('Run tests', test),
        'format': ('Format code with ruff', format_code),
        'lint': ('Check code with ruff', lint),
        'build': ('Build package for distribution', build),
        'list': ('List all conda/mamba environments', list_envs),
        'state': ('Show the state of the recipe catalog', catalog_state),
        'identity': (
            'Set how openplaces identifies itself to data providers',
            lambda: save_identity(get_env_name(), *ask_identity()),
        ),
        'qgis': ('Install QGIS processing scripts', install_qgis),
        'launcher': (
            'Install command to launch openplaces from the terminal',
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
