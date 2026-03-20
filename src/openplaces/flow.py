"""
src/openplaces/flow.py

Functions to manage the flow of scripts from prototyping to computing.

Includes conversion of notebooks to scripts.
"""

import inspect
import os
import re
import subprocess
from pathlib import Path

from openplaces.config import cfg
from openplaces.path import code_path

# Constants for notebook-to-script conversions

# Notebook to script: automatic text replacements
NOTEBOOK_REGEX_REPLACEMENTS = {
    # Remove single-line objects ('pass' or outputs to Jupyter)
    r'\n[a-zA-Z][a-zA-Z0-9_]+\n\n': (r'', None),
    # Remove 'In [*]:'
    r'# In\[[0-9 ]{,4}\]:\n\n': (r'', None),
    # Cut off notebook conversion code
    r'\n# ---\n# # Convert to script.*$': (r'', re.DOTALL),
    # Compress headings
    r'\n# (#+)(.*)?\n\n': (r'\n#\2\n', None),
    r'(?<!\n)\n# (#+)(.*)?\n': (r'\n\n#\2\n', None),
    r'\n# (#+)(.*)?\n': (r'\n#\2\n', None),
    # Remove formatting of title
    r'\*\*`(.*)?`\*\*\n': (r'\1\n', None),
    # Remove triple-empty rows
    r'\n\n\n[\n]+': (r'\n\n\n', None),
    # Bring continue back
    r'pass  # continue': ('continue', None),
}

# Markers used for automatic `for` loop indentation
END_LOOP_MARKER = (
    r"assert False, 'This marks the end of a flattened `for` loop.'" + '\n\n'
)
REGEX_FOR_LOOP = (
    r'\n *(for ([a-zA-Z0-9_]+) in .+?:\n)'
    r'( *pass\r?\n)'
    r'([\s\S]*)' + END_LOOP_MARKER
)

CODE_FORMATTERS = ['ruff']


def get_caller_path():
    """
    Dynamically extracts the full filepath of the calling notebook or script.
    Prioritizes specific environment markers (VS Code, ipynbname) before
    falling back to standard script detection.
    """
    # 1. Check for Jupyter/IPython environment
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is not None:
            # A. VS Code injects the notebook path into the IPython user namespace
            if '__vsc_ipynb_file__' in ipy.user_ns:
                return Path(ipy.user_ns['__vsc_ipynb_file__']).resolve()
            # B. Standard Jupyter / JupyterLab via server sessions API
            try:
                import re

                import ipykernel
                import requests
                from jupyter_server import serverapp

                kernel_id = re.search(
                    r'kernel-(.*?)\.json', ipykernel.connect.get_connection_file()
                ).group(1)

                for server in serverapp.list_running_servers():
                    try:
                        response = requests.get(
                            f'{server["url"]}api/sessions',
                            headers={'Authorization': f'token {server["token"]}'},
                            timeout=2,
                        )
                        if response.status_code != 200:
                            continue
                        for session in response.json():
                            if session['kernel']['id'] == kernel_id:
                                return (
                                    Path(server['root_dir'])
                                    / session['notebook']['path']
                                ).resolve()
                    except requests.exceptions.RequestException:
                        continue
            except Exception:
                pass
            # C. Fallback: ipynbname for JupyterHub / remote edge cases
            try:
                import ipynbname

                return ipynbname.path().resolve()
            except (ImportError, FileNotFoundError, Exception):
                pass
            # D. Last resort: CWD is usually the notebook's directory
            return Path.cwd().resolve()
    except ImportError:
        pass

    # 2. Standard Script (.py) Detection
    # Look back at the caller's frame (index 1 is the function calling this one)
    try:
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])

        if module and hasattr(module, '__file__'):
            return Path(module.__file__).resolve()
    except Exception:
        pass

    # 3. Final Fallback (Interactive shell or direct execution without file context)
    return Path.cwd().resolve()


def get_caller_path_in_code_directory():
    caller_path = get_caller_path()

    if caller_path.is_relative_to(cfg.code_root / 'notebooks'):
        return caller_path.relative_to(cfg.code_root / 'notebooks')
    elif caller_path.is_relative_to(cfg.code_root / 'scripts'):
        return caller_path.relative_to(cfg.code_root / 'scripts')
    else:
        raise NotImplementedError(
            'Caller path is not in /notebooks or /scripts: \n\n{caller_path}.'
        )


def convert_to_script(
    commit=False,
    fix=True,
    formatters=['ruff'],
    raise_error=True,
    show=False,
    verbose=False,
):
    """Convert Jupyter notebook to a script

    Includes numerous string edits:
    - 'Set arguments' > 'Get arguments'
      (currently for FIPS only)
    - Import standard imports from lib.core
    - Simplify headings
    - Remove 'In [*]'
    - Format header
    - Drop anything after 'Convert notebook to script'

    Parameters
    ----------
    commit : bool
        If True, will write to the Python script.
        This will trigger the script to be updated by the current
        scheduler (db/update.ipynb), which is bases its decisions on the
        last-modified timestamp of the scripts (not diffs to last run
        version, which would be better).
    fix : bool
        If True, will use `ruff check --fix` to try and fix all issues
    formatters : list of str
        Formatters to apply if fix=True.
        Only `ruff` is currently supported.
    raise_error : bool
        If True, will raise non-compliance errors.
        If False, will only print the error.
    show: bool
        If True, will print code regardless of whether error was found
    verbose : bool:
        If True, will output additional information to troubleshoot.
    """

    if fix:
        if not formatters:
            raise ValueError('You need to provide a code formatter if fix=True.')

    missing_formatters = set(formatters) - set(CODE_FORMATTERS)
    if missing_formatters and fix:
        raise ModuleNotFoundError(
            'Code formatters not yet supported: ' + ', '.join(missing_formatters)
        )

    caller_path = get_caller_path_in_code_directory()
    script_dirs = caller_path.parent.parts
    script = caller_path.stem

    # Convert script
    ipynb_filepath = code_path('notebooks', *script_dirs, script + '.ipynb')
    # print('ipynb_filepath', ipynb_filepath)

    command = [
        'jupyter',
        'nbconvert',
        '--to',
        'python',
        str(ipynb_filepath),
        '--log-level',
        'WARN',
    ]
    subprocess.run(command)

    # Load script
    conv_filepath = code_path('notebooks', *script_dirs, script + '.py')
    # print('conv_filepath', conv_filepath)

    with open(conv_filepath) as file:
        text = file.read()

    # Replace all notebook content from the heading `# Test arguments`
    # to the next top-level heading with `args=parser.parse_args()`
    text = re.sub(
        re.compile(r'# # Test arguments\n\n.*?# # ', re.DOTALL),
        '# Get arguments\n\nargs = parser.parse_args()\n\n\n# ',
        # repl + '\n\n\n# ',
        text,
    )

    # Text replacements
    for regex_from, (regex_to, flag) in NOTEBOOK_REGEX_REPLACEMENTS.items():
        if flag is not None:
            text = re.sub(re.compile(regex_from, flag), regex_to, text)
        else:
            text = re.sub(re.compile(regex_from), regex_to, text)

    # Indent flattened `for` loops
    for_loop_found = re.search(REGEX_FOR_LOOP, text)
    i = 0
    while for_loop_found:
        _, _, delete, indent = for_loop_found.groups()
        indented = re.sub(r'\n(?!\n)', '\\n    ', indent).rstrip() + '\n'
        if indented[:1] != '\n':
            indented = '    ' + indented
        end_marker = (
            END_LOOP_MARKER.replace(r'\(', r'(')
            .replace(r'\)', r')')
            .replace(r'\n', '\n')
        )
        text = text.replace(delete, '').replace(indent + end_marker, indented)
        i += 1
        for_loop_found = re.search(REGEX_FOR_LOOP, text)
        if i >= 5 and for_loop_found:
            raise ValueError('More than five for loops?')

    # Change header to comment string
    marker1, marker2 = r'utf-8\n\n#', r'\n\n# Configure'
    pos1 = re.search(marker1, text).start() + len(marker1) - 4
    try:
        pos2 = re.search(marker2, text).start()
    except AttributeError:
        raise Exception('Is your first header not `Configure`?')
    head = text[pos1:pos2]
    head_new = (
        '"""'
        + head.replace(
            '\n# ', '\n'
        )  # .replace('`Arguments`\n', 'Arguments\n---------')
        + '\n"""\n'
    )
    text = text[: pos1 + 1] + head_new + text[pos2:]

    if commit:
        py_filepath = code_path('scripts', *script_dirs, script + '.py')
        # Write to script folder
        if py_filepath.exists():
            print('File updated: ', end='')
        else:
            print('New file: ', end='')
        print(py_filepath.relative_to(cfg.code_root), '\n')
    else:
        py_filepath = code_path('scripts', '_test', *script_dirs, script + '.py')
        print(
            'Test run (commit=False).\n\n'
            + 'Main script has not been updated:\n'
            + code_path(*script_dirs, script + '.py')
            + '\n\nPass commit=True to overwrite main script.\n\n'
            + 'Test script written to:\n'
            + py_filepath
            + '\n'
        )

    py_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(py_filepath, 'w') as file:
        file.write(text)

    if fix:
        command = ['ruff', 'format', str(py_filepath)]
    else:
        command = ['ruff', 'check', str(py_filepath)]
    sp = subprocess.run(command, capture_output=True)

    if sp.returncode == 0:
        print('Code complies with code style.')
        error = False
    else:
        print('Code is not compliant with code style.')
        show = True
        error = True

    if show:
        if error and len(sp.stdout) > 0:
            print('\n---\n')
            for line in sp.stdout.decode('utf-8').split('\n'):
                print(line.replace(str(py_filepath) + ':', ''))
        title = str(py_filepath.relative_to(cfg.code_root))
        print('\n' + '-' * len(title) + '\n' + title + '\n' + '-' * len(title) + '\n')
        with open(py_filepath) as file:
            for i, line in enumerate(file.read().split('\n')):
                print(((str(i + 1).rjust(3) + '  ') if error else '') + line)

    if sp.returncode != 0 and raise_error:
        raise ValueError(
            'Not `ruff` compliant. Pass `raise_error=False` to convert_to_script() or '
            'try `fix=True` for automatic formatting.'
        )

    os.remove(conv_filepath)


def test_script(*args, verbose=False, committed=True):
    """Run a test of a script

    Parameters
    ----------
    *args : tuple of strings
        Unnamed arguments will be passed as strings
    verbose : bool
        If True, will print outputs.
        Will be set to True if '--verbose' is in `args`.
    committed : bool
        If True, will run the committed version of the script.
        If False, will run the test version.
    """

    # Set verbose to True if argument is found in args
    if not verbose and '--verbose' in args:
        verbose = True

    caller_path = get_caller_path_in_code_directory()
    script_dirs = caller_path.parent.parts
    script = caller_path.stem

    if committed:
        py_filepath = code_path('scripts', *script_dirs, script + '.py')
    else:
        py_filepath = code_path('scripts', '_test', *script_dirs, script + '.py')

    command = ['python', str(py_filepath)] + list(args)
    run_subprocess(command, verbose=verbose)


def run_subprocess(command, p={}, verbose=False, ignore_failures=False):
    import subprocess
    import threading

    for key, value in p.items():
        if isinstance(value, list):
            value = ','.join([str(v) for v in value])
        command += ['--' + key, str(value)]

    # Use -u (unbuffered) so subprocess print() output is not block-buffered.
    if command and Path(command[0]).stem in ('python', 'python3'):
        command.insert(1, '-u')

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_lines = []
    stderr_lines = []

    # Read stdout and stderr concurrently to prevent pipe-buffer deadlocks.
    # (Subprocess blocks if it fills a pipe whose read end is not being drained.)
    def _read_stderr():
        for line in process.stderr:
            stderr_lines.append(line)

    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

    for line in process.stdout:
        stdout_lines.append(line)
        if verbose:
            print(line, end='', flush=True)

    stderr_thread.join()
    process.wait()

    if process.returncode == 0:
        if verbose:
            print('\x1b[32mSubprocess terminated successfully\x1b[0m')
    else:
        print(
            '\x1b[31m-------------------------------\n'
            'Error while running subprocess:\n'
            '-------------------------------'
        )
        print('\n\x1b[34m' + ' '.join(command) + '\x1b[0m')
        print('\x1b[31m')
        for line in stderr_lines:
            print(line, end='')
        print('\x1b[0m')
        if not ignore_failures:
            raise Exception('Execution stopped, because subprocess failed.')
