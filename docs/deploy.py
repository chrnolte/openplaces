#!/usr/bin/env python3
"""Build the openplaces documentation from a commit and publish it.

Run without flags for a dry run (build, list what would change).
`--go` publishes after a typed confirmation, `--build-only` builds and
keeps the site without contacting the host, and `--go --force`
publishes even when nothing changed.

Deploys are manual: nothing runs this on a push. The site is built
from a commit (`--ref`, default `HEAD`), never from the working tree:
the recipe catalog reads every recipe YAML on disk, and a working tree
can hold git-ignored recipes whose terms forbid publishing their
schema. Uncommitted changes are therefore never published.

A deploy is skipped when the docs inputs did not change since the last
one. The inputs are the git trees of `docs/` and `src/openplaces/` at
the commit (autoapi documents the package, the catalog reads its
recipes). Each deploy leaves their ids in `INPUTS_FILE` on the host,
which the next deploy reads, so the check works from any machine.

Host, user, SSH key and pinned host key come from the value pages'
deploy settings (`openplaces.viz.web_scene.load_deploy_settings`), so
both sites deploy with the same key to the same host. Only the target
differs. It defaults to the folder `DEFAULT_FOLDER` beside the value
pages' folder and is overridden by `OPENPLACES_DOCS_DEPLOY_PATH` and
`OPENPLACES_DOCS_DEPLOY_URL`, in the environment or in the same
settings file. Adapted from the EE 508 course site's deploy script.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from openplaces.viz.web_scene import (
    RCLONE_SFTP_FLAGS,
    _rclone,
    _rclone_env,
    _rclone_output,
    default_deploy_env_file,
    load_deploy_settings,
)

REPO = Path(__file__).resolve().parents[1]
# Every input of the docs build: the pages and their extensions, and the
# package that autoapi documents and whose recipes the catalog lists.
INPUT_PATHS = ('docs', 'src/openplaces')
DEFAULT_FOLDER = 'openplaces-docs-preview'
DOCS_PREFIX = 'OPENPLACES_DOCS_DEPLOY_'
# Written to the host after a successful upload, never during it, so an
# interrupted upload is never mistaken for a finished deploy.
INPUTS_FILE = '.openplaces-docs-inputs'


def git(*args) -> str:
    """Output of a git command run in the repository."""
    return subprocess.run(
        ['git', '-C', str(REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def docs_inputs(commit: str) -> str:
    """One line naming the git trees the docs build reads at `commit`."""
    trees = git('rev-parse', *(f'{commit}:{path}' for path in INPUT_PATHS)).split()
    return ' '.join(f'{p}={t}' for p, t in zip(INPUT_PATHS, trees, strict=True))


def docs_target(settings: dict) -> tuple[str, str]:
    """Host folder and public URL of the docs site.

    Parameters
    ----------
    settings : dict
        The value pages' deploy settings, whose folder and URL the
        defaults sit beside.

    Returns
    -------
    tuple of str
        The absolute folder on the host and the public URL.
    """
    values = {}
    env_file = default_deploy_env_file()
    if env_file.exists():
        for raw in env_file.read_text(encoding='utf-8').splitlines():
            key, sep, value = raw.strip().partition('=')
            if sep and key.strip().startswith(DOCS_PREFIX):
                values[key.strip()] = value.strip().strip('"\'')
    values.update({k: v for k, v in os.environ.items() if k.startswith(DOCS_PREFIX)})
    path = values.get(DOCS_PREFIX + 'PATH') or (
        settings['path'].rsplit('/', 1)[0] + '/' + DEFAULT_FOLDER
    )
    url = values.get(DOCS_PREFIX + 'URL') or (
        settings['url'].rsplit('/', 1)[0] + '/' + DEFAULT_FOLDER
        if settings['url']
        else ''
    )
    return path.rstrip('/'), url.rstrip('/')


def export_commit(commit: str, dest: Path) -> None:
    """Write the committed docs inputs into `dest`, without the .git.

    `git archive` gives exactly what the commit holds: no git-ignored
    file, no uncommitted edit. Without a .git folder, conf.py's search
    for untracked modules finds none, as on Read the Docs.
    """
    archive = subprocess.run(
        ['git', '-C', str(REPO), 'archive', '--format=tar', commit, *INPUT_PATHS],
        capture_output=True,
        check=True,
    ).stdout
    tar_path = dest.parent / 'docs-inputs.tar'
    tar_path.write_bytes(archive)
    with tarfile.open(tar_path) as tar:
        tar.extractall(dest, filter='data')
    tar_path.unlink()


def build(tree: Path, strict: bool) -> Path | None:
    """Build the HTML site in an exported tree; return its folder."""
    html = tree / 'docs' / '_build' / 'html'
    command = [
        sys.executable,
        '-m',
        'sphinx',
        '-b',
        'html',
        # The doctree cache stays outside the published folder.
        '-d',
        str(tree / 'docs' / '_build' / 'doctrees'),
        str(tree / 'docs'),
        str(html),
    ]
    if strict:
        command.insert(3, '-W')
    print('-> ' + ' '.join(command))
    # Run from inside the tree: conf.py resolves '_ext' against the
    # working directory.
    ok = subprocess.run(command, cwd=tree / 'docs', check=False).returncode == 0
    return html if ok else None


def publish(env: dict, html: Path, target: str, inputs: str, commit: str, go: bool):
    """Mirror the built site to the host, then record its inputs."""
    dry = [] if go else ['--dry-run']
    progress = (
        ['--progress', '--stats-one-line']
        if sys.stdout.isatty()
        else ['--stats', '30s', '--stats-one-line', '--verbose']
    )
    # The inputs file is excluded from the sync, so the sync neither
    # uploads nor deletes it; it is replaced only after a full upload.
    _rclone(
        env,
        'sync',
        str(html),
        f'bh:{target}',
        '--create-empty-src-dirs',
        '--filter',
        '- .DS_Store',
        '--filter',
        f'- /{INPUTS_FILE}',
        *RCLONE_SFTP_FLAGS,
        *progress,
        *dry,
    )
    record = html.parent / INPUTS_FILE
    record.write_text(f'{inputs}\ncommit={commit}\n', encoding='utf-8')
    _rclone(env, 'copyto', str(record), f'bh:{target}/{INPUTS_FILE}', *dry)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build the openplaces documentation and publish it.'
    )
    parser.add_argument('--go', action='store_true', help='upload (default: dry run)')
    parser.add_argument('--ref', default='HEAD', help='commit to build (default HEAD)')
    parser.add_argument(
        '--force', action='store_true', help='build and upload even if unchanged'
    )
    parser.add_argument(
        '--strict', action='store_true', help='fail the build on Sphinx warnings'
    )
    parser.add_argument(
        '--build-only',
        action='store_true',
        help='build and keep the site, contact no host',
    )
    args = parser.parse_args()

    commit = git('rev-parse', '--verify', f'{args.ref}^{{commit}}')
    inputs = docs_inputs(commit)
    print(f'Commit : {commit}')
    if git('status', '--porcelain', '--', *INPUT_PATHS):
        print('Note   : uncommitted changes under docs/ or src/ are not built')

    if not args.build_only:
        settings = load_deploy_settings()
        target, url = docs_target(settings)
        env = _rclone_env(settings)
        print(f'Target : {settings["user"]}@{settings["host"]}:{target}')
        if url:
            print(f'URL    : {url}/')
        print(f'Mode   : {"PUBLISH" if args.go else "dry run"}\n')

        published = _rclone_output(env, 'cat', f'bh:{target}/{INPUTS_FILE}')
        last = (published or '').splitlines()
        if last and last[0] == inputs and not args.force:
            print(
                'Docs inputs unchanged since the last deploy '
                f'({last[1] if len(last) > 1 else "commit unknown"}); '
                'nothing to build or upload. --force overrides.'
            )
            return

        if args.go:
            folder = target.rsplit('/', 1)[-1]
            answer = input(f'This overwrites {url or target}. Type "{folder}" to go: ')
            if answer.strip() != folder:
                print('Cancelled.')
                return

    # Outside Dropbox, which would sync and lock the build tree.
    work = Path(tempfile.mkdtemp(prefix='openplaces-docs-'))
    tree = work / 'tree'
    tree.mkdir()
    try:
        export_commit(commit, tree)
        html = build(tree, args.strict)
        if html is None:
            print('Build failed; nothing was uploaded.')
            sys.exit(1)
        if args.build_only:
            print(f'\nBuilt: {html}')
            print(f'  Serve it:  python -m http.server -d "{html}" 8000')
            return
        publish(env, html, target, inputs, commit, args.go)
    finally:
        if not args.build_only:
            shutil.rmtree(work, ignore_errors=True)

    if args.go:
        print(f'\nPublished commit {commit[:8]}.' + (f'  {url}/' if url else ''))
    else:
        print('\nDry run complete. Re-run with --go to publish.')


if __name__ == '__main__':
    main()
