"""Hand out and withdraw access to the value site, one token per person.

    python -m openplaces.viz.web_scene.tokens add "<name>" ["<name>" ...]
    python -m openplaces.viz.web_scene.tokens add --from-file names.txt
    python -m openplaces.viz.web_scene.tokens list
    python -m openplaces.viz.web_scene.tokens revoke "<name>"

The tokens live in a file on this machine (`ACCESS_TOKENS` in the
deploy settings, by default beside them), one `token<TAB>name` per
line; only the tokens ever go to the host, never the names. `add`
prints each person's name and link, one per line. A change takes
effect online once the gate is uploaded again::

    python -m openplaces.viz.web_scene.site access --go
"""

import argparse
import secrets
from pathlib import Path

from openplaces.viz.web_scene import (
    ACCESS_TOKEN_PATTERN,
    load_deploy_settings,
)


def _entries(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    entries = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        token, _, name = line.partition('\t')
        entries.append((token.strip(), name.strip()))
    return entries


def _write(path: Path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Value site access tokens: token<TAB>name. Never commit this file.']
    lines += [f'{token}\t{name}' for token, name in entries]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def add(path: Path, name: str, url: str) -> str:
    """Mint a token for `name`, store it, and return the person's link."""
    entries = _entries(path)
    if any(n == name for _, n in entries):
        raise SystemExit(f'{name!r} already has a token; revoke it first.')
    token = secrets.token_urlsafe(16)
    assert ACCESS_TOKEN_PATTERN.match(token)
    _write(path, [*entries, (token, name)])
    return f'{url}/?t={token}'


def revoke(path: Path, name: str) -> None:
    """Drop every token of `name`."""
    entries = _entries(path)
    kept = [(t, n) for t, n in entries if n != name]
    if len(kept) == len(entries):
        raise SystemExit(f'No token for {name!r}.')
    _write(path, kept)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog='python -m openplaces.viz.web_scene.tokens')
    parser.add_argument('action', choices=['add', 'list', 'revoke'])
    parser.add_argument(
        'names', nargs='*', help='the people, as you will recognize them'
    )
    parser.add_argument(
        '--from-file',
        help='add: also one name per line from this file (blank lines and # skipped)',
    )
    args = parser.parse_args(argv)
    settings = load_deploy_settings()
    path = Path(settings['access_tokens']).expanduser()
    if args.action == 'list':
        for token, name in _entries(path):
            print(f'{name}\t{token[:4]}...')
        return
    names = list(args.names)
    if args.from_file:
        lines = Path(args.from_file).read_text(encoding='utf-8').splitlines()
        names += [n.strip() for n in lines if n.strip() and not n.startswith('#')]
    if not names:
        parser.error('name the people')
    for name in names:
        if args.action == 'add':
            # Name and link, one per line, ready to paste into messages.
            print(f'{name}\t{add(path, name, settings["url"])}')
        else:
            revoke(path, name)
            print(f'Revoked {name!r}.')
    print('Online once the gate is uploaded:')
    print('  python -m openplaces.viz.web_scene.site access --go')


if __name__ == '__main__':
    main()
