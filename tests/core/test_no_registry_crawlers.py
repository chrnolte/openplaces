"""No registry-of-deeds crawler comes back into the source tree.

The terms of the Massachusetts registries' public site prohibit scraping
and crawling "whether by automated means or manually", and the creation
of derivative works. The crawler that drove it, its sister adapter for
the registries on ALIS, and the recipe that ran them were removed
(maintainer, 2026-10-23), so that nobody runs them against those terms.
Massachusetts sale prices come from the Department of Revenue's LA3
sales export instead.

These tests fail if the hosts, the modules or a crawling recipe return.
"""

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / 'src' / 'openplaces'

# Host names and page signatures of the registry applications, and the
# modules that drove them. Matched without regard to case.
FORBIDDEN = (
    'masslandrecords',
    'uslandrecords',
    'titleview.org',
    'ww400r',
    'avenu_adapter',
    'avenu_selectors',
    'alis_adapter',
    'alis_selectors',
    'registry_ingester',
)
SUFFIXES = {'.py', '.yaml', '.yml', '.csv', '.json', '.md', '.txt'}


def test_no_registry_crawler_host_or_module_in_the_source():
    offenders = []
    for path in SRC.rglob('*'):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        text = path.read_text(encoding='utf-8', errors='replace').lower()
        hits = [name for name in FORBIDDEN if name in text]
        if hits:
            offenders.append(f'{path.relative_to(SRC).as_posix()}: {hits}')

    assert offenders == [], f'registry crawler references in: {offenders}'


def test_no_recipe_declares_a_scraper():
    offenders = [
        path.relative_to(SRC).as_posix()
        for path in (SRC / 'recipes').rglob('*.yaml')
        if any(
            line.startswith('scraper:')
            for line in path.read_text(encoding='utf-8').splitlines()
        )
    ]

    assert offenders == [], f'recipes with a scraper block: {offenders}'
