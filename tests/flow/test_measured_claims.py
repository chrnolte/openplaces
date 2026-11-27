"""The measured-claims report finds the convention's form, compares
dates the right way round, and never fails the build.

Every scope, date and number below is fabricated.
"""

from datetime import date
from pathlib import Path

from openplaces.flow.measured_claims import (
    find_claims,
    main,
    report,
    resolve_scopes,
    scan,
    verdict,
)


def _tree(tmp_path: Path) -> Path:
    module = tmp_path / 'pkg' / 'thing.py'
    module.parent.mkdir()
    module.write_text(
        '"""Docs.\n\n'
        'Measured 2026-01-15 on XX-AA-BBB: 12.5% of rows carry a value.\n'
        'Not a claim: measured on nothing in particular.\n'
        '"""\n\n'
        'def f():\n'
        '    # Measured 2026-02-01 on some-region: 3 of 4 lots agree.\n'
        '    return 1\n',
        encoding='utf8',
    )
    return tmp_path


def test_find_claims_reads_date_scope_and_text(tmp_path):
    claims = find_claims(_tree(tmp_path))
    assert [(c.measured, c.scope, c.line) for c in claims] == [
        (date(2026, 1, 15), 'XX-AA-BBB', 3),
        (date(2026, 2, 1), 'some-region', 8),
    ]
    assert claims[0].text == '12.5% of rows carry a value.'


def test_verdict_is_stale_only_when_the_build_is_newer():
    measured = date(2026, 1, 15)
    assert verdict(measured, date(2026, 1, 15)) == 'current'
    assert verdict(measured, date(2026, 1, 14)) == 'current'
    assert verdict(measured, date(2026, 1, 16)) == 'stale'
    assert verdict(measured, None) == 'no build'


def test_report_uses_the_resolver_and_survives_its_errors(tmp_path):
    claims = find_claims(_tree(tmp_path))

    def resolve(scope):
        if scope == 'XX-AA-BBB':
            return date(2026, 3, 1)
        raise RuntimeError('no such region')

    states = [state for _, state in report(claims, resolve=resolve)]
    assert states == ['stale', 'no build']


def test_main_reports_and_exits_zero(tmp_path, capsys):
    assert main(['--root', str(_tree(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert 'XX-AA-BBB' in out and 'some-region' in out


def test_main_says_so_when_there_is_nothing(tmp_path, capsys):
    assert main(['--root', str(tmp_path)]) == 0
    assert 'no measured claims' in capsys.readouterr().out


def _mixed_tree(tmp_path: Path) -> Path:
    recipe = tmp_path / 'recipes' / 'XX_thing-src-2026.yaml'
    recipe.parent.mkdir()
    recipe.write_text(
        '# Measured 2026-01-15 on XX-AA-BBB: 7 of 9 rows keep a key.\n'
        'notes: Measured 2026-01-15 on XX-AA-CCC: data, not a comment.\n'
        '# Measured 2026-01-20 on XX-AA-BBB and XX-AA-CCC: 3 lots.\n'
        '# Measured on the export: 4 rows.\n'
        '# a sentence that ends in Measured\n'
        '# 2026-01-21 on XX-AA-BBB: 5 rows.\n'
        '# Measured 2026-01-22 on nowhere-at-all: 6 rows.\n',
        encoding='utf8',
    )
    readme = tmp_path / 'pkg' / 'README.md'
    readme.parent.mkdir()
    readme.write_text(
        'Measured 2026-01-10 on XX-AA-DDD: 2 of 3 units.\n', encoding='utf8'
    )
    module = tmp_path / 'pkg' / 'thing.py'
    module.write_text(
        '"""Docs.\n\nMeasured 2026-01-15 on XX-AA-BBB: 12.5% of rows.\n"""\n',
        encoding='utf8',
    )
    return tmp_path


def test_scan_finds_recipe_comment_readme_and_python_claims(tmp_path):
    claims, _ = scan(_mixed_tree(tmp_path))
    found = sorted((c.kind, c.scope, c.text) for c in claims)
    assert found == [
        ('python', 'XX-AA-BBB', '12.5% of rows.'),
        ('readme', 'XX-AA-DDD', '2 of 3 units.'),
        ('recipe', 'XX-AA-BBB', '7 of 9 rows keep a key.'),
        ('recipe', 'nowhere-at-all', '6 rows.'),
    ]


def test_malformed_lines_are_listed_with_a_reason(tmp_path):
    root = _mixed_tree(tmp_path)
    claims, malformed = scan(root)
    claims, unknown = resolve_scopes(claims, known=lambda s: s != 'nowhere-at-all')
    assert [c.scope for c in claims if c.kind == 'recipe'] == ['XX-AA-BBB']
    reasons = {m.line: m.reason for m in malformed + unknown}
    assert set(reasons) == {3, 4, 5, 7}
    assert 'one token' in reasons[3]
    assert 'no date' in reasons[4]
    assert 'wraps' in reasons[5]
    assert 'names no unit' in reasons[7]
    assert all(m.kind == 'recipe' for m in malformed + unknown)


def test_strict_fails_only_on_malformed_claims(tmp_path, capsys):
    root = _mixed_tree(tmp_path)

    def known(scope):
        return scope != 'nowhere-at-all'

    assert main(['--root', str(root)], known=known) == 0
    out = capsys.readouterr().out
    assert '4 malformed' in out and 'claims by kind' in out
    assert main(['--root', str(root), '--strict'], known=known) == 1
    clean = tmp_path / 'clean'
    clean.mkdir()
    assert main(['--root', str(_tree(clean)), '--strict'], known=bool) == 0


def test_a_recipe_claim_goes_stale_like_a_python_one(tmp_path):
    claims, _ = scan(_mixed_tree(tmp_path))
    bbb = [c for c in claims if c.scope == 'XX-AA-BBB']
    assert {c.kind for c in bbb} == {'python', 'recipe'}

    def resolve(scope):
        return date(2026, 2, 1) if scope == 'XX-AA-BBB' else None

    states = {c.kind: state for c, state in report(bbb, resolve=resolve)}
    assert states == {'python': 'stale', 'recipe': 'stale'}

    def older(scope):
        return date(2026, 1, 1)

    states = {c.kind: state for c, state in report(bbb, resolve=older)}
    assert states == {'python': 'current', 'recipe': 'current'}
