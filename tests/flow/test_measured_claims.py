"""The measured-claims report finds the convention's form, compares
dates the right way round, and never fails the build.

Every scope, date and number below is fabricated.
"""

from datetime import date
from pathlib import Path

from openplaces.flow.measured_claims import find_claims, main, report, verdict


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
