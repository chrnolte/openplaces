"""Tests for the runnable spine rebuild.

The failure this guards is not a crash. Running the six phases out of
order, or stopping before the mint settles, produces a *plausible* spine
-- and every other dataset is keyed on it.
"""

from __future__ import annotations

import pytest

from openplaces.io.admin_codes import rebuild


@pytest.fixture(autouse=True)
def _never_touch_real_data(monkeypatch):
    """No test may reach the real population phases.

    Learned by doing it: one test called ``rebuild_spine(apply=True)``
    without ``skip_population``, so phase 1 ran real raster zonal-stats
    over 218,651 units and rewrote all three committed
    ``population-admin*.csv`` files. Killing it mid-run left them
    partial. ``skip_population`` is an argument a test can forget; this
    fixture is not.
    """
    for name in (
        'build_population',
        'fill_population_gaps',
        'repair_zero_weights',
        'resolve_stale_references',
    ):
        monkeypatch.setattr(
            rebuild.build,
            name,
            lambda *a, **k: pytest.fail('a test reached a real spine-writing phase'),
            raising=False,
        )
    monkeypatch.setattr(
        rebuild,
        'apply_population_overrides',
        lambda *a, **k: pytest.fail('a test reached the real overrides'),
        raising=False,
    )


def test_the_override_table_parses_and_declares_known_strategies():
    frame = rebuild.load_overrides()
    assert len(frame) > 0
    assert set(frame['level']) <= {2, 3, 4}
    assert set(frame['key']) - {''} <= set(rebuild.KEY_STRATEGIES)


def test_an_unknown_key_strategy_is_refused(tmp_path):
    p = tmp_path / 'o.csv'
    p.write_text(
        'recipe_id,scope,level,join_column,key,note\nr,US,3,,not_a_strategy,\n',
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='unknown key strategies'):
        rebuild.load_overrides(p)


def test_the_connecticut_key_survives_the_planning_region_renumbering():
    # A GEOID is state(2) + county(3) + subdivision(5). Connecticut
    # renumbered the middle in 2022, so only the ends may be joined on.
    key = rebuild.KEY_STRATEGIES['state_plus_subdivision']
    assert key('0900112345') == '0912345'
    assert key('0999912345') == '0912345'  # county renumbered, key unchanged


def test_missing_prerequisites_stop_the_rebuild_before_it_writes(monkeypatch):
    monkeypatch.setattr(rebuild, 'check_prerequisites', lambda **k: ['a-recipe'])
    with pytest.raises(RuntimeError, match='prerequisite'):
        rebuild.rebuild_spine(apply=True, verbose=False)


def test_a_dry_run_reports_without_writing(monkeypatch):
    seen = {}

    def fake_remint(levels=None, apply=None, backup_dir=None, verbose=None):
        seen['apply'] = apply
        return {2: {'units': 1, 'changed': 0, 'recycled': 0}}

    monkeypatch.setattr(rebuild, 'check_prerequisites', lambda **k: [])
    monkeypatch.setattr(rebuild.build, 'remint_spine', fake_remint)
    monkeypatch.setattr(
        rebuild.build,
        'resolve_stale_references',
        lambda **k: pytest.fail('a dry run must not write references'),
    )
    out = rebuild.rebuild_spine(apply=False, skip_population=True, verbose=False)
    assert seen['apply'] is False
    assert out['converged'] is True


def test_a_mint_that_never_settles_raises_rather_than_shipping(monkeypatch):
    monkeypatch.setattr(rebuild, 'check_prerequisites', lambda **k: [])
    monkeypatch.setattr(rebuild.build, 'repair_zero_weights', lambda **k: None)
    monkeypatch.setattr(rebuild.build, 'resolve_stale_references', lambda **k: None)
    monkeypatch.setattr(
        rebuild.build,
        'remint_spine',
        lambda **k: {2: {'units': 10, 'changed': 3, 'recycled': 1}},
    )
    with pytest.raises(RuntimeError, match='did not reach a fixed point'):
        rebuild.rebuild_spine(
            apply=True, max_passes=3, skip_population=True, verbose=False
        )


def test_convergence_stops_the_loop_and_sweeps_references(monkeypatch):
    calls = {'mint': 0, 'sweep': 0}

    def fake_remint(**k):
        calls['mint'] += 1
        changed = 5 if calls['mint'] < 3 else 0
        return {2: {'units': 10, 'changed': changed, 'recycled': 0}}

    monkeypatch.setattr(rebuild, 'check_prerequisites', lambda **k: [])
    monkeypatch.setattr(rebuild.build, 'repair_zero_weights', lambda **k: None)
    monkeypatch.setattr(rebuild.build, 'remint_spine', fake_remint)
    monkeypatch.setattr(
        rebuild.build,
        'resolve_stale_references',
        lambda **k: calls.__setitem__('sweep', calls['sweep'] + 1),
    )
    out = rebuild.rebuild_spine(apply=True, skip_population=True, verbose=False)
    assert out == {'passes': 3, 'history': [5, 5, 0], 'converged': True}
    assert calls['sweep'] == 1
