"""Tests for the size-based memory reservations of workflow jobs.

All admin ids, recipes and measurements here are fabricated.
"""

import os
from pathlib import Path

import pytest

from openplaces.flow import memory
from openplaces.flow.memory import (
    Envelope,
    MemoryModel,
    Observation,
    build_seed,
    fit_envelope,
    input_megabytes,
    load_seed,
    load_table,
    parse_rule_name,
    read_benchmark,
    record_input_size,
    refresh,
    write_seed,
    write_table,
)

REPO = Path(__file__).resolve().parents[2]
BENCH_HEADER = (
    's\th:m:s\tmax_rss\tmax_vms\tmax_uss\tmax_pss\tio_in\tio_out\tmean_load\tcpu_time\n'
)
RECIPE = 'US_widget-spine-2026'
NEW = f'harmonize_{RECIPE}_XX-AA-NEW'


def _obs(admin, peak, input_mb, recipe=RECIPE, stage='harmonize'):
    rule = f'{stage}_{recipe}_{admin}'
    return Observation(rule, stage, recipe, admin, peak, input_mb, '2026-01-01')


def _family(points, recipe=RECIPE, prefix='XX-AA'):
    return {
        o.rule: o
        for o in (
            _obs(f'{prefix}-{i:02d}', peak, size, recipe)
            for i, (peak, size) in enumerate(points)
        )
    }


def _reserve(mb):
    return memory._round_up(mb * memory.SAFETY_FACTOR + memory.HEADROOM_MB)


SIX = [(300.0 + 10 * i, 10.0 * (i + 1)) for i in range(6)]


@pytest.mark.parametrize(
    'rule, expected',
    [
        (
            'harmonize_US_widget-geospine-2026_XX-AA-BBB',
            ('harmonize', 'US_widget-geospine-2026', 'XX-AA-BBB'),
        ),
        (
            'harmonize_admin-openplaces-2026_admin4_XX-AA-BBB',
            ('harmonize', 'admin-openplaces-2026_admin4', 'XX-AA-BBB'),
        ),
        (
            'ingest_US_tile-widget-2025_block_global',
            ('ingest', 'US_tile-widget-2025_block', None),
        ),
        ('ingest_tile-widget-2025_batch', ('ingest', 'tile-widget-2025', None)),
        ('curate_XX-AA_widget-x-1_XX', ('curate', 'XX-AA_widget-x-1', 'XX')),
        ('deliver_US_widget-2026_XX-AA_some-region', None),
        ('all', None),
    ],
)
def test_parse_rule_name(rule, expected):
    assert parse_rule_name(rule) == expected


def test_input_megabytes_counts_geo_sidecar(tmp_path):
    attr = tmp_path / 'XX-AA-BBB_widget.parquet'
    attr.write_bytes(b'x' * 1_000_000)
    (tmp_path / 'XX-AA-BBB_widget_geo.parquet').write_bytes(b'x' * 2_000_000)
    other = tmp_path / 'table.csv'
    other.write_bytes(b'x' * 500_000)
    assert input_megabytes([attr, other]) == pytest.approx(3.5)
    assert input_megabytes([]) == 0


def test_input_megabytes_is_none_when_an_input_is_missing(tmp_path):
    present = tmp_path / 'a.parquet'
    present.write_bytes(b'x')
    assert input_megabytes([present, tmp_path / 'missing.parquet']) is None


def test_read_benchmark_takes_larger_of_rss_and_vms(tmp_path):
    bench = tmp_path / 'b.tsv'
    bench.write_text(
        BENCH_HEADER
        + '10.0\t0:00:10\t900.5\t1200.25\t0\t0\t0\t0\t90\t9\n'
        + '11.0\t0:00:11\t1500.0\t1000.0\t0\t0\t0\t0\t90\t9\n',
        encoding='utf-8',
    )
    assert read_benchmark(bench) == 1500.0
    assert read_benchmark(tmp_path / 'missing.tsv') is None


def test_measured_job_gets_its_peak_plus_margin():
    obs = _obs('XX-AA-BBB', 1000.0, 50.0)
    model = MemoryModel({obs.rule: obs})
    reserved = model.estimate(obs.rule, obs.stage, obs.recipe_id, 50.0, 16000)
    assert reserved == _reserve(1000.0)
    assert reserved >= 1000.0
    assert reserved % memory.ROUND_MB == 0
    # Far below the flat value it replaces.
    assert reserved < 16000


def test_measured_job_scales_with_grown_inputs_never_shrinks():
    obs = _obs('XX-AA-BBB', 1000.0, 50.0)
    model = MemoryModel({obs.rule: obs})
    args = (obs.rule, obs.stage, obs.recipe_id)
    assert model.estimate(*args, 100.0, 16000) == _reserve(2000.0)
    assert model.estimate(*args, 10.0, 16000) == _reserve(1000.0)
    assert model.estimate(*args, None, 16000) == _reserve(1000.0)


def test_measured_job_ignores_ceiling():
    obs = _obs('XX-AA-BBB', 1000.0, 50.0)
    model = MemoryModel({obs.rule: obs})
    reserved = model.estimate(
        obs.rule, obs.stage, obs.recipe_id, 500.0, 16000, ceiling_mb=48000
    )
    assert reserved == _reserve(10000.0)


def test_unmeasured_family_falls_back_to_flat():
    model = MemoryModel(_family([(500.0, 10.0)]))
    assert (
        model.estimate(
            'harmonize_US_other-2026_XX-AA-ZZ',
            'harmonize',
            'US_other-2026',
            10.0,
            16000,
        )
        == 16000
    )


def test_envelope_covers_every_unit_it_was_fitted_on():
    points = [
        (250.0, 10.0),
        (300.0, 40.0),
        (9000.0, 60.0),
        (800.0, 300.0),
        (400.0, 20.0),
        (1200.0, 100.0),
        (700.0, None),
    ]
    table = _family(points)
    env = fit_envelope(table.values())
    assert env.n_measured == 6
    assert env.max_input_mb == 300.0
    assert env.max_peak_mb == 9000.0
    for obs in table.values():
        if obs.input_mb is not None:
            assert env.need(obs.input_mb) >= obs.peak_mb
    model = MemoryModel(table)
    small = model.estimate(NEW, 'harmonize', RECIPE, 5.0, 16000)
    large = model.estimate(NEW, 'harmonize', RECIPE, 200.0, 16000)
    assert _reserve(250.0) <= small < large


def test_envelope_floor_covers_units_without_inputs():
    points = [(300.0 + i, 10.0 * (i + 1)) for i in range(5)] + [(2500.0, 0.0)]
    env = fit_envelope(_family(points).values())
    assert env.floor_mb == 2500.0
    assert env.need(1.0) >= 2500.0
    assert env.slope >= 0.0


def test_too_few_sized_units_have_no_envelope():
    assert fit_envelope(_family([(300.0, 10.0), (400.0, None)]).values()) is None


def test_thin_family_uses_largest_peak_and_never_less_than_flat():
    model = MemoryModel(_family([(300.0, 10.0), (700.0, 20.0)]))
    assert model.estimate(NEW, 'harmonize', RECIPE, 15.0, 4000) == 4000
    assert model.estimate(NEW, 'harmonize', RECIPE, 15.0, 256) == _reserve(700.0)


def test_extrapolated_job_gets_the_whole_ceiling():
    model = MemoryModel(_family(SIX))
    within = model.estimate(NEW, 'harmonize', RECIPE, 30.0, 16000, ceiling_mb=48000)
    beyond = model.estimate(NEW, 'harmonize', RECIPE, 61.0, 16000, ceiling_mb=48000)
    assert within < 48000
    assert beyond == 48000
    # Without a ceiling (a cluster run) the envelope stands.
    assert model.estimate(NEW, 'harmonize', RECIPE, 61.0, 16000) < 48000


def _seed_envelope(**kw):
    values = dict(
        stage='harmonize',
        recipe_id=RECIPE,
        n_measured=20,
        floor_mb=400.0,
        slope=10.0,
        max_peak_mb=5400.0,
        max_input_mb=500.0,
    )
    values.update(kw)
    return Envelope(**values)


def test_seed_sizes_a_family_this_machine_never_measured():
    env = _seed_envelope()
    model = MemoryModel({}, seed={(env.stage, env.recipe_id): env})
    assert model.estimate(NEW, 'harmonize', RECIPE, 100.0, 16000) == _reserve(1400.0)
    assert (
        model.estimate(NEW, 'harmonize', RECIPE, 600.0, 16000, ceiling_mb=48000)
        == 48000
    )
    assert model.estimate(NEW, 'harmonize', RECIPE, None, 16000) == _reserve(5400.0)
    # A family in neither falls through to the flat value.
    assert (
        model.estimate('curate_US_x-2026_XX-AA-B', 'curate', 'US_x-2026', 5.0, 8000)
        == 8000
    )


def test_order_local_job_then_local_family_then_seed():
    env = _seed_envelope(floor_mb=10000.0, slope=0.0, max_peak_mb=10000.0)
    seed = {(env.stage, env.recipe_id): env}
    table = _family(SIX)
    model = MemoryModel(table, seed=seed)
    own = next(iter(table.values()))
    assert model.estimate(own.rule, 'harmonize', RECIPE, own.input_mb, 16000) == (
        _reserve(own.peak_mb)
    )
    local = fit_envelope(table.values())
    assert model.estimate(NEW, 'harmonize', RECIPE, 30.0, 16000) == _reserve(
        local.need(30.0)
    )
    # A thinly measured local family defers to the seed.
    thin = MemoryModel(_family(SIX[:2]), seed=seed)
    assert thin.estimate(NEW, 'harmonize', RECIPE, 30.0, 16000) == _reserve(10000.0)


def test_table_round_trip(tmp_path):
    table = _family([(300.0, 10.0), (700.0, None)])
    path = tmp_path / 'sub' / 'mem.csv'
    write_table(path, table)
    loaded = load_table(path)
    assert set(loaded) == set(table)
    for rule, obs in table.items():
        assert loaded[rule].peak_mb == pytest.approx(obs.peak_mb, abs=0.5)
        assert loaded[rule].input_mb == obs.input_mb
        assert loaded[rule].admin_id == obs.admin_id
    assert load_table(tmp_path / 'missing.csv') == {}


def test_build_seed_fits_one_envelope_per_family_in_scope(tmp_path):
    table = {
        **_family(SIX, prefix='XX-AA'),
        **_family([(90000.0, 1.0)] * 6, prefix='XX-BB'),
        **_family(SIX, recipe='US_gadget-spine-2026', prefix='XX-AA'),
        **_family(SIX[:3], recipe='US_thin-spine-2026', prefix='XX-AA'),
    }
    seed = build_seed(table, scope='XX-AA')
    assert {e.recipe_id for e in seed} == {RECIPE, 'US_gadget-spine-2026'}
    widget = next(e for e in seed if e.recipe_id == RECIPE)
    assert widget.n_measured == 6
    assert widget.max_peak_mb == 350.0
    path = tmp_path / 'seed.csv'
    write_seed(path, seed)
    loaded = load_seed(path)
    assert loaded[('harmonize', RECIPE)].slope == pytest.approx(widget.slope, abs=1e-4)
    assert loaded[('harmonize', RECIPE)].max_input_mb == 60.0
    assert load_seed(tmp_path / 'missing.csv') == {}


def _bench(directory, rule, rss, mtime=None):
    path = directory / f'{rule}.tsv'
    path.write_text(
        BENCH_HEADER + f'10.0\t0:00:10\t{rss}\t0\t0\t0\t0\t0\t90\t9\n',
        encoding='utf-8',
    )
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_refresh_keeps_the_largest_peak_and_pairs_fresh_input_sizes(tmp_path):
    bench = tmp_path / 'benchmarks'
    bench.mkdir()
    table = tmp_path / 'mem' / 'mem.csv'
    rule = 'harmonize_US_widget-spine-2026_XX-AA-BBB'
    t0 = 1_800_000_000
    record_input_size(bench, rule, 42.0)
    os.utime(memory.ledger_path(bench, rule), (t0, t0))
    _bench(bench, rule, 2000.0, mtime=t0 + 60)
    _bench(bench, 'deliver_US_widget-2026_XX-AA_some-region', 50.0)
    assert refresh(table, bench) == (1, 0)
    obs = load_table(table)[rule]
    assert (obs.stage, obs.recipe_id, obs.admin_id) == (
        'harmonize',
        'US_widget-spine-2026',
        'XX-AA-BBB',
    )
    assert obs.peak_mb == 2000.0
    assert obs.input_mb == 42.0

    # A lower peak (say, a run resumed from a checkpoint) is ignored.
    _bench(bench, rule, 500.0, mtime=t0 + 120)
    assert refresh(table, bench) == (0, 0)
    assert load_table(table)[rule].peak_mb == 2000.0

    # A higher one raises the row. Its input size was recorded after
    # the benchmark (a later dry run), so it is not paired.
    _bench(bench, rule, 3000.0, mtime=t0 + 180)
    record_input_size(bench, rule, 99.0)
    os.utime(memory.ledger_path(bench, rule), (t0 + 240, t0 + 240))
    assert refresh(table, bench) == (0, 1)
    obs = load_table(table)[rule]
    assert obs.peak_mb == 3000.0
    assert obs.input_mb is None


def test_main_refresh_and_seed(tmp_path):
    bench = tmp_path / 'benchmarks'
    bench.mkdir()
    _bench(bench, 'curate_US_widget-2026_XX-AA-BBB', 700.0)
    table = tmp_path / 'mem.csv'
    lines = []
    code = memory.main(
        ['refresh', '--table', str(table), '--benchmarks', str(bench)],
        _print=lines.append,
    )
    assert code == 0
    assert '1 rows added' in lines[0]
    assert 'curate_US_widget-2026_XX-AA-BBB' in load_table(table)

    write_table(table, _family(SIX))
    out = tmp_path / 'seed.csv'
    code = memory.main(
        ['seed', '--table', str(table), '--out', str(out), '--scope', 'XX-AA'],
        _print=lines.append,
    )
    assert code == 0
    assert '1 families' in lines[1]
    assert ('harmonize', RECIPE) in load_seed(out)


def test_default_table_is_under_the_logs_directory():
    from openplaces.config import cfg

    path = memory.default_table_path()
    assert path.parent.parent == Path(cfg.get_dir('logs'))
    assert path.parent.name == 'mem'


def test_shipped_seed_is_well_formed():
    path = REPO / 'workflow' / 'profiles' / 'mem_seed.csv'
    seed = load_seed(path)
    assert seed, 'the seed table ships with the workflow'
    for (stage, _), env in seed.items():
        assert stage in ('ingest', 'harmonize', 'enrich', 'curate')
        assert env.n_measured >= memory.MIN_FAMILY_OBSERVATIONS
        assert env.slope >= 0
        assert env.floor_mb <= env.max_peak_mb
    # The measured table is per machine and stays out of git.
    assert not (REPO / 'workflow' / 'profiles' / 'mem_measured.csv').exists()
