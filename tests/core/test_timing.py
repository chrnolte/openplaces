import logging

import pytest

import openplaces.timing as timing
from openplaces.timing import Timer, clear_timers, get_timer


class _Clock:
    """Wall and CPU clocks that move only when a test advances them."""

    def __init__(self):
        self.wall = 1000.0
        self.cpu = 10.0

    def advance(self, wall: float = 0.0, cpu: float = 0.0) -> None:
        self.wall += wall
        self.cpu += cpu


@pytest.fixture
def clock(monkeypatch):
    # Patch the names the Timer looks up in its own module, so no real
    # time passing between two calls can change what a test sees.
    fake = _Clock()
    monkeypatch.setattr(timing, 'perf_counter', lambda: fake.wall)
    monkeypatch.setattr(timing, 'process_time', lambda: fake.cpu)
    return fake


def _timer_on(clock: _Clock, **kwargs) -> Timer:
    # The dataclass default factories hold the real clock functions,
    # bound at class definition, so the start is passed explicitly.
    return Timer(_created=clock.wall, _cpu_created=clock.cpu, **kwargs)


def test_mark_calls_are_contiguous_and_sum_to_total():
    timer = Timer(name='test_mark')
    timer.mark('a')
    timer.mark('b')
    timer.mark('c')
    timer.finish()
    assert timer.tracked_duration == pytest.approx(timer.total_duration, abs=1e-9)
    # No two consecutive records may leave a gap between them.
    for prev, nxt in zip(timer._records, timer._records[1:]):
        assert prev.end == nxt.start


def test_finish_is_idempotent_and_freezes_total_duration(clock):
    timer = _timer_on(clock, name='test_finish')
    timer.mark('a')
    timer.finish()
    frozen = timer.total_duration
    clock.advance(wall=5.0, cpu=5.0)
    timer.finish()  # no-op: already finished
    assert timer.total_duration == frozen


def test_finish_records_zero_length_trailing_gap():
    timer = Timer(name='test_zero_gap')
    timer.mark('a')
    n_records_before = len(timer._records)
    timer.finish()
    assert len(timer._records) == n_records_before + 1
    assert timer._records[-1].duration is not None


def _collect_log_messages(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    messages = []
    handler = logging.Handler()
    handler.emit = lambda record: messages.append(record.getMessage())
    logger.addHandler(handler)
    return logger, handler, messages


@pytest.mark.parametrize(
    ('wall', 'cpu'),
    [(0.02, 0.0), (0.0, 0.02)],
    ids=['wall', 'cpu'],
)
def test_finish_logs_a_nontrivial_gap_record_like_mark(clock, wall, cpu):
    logger, handler, messages = _collect_log_messages('test_finish_logs_nonzero')
    try:
        timer = _timer_on(clock, name='test_finish_logs_nonzero', logger=logger)
        timer.mark('a')
        clock.advance(wall=wall, cpu=cpu)
        timer.finish()
    finally:
        logger.removeHandler(handler)

    assert len(messages) == 2
    assert messages[0].startswith('a')
    assert messages[1].startswith('_final')


@pytest.mark.parametrize(
    ('wall', 'cpu'),
    [(0.0, 0.0), (0.004, 0.004)],
    ids=['zero', 'below-display-resolution'],
)
def test_finish_does_not_log_a_zero_length_gap(clock, wall, cpu):
    logger, handler, messages = _collect_log_messages('test_finish_logs_zero')
    try:
        timer = _timer_on(clock, name='test_finish_logs_zero', logger=logger)
        timer.mark('a')
        clock.advance(wall=wall, cpu=cpu)
        timer.finish()
    finally:
        logger.removeHandler(handler)

    assert len(messages) == 1
    assert messages[0].startswith('a')


def test_get_timer_overwrite_resets_tracked_duration():
    clear_timers()
    try:
        timer = get_timer('test_overwrite')
        timer.mark('a')
        fresh = get_timer('test_overwrite', overwrite=True)
        assert fresh.tracked_duration == 0
        assert fresh is not timer
    finally:
        clear_timers()
