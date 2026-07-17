import time

import pytest

from openplaces.timing import Timer, clear_timers, get_timer


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


def test_finish_is_idempotent_and_freezes_total_duration():
    timer = Timer(name='test_finish')
    timer.mark('a')
    timer.finish()
    frozen = timer.total_duration
    time.sleep(0.01)
    timer.finish()  # no-op: already finished
    assert timer.total_duration == frozen


def test_finish_records_zero_length_trailing_gap():
    timer = Timer(name='test_zero_gap')
    timer.mark('a')
    n_records_before = len(timer._records)
    timer.finish()
    assert len(timer._records) == n_records_before + 1
    assert timer._records[-1].duration is not None


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
