"""Tests for the transformed-axis tick helper."""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from openplaces.viz.axes import add_log_ticks  # noqa: E402


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def test_sub_one_axis_gets_ticks(axes):
    """Shares, rates and ratios live below 1 and got no tick at all."""
    axes.set_xlim(np.arcsinh(0.001), np.arcsinh(0.5))
    add_log_ticks(axes)
    labels = [text.get_text() for text in axes.get_xticklabels()]
    assert labels
    assert '0.001' in labels
    assert '0.01' in labels
    assert '0' not in labels


def test_ticks_are_left_alone_when_no_decade_is_in_range(axes):
    """Setting ticks unconditionally wiped matplotlib's own defaults."""
    axes.set_xlim(np.arcsinh(2.0), np.arcsinh(8.0))
    before = list(axes.get_xticks())
    add_log_ticks(axes)
    assert list(axes.get_xticks()) == before


def test_decades_above_one_still_use_short_number(axes):
    axes.set_xlim(np.arcsinh(1), np.arcsinh(2e6))
    add_log_ticks(axes, prefix='$')
    labels = [text.get_text() for text in axes.get_xticklabels()]
    assert '$1M' in labels
