"""A measured height counts the nearest whole number of floors."""

import numpy as np
import pandas as pd

from openplaces.io.curator.evidence import _measured_floors


def _floors(heights, stories=None):
    frame = pd.DataFrame({'height': heights})
    if stories is not None:
        frame['n_stories'] = stories
    return _measured_floors(
        frame,
        'n_stories' if stories is not None else None,
        'height',
        3.5,
    ).tolist()


def test_a_pitched_roof_is_not_a_floor():
    # 4 m over 3.5 m per floor is 1.14 floors: one, not two.
    assert _floors([4.0, 3.4, 7.1]) == [1.0, 1.0, 2.0]


def test_half_a_floor_rounds_up():
    assert _floors([5.25, 8.75]) == [2.0, 3.0]


def test_a_low_height_is_still_one_floor():
    assert _floors([1.0]) == [1.0]


def test_no_height_and_no_count_is_missing():
    out = _floors([np.nan, 0.0])
    assert np.isnan(out[0]) and np.isnan(out[1])


def test_the_larger_of_count_and_height_wins():
    # A count left at its default beside a tall measured height, and a
    # stated count above a low height.
    assert _floors([35.0, 4.0], stories=[1.0, 3.0]) == [10.0, 3.0]


def test_a_stated_fraction_rounds_up():
    assert _floors([np.nan], stories=[2.5]) == [3.0]
