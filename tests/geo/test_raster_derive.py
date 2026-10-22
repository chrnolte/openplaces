"""Tests for the declarative annual-raster derivation in geo.raster_derive."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from openplaces.geo.raster_derive import (
    NODATA,
    and_not,
    annual_transition,
    any_in_years,
    band_equals,
    cumulative_no_gain,
    derive_layers,
    lossyear_equals,
    lossyear_to_cover,
)

N = NODATA


def test_band_equals_marks_value_and_keeps_nodata():
    band = np.array([[1, 3], [3, 7]], dtype='uint8')
    out = band_equals(band, 3, valid=band != 7)
    assert out.tolist() == [[0, 1], [1, N]]


def test_annual_transition_loss_and_gain():
    before = np.array([[1, 1, 0, N]], dtype='uint8')
    after = np.array([[0, 1, 1, 1]], dtype='uint8')
    assert annual_transition(before, after, 1, 0).tolist() == [[1, 0, 0, N]]
    assert annual_transition(before, after, 0, 1).tolist() == [[0, 0, 1, N]]


def test_cumulative_no_gain_never_readmits_regrowth():
    # Pixel 0: forest throughout. Pixel 1: lost then regrown, must stay 0.
    # Pixel 2: non-forest throughout.
    y0 = np.array([[1, 1, 0]], dtype='uint8')
    y1 = np.array([[1, 0, 0]], dtype='uint8')
    y2 = np.array([[1, 1, 0]], dtype='uint8')
    ng1 = cumulative_no_gain(y0, y1)
    ng2 = cumulative_no_gain(ng1, y2)
    assert ng1.tolist() == [[1, 0, 0]]
    assert ng2.tolist() == [[1, 0, 0]]


def test_and_not_is_regrowing_forest():
    # Forest now: pixels 0, 1. Forest since the start: pixel 0 only.
    # Pixel 1 is regrowing; pixel 2 is not forest; pixel 3 unobserved.
    forest = np.array([[1, 1, 0, N]], dtype='uint8')
    since_start = np.array([[1, 0, 0, N]], dtype='uint8')
    assert and_not(forest, since_start).tolist() == [[0, 1, 0, N]]


def test_any_in_years_union_and_nodata():
    a = np.array([[1, 0, N, N]], dtype='uint8')
    b = np.array([[0, 0, 0, N]], dtype='uint8')
    assert any_in_years([a, b]).tolist() == [[1, 0, 0, N]]


def test_lossyear_helpers():
    lossyear = np.array([[0, 5, 12]], dtype='uint8')  # none, 2005, 2012
    cover = np.array([[1, 1, 0]], dtype='uint8')
    assert lossyear_equals(lossyear, 2005).tolist() == [[0, 1, 0]]
    assert lossyear_to_cover(lossyear, cover, 2004).tolist() == [[1, 1, 0]]
    assert lossyear_to_cover(lossyear, cover, 2005).tolist() == [[1, 0, 0]]


@pytest.fixture
def multiband_source(tmp_path):
    """A 4-band (years 2000-2003) 2x2 land-cover product, class 1 = forest.

    Pixel (0,0): forest all years. Pixel (0,1): forest, lost in 2002,
    regrown 2003. Pixel (1,0): non-forest, gained 2001. Pixel (1,1):
    nodata in 2001 only.
    """
    stack = np.array(
        [
            [[1, 1], [2, 1]],  # 2000
            [[1, 1], [1, 0]],  # 2001 (0 = nodata below)
            [[1, 2], [1, 1]],  # 2002
            [[1, 1], [1, 1]],  # 2003
        ],
        dtype='uint8',
    )
    path = tmp_path / 'product.tif'
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        width=2,
        height=2,
        count=4,
        dtype='uint8',
        nodata=0,
        crs='EPSG:3116',
        transform=from_origin(0, 2, 1, 1),
    ) as dst:
        for band in range(4):
            dst.write(stack[band], band + 1)
    return path


def test_derive_layers_end_to_end(multiband_source, tmp_path):
    specs = [
        {'layer': 'forest', 'op': 'band_equals', 'value': 1, 'years': [2000, 2003]},
        {
            'layer': 'deforestation',
            'op': 'annual_transition',
            'from': 'forest',
            'before': 1,
            'after': 0,
            'years': [2001, 2003],
        },
        {
            'layer': 'gain',
            'op': 'annual_transition',
            'from': 'forest',
            'before': 0,
            'after': 1,
            'years': [2001, 2003],
        },
        {
            'layer': 'forest_no_gain',
            'op': 'cumulative_no_gain',
            'from': 'forest',
            'years': [2000, 2003],
        },
        {
            'layer': 'deforestation_prior_2y',
            'op': 'any_in_prior_years',
            'from': 'deforestation',
            'n_years': 2,
            'years': [2003, 2003],
        },
    ]
    out = tmp_path / 'derived'
    written = derive_layers(multiband_source, out, specs, first_year=2000)
    assert set(written) == {
        *[('forest', y) for y in range(2000, 2004)],
        *[('deforestation', y) for y in range(2001, 2004)],
        *[('gain', y) for y in range(2001, 2004)],
        *[('forest_no_gain', y) for y in range(2000, 2004)],
        ('deforestation_prior_2y', 2003),
    }

    def read(layer, year):
        with rasterio.open(written[(layer, year)]) as src:
            assert src.nodata == N
            assert src.tags()['OPENPLACES_LAYER'] == layer
            return src.read(1).tolist()

    assert read('forest', 2000) == [[1, 1], [0, 1]]
    assert read('forest', 2001) == [[1, 1], [1, N]]
    assert read('deforestation', 2002) == [[0, 1], [0, N]]
    assert read('gain', 2001) == [[0, 0], [1, N]]
    assert read('gain', 2003) == [[0, 1], [0, 0]]
    # Regrowth in 2003 does not re-enter the no-gain layer.
    assert read('forest_no_gain', 2003) == [[1, 0], [0, 1]]
    # 2003 looks back at 2001-2002 deforestation: pixel (0,1) lost in 2002.
    assert read('deforestation_prior_2y', 2003) == [[0, 1], [0, N]]


def test_unknown_op_raises(multiband_source, tmp_path):
    with pytest.raises(ValueError, match='unknown op'):
        derive_layers(
            multiband_source,
            tmp_path,
            [{'layer': 'x', 'op': 'nope', 'years': [2000, 2000]}],
            first_year=2000,
        )
