"""Tests for the graduated-renderer retarget in the QGIS generator."""

import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from openplaces.viz.qgis_map.generator import _retarget_graduated


def _clone():
    clone = ET.Element('maplayer')
    renderer = ET.SubElement(clone, 'renderer-v2', {'type': 'graduatedSymbol'})
    ET.SubElement(renderer, 'ranges')
    ET.SubElement(renderer, 'symbols')
    return clone


@pytest.fixture
def values_parquet(tmp_path):
    """A value column in stored units, spread over four decades."""
    path = tmp_path / 'values.parquet'
    pd.DataFrame({'structure_value': [1e4 * i for i in range(1, 121)]}).to_parquet(path)
    return path


def test_quantile_breaks_are_in_the_rendered_unit(values_parquet):
    """The renderer classifies a scaled expression, so breaks must scale.

    A registry row with a scale and no explicit breaks classified in raw
    units, so every feature fell outside every class and the layer
    rendered empty.
    """
    clone = _clone()
    _retarget_graduated(
        clone,
        values_parquet,
        'structure_value',
        scale='0.001',
        verbose=False,
    )
    renderer = clone.find('.//renderer-v2')
    assert renderer.get('attr') == '"structure_value" * 0.001'

    ranges = renderer.find('ranges').findall('range')
    assert ranges
    lower = float(ranges[0].get('lower'))
    upper = float(ranges[-1].get('upper'))
    # Data runs 10,000 to 1,200,000; displayed in thousands.
    assert lower == pytest.approx(10.0)
    assert upper == pytest.approx(1200.0)


def test_unscaled_breaks_are_unchanged(values_parquet):
    clone = _clone()
    _retarget_graduated(clone, values_parquet, 'structure_value', verbose=False)
    ranges = clone.find('.//renderer-v2').find('ranges').findall('range')
    assert float(ranges[0].get('lower')) == pytest.approx(10000.0)
    assert float(ranges[-1].get('upper')) == pytest.approx(1200000.0)
