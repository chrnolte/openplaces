"""Tests for the generic ground-truth validation helpers."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from openplaces.io.curator.validation import (
    classify_validation_result,
    link_points_to_entities,
    normalize_house_number,
    score_classification,
    summarize_sources,
)


def _box(lon, lat, size=0.0002):
    return Polygon(
        [(lon, lat), (lon + size, lat), (lon + size, lat + size), (lon, lat + size)]
    )


def test_normalize_house_number_bridges_float_and_string():
    # The trap this exists for: a number that round-tripped through a float
    # column never matches the string form, and fails silently.
    assert normalize_house_number(5114.0) == '5114'
    assert normalize_house_number('5114') == '5114'
    assert normalize_house_number('  5114 ') == '5114'
    assert normalize_house_number(None) is None
    assert normalize_house_number(float('nan')) is None


def test_address_match_beats_a_nearer_wrong_building():
    # The pin sits almost on top of entity 0 (a shed), but the address
    # identifies entity 1. Identity must win over proximity.
    entities = gpd.GeoDataFrame(
        {
            'address_number': ['99', '5114'],
            'address_street': ['OTHER ST', 'S VIRGINIA DARE TRL'],
            'occupancy_type': ['Secondary', 'Single-Family'],
        },
        geometry=[_box(-78.0, 35.0), _box(-78.01, 35.0)],
        crs='EPSG:4326',
    )
    points = pd.DataFrame(
        {
            'address_number': [5114.0],
            'address_street': ['SOUTH VIRGINIA DARE TRAIL'],
            'lon': [-78.0],
            'lat': [35.0],
        }
    )
    linked = link_points_to_entities(points, entities, admin1_id='US-NC')

    assert len(linked) == 1
    assert linked['matched_by'].iloc[0] == 'address'
    assert linked['occupancy_type_inv'].iloc[0] == 'Single-Family'


def test_distance_fallback_applies_and_respects_the_radius():
    entities = gpd.GeoDataFrame(
        {'occupancy_type': ['Single-Family']},
        geometry=[_box(-78.0, 35.0)],
        crs='EPSG:4326',
    )
    near = pd.DataFrame({'lon': [-78.0], 'lat': [35.0]})
    far = pd.DataFrame({'lon': [-78.5], 'lat': [35.5]})

    assert link_points_to_entities(near, entities)['matched_by'].iloc[0] == 'distance'
    assert link_points_to_entities(far, entities).empty


def test_entity_columns_are_suffixed_so_they_cannot_shadow_the_truth():
    # Both sides carry occupancy_type. If the entity value overwrote the
    # ground-truth one, every comparison would agree with itself.
    entities = gpd.GeoDataFrame(
        {'occupancy_type': ['Multi-Family']},
        geometry=[_box(-78.0, 35.0)],
        crs='EPSG:4326',
    )
    points = pd.DataFrame(
        {'occupancy_type': ['Single-Family'], 'lon': [-78.0], 'lat': [35.0]}
    )
    linked = link_points_to_entities(points, entities)

    assert linked['occupancy_type'].iloc[0] == 'Single-Family'
    assert linked['occupancy_type_inv'].iloc[0] == 'Multi-Family'


def test_validation_result_separates_omission_from_commission():
    truth = pd.Series(['Single-Family', 'Single-Family', 'Multi-Family'])
    predicted = pd.Series(['Single-Family', None, 'Single-Family'])
    result = classify_validation_result(truth, predicted)

    # Declining to answer and answering wrongly have different fixes, so they
    # get different labels.
    assert result.tolist() == ['correct', 'omission', 'commission']


def test_summarize_sources_renders_missing_values_rather_than_dropping_them():
    summary = summarize_sources(
        {
            'truth': pd.Series(['Multi-Family']),
            'nsi': pd.Series([None]),
            'fema': pd.Series(['Single-Family']),
        }
    )
    # A source that said nothing is itself evidence about the disagreement.
    assert summary.iloc[0] == 'truth: Multi-Family | nsi: - | fema: Single-Family'


def test_score_classification_separates_precision_from_recall():
    # Predicts Single-Family for everything: perfect recall for that class,
    # poor precision. Recall alone would call this a success.
    truth = pd.Series(['Single-Family', 'Single-Family', 'Multi-Family'])
    predicted = pd.Series(['Single-Family'] * 3)
    table = score_classification(
        truth, predicted, ['Single-Family', 'Multi-Family']
    ).set_index('class')

    # Scores are rounded to 4 places, so compare at that resolution.
    assert table.loc['Single-Family', 'recall'] == 1.0
    assert table.loc['Single-Family', 'precision'] == pytest.approx(2 / 3, abs=1e-4)
    assert table.loc['Multi-Family', 'recall'] == 0.0
    assert table.loc['ALL', 'recall'] == pytest.approx(2 / 3, abs=1e-4)


def test_score_classification_ignores_unpredicted_rows_in_recall():
    truth = pd.Series(['Single-Family', 'Single-Family'])
    predicted = pd.Series(['Single-Family', None])
    table = score_classification(truth, predicted, ['Single-Family']).set_index('class')

    # One of two was scored, and it was right: recall over scored rows is 1.0
    # while n_scored records that half the population went unanswered.
    assert table.loc['Single-Family', 'n_truth'] == 2
    assert table.loc['Single-Family', 'n_scored'] == 1
    assert table.loc['Single-Family', 'recall'] == 1.0
