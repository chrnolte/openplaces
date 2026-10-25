"""Tests for the region verification delivery gate."""

from unittest.mock import Mock

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, box

import openplaces as op
from openplaces.flow import dag, freshness, region_verify

REGION = 'fabricated-region'
RECIPE = 'ZZ_property-fabricated-2026'
UNITS = ('ZZ-AA-ALP', 'ZZ-AA-BET', 'ZZ-AA-GAM')
EXCLUDED_BY_PROFILE = 'ZZ_parcel-optional-lane-2026'


def _geodataframe(*geometries):
    return gpd.GeoDataFrame(geometry=list(geometries), crs='EPSG:4326')


def _admin_frame(units=UNITS):
    return gpd.GeoDataFrame(
        geometry=[box(0, 0, 2, 2) for _ in units],
        index=list(units),
        crs='EPSG:4326',
    )


def _patch_region(monkeypatch, entities, units=UNITS):
    get_admin = Mock(return_value=_admin_frame(units))

    def get_entities(recipe_id, unit, geom):
        value = entities[(recipe_id, unit)]
        if isinstance(value, Exception):
            raise value
        assert geom is True
        return value

    monkeypatch.setattr(op, 'get_region_admin_ids', lambda region: list(units))
    monkeypatch.setattr(op, 'get_admin', get_admin)
    monkeypatch.setattr(op, 'get_entities', get_entities)
    return get_admin


def test_bbox_outliers_counts_only_rows_beyond_margin():
    gdf = _geodataframe(
        Point(1, 1),
        Point(-1.1, 1),
        Point(3.1, 1),
        Point(1, -1.1),
        Point(1, 3.1),
    )

    assert region_verify._bbox_outliers(gdf, (0, 0, 2, 2), margin=1) == 4


def test_bbox_outliers_does_not_count_rows_exactly_at_margin():
    gdf = _geodataframe(
        Point(-1, 1),
        Point(3, 1),
        Point(1, -1),
        Point(1, 3),
    )

    assert region_verify._bbox_outliers(gdf, (0, 0, 2, 2), margin=1) == 0


def test_verify_region_reports_pass_empty_and_missing_units(monkeypatch, capsys):
    entities = {
        (RECIPE, UNITS[0]): _geodataframe(Point(1, 1)),
        (RECIPE, UNITS[1]): _geodataframe(),
        (RECIPE, UNITS[2]): FileNotFoundError('fabricated missing output'),
    }
    get_admin = _patch_region(monkeypatch, entities)

    bad, total_rows = region_verify.verify_region(RECIPE, REGION)

    assert bad == [
        (UNITS[1], 'empty'),
        (UNITS[2], 'missing: FileNotFoundError'),
    ]
    assert total_rows == 1
    assert '1/3 verified, 1 rows total' in capsys.readouterr().out
    get_admin.assert_called_once_with('ZZ-AA', level=3, geom=True)


@pytest.mark.parametrize(
    ('n_outliers', 'expected_bad'),
    [
        (1, []),
        (2, []),
        (3, [(UNITS[0], '3 rows beyond 0.1 deg')]),
    ],
)
def test_verify_region_applies_max_outliers(
    monkeypatch, capsys, n_outliers, expected_bad
):
    outside = [Point(3, 1) for _ in range(n_outliers)]
    entities = {(RECIPE, UNITS[0]): _geodataframe(Point(1, 1), *outside)}
    _patch_region(monkeypatch, entities, units=(UNITS[0],))

    bad, total_rows = region_verify.verify_region(
        RECIPE,
        REGION,
        margin=0.1,
        max_outliers=2,
    )

    assert bad == expected_bad
    assert total_rows == n_outliers + 1
    output = capsys.readouterr().out
    if n_outliers <= 2:
        assert f'NOTE {UNITS[0]}: {n_outliers} outlier row(s) kept' in output
    else:
        assert 'NOTE' not in output


def test_verify_region_fails_bad_input_when_curated_output_passes(monkeypatch, capsys):
    input_recipe = 'ZZ_property-fabricated-input-2026'
    entities = {
        (RECIPE, UNITS[0]): _geodataframe(Point(1, 1)),
        (input_recipe, UNITS[0]): _geodataframe(Point(3, 1), Point(4, 1)),
    }
    _patch_region(monkeypatch, entities, units=(UNITS[0],))

    bad, total_rows = region_verify.verify_region(
        RECIPE,
        REGION,
        input_layers=[input_recipe],
        max_outliers=1,
    )

    assert bad == [(UNITS[0], f'{input_recipe}: 2 rows out of bounds')]
    assert total_rows == 1
    assert f'FAILED {UNITS[0]}' in capsys.readouterr().out


def test_verify_region_fails_an_input_layer_that_does_not_load(monkeypatch, capsys):
    input_recipe = 'ZZ_property-fabricated-input-2026'
    entities = {
        (RECIPE, UNITS[0]): _geodataframe(Point(1, 1)),
        (input_recipe, UNITS[0]): OSError('fabricated unreadable file'),
    }
    _patch_region(monkeypatch, entities, units=(UNITS[0],))

    bad, total_rows = region_verify.verify_region(
        RECIPE, REGION, input_layers=[input_recipe]
    )

    assert bad == [
        (
            UNITS[0],
            f'{input_recipe}: failed to load: OSError: fabricated unreadable file',
        )
    ]
    assert total_rows == 1
    assert f'FAILED {UNITS[0]}' in capsys.readouterr().out


def test_main_passes_graph_and_override_to_the_export(monkeypatch, main_dependencies):
    export_delivery, fake_dag, _, _ = main_dependencies
    monkeypatch.setattr(region_verify, 'verify_region', lambda *args, **kwargs: ([], 1))
    monkeypatch.setattr(
        'openplaces.io.delivery.delivery_paths', lambda *args, **kwargs: {}
    )

    region_verify.main([RECIPE, REGION, '--deliver', '--allow-stale'])

    export_delivery.assert_called_once_with(
        RECIPE, region=REGION, verbose=True, allow_stale=True, graph=fake_dag
    )


@pytest.fixture
def main_dependencies(monkeypatch):
    export_delivery = Mock()
    fake_dag = object()
    recipe_dag = Mock(return_value=fake_dag)
    out_of_date = Mock(return_value=pd.DataFrame())

    monkeypatch.setattr(op, 'get_region_admin_ids', lambda region: [UNITS[0]])
    monkeypatch.setattr(op, 'export_delivery', export_delivery)
    monkeypatch.setattr(dag, 'RecipeDAG', recipe_dag)
    monkeypatch.setattr(freshness, 'out_of_date', out_of_date)
    monkeypatch.setattr(freshness, 'freshness_graph', lambda graph: graph)
    monkeypatch.setattr(
        freshness, 'default_exclude_recipe_ids', lambda: {EXCLUDED_BY_PROFILE}
    )
    return export_delivery, fake_dag, recipe_dag, out_of_date


def test_main_exits_one_when_verification_fails(monkeypatch, main_dependencies):
    export_delivery, _, recipe_dag, out_of_date = main_dependencies
    monkeypatch.setattr(
        region_verify,
        'verify_region',
        lambda *args, **kwargs: ([(UNITS[0], 'empty')], 0),
    )

    with pytest.raises(SystemExit) as excinfo:
        region_verify.main([RECIPE, REGION])

    assert excinfo.value.code == 1
    recipe_dag.assert_not_called()
    out_of_date.assert_not_called()
    export_delivery.assert_not_called()


def test_main_exits_one_for_stale_outputs(monkeypatch, main_dependencies):
    export_delivery, fake_dag, recipe_dag, out_of_date = main_dependencies
    monkeypatch.setattr(region_verify, 'verify_region', lambda *args, **kwargs: ([], 1))
    out_of_date.return_value = pd.DataFrame(
        [{'recipe_id': RECIPE, 'reason': 'fabricated newer input'}]
    )

    with pytest.raises(SystemExit) as excinfo:
        region_verify.main([RECIPE, REGION])

    assert excinfo.value.code == 1
    recipe_dag.assert_called_once_with(
        RECIPE,
        admin_ids=[UNITS[0]],
        exclude_recipe_ids={EXCLUDED_BY_PROFILE},
        deliver=False,
        validate=False,
    )
    out_of_date.assert_called_once_with(fake_dag)
    export_delivery.assert_not_called()


@pytest.mark.parametrize(
    ('flag', 'expected'),
    [
        ('ZZ_lane-a-2026,ZZ_lane-b-2026', {'ZZ_lane-a-2026', 'ZZ_lane-b-2026'}),
        ('', set()),
    ],
)
def test_main_takes_the_run_exclusions(monkeypatch, main_dependencies, flag, expected):
    _, _, recipe_dag, _ = main_dependencies
    monkeypatch.setattr(region_verify, 'verify_region', lambda *args, **kwargs: ([], 1))

    region_verify.main([RECIPE, REGION, f'--exclude-recipe-ids={flag}'])

    assert recipe_dag.call_args.kwargs['exclude_recipe_ids'] == expected


def test_main_allow_stale_continues_without_delivery(
    monkeypatch, main_dependencies, capsys
):
    export_delivery, _, _, out_of_date = main_dependencies
    monkeypatch.setattr(region_verify, 'verify_region', lambda *args, **kwargs: ([], 1))
    out_of_date.return_value = pd.DataFrame(
        [{'recipe_id': RECIPE, 'reason': 'fabricated newer input'}]
    )

    region_verify.main([RECIPE, REGION, '--allow-stale'])

    assert 'dry verification only' in capsys.readouterr().out
    export_delivery.assert_not_called()


def test_main_never_delivers_without_deliver_flag(
    monkeypatch, main_dependencies, capsys
):
    export_delivery, _, _, _ = main_dependencies
    monkeypatch.setattr(region_verify, 'verify_region', lambda *args, **kwargs: ([], 1))

    region_verify.main([RECIPE, REGION])

    assert 'dry verification only' in capsys.readouterr().out
    export_delivery.assert_not_called()
