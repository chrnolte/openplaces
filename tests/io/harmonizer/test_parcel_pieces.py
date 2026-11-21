"""Outlines cut into one building per parcel piece (b3 step 2).

plans/buildings-from-parcel-pieces.md, sections 3 and 4 (steps 2, 3
and 5). Every geometry, id and value below is fabricated: boxes in an
equal-area projection near the origin, made-up lot ids and values; no
record of any source is used (docs/5_contribute/no-personal-data.rst).
The overlay rows are made by the same functions the footprint geospine
runs (`overlay_polygons`, `_build_crosswalk`), so the input has the
shape of a real sidecar.
"""

from __future__ import annotations

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, box

from openplaces.core.schema import Entity
from openplaces.geo.polygon import overlay_polygons
from openplaces.io.harmonizer import HarmonizeState
from openplaces.io.harmonizer import load as load_module
from openplaces.io.harmonizer.apportion import apportion_reference_values
from openplaces.io.harmonizer.entity_links import build_id_links
from openplaces.io.harmonizer.links import parcel_pieces as pp
from openplaces.io.harmonizer.links.spatial import _build_crosswalk

CRS = 'epsg:6933'
MIN_FRACTION = 0.1667
AREA_MIN = 10.0
RULE = pp.parse_keep_piece(
    {
        'improvement_value_min': 5000,
        'structure_flag_values': ['Y', 'TRUE'],
        'compactness_max': 0.95,
    }
)
CONTEXT = pp.SplitContext(
    links_recipe={},
    links_key='footprint_id',
    overlay_step={},
    min_fraction=MIN_FRACTION,
    area_min_m2=AREA_MIN,
    values_recipe=None,
    fingerprint_config={},
)


def _lots(**geoms):
    return gpd.GeoDataFrame(
        geometry=list(geoms.values()),
        index=pd.Index(list(geoms), name='parcel_id'),
        crs=CRS,
    )


def _spine(sources=None, **outlines):
    sources = sources or {}
    ids = list(outlines)
    return gpd.GeoDataFrame(
        {
            'footprint_id': ids,
            'geometry_source': [sources.get(i, 'obm') for i in ids],
            'admin4_id': 'US-XX-YY-A',
        },
        geometry=list(outlines.values()),
        index=pd.Index(ids, name='building_id'),
        crs=CRS,
    )


def _flat(spine, lots):
    """The footprint geospine's overlay sidecar for *spine* on *lots*."""
    outlines = gpd.GeoDataFrame(
        geometry=spine.geometry.to_numpy(),
        index=pd.Index(spine['footprint_id'].to_numpy(), name='footprint_id'),
        crs=CRS,
    )
    overlay = overlay_polygons(
        outlines,
        lots,
        suffixes=('_spine', '_ref'),
        how='identity',
        iou=True,
        geom=False,
    )
    crosswalk = _build_crosswalk(overlay, 'footprint_id', MIN_FRACTION, AREA_MIN)
    return overlay.join(crosswalk['link']).reset_index()


def _values(**lots):
    """Lot evidence: improvement value, structure flag."""
    return pd.DataFrame(
        [(v, f) for v, f in lots.values()],
        columns=[pp.IMPROVEMENT_VALUE_COLUMN, pp.STRUCTURE_FLAG_COLUMN],
        index=pd.Index(list(lots), name='parcel_id'),
    )


def _evidence(points, values):
    point_layer = gpd.GeoSeries(points, crs=CRS)

    def evidence_for(others):
        lot = values.reindex(others['parcel_id'].to_numpy())
        return pd.DataFrame(
            {
                'has_point': pp.points_inside(
                    others.geometry, [point_layer]
                ).to_numpy(),
                'improvement_value': lot[pp.IMPROVEMENT_VALUE_COLUMN].to_numpy(),
                'structure_flag': lot[pp.STRUCTURE_FLAG_COLUMN].to_numpy(),
                'compactness': pp.compactness(others.geometry).to_numpy(),
            },
            index=others.index,
        )

    return evidence_for


def _split(spine, lots, points=(), values=None, rule=RULE):
    flat = _flat(spine, lots)
    values = values if values is not None else _values()
    taken = set(spine.index) | set(flat['footprint_id'].dropna())
    return pp.split_overlay(
        spine,
        flat,
        lots.geometry,
        CONTEXT,
        _evidence(list(points), values),
        rule,
        taken,
    )


def _row_of_four():
    spine = _spine(f1=box(0, 0, 40, 10))
    # Lots of slightly different depth: `_build_crosswalk` drops
    # overlay rows equal in every column (it reads them as copies of
    # one pair), so equal lots would lose pairs before the cut.
    lots = _lots(**{f'P{i}': box(10 * i, -5 - i, 10 * i + 10, 15) for i in range(4)})
    points = [Point(10 * i + 5, 5) for i in range(4)]
    values = _values(**{f'P{i}': (100000.0, None) for i in range(4)})
    return spine, lots, points, values


# The cut (section 3, items 1 to 3)


def test_a_row_of_four_on_four_lots_gives_four_buildings():
    spine, lots, points, values = _row_of_four()
    out, flat, stats = _split(spine, lots, points, values)
    assert len(out) == 4
    assert set(out['footprint_id']) == {'f1'}
    assert set(out[pp.BUILDING_GEOMETRY_COLUMN]) == {pp.PARCEL_PIECE}
    assert np.allclose(sorted(out.geometry.area), [100.0] * 4)
    assert stats['outlines_cut'] == 1
    assert stats['buildings_added'] == 3
    # Each building holds one lot's pair, labeled a unique parcel now.
    labeled = flat[flat['link'].notna()]
    assert labeled.groupby('building_id')['parcel_id'].nunique().eq(1).all()
    assert set(labeled['link']) == {'unique parcel'}


def test_the_pieces_sum_to_the_outline():
    spine, lots, points, values = _row_of_four()
    out, _, _ = _split(spine, lots, points, values)
    assert out.geometry.area.sum() == pytest.approx(spine.geometry.area.sum())
    assert out.geometry.union_all().symmetric_difference(
        spine.geometry.iloc[0]
    ).area == pytest.approx(0.0, abs=1e-6)


def test_a_piece_below_the_area_threshold_after_subtraction_folds():
    # The second lot overlaps the first; what is left of it on the
    # outline is 5 m2, under the overlay's 10 m2.
    spine = _spine(f1=box(0, 0, 20, 10))
    lots = _lots(L1=box(0, -5, 12, 15), L2=box(1, -5, 12.5, 15))
    out, flat, stats = _split(
        spine, lots, [Point(12.2, 5)], _values(L2=(90000.0, None))
    )
    assert out.index.tolist() == ['f1']
    assert out[pp.BUILDING_GEOMETRY_COLUMN].tolist() == [pp.OUTLINE]
    assert stats['outlines_cut'] == 0
    assert stats['pieces_folded'] == 1


def test_a_piece_below_the_fraction_threshold_after_subtraction_folds():
    # 70 m2 left of the second lot against 530 m2 on the first: above
    # 10 m2, under a sixth.
    spine = _spine(f1=box(0, 0, 60, 10))
    lots = _lots(L1=box(0, -5, 53, 15), L2=box(44, -5, 60, 15))
    out, _, stats = _split(spine, lots, [Point(57, 5)], _values(L2=(90000.0, None)))
    assert out.index.tolist() == ['f1']
    assert stats['pieces_folded'] == 1


def test_the_off_parcel_remainder_joins_the_dominant_piece():
    # The lots stop 2 m short of the outline's top edge (a street).
    spine = _spine(f1=box(0, 0, 40, 10))
    lots = _lots(L1=box(0, -5, 20, 8), L2=box(20, -6, 40, 8))
    out, flat, _ = _split(spine, lots, [Point(30, 4)], _values(L2=(90000.0, None)))
    assert len(out) == 2
    areas = sorted(out.geometry.area)
    assert areas == pytest.approx([160.0, 240.0])
    # The remainder's row (no parcel) moved to the dominant piece.
    remainder = flat[flat['parcel_id'].isna()]
    dominant = out.index[out.geometry.area.round() == 240][0]
    assert remainder['building_id'].tolist() == [dominant]
    assert remainder['footprint_id'].tolist() == ['f1']


def test_overlapping_lots_partition_without_double_area():
    spine = _spine(f1=box(0, 0, 40, 10))
    lots = _lots(A=box(0, -5, 25, 15), B=box(15, -6, 40, 15))
    out, _, _ = _split(spine, lots, [Point(35, 5)], _values(B=(90000.0, None)))
    assert len(out) == 2
    assert sorted(out.geometry.area) == pytest.approx([150.0, 250.0])
    first, second = out.geometry.iloc[0], out.geometry.iloc[1]
    assert first.intersection(second).area == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize('source', ['condo_cluster.nconemap', 'parcel.spine'])
def test_condo_cluster_and_inferred_rows_are_not_cut(source):
    spine, lots, points, values = _row_of_four()
    spine['geometry_source'] = source
    out, flat, stats = _split(spine, lots, points, values)
    assert out.index.tolist() == ['f1']
    assert stats['candidates'] == 0
    assert (flat['building_id'] == 'f1').all()


def test_a_snapped_outline_is_not_cut():
    spine = _spine(f1=box(0, 0, 20, 10))
    lots = _lots(L1=box(0, -5, 12, 15), L2=box(12, -5, 20, 15))
    flat = _flat(spine, lots)
    # As snap_chained_links leaves it: the dominant pair relabeled,
    # the minor one unlabeled and marked.
    flat['link_chain'] = pd.Series(pd.NA, index=flat.index, dtype=object)
    flat.loc[flat['parcel_id'] == 'L1', 'link'] = 'unique parcel (snapped from chain)'
    flat.loc[flat['parcel_id'] == 'L1', 'link_chain'] = 'snapped dominant'
    flat.loc[flat['parcel_id'] == 'L2', 'link'] = None
    flat.loc[flat['parcel_id'] == 'L2', 'link_chain'] = 'snapped minor'
    out, new_flat, stats = pp.split_overlay(
        spine,
        flat,
        lots.geometry,
        CONTEXT,
        _evidence([Point(16, 5)], _values(L2=(90000.0, None))),
        RULE,
        {'f1'},
    )
    assert stats['candidates'] == 0
    assert out.index.tolist() == ['f1']
    assert new_flat['link_chain'].tolist() == flat['link_chain'].tolist()


def test_a_cross_town_piece_is_cut_again_and_keeps_its_envelope():
    spine, lots, points, values = _row_of_four()
    spine['envelope_id'] = 'ENVELOPE0+00'
    out, _, _ = _split(spine, lots, points, values)
    assert len(out) == 4
    assert set(out['envelope_id']) == {'ENVELOPE0+00'}


# Ids (item 4)


def test_ids_are_distinct_unsuffixed_and_identical_on_a_rerun():
    spine, lots, points, values = _row_of_four()
    first, flat_one, _ = _split(spine, lots, points, values)
    second, flat_two, _ = _split(spine, lots, points, values)
    ids = first.index.tolist()
    assert len(set(ids)) == len(ids)
    assert 'f1' not in ids
    assert not any(re.search(r'-\d+$', i) for i in ids)
    assert ids == second.index.tolist()
    pd.testing.assert_frame_equal(flat_one, flat_two)


def test_a_piece_id_never_takes_an_existing_id():
    spine, lots, points, values = _row_of_four()
    first, _, _ = _split(spine, lots, points, values)
    # Name another outline after one of the pieces' codes: the cut has
    # to choose a finer code for that piece.
    clash = first.index[0]
    other = _spine(**{clash: box(100, 0, 110, 10)})
    both = pd.concat([spine, other])
    out, _, _ = _split(both, lots, points, values)
    pieces = out[out['footprint_id'] == 'f1'].index
    assert clash not in pieces
    assert len(pieces) == 4


# Re-keying at the pair (item 5)


def test_ghost_pairs_keep_their_id():
    """An outline in the sidecar but not on the spine (a later geometry
    step of the footprint geospine removed it) keeps its pairs as they
    were."""
    spine, lots, points, values = _row_of_four()
    ghost = _spine(g9=box(0, 20, 5, 25))
    flat = _flat(pd.concat([spine, ghost]), lots)
    out, new_flat, _ = pp.split_overlay(
        spine,
        flat,
        lots.geometry,
        CONTEXT,
        _evidence(points, values),
        RULE,
        set(flat['footprint_id']),
    )
    rows = new_flat[new_flat['footprint_id'] == 'g9']
    assert rows['building_id'].tolist() == ['g9'] * len(rows)
    assert len(new_flat) == len(flat)


def test_a_moved_row_is_measured_on_its_piece():
    spine = _spine(f1=box(0, 0, 40, 10))
    lots = _lots(L1=box(0, -5, 20, 15), L2=box(20, -6, 40, 15))
    out, flat, _ = _split(spine, lots, [Point(30, 5)], _values(L2=(90000.0, None)))
    assert len(out) == 2
    flat = flat.sort_values('parcel_id')
    assert flat['area_spine_m2'].tolist() == pytest.approx([200.0, 200.0])
    assert flat['area_intersection_m2'].tolist() == pytest.approx([200.0, 200.0])
    # A 200 m2 piece on a 400 m2 and on a 420 m2 lot; the outline's
    # own iou was 200 / 600 and 200 / 620.
    assert flat['iou'].tolist() == pytest.approx([200 / 400, 200 / 420])


def test_the_one_parcel_case_is_todays_building_geospine():
    """Outlines on one lot each: the spine gains only the marker, and the
    overlay is the footprint geospine's, renamed and re-keyed one to one,
    with the outline id kept as the last column."""
    spine = _spine(f1=box(0, 0, 10, 10), f2=box(20, 0, 30, 10))
    lots = _lots(L1=box(-5, -5, 15, 15), L2=box(15, -6, 35, 15))
    flat = _flat(spine, lots)
    out, new_flat, stats = _split(spine, lots)
    assert stats['candidates'] == 0
    pd.testing.assert_frame_equal(out.drop(columns=pp.BUILDING_GEOMETRY_COLUMN), spine)
    assert set(out[pp.BUILDING_GEOMETRY_COLUMN]) == {pp.OUTLINE}
    today = load_module._rekey_frame(
        flat,
        'footprint_id',
        'building_id',
        pd.Series(spine.index, index=spine['footprint_id']),
    )
    pd.testing.assert_frame_equal(
        new_flat.drop(columns='footprint_id'), today, check_dtype=False
    )
    assert list(new_flat.columns)[-1] == 'footprint_id'


# The refusal rule (decision D2)


def _duplex(point=(15, 5), value=(80000.0, None), right=None):
    spine = _spine(f1=box(0, 0, 20, 10))
    lots = _lots(L1=box(0, -5, 10, 15), L2=right or box(10, -6, 20, 15))
    points = [Point(*point)] if point is not None else []
    return _split(spine, lots, points, _values(L1=(80000.0, None), L2=value))


def test_a_duplex_half_with_a_point_and_a_value_is_a_building():
    out, _, stats = _duplex()
    assert len(out) == 2
    assert stats['pieces_refused'] == 0


def test_a_piece_without_a_point_folds():
    out, _, stats = _duplex(point=None)
    assert out.index.tolist() == ['f1']
    assert stats['pieces_refused'] == 1


def test_a_point_on_the_dominant_piece_only_does_not_keep_the_other():
    out, _, _ = _duplex(point=(5, 5))
    assert out.index.tolist() == ['f1']


def test_a_lot_valued_below_the_floor_folds():
    out, _, _ = _duplex(value=(3000.0, None))
    assert out.index.tolist() == ['f1']


def test_a_lot_without_a_value_counts_with_a_structure_flag():
    out, _, _ = _duplex(value=(np.nan, 'Y'))
    assert len(out) == 2
    out, _, _ = _duplex(value=(np.nan, True))
    assert len(out) == 2
    out, _, _ = _duplex(value=(np.nan, 'N'))
    assert out.index.tolist() == ['f1']


def test_a_stated_zero_is_not_overridden_by_the_flag():
    out, _, _ = _duplex(value=(0.0, 'Y'))
    assert out.index.tolist() == ['f1']


def test_a_round_piece_folds():
    """A share drawn as a circle inside the home, on a lot of its own."""
    circle = Point(10, 5).buffer(4, quad_segs=32)
    spine = _spine(f1=box(0, 0, 20, 10))
    lots = _lots(L1=box(-5, -5, 25, 15).difference(circle), L2=circle)
    evidence = _evidence([Point(10, 5)], _values(L2=(80000.0, None)))
    flat = _flat(spine, lots)
    out, _, stats = pp.split_overlay(
        spine, flat, lots.geometry, CONTEXT, evidence, RULE, {'f1'}
    )
    assert out.index.tolist() == ['f1']
    assert stats['pieces_refused'] == 1


def test_the_rule_needs_its_thresholds_from_the_recipe():
    with pytest.raises(ValueError, match='missing'):
        pp.parse_keep_piece({'improvement_value_min': 5000})
    with pytest.raises(ValueError, match='compactness_max'):
        pp.parse_keep_piece(
            {
                'improvement_value_min': 1,
                'structure_flag_values': [],
                'compactness_max': 2,
            }
        )


def test_keep_pieces_reads_each_condition():
    evidence = pd.DataFrame(
        {
            'has_point': [True, False, True, True, True],
            'improvement_value': [6000, 6000, 4999, np.nan, 6000],
            'structure_flag': [None, None, None, 'y', None],
            'compactness': [0.7, 0.7, 0.7, 0.7, 0.99],
        }
    )
    assert pp.keep_pieces(evidence, RULE).tolist() == [
        True,
        False,
        False,
        True,
        False,
    ]


def test_lot_values_run_the_value_recipes_passes_on_the_raw_lots(monkeypatch):
    """The passes see each lot as resolve_spine keeps it: its first row,
    a missing value missing (not summed to 0, so a later pass can fill
    it), and the reference as its geometry source."""
    seen = []

    def fake_link_by_id(state, **params):
        seen.append((params, state.spine.copy()))
        if params.get('entity_type') == 'property':
            state.spine.loc['L2', 'improvement_value'] = 7000.0
        return state

    monkeypatch.setattr(pp, 'link_by_id', fake_link_by_id)
    values_recipe = {
        'pipeline': [
            {
                'step': 'resolve_spine',
                'keep_columns': ['parcel_id_local', 'improvement_value'],
                'track_provenance': ['improvement_value'],
            },
            {'step': 'link_by_id', 'auto_discover': True, 'entity_type': 'parcel'},
            {'step': 'link_by_id', 'recipe_id': 'US_property-spine-2026'},
            {
                'step': 'link_by_id',
                'auto_discover': True,
                'entity_type': 'property',
                'mode': 'aggregate',
            },
        ]
    }
    raw = pd.DataFrame(
        {
            'geo_id': ['L1', 'L1', 'L2'],
            'parcel_id_local': ['A-1', 'A-1', 'A-2'],
            'improvement_value': [9000.0, 1.0, np.nan],
            'owner_name': ['x', 'y', 'z'],
        }
    )
    state = HarmonizeState(recipe={}, admin_id='US-XX-YY', verbose=False, timer=None)
    out = pp.lot_structure_evidence(state, raw, 'US-XX_parcel-test-2026', values_recipe)
    # Only the two auto_discover passes ran, in order.
    assert [p.get('entity_type') for p, _ in seen] == ['parcel', 'property']
    first = seen[0][1]
    assert first.index.tolist() == ['L1', 'L2']
    assert 'owner_name' not in first.columns
    assert first.loc['L1', 'improvement_value'] == 9000.0
    assert pd.isna(first.loc['L2', 'improvement_value'])
    assert set(first['geometry_source']) == {'test'}
    assert first['improvement_value_source'].tolist()[0] == 'test'
    assert out.loc['L2', pp.IMPROVEMENT_VALUE_COLUMN] == 7000.0
    assert pd.isna(out.loc['L1', pp.STRUCTURE_FLAG_COLUMN])


# Consumers (step 3)


def test_a_point_on_a_piece_falls_in_that_building():
    """The point joins' first pass is containment in the building's
    polygon, which is now the piece."""
    spine, lots, points, values = _row_of_four()
    out, _, _ = _split(spine, lots, points, values)
    hits = gpd.sjoin(
        gpd.GeoDataFrame(geometry=points, crs=CRS),
        out[['geometry']],
        predicate='within',
    )
    assert hits['building_id'].nunique() == 4


def test_each_piece_takes_its_own_lots_value_and_lot_totals_hold():
    spine, lots, points, values = _row_of_four()
    out, flat, _ = _split(spine, lots, points, values)
    pairs = flat[flat['link'].notna()][
        ['building_id', 'parcel_id', 'area_intersection_m2']
    ]
    lot_values = pd.DataFrame(
        {'improvement_value': [100.0, 200.0, 300.0, 400.0]},
        index=pd.Index(['P0', 'P1', 'P2', 'P3'], name='parcel_id'),
    )
    result = apportion_reference_values(pairs, lot_values, spine_id_col='building_id')
    by_lot = pairs.set_index('building_id')['parcel_id']
    got = result['improvement_value'].reindex(by_lot.index)
    assert got.tolist() == lot_values['improvement_value'].reindex(by_lot).tolist()
    assert got.sum() == pytest.approx(lot_values['improvement_value'].sum())


def test_the_footprint_link_has_one_row_per_building():
    spine, lots, points, values = _row_of_four()
    out, _, _ = _split(spine, lots, points, values)
    footprints = pd.DataFrame(index=pd.Index(['f1'], name='footprint_id'))
    links = build_id_links(
        out,
        footprints,
        'footprint_id',
        'footprint_id',
        'building_id',
        'footprint_id',
        'footprint_outline',
        source_column=None,
    )
    assert len(links) == 4
    assert set(links['footprint_id']) == {'f1'}
    assert set(links['link_method']) == {'footprint_outline'}


def test_a_piece_keeps_its_outlines_source_token():
    """bundle_terms derives a bundle's terms from geometry_source."""
    spine, lots, points, values = _row_of_four()
    out, _, _ = _split(spine, lots, points, values)
    assert set(out['geometry_source']) == {'obm'}


# The step, its sidecar and the building spine's restore


REFERENCE = 'US-XX_parcel-test-2026'
FOOTPRINT_GEOSPINE = {
    'recipe_id': 'US_footprint-geospine-2026',
    'entity': Entity('footprint', 'geospine', '2026'),
    'pipeline': [
        {'step': 'resolve_spine'},
        {
            'step': 'link_to_reference',
            'entity_type': 'parcel',
            'join': 'spatial_overlay',
            'thresholds': {
                'min_fraction_of_largest': MIN_FRACTION,
                'area_intersection_m2_min': AREA_MIN,
                'snap_chains': True,
            },
            'source_geometry_type': 'mixed_type_footprint',
        },
    ],
}
SPLIT_CFG = {
    'step': 'split_outlines_at_parcel_lines',
    'entity_type': 'parcel',
    'save_link': True,
    'lot_values_from': 'US_parcel-geospine-2026',
    'points': [{'recipe_id': 'US_building-nsi-2026'}],
    'keep_piece': {
        'improvement_value_min': 5000,
        'structure_flag_values': ['Y'],
        'compactness_max': 0.95,
    },
}
BUILDING_GEOSPINE = {
    'recipe_id': 'US_building-geospine-2026',
    'entity': Entity('building', 'geospine', '2026'),
    'pipeline': [
        {'step': 'resolve_spine'},
        {'step': 'adopt_source_entity_id', 'source_id_column': 'footprint_id'},
        {
            'step': 'load_links',
            'links_recipe_id': 'US_footprint-geospine-2026',
            'links_key': 'footprint_id',
        },
        SPLIT_CFG,
    ],
}
FINGERPRINT = {'format': 2, 'spine_recipe_id': 'x', 'sources': []}


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """The step's reads and writes pointed at fabricated inputs."""
    spine, lots, points, values = _row_of_four()
    flat = _flat(spine, lots)
    paths = {
        ('US_footprint-geospine-2026', REFERENCE): tmp_path / 'fp_parcel.parquet',
        ('US_building-geospine-2026', REFERENCE): tmp_path / 'bld_parcel.parquet',
    }
    flat.to_parquet(paths[('US_footprint-geospine-2026', REFERENCE)], index=False)
    recipes = {
        'US_footprint-geospine-2026': FOOTPRINT_GEOSPINE,
        'US_parcel-geospine-2026': {'recipe_id': 'US_parcel-geospine-2026'},
    }
    monkeypatch.setattr(pp, 'get_recipe_by_id', recipes.__getitem__)
    monkeypatch.setattr(pp, '_as_run_for', lambda recipe, admin_id: recipe)
    monkeypatch.setattr(
        pp,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: (
            (recipe_id, 'parcel') if recipe_id else (REFERENCE, entity_type)
        ),
    )
    monkeypatch.setattr(pp, 'get_entity_link_path', lambda a, b, admin: paths[(a, b)])
    monkeypatch.setattr(
        pp,
        'get_entities',
        lambda *a, **k: gpd.GeoDataFrame(geometry=points, crs=CRS),
    )
    monkeypatch.setattr(
        pp, 'lot_structure_evidence', lambda state, lots, ref, recipe: values
    )
    monkeypatch.setattr(
        pp, 'reference_rows', lambda rid, admin, kept_by=None: (None, None)
    )
    monkeypatch.setattr(pp, '_link_fingerprint', lambda *a, **k: dict(FINGERPRINT))
    raw_lots = lots.assign(geo_id=lots.index.to_numpy()).reset_index(drop=True)
    return spine, lots, raw_lots, flat, paths


def _geospine_state(spine, lots, flat):
    state = HarmonizeState(
        recipe=BUILDING_GEOSPINE, admin_id='US-XX-YY', verbose=False, timer=None
    )
    state.spine = spine
    state.step_index = 3
    indexed = flat.rename(columns={'footprint_id': 'building_id'}).set_index(
        ['building_id', 'parcel_id']
    )
    state.crosswalks[REFERENCE] = indexed[indexed['link'].notna()][
        ['link', 'area_intersection_m2', 'iou']
    ]
    state.overlays[REFERENCE] = indexed.drop(columns='link')
    state.references[REFERENCE] = lots
    state.reference_types[REFERENCE] = 'parcel'
    return state


def test_the_step_writes_the_cut_overlay_and_rekeys_the_state(wired):
    from openplaces.io.aggregate import read_file_metadata
    from openplaces.io.harmonizer.links._shared import _LINK_METADATA_KEY

    spine, lots, _, flat, paths = wired
    state = _geospine_state(spine, lots, flat)
    state = pp.split_outlines_at_parcel_lines(
        state, **{k: v for k, v in SPLIT_CFG.items() if k != 'step'}
    )
    assert len(state.spine) == 4
    written = pd.read_parquet(paths[('US_building-geospine-2026', REFERENCE)])
    assert list(written.columns)[0] == 'building_id'
    assert set(written['building_id']) == set(state.spine.index)
    footer = read_file_metadata(paths[('US_building-geospine-2026', REFERENCE)])
    assert json.loads(footer[_LINK_METADATA_KEY])['format'] == 2
    assert footer[pp.CUT_FROM_KEY] == 'US_footprint-geospine-2026'
    crosswalk = state.crosswalks[REFERENCE]
    assert crosswalk.index.names == ['building_id', 'parcel_id']
    assert set(crosswalk.index.get_level_values(0)) == set(state.spine.index)
    assert 'footprint_id' not in state.overlays[REFERENCE].columns


def test_the_cut_sidecar_is_the_same_file_on_a_rerun(wired):
    from openplaces.io.harmonizer.links.sidecars import _hash_file

    spine, lots, _, flat, paths = wired
    params = {k: v for k, v in SPLIT_CFG.items() if k != 'step'}
    path = paths[('US_building-geospine-2026', REFERENCE)]
    pp.split_outlines_at_parcel_lines(_geospine_state(spine, lots, flat), **params)
    first = _hash_file(path)
    pp.split_outlines_at_parcel_lines(_geospine_state(spine, lots, flat), **params)
    assert _hash_file(path) == first


def test_the_building_spine_restores_the_cut_overlay(wired, monkeypatch):
    spine, lots, raw_lots, flat, paths = wired
    state = pp.split_outlines_at_parcel_lines(
        _geospine_state(spine, lots, flat),
        **{k: v for k, v in SPLIT_CFG.items() if k != 'step'},
    )
    monkeypatch.setattr(
        load_module,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: (REFERENCE, entity_type),
    )
    monkeypatch.setattr(load_module, '_link_fingerprint', lambda *a, **k: FINGERPRINT)
    monkeypatch.setattr(
        load_module, 'reference_rows', lambda rid, admin, kept_by=None: (raw_lots, None)
    )
    restored = HarmonizeState(
        recipe={'entity': Entity('building', 'spine', '2026')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
    )
    restored.spine = state.spine
    restored = load_module._restore_parcel_pieces(
        restored, BUILDING_GEOSPINE, 3, SPLIT_CFG, 'building_id'
    )
    crosswalk = restored.crosswalks[REFERENCE]
    assert set(crosswalk.index.get_level_values('building_id')) == set(
        state.spine.index
    )
    assert set(crosswalk['link']) == {'unique parcel'}
    assert 'footprint_id' not in restored.overlays[REFERENCE].columns
    assert restored.reference_types[REFERENCE] == 'parcel'
    assert REFERENCE in restored.source_geometry_types


def test_a_stale_cut_overlay_fails_closed(wired, monkeypatch):
    spine, lots, raw_lots, flat, _ = wired
    pp.split_outlines_at_parcel_lines(
        _geospine_state(spine, lots, flat),
        **{k: v for k, v in SPLIT_CFG.items() if k != 'step'},
    )
    monkeypatch.setattr(
        load_module,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: (REFERENCE, entity_type),
    )
    # The footprint geospine was rerun: its overlay hash moved.
    monkeypatch.setattr(
        load_module,
        '_link_fingerprint',
        lambda *a, **k: {**FINGERPRINT, 'restored_links': {'x': 'changed'}},
    )
    monkeypatch.setattr(
        load_module, 'reference_rows', lambda rid, admin, kept_by=None: (raw_lots, None)
    )
    state = HarmonizeState(
        recipe={'entity': Entity('building', 'spine', '2026')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
    )
    with pytest.raises(RuntimeError, match='stale'):
        load_module._restore_parcel_pieces(
            state, BUILDING_GEOSPINE, 3, SPLIT_CFG, 'building_id'
        )


def test_a_unit_without_a_parcel_overlay_marks_every_row_an_outline(wired):
    spine, _, _, flat, paths = wired
    # The footprint geospine found no parcel in the unit and wrote its
    # overlay sidecar without rows.
    flat.iloc[:0].to_parquet(
        paths[('US_footprint-geospine-2026', REFERENCE)], index=False
    )
    state = HarmonizeState(
        recipe=BUILDING_GEOSPINE, admin_id='US-XX-YY', verbose=False, timer=None
    )
    state.spine = spine
    state.step_index = 3
    state = pp.split_outlines_at_parcel_lines(
        state, **{k: v for k, v in SPLIT_CFG.items() if k != 'step'}
    )
    assert state.spine[pp.BUILDING_GEOMETRY_COLUMN].tolist() == [pp.OUTLINE]
    assert REFERENCE not in state.crosswalks


def test_an_unbuilt_parcel_overlay_writes_no_cut_sidecar(wired):
    spine, _, _, _, paths = wired
    # The footprint geospine wrote no overlay sidecar: the reference
    # was not built, which is not the same as built without rows.
    paths[('US_footprint-geospine-2026', REFERENCE)].unlink()
    state = HarmonizeState(
        recipe=BUILDING_GEOSPINE, admin_id='US-XX-YY', verbose=False, timer=None
    )
    state.spine = spine
    state.step_index = 3
    pp.split_outlines_at_parcel_lines(
        state, **{k: v for k, v in SPLIT_CFG.items() if k != 'step'}
    )
    assert not paths[('US_building-geospine-2026', REFERENCE)].exists()


def test_a_unit_without_a_parcel_overlay_writes_the_cut_sidecar_empty(
    wired, monkeypatch
):
    from openplaces.io.curator import CurateState, evidence

    spine, lots, _, flat, paths = wired
    params = {k: v for k, v in SPLIT_CFG.items() if k != 'step'}
    cut_path = paths[('US_building-geospine-2026', REFERENCE)]
    pp.split_outlines_at_parcel_lines(_geospine_state(spine, lots, flat), **params)
    full = pd.read_parquet(cut_path)

    flat.iloc[:0].to_parquet(
        paths[('US_footprint-geospine-2026', REFERENCE)], index=False
    )
    state = HarmonizeState(
        recipe=BUILDING_GEOSPINE, admin_id='US-XX-YY', verbose=False, timer=None
    )
    state.spine = spine
    state.step_index = 3
    pp.split_outlines_at_parcel_lines(state, **params)
    empty = pd.read_parquet(cut_path)
    assert len(empty) == 0
    assert list(empty.columns) == list(full.columns)
    assert dict(empty.dtypes) == dict(full.dtypes)

    # The building spine restores it as a unit without parcels.
    monkeypatch.setattr(
        load_module,
        '_resolve_reference_recipe',
        lambda recipe_id, entity_type, admin_id: (REFERENCE, entity_type),
    )
    monkeypatch.setattr(load_module, '_link_fingerprint', lambda *a, **k: FINGERPRINT)
    monkeypatch.setattr(
        load_module, 'reference_rows', lambda rid, admin, kept_by=None: (None, None)
    )
    restored = HarmonizeState(
        recipe={'entity': Entity('building', 'spine', '2026')},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
    )
    restored.spine = state.spine
    restored = load_module._restore_parcel_pieces(
        restored, BUILDING_GEOSPINE, 3, SPLIT_CFG, 'building_id'
    )
    assert REFERENCE not in restored.crosswalks

    # And the curate collects no parcel ids from it, without raising.
    monkeypatch.setattr(evidence, 'get_entity_link_path', lambda *a: cut_path)
    monkeypatch.setattr(evidence, 'get_link_owner_recipe_id', lambda r: 'owner')
    curated = pd.DataFrame(
        {'parcel_id': [None] * len(state.spine)},
        index=pd.Index(state.spine.index, name='building_id'),
    )
    curate = CurateState(
        recipe={'recipe_id': 'US_building-test-2026'},
        entity_recipe={},
        admin_id='US-XX-YY',
        verbose=False,
        timer=None,
        curated=curated,
    )
    curate = evidence.collect_link_ids(curate, link_recipe_id=REFERENCE)
    assert curate.curated['parcel_id_all'].isna().all()


# Recipes


def test_the_split_reads_the_overlay_thresholds_from_the_footprint_geospine():
    from openplaces.recipe import get_recipe_by_id

    building = get_recipe_by_id('US_building-geospine-2026')
    footprint = get_recipe_by_id('US_footprint-geospine-2026')
    steps = [s['step'] for s in building['pipeline']]
    index = steps.index('split_outlines_at_parcel_lines')
    context = pp.split_context(building, index, building['pipeline'][index], None)
    overlay = next(
        s
        for s in footprint['pipeline']
        if s.get('step') == 'link_to_reference' and s.get('entity_type') == 'parcel'
    )
    assert context.min_fraction == overlay['thresholds']['min_fraction_of_largest']
    assert context.area_min_m2 == overlay['thresholds']['area_intersection_m2_min']
    split = building['pipeline'][index]
    assert 'thresholds' not in split
    assert 'min_fraction_of_largest' not in json.dumps(split)
    pp.parse_keep_piece(split['keep_piece'])


def test_the_lot_values_come_from_the_parcel_geospines_ingests_not_its_output():
    from openplaces.recipe import get_recipe_by_id, get_recipe_dependencies

    building = get_recipe_by_id('US_building-geospine-2026')
    edges = get_recipe_dependencies(building, admin_id='US-NC-CUR')
    upstream = {e.upstream_recipe_id for e in edges}
    assert 'US_parcel-geospine-2026' not in upstream
    assert 'US_parcel-spine-2026' not in upstream
    assert any(e.step == 'link_by_id' for e in edges)
