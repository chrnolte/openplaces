"""Tests for building attribute imputation (fabricated data only)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from openplaces.io import building_imputation as bi


def test_nsi_token_matches_route_parts_only():
    tokens = pd.Series(['nsi', 'nsi/fema/parcel', 'parcel', 'nsight', None, 'a+nsi'])
    flags = bi.is_nsi_token(tokens.fillna(''))
    assert flags.tolist() == [True, True, False, False, False, True]


def test_year_labels_drop_nsi_and_out_of_range():
    frame = pd.DataFrame(
        {
            'year_built': [1950, 1975, 1000, 2001],
            'year_built_source': ['parcel', 'nsi', 'parcel', 'ncdps'],
        }
    )
    labels = bi.year_labels(frame)
    assert labels.tolist()[0] == 1950
    assert np.isnan(labels.tolist()[1])
    assert np.isnan(labels.tolist()[2])
    assert labels.tolist()[3] == 2001


def test_value_labels_keep_zero_and_drop_withheld_and_nsi():
    frame = pd.DataFrame(
        {
            'structure_value': [0.0, 100.0, 200.0, 300.0],
            'structure_value_source': ['parcel', 'nsi', 'withheld', 'parcel.x'],
        }
    )
    labels = bi.value_labels(frame)
    assert labels.iloc[0] == 0.0
    assert labels.iloc[1:3].isna().all()
    assert labels.iloc[3] == 300.0


def _homes():
    # Fabricated manufactured homes, 100 m2 each, one per setting.
    return pd.DataFrame(
        {
            'occupancy_type': ['Manufactured Home'] * 6 + ['Single-Family'],
            'area_m2': [100.0] * 7,
            'structure_value': [
                40000.0,  # alone on its parcel, $400/m2: listed
                0.0,  # alone, zero: personal property
                3000.0,  # alone, $30/m2: site improvements only
                12000.0,  # alone, $120/m2: between the thresholds
                40000.0,  # a park's share of the park's improvements
                np.nan,  # no value
                0.0,  # a house with zero value stays a label
            ],
            'structure_value_source': ['parcel'] * 5 + [None, 'parcel'],
            'n_manufactured_homes_per_parcel': [1, 1, 1, 1, 40, 1, 0],
        }
    )


def test_manufactured_home_listing_follows_the_roll():
    listing = bi.manufactured_home_listing(_homes())
    assert listing.tolist()[:6] == [
        'listed',
        'unlisted',
        'unlisted',
        'unclear',
        'unlisted',
        'unlisted',
    ]
    assert pd.isna(listing.iloc[6])


def test_a_low_value_with_a_building_record_is_kept():
    # Fabricated: $30/m2 on a parcel that records the home's year built
    # is a listed home valued cheaply; a zero stays unlisted either way.
    frame = _homes()
    frame['year_built_source'] = ['parcel'] * 7
    listing = bi.manufactured_home_listing(frame)
    assert listing.iloc[2] == 'unclear'
    assert listing.iloc[1] == 'unlisted'


def test_only_listed_homes_train_and_unclear_ones_are_kept():
    frame = _homes()
    train = bi.value_labels(frame)
    keep = bi.value_labels(frame, train=False)
    assert train.iloc[0] == 40000.0 and train.iloc[1:6].isna().all()
    assert keep.iloc[3] == 12000.0 and keep.iloc[[1, 2, 4, 5]].isna().all()
    # Zero stays a valid label for buildings that are not manufactured.
    assert train.iloc[6] == 0.0


def test_manufactured_value_model_reads_the_home_not_the_parcel():
    inputs = bi.input_columns(bi.MANUFACTURED_VALUE)
    assert 'log_area_m2' in inputs and 'chain_year_built' in inputs
    for parcel_level in (
        'n_footprints_per_parcel',
        'n_manufactured_homes_per_parcel',
        'log_land_value_parcel',
        'log_structure_value_building_nsi',
        'region_type',
    ):
        assert parcel_level not in inputs


def test_park_factor_is_the_median_over_counties_with_enough_homes():
    # Fabricated: county A values park homes at half the prediction,
    # county B at 0.7; county C has too few homes to count.
    county = pd.Series(['A'] * 30 + ['B'] * 25 + ['C'] * 5)
    predicted = pd.Series([100.0] * 60)
    observed = pd.Series([50.0] * 30 + [70.0] * 25 + [500.0] * 5)
    factor = bi.park_value_factor(observed, predicted, county, min_homes=20)
    assert factor == pytest.approx(0.6)
    assert bi.park_value_factor(observed[:5], predicted[:5], county[:5]) == 1.0


def test_appraisal_year_falls_back_to_previous_revaluation():
    reval = pd.DataFrame(
        {
            'county_fips': ['99001', '99003'],
            'last_reval_year': [2026, 2023],
            'previous_reval_year': [2018, 2015],
        }
    )
    years = bi.appraisal_years(reval, roll_year=2025)
    assert years.to_dict() == {'99001': 2018, '99003': 2023}


def test_value_scales_use_index_ratio_and_fall_back_to_median():
    hpi = pd.DataFrame(
        {
            'fips': ['99001', '99001', '99003', '99003', '99005'],
            'year': [2020, 2025, 2020, 2025, 2020],
            'hpi': [100.0, 150.0, 100.0, 130.0, 100.0],
        }
    )
    appraisal = pd.Series({'99001': 2020, '99003': 2020, '99005': 2020})
    scales = bi.value_scales(hpi, appraisal, common_year=2025)
    assert scales.at['99001', 'scale'] == pytest.approx(1.5)
    assert scales.at['99003', 'scale'] == pytest.approx(1.3)
    assert bool(scales.at['99005', 'fallback'])
    assert scales.at['99005', 'scale'] == pytest.approx(1.4)


def _toy_frame():
    rng = np.random.default_rng(0)
    rows = []
    for i, county in enumerate(['C1', 'C2', 'C3']):
        for _ in range(20):
            rows.append(
                {
                    'admin3_id': county,
                    'region_id': f'r{i}',
                    'x': 1000.0 * 100 * i + rng.normal(),
                    'y': 0.0,
                    'structure_value_building_nsi_per_area': 100.0 + i,
                    'year_built_block_median_building_nsi': 1980 + i,
                    'occupancy_type': 'Single-Family',
                    'area_m2': 150.0,
                    'land_value_2010': 1000.0,
                    'region_type': 'metro',
                    'distance_to_core_km': 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_own_county_weight_is_one_and_far_regions_decay():
    frame = _toy_frame()
    table = bi.county_table(frame, pd.Series({'C1': 10.0, 'C2': 10.0, 'C3': 10.0}))
    orders = bi.region_orders(
        {'r0': {'r1'}, 'r1': {'r0', 'r2'}, 'r2': {'r1'}}, max_order=6
    )
    donors = bi.DonorWeights(frame, table, orders)
    config = bi.DonorConfig(mode='region', decay=0.5)
    w = donors.weights('C1', frame, config)
    by_county = pd.Series(w).groupby(frame['admin3_id']).first()
    assert by_county['C1'] == 1.0
    assert by_county['C2'] == pytest.approx(0.5)
    assert by_county['C3'] == pytest.approx(0.25)


def test_radius_drops_distant_donor_counties():
    frame = _toy_frame()
    table = bi.county_table(frame, pd.Series({'C1': 10.0, 'C2': 10.0, 'C3': 10.0}))
    donors = bi.DonorWeights(frame, table, bi.region_orders({}, 6))
    config = bi.DonorConfig(mode='equal', radius_km=150)
    w = pd.Series(donors.weights('C1', frame, config)).groupby(frame['admin3_id'])
    assert w.first()['C3'] == 0.0
    assert w.first()['C2'] == 1.0


def test_donor_weights_depend_on_distance_only():
    # Design limit L30: no county is paired with another by how alike
    # their buildings are, so changing a donor's buildings leaves its
    # weight unchanged, and a recipe asking for similarity fails.
    frame = _toy_frame()
    area = pd.Series({'C1': 10.0, 'C2': 10.0, 'C3': 10.0})
    config = bi.DonorConfig(mode='km', bandwidth_km=100.0)
    before = bi.DonorWeights(frame, bi.county_table(frame, area), {})
    changed = frame.copy()
    changed.loc[changed['admin3_id'] == 'C2', 'area_m2'] *= 10
    after = bi.DonorWeights(changed, bi.county_table(changed, area), {})
    np.testing.assert_array_equal(
        before.weights('C1', frame, config), after.weights('C1', frame, config)
    )
    with pytest.raises(ValueError, match='similarity_bandwidth'):
        bi.delivery_config({'donors': {'mode': 'km', 'similarity_bandwidth': 1.0}})


def test_value_model_takes_no_core_type():
    # Design limit L24: no core size or type in the structure value model.
    spec = next(s for s in bi.TARGETS if s.column == 'structure_value')
    assert 'region_type' not in bi.input_columns(spec)


def test_zero_size_or_value_reads_as_missing():
    out = bi._log(pd.Series([0.0, -1.0, 9.0]))
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(np.log(10.0))


def test_per_county_models_take_no_county_features_or_coordinates():
    spec = next(s for s in bi.TARGETS if s.column == 'year_built')
    columns = bi.input_columns(spec)
    assert not set(bi.COUNTY_FEATURES) & set(columns)
    assert not bi.LOCATION & set(columns)


def test_label_evidence_is_never_an_input_of_its_own_column():
    for spec in bi.TARGETS:
        evidence = bi.LABEL_EVIDENCE.get(spec.column, set())
        assert not evidence & set(bi.input_columns(spec)), spec.column


def test_cost_measures_and_market_value_never_read_each_other():
    # NSI's cost measures ship beside the market value, never mixed with
    # it (maintainer, 2026-10-28): no measure is an input, chain input
    # or label source of structure_value, nor the roll's values of a
    # measure (design limit L13: no prediction feeds another value
    # model).
    measures = {s.column for s in bi.MEASURE_TARGETS}
    value = next(s for s in bi.TARGETS if s.column == 'structure_value')
    assert not measures & set(bi.CHAIN_INPUTS['structure_value'])
    assert not {f'chain_{m}' for m in measures} & set(bi.input_columns(value))
    for spec in bi.MEASURE_TARGETS:
        columns = set(bi.input_columns(spec))
        assert 'chain_structure_value' not in columns
        assert not (measures - {spec.column}) & set(bi.CHAIN_INPUTS[spec.column])
        assert not set(bi.PARCEL_VALUE_INPUTS) & columns
        assert 'log_structure_value_building_nsi' not in columns
        assert 'region_type' not in columns


def test_cost_measure_labels_are_positive_and_not_withheld():
    frame = pd.DataFrame(
        {
            'replacement_cost': [0.0, 250000.0, 300000.0, np.nan],
            'replacement_cost_source': ['nsi', 'nsi', 'withheld', None],
        }
    )
    spec = next(s for s in bi.MEASURE_TARGETS if s.column == 'replacement_cost')
    labels = bi.labels_for(spec, frame)
    assert labels.iloc[1] == 250000.0
    assert labels.drop(index=1).isna().all()


def test_a_county_trains_a_cost_measure_on_its_own_values():
    # The own-only configuration Imputer._fill_measure uses: a radius
    # of 0 km leaves every other county a weight of 0.
    frame = _toy_frame()
    area = pd.Series({'C1': 10.0, 'C2': 10.0, 'C3': 10.0})
    donors = bi.DonorWeights(frame, bi.county_table(frame, area), {})
    config = bi.DonorConfig(mode='km', bandwidth_km=250.0, radius_km=0.0)
    w = pd.Series(donors.weights('C1', frame, config)).groupby(frame['admin3_id'])
    assert w.max().to_dict() == {'C1': 1.0, 'C2': 0.0, 'C3': 0.0}


def _block(index):
    block = pd.DataFrame(index=index)
    for spec in bi.TARGETS:
        numeric = spec.column not in {
            'construction_type',
            'foundation_type',
            'roof_shape',
        }
        block[spec.column] = 9.0 if numeric else 'Wood'
        block[f'{spec.column}_source'] = 'm+imputed'
    return block


SCALES = {'99001': {'scale': 2.0, 'appraisal_year': 2020.0}}


def test_public_overlay_takes_the_block_everywhere():
    pooled = pd.DataFrame(
        {
            'year_built': [1990.0, np.nan],
            'year_built_source': ['parcel', None],
            'county_fips': ['99001', '99001'],
        },
        index=pd.Index(['a', 'b'], name='footprint_id'),
    )
    out = bi.overlay_delivery(
        pooled, _block(pooled.index), keep_observed=False, scales=SCALES
    )
    assert out['year_built'].tolist() == [9.0, 9.0]
    assert out['structure_value_adjusted'].tolist() == [18.0, 18.0]


def test_team_overlay_keeps_its_own_observed_values():
    pooled = pd.DataFrame(
        {
            'year_built': [1990.0, 1980.0, np.nan],
            'year_built_source': ['parcel', 'nsi', None],
            'county_fips': ['99001'] * 3,
        },
        index=pd.Index(['a', 'b', 'c'], name='footprint_id'),
    )
    out = bi.overlay_delivery(
        pooled, _block(pooled.index), keep_observed=True, scales=SCALES
    )
    assert out['year_built'].tolist() == [1990.0, 9.0, 9.0]
    assert out['year_built_source'].tolist() == ['parcel', 'm+imputed', 'm+imputed']


def test_team_overlay_restores_only_what_the_public_twin_withheld():
    # Row a: withheld in the public twin, so the team's own value comes
    # back. Row b: the public imputation refilled an observed value on
    # purpose, so the block wins. Row c: the team has a row the public
    # twin lacks, so its own value stays.
    pooled = pd.DataFrame(
        {
            'structure_value': [50_000.0, 3_000.0, 70_000.0],
            'structure_value_source': ['parcel', 'parcel', 'parcel'],
            'county_fips': ['99001'] * 3,
        },
        index=pd.Index(['a', 'b', 'c'], name='footprint_id'),
    )
    block = _block(pd.Index(['a', 'b'], name='footprint_id'))
    block['structure_value_withheld'] = [True, False]
    out = bi.overlay_delivery(pooled, block, keep_observed=True, scales=SCALES)
    assert out['structure_value'].tolist() == [50_000.0, 9.0, 70_000.0]
    # The adjusted value follows the value shipped beside it, on rows
    # the public twin lacks too.
    assert out['structure_value_adjusted'].tolist() == [100_000.0, 18.0, 140_000.0]
    assert out['appraisal_year'].tolist() == [2020.0] * 3
    assert bi.imputed_cells(out, 'm')['structure_value'] == 1


def test_an_excluded_location_input_takes_its_county_summary_with_it():
    for feature, (_, _, county) in bi.LOCATION_INPUTS.items():
        assert county in bi.COUNTY_FEATURES
        assert {feature, county} <= set(bi.UNUSED_LOCATION_INPUTS)


def test_value_model_fits_with_a_single_zero_label():
    rng = np.random.default_rng(0)
    x = pd.DataFrame({'a': rng.normal(size=400)})
    y = pd.Series(np.exp(11 + x['a'].to_numpy()))
    y.iloc[0] = 0.0
    model = bi.fit_value_model(x, y, np.ones(len(y)), seed=1)
    assert model.zero is None


def test_pooled_scores_count_classes_and_shares():
    results = pd.DataFrame(
        {
            'admin3_id': ['A'] * 4,
            'config': ['model'] * 4,
            'observed': ['Wood', 'Wood', 'Masonry', 'Masonry'],
            'predicted': ['Masonry', 'Wood', 'Wood', 'Masonry'],
        }
    )
    row = bi.score_pooled(results, 'class').loc['model']
    assert row['accuracy'] == 0.5
    assert row['macro_f1'] == pytest.approx(0.5)
    # Right shares, wrong buildings.
    assert row['county_share_gap_median'] == 0.0
    counts = results.assign(observed=[1, 2, 2, 1], predicted=[1.0, 2.0, 3.0, 1.0])
    row = bi.score_pooled(counts, 'count').loc['model']
    assert row['accuracy'] == 0.75
    assert row['mae'] == 0.25


def test_nsi_anchor_moves_a_county_to_its_donor_calibrated_nsi_total():
    # Donor A's roll totals are half its NSI totals; C has no labels,
    # NSI of 100 per building, and filled values of 31.25: its target
    # total is 0.5 * 100 per building, so the factor is 1.6.
    n = bi.ANCHOR_MIN_HOMES
    index = pd.RangeIndex(2 * n)
    frame = pd.DataFrame(
        {
            'admin3_id': ['A'] * n + ['C'] * n,
            'structure_value_building_nsi': [100.0] * (2 * n),
        },
        index=index,
    )
    y = pd.Series([50.0] * n + [np.nan] * n, index=index)
    stub = SimpleNamespace(
        frame=frame,
        value_scale=pd.Series(1.0, index=index),
        donors=SimpleNamespace(
            weights=lambda county, donors, config: np.ones(len(donors))
        ),
        config=None,
    )
    values = pd.Series(31.25, index=index[n:])
    factor = bi.Imputer._nsi_anchor(stub, 'C', values, y, exclude_own=True)
    assert factor == pytest.approx(1.6)


def test_an_appraisal_after_the_index_is_measured_from_its_last_year():
    hpi = pd.DataFrame(
        {'fips': ['99001'] * 2, 'year': [2019, 2025], 'hpi': [100.0, 171.0]}
    )
    scales = bi.value_scales(hpi, pd.Series({'99001': 2026}), 2025)
    assert scales.at['99001', 'scale'] == pytest.approx(1.0)
    assert not scales.at['99001', 'fallback']


def test_roll_level_check_flags_a_county_far_from_its_acs_level():
    n = bi.ANCHOR_MIN_HOMES
    frame = pd.DataFrame(
        {
            'admin3_id': ['A'] * n + ['B'] * n,
            'occupancy_type': ['Single-Family'] * (2 * n),
        }
    )
    labels = pd.Series([100.0] * n + [170.0] * n)
    stub = SimpleNamespace(
        frame=frame,
        acs_value=pd.Series({'A': 150.0, 'B': 110.0}),
        _labels=lambda spec: labels,
    )
    level = bi.roll_level_check(stub)
    assert level['ratio'] == {'A': 0.667, 'B': 1.545}
    assert level['flagged'] == {'B': 1.545}


def test_appraisal_scales_require_every_county():
    years = pd.DataFrame({'county_fips': ['99001'], 'appraisal_year': [2020]})
    hpi = pd.DataFrame(
        {
            'county_fips': ['99001', '99001'],
            'index_year': [2020, 2025],
            'house_price_index': [100.0, 120.0],
        }
    )
    scales = bi.appraisal_scales(pd.Series(['99001']), years, hpi, 2025)
    assert scales.at['99001', 'scale'] == pytest.approx(1.2)
    with pytest.raises(ValueError):
        bi.appraisal_scales(pd.Series(['99001', '99003']), years, hpi, 2025)


def test_rare_classes_are_masked_for_training():
    y = pd.Series(['a'] * 60 + ['b'] * 3)
    masked = bi.drop_rare_classes(y)
    assert masked.isna().sum() == 3


def test_positive_values_stay_in_range_with_tiny_labels():
    # Labels from $100 to $100M, as on a roll: the positive model must
    # not leave the labels' range (gamma loss diverged on such data).
    rng = np.random.default_rng(0)
    n = 4000
    x = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)})
    y = pd.Series(np.exp(rng.uniform(np.log(100), np.log(1e8), size=n)))
    y[rng.random(n) < 0.05] = 0.0
    spec = bi.TargetSpec('structure_value', 'value', per_county=True)
    model = bi.fit_target(spec, x, y, np.ones(n), seed=1)
    pred = bi._predict_value(model, x)
    assert np.isfinite(pred).all()
    assert pred.max() <= 10 * y.max()
    # Training totals are matched by construction.
    assert pred.sum() == pytest.approx(y.sum(), rel=1e-6)


def _skewed_values(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)})
    log_value = 11.5 + 0.8 * x['b'] + rng.normal(scale=0.9, size=n)
    y = pd.Series(np.exp(log_value))
    y[x['a'] > 0.8] = 0.0
    return x, y


def test_drawn_values_keep_totals_and_spread():
    # Drawn values are zero or positive with the roll's scatter: totals
    # match the expectation, and the spread is not collapsed onto the
    # mean of buildings like it.
    x, y = _skewed_values()
    spec = bi.TargetSpec('structure_value', 'value', per_county=True)
    model = bi.fit_target(spec, x, y, np.ones(len(y)), seed=1)
    drawn = bi._predict_value(model, x, np.random.default_rng(2))
    expected = bi._predict_value(model, x)
    assert drawn.sum() == pytest.approx(expected.sum(), rel=0.05)
    is_zero = drawn == 0
    assert 0.1 < is_zero.mean() < 0.3
    observed_spread = np.std(np.log(y[y > 0]))
    drawn_spread = np.std(np.log(drawn[~is_zero]))
    assert drawn_spread > 0.8 * observed_spread


def test_sparse_own_labels_give_way_to_donors():
    imputer = object.__new__(bi.Imputer)
    imputer.frame = pd.DataFrame({'admin3_id': ['A'] * 100 + ['B'] * 100 + ['C'] * 100})
    y = pd.Series(np.nan, index=imputer.frame.index)
    y.iloc[:10] = 2005.0  # A: 10% labeled
    y.iloc[100:160] = 1970.0  # B: 60% labeled; C: none
    applicable = pd.Series(True, index=imputer.frame.index)
    spec = bi.TargetSpec('year_built', 'year', per_county=True)
    assert imputer._own_labels_too_sparse(spec, y, applicable, 'A')
    assert not imputer._own_labels_too_sparse(spec, y, applicable, 'B')
    assert not imputer._own_labels_too_sparse(spec, y, applicable, 'C')
    assert not imputer._own_labels_too_sparse(bi.MANUFACTURED_VALUE, y, applicable, 'A')


def _census_fixture():
    # Fabricated levels for two tracts, one block group and two places.
    ids = [
        '1400000US37015960100',
        '1400000US37015960200',
        '1500000US370159601001',
        '1600000US3700001',
        '1600000US3700002',
        '0500000US37015',
    ]
    acs = {
        'b25077': pd.DataFrame(
            {'B25077_E001': [98_500, -666666666, 90_000, 0, 0, 98_000]}, index=ids
        ),
        'b25035': pd.DataFrame(
            {'med_year_built': [1976, 1988, 1980, 0, 0, 1979]}, index=ids
        ),
        'b19013': pd.DataFrame({'B19013_E001': [0, 0, 41_000, 0, 0, 0]}, index=ids),
        'b01003': pd.DataFrame(
            {'B01003_E001': [4_000, 2_000, 900, 60_000, 12_000, 20_000]}, index=ids
        ),
    }
    gazetteer = {
        'tracts': pd.DataFrame(
            {'land_area_m2': [40e6, 0.0], 'lat': [36.0, 36.2], 'long': [-77.0, -77.1]},
            index=['37015960100', '37015960200'],
        ),
        'place': pd.DataFrame(
            {'land_area_m2': [9e7, 2e7], 'lat': [36.0, 36.5], 'long': [-77.5, -77.0]},
            index=['3700001', '3700002'],
        ),
    }
    return bi.census_levels(acs, gazetteer)


def test_census_levels_are_read_per_unit_without_jam_values():
    census = _census_fixture()
    tract = census['tract']
    assert tract.loc['37015960100', 'acs_median_home_value_tract'] == 98_500
    assert np.isnan(tract.loc['37015960200', 'acs_median_home_value_tract'])
    assert tract.loc['37015960100', 'population_density_tract'] == pytest.approx(100)
    # No land area: no density, rather than a division by zero.
    assert np.isnan(tract.loc['37015960200', 'population_density_tract'])
    assert census['blockgroup'].loc['370159601001', 'acs_median_income_blockgroup'] == (
        41_000
    )
    assert census['place']['population'].to_dict() == {
        '3700001': 60_000,
        '3700002': 12_000,
    }


def test_settlement_inputs_have_no_radius():
    census = _census_fixture()
    frame = pd.DataFrame(
        {
            'census_tract_id': ['37015960200', '37015960100', None],
            'census_blockgroup_id': [None, '370159601001', None],
            'lat': [36.2, 36.0, 30.0],
            'long': [-77.1, -77.0, -90.0],
        }
    )
    bi.attach_census_inputs(frame, census)
    assert frame['acs_median_year_built_tract'].tolist()[:2] == [1988, 1976]
    assert frame['acs_median_income_blockgroup'].tolist()[1] == 41_000
    # A building far from every place still gets its nearest distance
    # and the city: no radius cuts either off (design limit L24).
    assert frame['distance_to_place_10k_km'].notna().all()
    assert frame['distance_to_place_50k_km'].iloc[2] > 1000
    # Only the 60,000-person place counts for the 50k distance.
    assert frame['distance_to_place_50k_km'].iloc[1] == pytest.approx(45, abs=1)
    assert (frame['place_gravity_tract'].iloc[:2] > 0).all()


def test_acs_levels_move_from_the_survey_period_to_the_common_year():
    # County index 100 on average over 2020-2024 and 150 in 2025: the
    # ACS value is moved by 1.5; a county without an index gets the
    # median factor.
    years = list(range(2018, 2026))
    hpi = pd.DataFrame(
        {
            'county_fips': ['37001'] * len(years),
            'index_year': years,
            'house_price_index': [90, 95, 80, 90, 100, 110, 120, 150],
        }
    )
    fips = pd.Series({'US-NC-ALA': '37001', 'US-NC-XXX': '37999'})
    acs = pd.Series({'37001': 200_000.0, '37999': 100_000.0})
    levels = bi.acs_value_levels(acs, fips, hpi, 2025)
    assert levels['US-NC-ALA'] == pytest.approx(300_000)
    assert levels['US-NC-XXX'] == pytest.approx(150_000)


def test_price_anchor_matches_the_donors_ratio_to_acs():
    # Two donor counties value single-family homes at 0.6 of their ACS
    # home value; the target's filled values sit at 0.9 of its own, so
    # the anchor scales them by 0.6 / 0.9.
    n = 300
    frame = pd.DataFrame(
        {
            'admin3_id': ['A'] * n + ['B'] * n + ['T'] * n,
            'occupancy_type': ['Single-Family'] * (3 * n),
        }
    )
    imputer = object.__new__(bi.Imputer)
    imputer.frame = frame
    imputer.config = bi.DonorConfig(mode='equal')
    distance = pd.DataFrame(0.0, index=['A', 'B', 'T'], columns=['A', 'B', 'T'])

    class Donors:
        def weights(self, target, donors, config):
            return np.ones(len(donors))

    imputer.donors = Donors()
    imputer.distance = distance
    imputer.acs_value = pd.Series({'A': 100_000.0, 'B': 200_000.0, 'T': 150_000.0})
    y = pd.Series([60_000.0] * n + [120_000.0] * n + [np.nan] * n)
    values = pd.Series(135_000.0, index=frame.index[2 * n :])
    factor = imputer._price_anchor('T', values, y, exclude_own=True)
    assert factor == pytest.approx(0.6 / 0.9)


def test_filled_values_are_clipped_to_their_class_bound():
    occupancy = pd.Series(['Single-Family'] * 60 + ['Retail'] * 3 + ['Single-Family'])
    area = pd.Series([100.0] * 63 + [200.0])
    labels = pd.Series([100_000.0] * 63 + [np.nan])
    caps, overall = bi.value_per_area_caps(labels, area, occupancy)
    # Retail has too few labels: the bound over all labels applies.
    assert 'Retail' not in caps.index
    assert overall == pytest.approx(1000.0)
    values = pd.Series([50_000.0] * 63 + [9e8])
    rows = pd.Series([False] * 63 + [True])
    clipped_values, clipped = bi.clip_to_caps(
        values, rows, area, occupancy, caps, overall
    )
    assert clipped.tolist() == [False] * 63 + [True]
    assert clipped_values.iloc[-1] == pytest.approx(200_000.0)
    assert clipped_values.iloc[0] == 50_000.0
