"""Impute building attributes from parcel-recorded evidence, without NSI.

Fills the primary columns of a curated footprint inventory (year built,
structure value, roof shape, story count, construction, foundation,
garage, sections, dwellings, footprint area) for primary buildings. NSI
values never reach the output columns; NSI fields enter only as named
model inputs, so a model learns how parcel-recorded values relate to
them where both exist and applies that relation where parcel values
are missing.

Credit. The method choices below come from Mohammad Askari's (UCLA)
research, which he leads: "Imputation Model Selection for CoPe-CHEER:
North Carolina" (technical memo, 2026-09-10; repository
mhmdaskari/building-inventory-imputation v1.0.0) and "Imputing All
Nine Building Attributes in CHEER Places North Carolina" (2026-10-15;
CHEER-Hub/building-inventory-imputation v2.0). No code from either
repository is used here; this module implements its own steps on
scikit-learn. Taken from that work:
- histogram gradient boosting as the learner, which beat KNN, MICE and
  the county median under spatially blocked holdout;
- spatially blocked evaluation only, since random masking selects the
  wrong method;
- NSI-assigned years are area lookups that flatter accuracy, so
  accuracy is reported by data source and NSI values are not labels.
His implemented contributions (the validity repairs, gap-trained
boosting, and the Stage 2b evaluation on buildings as sparse as the
real gaps) are not in this module: they plug in through
ImputationExtensions, which a recipe names (`impute: extensions:`).

Added here, for the production inventory:
- one model per county, trained on the county's own labels plus donor
  counties weighted by their distance, so a county
  without parcel values can be filled; donors only where a county's
  own labels are too sparse to be a sample of it;
- parcel-recorded labels only, with NSI as named inputs;
- values in a common year (county price index, revaluation years), a
  separate zero part and a manufactured-home model;
- a log-value model with unbiased totals and residual draws, so the
  spread of values follows the roll's;
- Census tract and block-group levels and settlement distances as
  inputs, and an ACS price anchor for counties without roll values;
- running inside the delivery export, with every filled cell marked.

Design limits checked (plans/design-limits/DESIGN_LIMITS.md): L4, L12,
L13, L15, L23, L24, L25, L29, L30. Histogram boosting has no row
subsampling and keeps max_features at 1.0 (L12). Donor weights are
between counties, never between pairs of buildings, and no estimate
is an adjusted average of comparables (L23). They depend on geometry
alone (centroid distance, or region-graph order): no county is paired
with another by the similarity of its buildings or values (L30). The structure value
model takes no core size or type, and nothing here computes feature
importance or correlation for it (L24). Roof shape uses one
multiclass model (L15). No published index is built here (L29).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

MODEL_VERSION = 'building-imputation-v0.1'
# Base seed of every random step (samples, gap patterns, fits, draws);
# a recipe's `impute: seed:` overrides it and the counts record it.
DEFAULT_SEED = 2026
YEAR_RANGE = (1700, 2026)
VALUE_CAP = 5e8
# Filled structure values are clipped to this quantile of the labels'
# value per m2 of footprint in their occupancy class
# (value_per_area_caps).
VALUE_PER_AREA_QUANTILE = 0.999
LOCATION = {'x_km', 'y_km'}
# Evidence each canonical column is reconciled from in the curate
# recipe (US_building-openplaces-2026), hence its labels' source.
LABEL_EVIDENCE = {
    'n_stories': {
        'n_stories_building_nsi',
        'n_stories_building_cheer',
        'n_stories_brails',
        'n_stories_parcel',
    },
    'construction_type': {
        'construction_type_building_nsi',
        'construction_type_building_cheer',
    },
    'foundation_type': {
        'foundation_type_building_nsi',
        'foundation_type_building_cheer',
    },
    'roof_shape': {'roof_shape_building_cheer'},
    'n_dwellings': {'n_dwellings_nsi'},
}

# Evidence columns entering the models as inputs. Target-derived
# fields (improvement and total values, per-area values derived from
# them, the parcel and NCDPS years that are labels) are excluded so no
# model sees its own answer.
NUMERIC_INPUTS = [
    'log_area_m2',
    'x_km',
    'y_km',
    'log_land_value_2010',
    'distance_to_core_km',
    'log_structure_value_building_nsi',
    'log_structure_value_building_nsi_per_area',
    'log_gross_floor_area_sqft_building_nsi',
    'year_built_block_median_building_nsi',
    'n_stories_building_nsi',
    'n_dwellings_nsi',
    'n_buildings_nsi',
    'log_land_value_parcel',
    'log_living_area_sqft_parcel',
    'n_stories_parcel',
    'n_footprints_per_parcel',
    'n_parcels_per_footprint',
    'overlap_fraction_parcel',
    'n_manufactured_homes_per_parcel',
    'has_homestead_exemption_parcel',
    'n_stories_building_cheer',
    'n_stories_brails',
    'log_acs_median_home_value_tract',
    'acs_median_year_built_tract',
    'log_acs_median_home_value_blockgroup',
    'log_acs_median_income_blockgroup',
    'log_population_density_tract',
    'log_place_gravity_tract',
    'log_distance_to_place_10k_km',
    'log_distance_to_place_50k_km',
]
# Tract levels from the ACS 5-year estimates (public domain), read at
# delivery from `impute: census_recipe`: a price and an age level for
# every tract, including tracts in counties whose roll has no values
# or years. The per-county models leave out coordinates and county
# features (input_columns), so without these a county with no labels
# of its own has no local level and drifts toward its donors'.
# Measured 2026-10-20 on cheer-eastern-nc: imputed single-family
# values in Halifax (no roll values) were 0.94 of the ACS median home
# value, against 0.62 to 0.99 in counties with roll values.
#
# Settlement and density inputs (land-market proxies) come from the
# same public sources: Census places weighted by their ACS population,
# located by the Census Gazetteer, and ACS population over Gazetteer
# land area. Design limit L24 (and its continuations US11983203,
# US12361034, US12705263, read 2026-10-21) stays at least two elements
# away, and these rules keep it so: no radius or boundary around a
# building (the nearest place of a size class, and gravity summed over
# every place); one summary number per input, never one entry per
# place; no correlation or importance reported per input; no selection
# of properties by these inputs; no service answering requests.
#
# ACS tables read, each with the column names its parquet may carry
# (renamed by the recipe, or raw).
ACS_TABLES = {
    'home_value': ('b25077', ('med_prop_val', 'B25077_E001')),
    'year_built': ('b25035', ('med_year_built', 'B25035_E001')),
    'income': ('b19013', ('med_household_inc', 'B19013_E001')),
    'population': ('b01003', ('total_pop', 'B01003_E001')),
}
# Raw per-row columns the census inputs are derived from.
CENSUS_COLUMNS = [
    'acs_median_home_value_tract',
    'acs_median_year_built_tract',
    'acs_median_home_value_blockgroup',
    'acs_median_income_blockgroup',
    'population_density_tract',
    'place_gravity_tract',
    'distance_to_place_10k_km',
    'distance_to_place_50k_km',
]
CENSUS_INPUTS = [
    'log_acs_median_home_value_tract',
    'acs_median_year_built_tract',
    'log_acs_median_home_value_blockgroup',
    'log_acs_median_income_blockgroup',
    'log_population_density_tract',
    'log_place_gravity_tract',
    'log_distance_to_place_10k_km',
    'log_distance_to_place_50k_km',
]
# Price anchor of counties without value labels (Imputer._price_anchor):
# the fewest single-family homes a county's median is read from, and
# the range a factor is held to, so that one odd county or a thin ACS
# estimate cannot move a whole county's values far.
ANCHOR_MIN_HOMES = 200
ANCHOR_RANGE = (0.5, 2.0)
# Pools of incomplete buildings for the extension hooks in
# leave-county-out (Imputer._gap_pools): a county with fewer draws
# from the whole region, and at most this many per half are kept.
MIN_GAP_ROWS = 50
MAX_GAP_ROWS = 20_000
# The ACS 5-year period whose dollar values acs_value_levels reads.
ACS_PERIOD = (2020, 2024)
# Population thresholds of the nearest-place distances: a town with
# services, and a city.
PLACE_SIZES = {'10k': 10_000, '50k': 50_000}
# Gravity counts a place no closer than this, so a tract inside a city
# does not divide by a distance near zero.
MIN_GRAVITY_KM = 1.0
# ACS median year built is bottom-coded at 1939 and jam values are
# negative; outside this range it is not a year.
ACS_YEAR_RANGE = (1939, 2030)
# A county whose own labels cover less than this share of the rows a
# target applies to fills those rows from donor counties only. Labels
# that sparse are rarely a sample of the county: they are the
# buildings a recent record reached.
# Measured 2026-10-20 on cheer-eastern-nc: Hertford's 1,092 observed
# years (9% of its buildings) and Bertie's 3,215 (24%) had medians of
# 2002 and 2003, against about 1970 in Edgecombe and Halifax, and the
# county-weighted model carried that to the rest (imputed medians 1986
# and 1992; Martin, from donors, 1964). The next lowest county, Robeson
# at 48%, has observed and imputed medians within 3 years.
MIN_OWN_LABEL_SHARE = 0.30
CATEGORICAL_INPUTS = [
    'occupancy_type',
    'occupancy_type_building_nsi',
    'foundation_type_building_nsi',
    'construction_type_building_nsi',
    'foundation_type_building_cheer',
    'construction_type_building_cheer',
    'roof_shape_building_cheer',
    'geometry_kind',
    'land_use_class_parcel',
    'region_type',
]
# County-level features: available in every county whether or not its
# parcel data carry values. They enter the regional models as inputs,
# never the donor weights (design limit L30: no county is paired with
# another by the similarity of its buildings or values).
COUNTY_FEATURES = [
    'county_nsi_value_per_area',
    'county_nsi_year',
    'county_share_manufactured',
    'county_log_area',
    'county_log_density',
    'county_log_land_value_2010',
    'county_share_metro',
    'county_distance_to_core',
]
# Parcel value fields a county without parcel values lacks; masked for
# a held-out county so leave-county-out mimics that case.
PARCEL_VALUE_INPUTS = ['log_land_value_parcel']
# Manufactured homes the roll does not list (manufactured_home_listing).
# A parcel with this many homes is a park: homes on another's land are
# personal property, and the parcel's improvements are the park's.
PARK_MIN_HOMES = 5
# Value per m2 of footprint below which a single home's parcel value is
# site improvements only, and at or above which it is a listed home.
# Between the two, the value is kept but trains nothing.
UNLISTED_MAX_VALUE_PER_M2 = 100.0
LISTED_MIN_VALUE_PER_M2 = 150.0
# A county's park homes inform the park factor (park_value_factor) only
# where at least this many are on the roll with their park.
PARK_FACTOR_MIN_HOMES = 20
# Inputs of the manufactured-home value model: the home's own size,
# age, sections and setting. Parcel-level inputs (footprint and home
# counts, the parcel's land value) describe a park, not its homes, and
# read like an apartment complex to a pooled model; NSI values stay
# out so that NSI remains an independent benchmark.
MANUFACTURED_VALUE_INPUTS = [
    'log_area_m2',
    'log_land_value_2010',
    'distance_to_core_km',
    'log_acs_median_home_value_tract',
    'geometry_kind',
]


@dataclass
class DonorConfig:
    """Donor weighting for the per-county models.

    Parameters
    ----------
    mode : str
        'region' weights a donor building by `decay ** order`, where
        order is the region-graph distance between its region and the
        target county's regions (summed over the target's regions,
        weighted by their share of its buildings); 'km' by a Gaussian
        of the distance between county centroids; 'equal' gives every
        donor weight 1.
    decay : float
        Per-order factor in 'region' mode, in (0, 1].
    bandwidth_km : float
        Gaussian bandwidth in 'km' mode.
    max_order : int
        Region orders beyond this get weight 0.
    radius_km : float
        Donor counties farther than this (centroid distance) are
        dropped in every mode.
    own_cap, donor_cap : int
        Maximum labeled rows sampled from the target county and from
        each donor county.
    """

    mode: str = 'region'
    decay: float = 0.5
    bandwidth_km: float = 100.0
    max_order: int = 6
    radius_km: float = 250.0
    own_cap: int = 40_000
    donor_cap: int = 4_000
    label: str = field(default='')

    def name(self) -> str:
        if self.label:
            return self.label
        if self.mode == 'region':
            return f'region_d{self.decay:g}'
        if self.mode == 'km':
            return f'km{self.bandwidth_km:g}'
        return 'equal'


@dataclass
class ImputationExtensions:
    """Optional steps a separate module adds to the imputation.

    The imputation runs without any of them. A recipe names a module
    attribute holding an instance (`impute: extensions:`), so a method
    developed elsewhere plugs in without changing this module.

    Parameters
    ----------
    repair : callable, optional
        `frame -> (frame, counts)`: empties observed values that cannot
        be true before any label is read; counts are cells per rule.
    train_rows : callable, optional
        `(x, y, w, patterns, rng) -> (x, y, w)`: changes a model's
        training rows before it is fitted. *patterns* are the input rows
        of buildings that genuinely lack the target. With it, one model
        fills all target rows, and the guard that leaves out inputs the
        target rows lack (Imputer._guarded_fit_predict) is not used.
    holdout_rows : callable, optional
        `(xt, patterns, rng) -> xt`: thins held-out rows to the input
        patterns of genuinely incomplete buildings, for evaluation.
    """

    repair: Callable | None = None
    train_rows: Callable | None = None
    holdout_rows: Callable | None = None


def load_extensions(path: str | None) -> ImputationExtensions:
    """The ImputationExtensions a recipe names as 'module:attribute'."""
    if not path:
        return ImputationExtensions()
    module, _, attribute = str(path).partition(':')
    return getattr(importlib.import_module(module), attribute or 'EXTENSIONS')


def is_nsi_token(tokens: pd.Series) -> pd.Series:
    """Flag provenance tokens that name NSI as a route part."""
    parts = tokens.astype('string').str.lower().str.split(r'[+/]', regex=True)
    return parts.map(lambda p: isinstance(p, list) and 'nsi' in p).astype(bool)


def year_labels(frame: pd.DataFrame) -> pd.Series:
    """Return parcel- or NCDPS-recorded years in range, else NaN."""
    year = pd.to_numeric(frame['year_built'], errors='coerce')
    ok = (
        year.between(*YEAR_RANGE)
        & frame['year_built_source'].notna()
        & ~is_nsi_token(frame['year_built_source'].fillna(''))
    )
    return year.where(ok)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors='coerce')


def manufactured_home_listing(frame: pd.DataFrame) -> pd.Series:
    """Classify each manufactured home by how the roll lists it.

    In North Carolina a manufactured home is real property only when it
    stands on a permanent foundation on land its owner owns (or holds
    on a lease of 20 years or more, G.S. 105-273(13)); every other home
    is personal property and absent from the real-property roll. The
    parcel then carries only the site improvements (well, septic, a
    deck), or, in a park, the park owner's improvements split across
    the homes. Neither is the home's value. The NCDOR appraisal manual
    (2019, Section VII) asks counties to appraise those homes "the
    same as other homes", from one county schedule for manufactured
    housing, which is what a model trained on the listed homes does.

    Texas works the same way in effect: a home is real property only by
    the owner's election, and only on land the owner owns or holds on a
    long-term lease (Occupations Code 1201.2055); otherwise it is
    personal property taxed apart from the land.

    Returns 'listed' (a usable label: the only manufactured home on its
    parcel, valued at LISTED_MIN_VALUE_PER_M2 or more), 'unlisted' (the
    value is not the home's and is filled: no value, a zero, an NSI
    value, a park of PARK_MIN_HOMES or more, or under
    UNLISTED_MAX_VALUE_PER_M2 on a parcel with no building record, read
    as no parcel year built), 'unclear' (kept as observed, never a
    label), and NA for rows that are not primary manufactured homes.

    A low value is read as "not on the roll" only without a building
    record, because rolls differ in what a low value means. Measured
    2026-10-18 on cheer-eastern-nc: of single-home parcels, homes at $1-50
    per m2 have a parcel year built 6% of the time and those at $400 or
    more 75% of the time. Measured 2026-10-19 on cheer-coastal-tx: 54% at
    $1-50 and 79% at $50-100 (median year 1985), so most low-valued
    Texas homes are listed homes valued cheaply, not missing ones.
    """
    out = pd.Series(pd.NA, index=frame.index, dtype='string')
    if 'occupancy_type' not in frame.columns:
        return out
    occupancy = frame['occupancy_type'].astype('string')
    mh = occupancy.eq('Manufactured Home').fillna(False)
    value = pd.to_numeric(frame['structure_value'], errors='coerce')
    source = frame['structure_value_source'].astype('string').fillna('')
    area = _numeric_column(frame, 'area_m2')
    per_m2 = value / area.where(area > 0)
    homes = _numeric_column(frame, 'n_manufactured_homes_per_parcel')
    from_parcel = source.str.startswith('parcel') & ~is_nsi_token(source)
    year_source = (
        frame['year_built_source'].astype('string').fillna('')
        if 'year_built_source' in frame.columns
        else pd.Series('', index=frame.index, dtype='string')
    )
    has_record = year_source.str.startswith('parcel')
    unlisted = (
        value.isna()
        | value.eq(0)
        | ~from_parcel
        | homes.ge(PARK_MIN_HOMES)
        | (per_m2.lt(UNLISTED_MAX_VALUE_PER_M2) & ~has_record)
    )
    listed = ~unlisted & homes.eq(1) & per_m2.ge(LISTED_MIN_VALUE_PER_M2)
    out[mh] = 'unclear'
    out[mh & unlisted.fillna(True)] = 'unlisted'
    out[mh & listed.fillna(False)] = 'listed'
    return out


def park_value_factor(
    observed: pd.Series,
    predicted: pd.Series,
    county: pd.Series,
    min_homes: int = PARK_FACTOR_MIN_HOMES,
) -> float:
    """Value of park homes relative to listed homes like them.

    Some parks carry their homes on the roll: the park owner owns and
    rents them out, which makes them the landowner's real property, and
    the assessor appraises them. Their roll value over what the
    manufactured-home model predicts for a listed home of the same size,
    age and sections measures how the county values homes in parks. The
    NCDOR manual (2019, Section VII) names rental units in parks as the
    exception to appraising all homes alike, and a household that does
    not own the land has less reason to invest in the home.

    Returns the median over counties with at least *min_homes* such
    homes of each county's median ratio, or 1.0 where no county has
    enough. *observed* and *predicted* are on the same value scale and
    index the park homes whose roll value is in the listed range.

    Measured 2026-10-18 on cheer-eastern-nc, out of sample: 0.531 across
    26 counties (IQR 0.378-0.723); rental clusters of 2-4 homes 0.942.
    """
    ratio = observed / predicted.where(predicted > 0)
    frame = pd.DataFrame({'ratio': ratio, 'county': county}).dropna()
    counts = frame.groupby('county').size()
    keep = counts.index[counts >= min_homes]
    if len(keep) == 0:
        return 1.0
    medians = frame[frame['county'].isin(keep)].groupby('county')['ratio'].median()
    return float(medians.median())


def value_per_area_caps(
    labels: pd.Series,
    area_m2: pd.Series,
    occupancy: pd.Series,
    quantile: float = VALUE_PER_AREA_QUANTILE,
) -> tuple[pd.Series, float]:
    """Upper bound on a filled value per square meter, by occupancy.

    The *quantile* of positive labels' value per m2 of footprint in each
    occupancy class with at least MIN_CLASS_ROWS of them, and over all
    labels for the rest. A filled value above its class bound is not a
    finding but a model outside its data (a sub-square-meter sliver, or
    a fit that left its training range), so it is clipped to the bound
    and its source says so.

    Returns
    -------
    caps : Series
        Bound by occupancy class.
    overall : float
        Bound for classes without enough labels.
    """
    area = pd.to_numeric(area_m2, errors='coerce')
    ok = labels.gt(0) & area.gt(0)
    per_m2 = labels[ok] / area[ok]
    overall = float(per_m2.quantile(quantile)) if len(per_m2) else np.inf
    occ = occupancy[ok].astype('string')
    counts = occ.value_counts()
    enough = counts.index[counts >= MIN_CLASS_ROWS]
    caps = per_m2[occ.isin(enough)].groupby(occ[occ.isin(enough)]).quantile(quantile)
    return caps, overall


def clip_to_caps(
    values: pd.Series,
    rows: pd.Series,
    area_m2: pd.Series,
    occupancy: pd.Series,
    caps: pd.Series,
    overall: float,
) -> tuple[pd.Series, pd.Series]:
    """Clip *values* on *rows* to their class bound; return them and a flag.

    Rows without a positive area are left to VALUE_CAP.
    """
    area = pd.to_numeric(area_m2, errors='coerce')
    bound = occupancy.astype('string').map(caps).astype(float).fillna(overall)
    limit = bound * area.where(area > 0)
    values = pd.to_numeric(values, errors='coerce')
    clipped = (rows & values.gt(limit)).fillna(False).astype(bool)
    return values.where(~clipped, limit), clipped


def value_labels(frame: pd.DataFrame, train: bool = True) -> pd.Series:
    """Return parcel-recorded structure values (>= 0), else NaN.

    A manufactured home's value counts only where the roll lists the
    home (:func:`manufactured_home_listing`): with *train*, only
    'listed' homes are labels; without it, 'unclear' homes also keep
    their observed value. 'unlisted' homes are always refilled.
    """
    value = pd.to_numeric(frame['structure_value'], errors='coerce')
    src = frame['structure_value_source'].fillna('')
    ok = value.ge(0) & src.ne('') & ~is_nsi_token(src) & src.ne('withheld')
    listing = manufactured_home_listing(frame)
    keep = {'listed'} if train else {'listed', 'unclear'}
    ok &= listing.isna() | listing.isin(keep)
    return value.where(ok)


def _log(series: pd.Series) -> pd.Series:
    # Rolls code an unknown size or value as 0; read it as missing.
    values = pd.to_numeric(series, errors='coerce')
    return np.log1p(values.where(values > 0))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive model inputs from the canonical and evidence columns."""
    out = pd.DataFrame(index=frame.index)
    out['log_area_m2'] = _log(frame['area_m2'])
    out['x_km'] = frame['x'] / 1000.0
    out['y_km'] = frame['y'] / 1000.0
    # Already a log (ln USD/ha, land_value_per_ha_log_2010).
    out['log_land_value_2010'] = pd.to_numeric(
        frame['land_value_2010'], errors='coerce'
    )
    out['distance_to_core_km'] = frame['distance_to_core_km']
    for column in [
        'structure_value_building_nsi',
        'structure_value_building_nsi_per_area',
        'gross_floor_area_sqft_building_nsi',
        'land_value_parcel',
        'living_area_sqft_parcel',
    ]:
        out[f'log_{column}'] = _log(frame[column])
    for column in [
        'year_built_block_median_building_nsi',
        'n_stories_building_nsi',
        'n_dwellings_nsi',
        'n_buildings_nsi',
        'n_stories_parcel',
        'n_footprints_per_parcel',
        'n_parcels_per_footprint',
        'overlap_fraction_parcel',
        'n_manufactured_homes_per_parcel',
        'n_stories_building_cheer',
        'n_stories_brails',
    ]:
        out[column] = pd.to_numeric(frame[column], errors='coerce')
    out['has_homestead_exemption_parcel'] = pd.to_numeric(
        frame['has_homestead_exemption_parcel'].astype('float'), errors='coerce'
    )
    missing = pd.Series(np.nan, index=frame.index)
    for column in CENSUS_COLUMNS:
        values = pd.to_numeric(frame.get(column, missing), errors='coerce')
        if column == 'acs_median_year_built_tract':
            out[column] = values.where(values.between(*ACS_YEAR_RANGE))
        else:
            out[f'log_{column}'] = _log(values)
    for column in CATEGORICAL_INPUTS:
        if column == 'geometry_kind':
            out[column] = geometry_kind(frame['geometry_source'])
        else:
            out[column] = frame[column].astype('string')
    return out


def geometry_kind(tokens: pd.Series) -> pd.Series:
    """Collapse geometry source tokens to their kind (obm, parcel, ...)."""
    first = tokens.astype('string').str.split('.', n=1).str[0]
    return first.fillna('unknown')


def county_table(frame: pd.DataFrame, county_area_km2: pd.Series) -> pd.DataFrame:
    """Summarize each county with features available without parcel data."""
    g = frame.groupby('admin3_id')
    table = pd.DataFrame(
        {
            'x': g['x'].mean(),
            'y': g['y'].mean(),
            'n': g.size(),
            'county_nsi_value_per_area': g[
                'structure_value_building_nsi_per_area'
            ].median(),
            'county_nsi_year': g['year_built_block_median_building_nsi'].median(),
            'county_share_manufactured': g['occupancy_type'].agg(
                lambda s: s.astype('string').eq('Manufactured Home').mean()
            ),
            'county_log_area': g['area_m2'].agg(
                lambda s: np.log(pd.to_numeric(s, errors='coerce')).median()
            ),
            'county_log_land_value_2010': g['land_value_2010'].median(),
            'county_share_metro': g['region_type'].agg(
                lambda s: s.astype('string').eq('metro').mean()
            ),
            'county_distance_to_core': g['distance_to_core_km'].median(),
        }
    )
    area = county_area_km2.reindex(table.index)
    table['county_log_density'] = np.log(table['n'] / area)
    return table


def county_distance_km(table: pd.DataFrame) -> pd.DataFrame:
    """Distance between county centroids, in kilometers."""
    xy = table[['x', 'y']].to_numpy() / 1000.0
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    return pd.DataFrame(d, index=table.index, columns=table.index)


def region_orders(neighbors: dict[str, set[str]], max_order: int) -> dict:
    """Breadth-first graph order between every pair of regions."""
    orders: dict[str, dict[str, int]] = {}
    for start in neighbors:
        seen = {start: 0}
        frontier = [start]
        for order in range(1, max_order + 1):
            nxt = []
            for region in frontier:
                for other in neighbors.get(region, ()):
                    if other not in seen:
                        seen[other] = order
                        nxt.append(other)
            frontier = nxt
        orders[start] = seen
    return orders


class DonorWeights:
    """Per-building donor weights for one target county.

    Parameters
    ----------
    frame : DataFrame
        Buildings with 'admin3_id' and 'region_id'.
    table : DataFrame
        Output of :func:`county_table`.
    orders : dict
        Output of :func:`region_orders`.
    """

    def __init__(self, frame: pd.DataFrame, table: pd.DataFrame, orders: dict):
        self.table = table
        self.distance = county_distance_km(table)
        self.orders = orders
        shares = (
            frame.groupby(['admin3_id', 'region_id']).size()
            / frame.groupby('admin3_id').size()
        )
        self.region_shares = {
            county: grp.droplevel(0) for county, grp in shares.groupby(level=0)
        }

    def weights(
        self,
        target: str,
        donors: pd.DataFrame,
        config: DonorConfig,
    ) -> np.ndarray:
        """Return the weight of each donor row for *target*."""
        county = donors['admin3_id'].to_numpy()
        w = np.ones(len(donors))
        if config.mode == 'region':
            shares = self.region_shares.get(target, pd.Series(dtype=float))
            lookup = {}
            for region in donors['region_id'].dropna().unique():
                total = 0.0
                for target_region, share in shares.items():
                    order = self.orders.get(target_region, {}).get(region)
                    if order is not None and order <= config.max_order:
                        total += share * config.decay**order
                lookup[region] = total
            w = donors['region_id'].map(lookup).fillna(0.0).to_numpy(float)
        elif config.mode == 'km':
            d = self.distance.loc[target].reindex(county).to_numpy()
            w = np.exp(-((d / config.bandwidth_km) ** 2))
        far = self.distance.loc[target].reindex(county).to_numpy()
        w = np.where(far > config.radius_km, 0.0, w)
        w = np.where(county == target, 1.0, w)
        return w


def training_sample(
    labeled: pd.DataFrame,
    target: str,
    config: DonorConfig,
    exclude_target: bool,
    seed: int,
) -> pd.DataFrame:
    """Sample labeled rows: the target county's own plus capped donors."""
    rng = np.random.default_rng(seed)
    parts = []
    for county, grp in labeled.groupby('admin3_id', sort=True):
        if county == target and exclude_target:
            continue
        cap = config.own_cap if county == target else config.donor_cap
        if len(grp) > cap:
            grp = grp.iloc[rng.choice(len(grp), cap, replace=False)]
        parts.append(grp)
    return pd.concat(parts) if parts else labeled.iloc[:0]


def _category_frame(x: pd.DataFrame, categories: dict) -> pd.DataFrame:
    x = x.copy()
    for column, cats in categories.items():
        if column in x.columns:
            x[column] = pd.Categorical(x[column].astype('string'), categories=cats)
    return x


def _boost_kwargs(seed: int) -> dict:
    # No row subsampling exists in histogram boosting, and max_features
    # stays at 1.0 (design limit L12).
    return dict(
        max_iter=300,
        learning_rate=0.1,
        max_leaf_nodes=63,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        categorical_features='from_dtype',
        random_state=seed,
    )


def fit_regressor(x, y, w, loss: str, seed: int):
    model = HistGradientBoostingRegressor(loss=loss, **_boost_kwargs(seed))
    model.fit(x, y, sample_weight=w)
    return model


def fit_classifier(x, y, w, seed: int):
    model = HistGradientBoostingClassifier(**_boost_kwargs(seed))
    model.fit(x, y, sample_weight=w)
    return model


@dataclass
class TargetSpec:
    """One column to impute.

    Parameters
    ----------
    column : str
        Canonical column filled.
    kind : str
        'year', 'value', 'log' (positive continuous on a log scale),
        'class' (categorical draw), 'count' (integer, drawn as a class).
    per_county : bool
        True for a model per county with donor weights; False for one
        pooled model across all counties.
    rows : str
        'primary' or 'manufactured' (rows the column applies to).
    """

    column: str
    kind: str
    per_county: bool
    rows: str = 'primary'


TARGETS = [
    TargetSpec('area_m2', 'log', per_county=False),
    TargetSpec('year_built', 'year', per_county=True),
    TargetSpec('n_stories', 'count', per_county=False),
    TargetSpec('construction_type', 'class', per_county=False),
    TargetSpec('foundation_type', 'class', per_county=False),
    TargetSpec('roof_shape', 'class', per_county=False),
    TargetSpec('has_garage', 'class', per_county=False),
    TargetSpec('n_sections', 'count', per_county=False, rows='manufactured'),
    TargetSpec('n_dwellings', 'count', per_county=False),
    TargetSpec('structure_value', 'value', per_county=True),
]
# The manufactured-home pass of structure_value: fitted on listed homes
# only and applied to every manufactured home whose value the roll does
# not give (manufactured_home_listing). Imputer.run calls it for those
# rows in place of the general value model.
MANUFACTURED_VALUE = TargetSpec(
    'structure_value', 'value', per_county=True, rows='manufactured'
)
# Canonical columns that become inputs once filled, in chain order.
CHAIN_INPUTS = {
    'area_m2': [],
    'year_built': [],
    'n_stories': ['year_built'],
    'construction_type': ['year_built', 'n_stories'],
    'foundation_type': ['year_built', 'n_stories', 'construction_type'],
    'roof_shape': ['year_built', 'n_stories', 'construction_type'],
    'has_garage': ['year_built', 'n_stories', 'roof_shape'],
    'n_sections': ['year_built', 'roof_shape'],
    'n_dwellings': ['year_built', 'n_stories'],
    'structure_value': [
        'year_built',
        'n_stories',
        'construction_type',
        'foundation_type',
        'roof_shape',
        'has_garage',
        'n_dwellings',
    ],
}
MANUFACTURED_CHAIN_INPUTS = ['year_built', 'n_sections', 'roof_shape']
CLASS_COLUMNS = {
    'construction_type',
    'foundation_type',
    'roof_shape',
    'has_garage',
    'n_stories',
    'n_sections',
    'n_dwellings',
}
NUMERIC_CLASS_COLUMNS = {'has_garage', 'n_stories', 'n_sections', 'n_dwellings'}


def input_columns(spec: TargetSpec) -> list[str]:
    """Inputs for *spec*: evidence, county features and chain columns."""
    if spec == MANUFACTURED_VALUE:
        chain = [f'chain_{c}' for c in MANUFACTURED_CHAIN_INPUTS]
        return list(MANUFACTURED_VALUE_INPUTS) + chain
    numeric = list(NUMERIC_INPUTS) + list(COUNTY_FEATURES)
    categorical = list(CATEGORICAL_INPUTS)
    if spec.per_county:
        # A county without labels of its own carries county features
        # and coordinates no training row shares, so trees route the
        # whole county to whichever donor those splits isolate, and it
        # inherits that donor's level (Askari 2026: location alone does
        # not transfer). Donors reach it through distance weights.
        # Measured 2026-10-15 on US-NC-CHO: with them, the median
        # probability of a zero structure value was 0.989.
        numeric = [c for c in numeric if c not in COUNTY_FEATURES and c not in LOCATION]
    if spec.column == 'structure_value':
        # Design limit L24: no core size or type in the value model.
        categorical = [c for c in categorical if c != 'region_type']
    if spec.column == 'area_m2':
        numeric = [c for c in numeric if c != 'log_area_m2']
    # The evidence a column's labels were reconciled from would teach
    # the model to copy it, and the rows to fill lack it by definition.
    label_evidence = LABEL_EVIDENCE.get(spec.column, set())
    numeric = [c for c in numeric if c not in label_evidence]
    categorical = [c for c in categorical if c not in label_evidence]
    chain = [f'chain_{c}' for c in CHAIN_INPUTS[spec.column]]
    return numeric + categorical + chain


def labels_for(spec: TargetSpec, frame: pd.DataFrame, train: bool = True) -> pd.Series:
    """Observed values usable as labels for *spec*.

    With *train* False, the values kept as observed (not refilled),
    which for structure value also include 'unclear' manufactured homes.
    """
    if spec.kind == 'year':
        return year_labels(frame)
    if spec.kind == 'value':
        return value_labels(frame, train=train)
    values = frame[spec.column]
    if spec.column in {'n_stories', 'n_dwellings', 'n_sections'}:
        values = pd.to_numeric(values, errors='coerce')
        values = values.where(values >= 1)
        cap = {'n_stories': 6, 'n_dwellings': 20, 'n_sections': 3}[spec.column]
        return values.clip(upper=cap).round()
    if spec.column == 'area_m2':
        return pd.to_numeric(values, errors='coerce').where(lambda v: v > 0)
    if spec.column == 'has_garage':
        return pd.to_numeric(values, errors='coerce')
    # A combined value ("Slab; Crawl") names its first-listed class as
    # the primary one; models learn primary classes only.
    return values.astype('string').str.split(';').str[0].str.strip()


MIN_CLASS_ROWS = 50


def drop_rare_classes(y: pd.Series) -> pd.Series:
    """Mask classes with fewer than MIN_CLASS_ROWS labels.

    Early stopping holds out a stratified validation split, which
    fails on a class with a single member, and a class this rare
    cannot be learned anyway.
    """
    counts = y.value_counts()
    rare = counts[counts < MIN_CLASS_ROWS].index
    return y.where(~y.isin(rare))


def applies_to(spec: TargetSpec, frame: pd.DataFrame) -> pd.Series:
    occ = frame['occupancy_type'].astype('string')
    if spec.rows == 'manufactured':
        return occ.eq('Manufactured Home')
    return occ.ne('Secondary') & occ.notna()


def chain_value(spec_column: str, values: pd.Series) -> pd.Series:
    """Encode a filled canonical column as a model input."""
    if spec_column in {'construction_type', 'foundation_type', 'roof_shape'}:
        return values.astype('string')
    if spec_column == 'area_m2':
        return np.log1p(pd.to_numeric(values, errors='coerce'))
    return pd.to_numeric(values, errors='coerce')


# Bands of the predicted log value whose training residuals a filled
# value draws from; fewer where a model has few positive labels.
VALUE_RESIDUAL_BANDS = 10
MIN_BAND_ROWS = 200


@dataclass
class ValueModel:
    """The fitted parts of a structure value model (:func:`fit_value_model`).

    Parameters
    ----------
    zero : classifier or None
        P(value = 0); None where the labels have no zeros.
    positive : regressor
        Mean log value given positive.
    edges : ndarray
        Inner band edges on the predicted log value.
    residuals, weights : list of ndarray
        Training log residuals per band, and their sample weights.
    smear : ndarray
        Weighted mean of exp(residual) per band.
    scale : float
        Factor equating expected and observed weighted training totals.
    """

    zero: object | None
    positive: object
    edges: np.ndarray
    residuals: list
    weights: list
    smear: np.ndarray
    scale: float


def fit_value_model(x, y, w, seed: int) -> ValueModel:
    """Fit a two-part structure value model with unbiased totals.

    Why this design: building loss models need both the mean (totals)
    and the spread of values right.

    - The positive part is boosting on log value with squared error.
      Poisson loss (v0.4 draft) keeps totals but fits buildings worse,
      and gamma loss fits them better on the log scale but its Newton
      step divides by sum(y / mean), which vanishes in a leaf of small
      labels with a large mean, so leaf values diverge. Log values
      bound every leaf by the labels' range.
    - exp(predicted log value) is a median, not a mean; each band of
      predictions carries the mean of exp(residual) over its training
      rows (a smearing factor, Duan 1983), so expectations are means.
    - One factor then equates the expected and the observed weighted
      training totals, which the zero part and the bands leave close
      but not equal.
    - A filled value is drawn from the same band's residuals, so the
      spread follows the roll's and totals stay unbiased.

    Measured 2026-10-21 on cheer-eastern-nc: with 14 counties held out
    on the same inputs, Poisson loss gave a median county total bias of
    +0.6%, building MAE $172,488 and a 20.0% typical county miss; gamma
    loss gave -8.8%, $138,533 and 17.1%.
    """
    zero = (y <= 0).astype(int)
    # Early stopping's stratified split fails on a class with one
    # member; a part with this few labels is left out, like a rare
    # class (drop_rare_classes).
    zero_model = (
        fit_classifier(x, zero, w, seed)
        if zero.nunique() > 1 and zero.value_counts().min() >= MIN_CLASS_ROWS
        else None
    )
    pos = (y > 0).to_numpy()
    log_y = np.log(y[pos].to_numpy(float))
    positive = fit_regressor(x[pos], log_y, w[pos], loss='squared_error', seed=seed)
    fitted = positive.predict(x[pos])
    residual = log_y - fitted
    n_bands = int(max(1, min(VALUE_RESIDUAL_BANDS, pos.sum() // MIN_BAND_ROWS)))
    edges = np.quantile(fitted, np.linspace(0, 1, n_bands + 1)[1:-1])
    band = np.digitize(fitted, edges)
    weights = w[pos]
    residuals = [residual[band == b] for b in range(n_bands)]
    band_weights = [weights[band == b] for b in range(n_bands)]
    smear = np.array(
        [
            np.average(np.exp(r), weights=bw) if len(r) else 1.0
            for r, bw in zip(residuals, band_weights, strict=True)
        ]
    )
    model = ValueModel(zero_model, positive, edges, residuals, band_weights, smear, 1.0)
    expected = _predict_value(model, x)
    total = float(np.sum(w * expected))
    model.scale = float(np.sum(w * y.to_numpy(float))) / total if total > 0 else 1.0
    return model


def _predict_value(model: ValueModel, x: pd.DataFrame, rng=None) -> np.ndarray:
    """Structure value per row: a draw, or the expectation without *rng*.

    With *rng*, each building is zero with its predicted probability and
    otherwise gets its predicted log value plus a residual drawn from
    its band. The expectation, (1 - p_zero) times the mean positive
    value, is unbiased for totals too, but it gives a building a value
    no building has (a house with a 95% chance of a zero roll value got
    5% of a house's value), and loss models read buildings one by one.

    Measured 2026-10-20 on cheer-eastern-nc: the expectation's median
    imputed single-family value was $4,840 on 164 homes in Wake.
    """
    p_zero = (
        model.zero.predict_proba(x)[:, 1]
        if model.zero is not None
        else np.zeros(len(x))
    )
    log_value = model.positive.predict(x)
    band = np.digitize(log_value, model.edges)
    if rng is None:
        value = (1.0 - p_zero) * np.exp(log_value) * model.smear[band]
    else:
        residual = np.zeros(len(x))
        for b in np.unique(band):
            rows = np.flatnonzero(band == b)
            pool, weights = model.residuals[b], model.weights[b]
            if len(pool):
                residual[rows] = rng.choice(
                    pool, size=len(rows), p=weights / weights.sum()
                )
        value = np.where(rng.random(len(x)) < p_zero, 0.0, np.exp(log_value + residual))
    # No building is worth more than the cap.
    return np.minimum(value * model.scale, VALUE_CAP)


def fit_target(
    spec: TargetSpec,
    x: pd.DataFrame,
    y: pd.Series,
    w: np.ndarray,
    seed: int,
):
    """Fit the model(s) for one target."""
    if spec.kind == 'value':
        return fit_value_model(x, y, w, seed)
    if spec.kind == 'year':
        return fit_regressor(x, y, w, loss='squared_error', seed=seed)
    if spec.kind == 'log':
        return fit_regressor(x, np.log(y), w, loss='squared_error', seed=seed)
    return fit_classifier(x, y.astype('string'), w, seed)


def predict_target(spec: TargetSpec, model, x: pd.DataFrame, rng) -> np.ndarray:
    """Predict a filled value for each row of *x*.

    Continuous targets get the model mean, except that a structure
    value is first drawn zero or positive (:func:`_predict_value`).
    Categorical and count targets get one draw from the predicted class
    probabilities, which keeps class shares unbiased in aggregate where
    the most likely class would overstate the majority class.
    """
    if spec.kind == 'value':
        return _predict_value(model, x, rng)
    if spec.kind == 'year':
        return np.clip(model.predict(x), *YEAR_RANGE).round()
    if spec.kind == 'log':
        return np.exp(model.predict(x))
    proba = model.predict_proba(x)
    cumulative = proba.cumsum(axis=1)
    u = rng.random(len(x))[:, None]
    picks = (u > cumulative).sum(axis=1).clip(max=proba.shape[1] - 1)
    return model.classes_[picks]


def appraisal_years(revaluations: pd.DataFrame, roll_year: int) -> pd.Series:
    """Return the appraisal year a county's roll reflects, by county FIPS.

    A roll of *roll_year* carries the latest revaluation effective on
    or before January 1 of that year; a revaluation scheduled after it
    is not yet on the roll, so the previous one applies.
    """
    last = revaluations['last_reval_year'].astype(int)
    previous = pd.to_numeric(revaluations['previous_reval_year'], errors='coerce')
    year = last.where(last <= roll_year, previous)
    return pd.Series(
        year.to_numpy(), index=revaluations['county_fips'].astype(str).str.zfill(5)
    )


def value_scales(
    hpi: pd.DataFrame, appraisal: pd.Series, common_year: int
) -> pd.Series:
    """Factor taking each county's appraised values to *common_year*.

    Uses a published county index (FHFA all-transactions HPI); no index
    is built here (design limit L29). A county missing either year
    falls back to the median factor of the counties appraised in the
    same year. An appraisal year after the index's last year (a
    reappraisal effective 1 January of the year after it) is measured
    from that last year: no movement is known beyond it.

    Parameters
    ----------
    hpi : DataFrame
        Columns 'fips' (5-digit string), 'year', 'hpi'.
    appraisal : Series
        Appraisal year by county FIPS.
    common_year : int
        Year all values are expressed in.
    """
    table = hpi.pivot_table(index='fips', columns='year', values='hpi')
    last = max(table.columns) if len(table.columns) else common_year

    def level(fips, year):
        year = min(int(year), last)
        if fips not in table.index or year not in table.columns:
            return np.nan
        return table.at[fips, year]

    rows = []
    for fips, year in appraisal.items():
        base = level(fips, year)
        top = level(fips, common_year)
        rows.append((fips, year, top / base if base and base > 0 else np.nan))
    out = pd.DataFrame(rows, columns=['fips', 'year', 'scale']).set_index('fips')
    fallback = out.groupby('year')['scale'].transform('median')
    out['fallback'] = out['scale'].isna()
    out['scale'] = out['scale'].fillna(fallback)
    return out


def _prepare_inputs(frame: pd.DataFrame, features: pd.DataFrame, table):
    x = features.copy()
    county = frame['admin3_id']
    for column in COUNTY_FEATURES:
        x[column] = county.map(table[column]).to_numpy()
    return x


class Imputer:
    """Fill the primary columns of a footprint inventory, county by county.

    Parameters
    ----------
    frame : DataFrame
        Canonical plus evidence columns, with 'x', 'y' (EPSG:5070
        meters), 'region_id', 'region_type', 'distance_to_core_km' and
        'land_value_2010' added by the caller.
    county_area_km2 : Series
        Land area by admin3_id, for the density feature.
    orders : dict
        Region-graph orders (:func:`region_orders`).
    config : DonorConfig
        Donor weighting for the per-county targets.
    value_scale : Series, optional
        Factor by admin3_id taking appraised values to a common year;
        1 where absent.
    seed : int
        Base seed.
    exclude_inputs : tuple of str
        Inputs left out of every model.
    acs_value : Series, optional
        ACS median home value by admin3_id in the common year
        (:func:`acs_value_levels`); without it, no county's price level
        is anchored.
    extensions : ImputationExtensions, optional
        Steps a separate module adds (repairs, training-row and held-out
        hooks); none by default.
    anchor : str
        How a county without value labels gets its price level: 'acs'
        (:meth:`_price_anchor`), 'nsi' (:meth:`_nsi_anchor`) or 'both'
        (their geometric mean). Anything else raises ValueError.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        county_area_km2: pd.Series,
        orders: dict,
        config: DonorConfig,
        value_scale: pd.Series | None = None,
        seed: int = DEFAULT_SEED,
        exclude_inputs: tuple[str, ...] = (),
        acs_value: pd.Series | None = None,
        extensions: ImputationExtensions | None = None,
        anchor: str = 'acs',
    ):
        if anchor not in {'acs', 'nsi', 'both'}:
            raise ValueError(f"anchor must be 'acs', 'nsi' or 'both', not {anchor!r}")
        self.anchor = anchor
        self.extensions = extensions or ImputationExtensions()
        self.voided = {}
        if self.extensions.repair is not None:
            frame, self.voided = self.extensions.repair(frame)
        self.frame = frame
        self.acs_value = acs_value
        self.config = config
        self.seed = seed
        self.table = county_table(frame, county_area_km2)
        self.donors = DonorWeights(frame, self.table, orders)
        self.features = _prepare_inputs(frame, build_features(frame), self.table)
        self.value_scale = (
            frame['admin3_id'].map(value_scale).fillna(1.0)
            if value_scale is not None
            else pd.Series(1.0, index=frame.index)
        )
        cats = {}
        for column in CATEGORICAL_INPUTS:
            cats[column] = sorted(self.features[column].dropna().unique().tolist())
        self.categories = cats
        # An input with no value anywhere (a source absent from the
        # region) breaks histogram binning; leave it out of every model.
        self.empty_inputs = {
            c for c in self.features.columns if self.features[c].notna().sum() == 0
        } | set(exclude_inputs)

    def _x(self, rows: pd.Index, spec: TargetSpec, filled: pd.DataFrame):
        x = self.features.loc[rows].copy()
        chain = (
            MANUFACTURED_CHAIN_INPUTS
            if spec == MANUFACTURED_VALUE
            else CHAIN_INPUTS[spec.column]
        )
        for column in chain:
            x[f'chain_{column}'] = chain_value(column, filled.loc[rows, column])
        cats = dict(self.categories)
        for column in chain:
            if column in {'construction_type', 'foundation_type', 'roof_shape'}:
                cats[f'chain_{column}'] = self.chain_categories[column]
        columns = [c for c in input_columns(spec) if c not in self.empty_inputs]
        return _category_frame(x[columns], cats)

    def _labels(self, spec: TargetSpec) -> pd.Series:
        y = labels_for(spec, self.frame)
        if spec.kind == 'value':
            y = y * self.value_scale
        y = y.where(applies_to(spec, self.frame))
        if spec.column in CLASS_COLUMNS:
            y = drop_rare_classes(y.astype('string')).astype('string')
        return y

    def run(self, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fill every target; return filled columns and imputed flags."""
        filled = pd.DataFrame(index=self.frame.index)
        flags = pd.DataFrame(index=self.frame.index)
        self.chain_categories = {}
        # Counties filled from donors only, by column (MIN_OWN_LABEL_SHARE).
        self.donors_only = {}
        # Price-level factors of counties without value labels.
        self.price_anchor = {}
        for spec in TARGETS:
            y = self._labels(spec)
            rows = applies_to(spec, self.frame)
            observed = labels_for(spec, self.frame, train=False)
            missing = rows & observed.isna()
            parts = [(spec, y, pd.Series(True, index=self.frame.index))]
            if spec.column == 'structure_value':
                # Manufactured homes get their own model, trained on
                # listed homes only (manufactured_home_listing).
                is_mh = applies_to(MANUFACTURED_VALUE, self.frame)
                parts = [
                    (spec, y.where(~is_mh), ~is_mh),
                    (MANUFACTURED_VALUE, self._labels(MANUFACTURED_VALUE), is_mh),
                ]
            # Observed cells keep their delivered value unchanged; only
            # missing cells (and NSI-sourced years and values, which
            # labels_for treats as missing) are written.
            current = self.frame[spec.column].copy().astype(object)
            current[missing] = np.nan
            if spec.kind in {'year', 'value'}:
                nsi = is_nsi_token(self.frame[f'{spec.column}_source'].fillna(''))
                current[nsi] = np.nan
            if spec.kind == 'value':
                current = pd.to_numeric(current, errors='coerce') * self.value_scale
            if spec.per_county:
                for county in sorted(self.frame['admin3_id'].unique()):
                    in_county = missing & self.frame['admin3_id'].eq(county)
                    for part_spec, part_y, part_rows in parts:
                        target_rows = in_county & part_rows
                        if not target_rows.any():
                            continue
                        donors_only = self._own_labels_too_sparse(
                            part_spec, part_y, rows & part_rows, county
                        )
                        if donors_only:
                            self.donors_only.setdefault(spec.column, []).append(county)
                        pred = self._fit_predict_county(
                            part_spec,
                            part_y,
                            filled,
                            county,
                            target_rows,
                            exclude=False,
                            drop_own=donors_only,
                        )
                        current.loc[target_rows] = pred
                        log(
                            f'{spec.column} ({part_spec.rows}) {county}: '
                            f'filled {int(target_rows.sum())}'
                        )
                    if spec.column == 'structure_value' and self._anchors_price(
                        parts[0][1], county
                    ):
                        values = pd.to_numeric(current[in_county], errors='coerce')
                        factor = self._anchor_factor(county, values, y)
                        self.price_anchor[county] = factor
                        current.loc[in_county] = values * factor
                        log(f'structure_value {county}: price anchor {factor:.3f}')
                if spec.column == 'structure_value':
                    current = self._apply_park_factor(current, missing, log)
                    caps, overall = value_per_area_caps(
                        y, filled['area_m2'], self.frame['occupancy_type']
                    )
                    current, self.clipped = clip_to_caps(
                        current,
                        missing,
                        filled['area_m2'],
                        self.frame['occupancy_type'],
                        caps,
                        overall,
                    )
                    log(
                        f'structure_value: clipped {int(self.clipped.sum())} '
                        'filled values to their value per m2 bound'
                    )
            elif missing.any():
                labeled = y.notna() & rows
                train = self._pooled_sample(labeled)
                x = self._x(train, spec, filled)
                xt = self._x(missing[missing].index, spec, filled)
                current.loc[missing] = self._guarded_fit_predict(
                    spec, x, y.loc[train], np.ones(len(train)), xt
                )
                log(f'{spec.column}: filled {int(missing.sum())}')
            if spec.kind == 'value':
                current = current / self.value_scale
            if spec.column in NUMERIC_CLASS_COLUMNS or spec.kind in {
                'year',
                'value',
                'log',
            }:
                current = pd.to_numeric(current, errors='coerce')
            if spec.column == 'area_m2':
                self.features['log_area_m2'] = np.log1p(current)
            filled[spec.column] = current
            flags[spec.column] = missing
            if spec.column in {'construction_type', 'foundation_type', 'roof_shape'}:
                self.chain_categories[spec.column] = sorted(
                    current.dropna().astype(str).unique().tolist()
                )
        return filled, flags

    def _apply_park_factor(self, current, missing, log) -> pd.Series:
        """Scale filled park homes by the region's park_value_factor.

        *current* holds structure values on the common-year scale, the
        filled ones included. The factor compares the roll values of
        park homes the roll lists with the model's prediction for them,
        then applies to every filled home in a park.
        """
        frame = self.frame
        homes = _numeric_column(frame, 'n_manufactured_homes_per_parcel')
        is_mh = applies_to(MANUFACTURED_VALUE, frame)
        park = is_mh & homes.ge(PARK_MIN_HOMES) & missing
        value = pd.to_numeric(frame['structure_value'], errors='coerce')
        source = frame['structure_value_source'].astype('string').fillna('')
        area = _numeric_column(frame, 'area_m2')
        on_roll = (
            park
            & source.str.startswith('parcel')
            & (value / area.where(area > 0)).ge(LISTED_MIN_VALUE_PER_M2)
        )
        self.park_factor = park_value_factor(
            value[on_roll] * self.value_scale[on_roll],
            pd.to_numeric(current[on_roll], errors='coerce'),
            frame.loc[on_roll, 'admin3_id'],
        )
        current = current.copy()
        current[park] = pd.to_numeric(current[park], errors='coerce') * self.park_factor
        log(
            f'park factor {self.park_factor:.3f} applied to '
            f'{int(park.sum())} park homes'
        )
        return current

    def _pooled_sample(self, labeled: pd.Series, cap: int = 400_000) -> pd.Index:
        idx = labeled[labeled].index
        if len(idx) > cap:
            rng = np.random.default_rng(self.seed)
            idx = idx[np.sort(rng.choice(len(idx), cap, replace=False))]
        return idx

    def _anchors_price(self, y: pd.Series, county: str) -> bool:
        """Whether *county*'s filled values get a price anchor.

        True where its own value labels (the general model's, *y*) do
        not set its level: it has none, or too few to train on
        (MIN_OWN_LABEL_SHARE).
        """
        if self.acs_value is None and self.anchor != 'nsi':
            return False
        if county in self.donors_only.get('structure_value', []):
            return True
        return not (y.notna() & self.frame['admin3_id'].eq(county)).any()

    def _price_anchor(
        self, county: str, values: pd.Series, y: pd.Series, exclude_own: bool
    ) -> float:
        """Factor setting *county*'s price level from its ACS home value.

        A county without value labels gets its level from donor counties
        whose prices may differ. The anchor carries over what transfers
        better: the ratio of a county's median single-family structure
        value to its ACS median home value. The target ratio is the
        donor-weighted geometric mean of that ratio over donor counties
        with labels; the factor moves the county's filled single-family
        median to it, and applies to all its filled values. Both
        medians are in the common year (:func:`acs_value_levels`).

        Parameters
        ----------
        county : str
            The county filled.
        values : Series
            Its filled values on the common-year scale, by row.
        y : Series
            Value labels on the common-year scale.
        exclude_own : bool
            Leave the county's own labels out (leave-county-out).
        """
        acs = self.acs_value
        if acs is None or not np.isfinite(acs.get(county, np.nan)):
            return 1.0
        admin = self.frame['admin3_id']
        single = self.frame['occupancy_type'].astype('string').eq('Single-Family')
        labeled = single & (y > 0).fillna(False)
        if exclude_own:
            labeled &= admin.ne(county)
        grouped = y[labeled].groupby(admin[labeled])
        medians = grouped.median()[grouped.size() >= ANCHOR_MIN_HOMES]
        ratio = (medians / acs.reindex(medians.index)).dropna()
        ratio = ratio[ratio > 0]
        if ratio.empty:
            return 1.0
        donors = pd.DataFrame(
            {'admin3_id': ratio.index, 'region_id': pd.NA}, index=ratio.index
        )
        w = self.donors.weights(county, donors, self.config)
        keep = (w > 1e-3) & (ratio.index != county)
        if not keep.any():
            return 1.0
        target = float(np.exp(np.average(np.log(ratio[keep]), weights=w[keep])))
        filled = values[single.reindex(values.index).fillna(False) & (values > 0)]
        if len(filled) < ANCHOR_MIN_HOMES:
            return 1.0
        current = float(filled.median()) / float(acs[county])
        return float(np.clip(target / current, *ANCHOR_RANGE))

    def _anchor_factor(self, county: str, values: pd.Series, y: pd.Series) -> float:
        """The delivery's level factor for a county without value labels.

        'acs' (:meth:`_price_anchor`), 'nsi' (:meth:`_nsi_anchor`), or
        'both', their geometric mean.
        """
        if self.anchor == 'nsi':
            return self._nsi_anchor(county, values, y, exclude_own=False)
        acs = self._price_anchor(county, values, y, exclude_own=False)
        if self.anchor == 'both':
            return float(
                np.sqrt(acs * self._nsi_anchor(county, values, y, exclude_own=False))
            )
        return acs

    def _nsi_anchor(
        self, county: str, values: pd.Series, y: pd.Series, exclude_own: bool
    ) -> float:
        """Factor setting *county*'s filled total from its NSI total.

        NSI's replacement-cost values see what stands in a county (a
        coast of large houses) that donors from elsewhere may not, but
        their level is off by a county-specific factor. The target is
        the county's NSI total over its filled rows with an NSI value,
        times the donor-weighted geometric mean of the ratio of roll
        totals to NSI totals in donor counties; the factor moves the
        filled total of those rows to it and applies to all filled
        values. Same inputs as :meth:`_price_anchor`.

        Measured 2026-10-24 on cheer-eastern-nc, 14 counties held out:
        NSI's county totals were 0.75 to 2.70 times the roll's (Pender
        1.02), where the donor model alone gave Pender 0.52.
        """
        admin = self.frame['admin3_id']
        nsi = _numeric_column(self.frame, 'structure_value_building_nsi')
        nsi = (nsi * self.value_scale).where(nsi > 0)
        labeled = y.notna() & nsi.notna()
        if exclude_own:
            labeled &= admin.ne(county)
        sums = pd.DataFrame({'y': y[labeled], 'nsi': nsi[labeled]}).groupby(
            admin[labeled]
        )
        totals = sums.sum()[sums.size() >= ANCHOR_MIN_HOMES]
        ratio = (totals['y'] / totals['nsi'])[lambda r: r > 0]
        if ratio.empty:
            return 1.0
        donors = pd.DataFrame(
            {'admin3_id': ratio.index, 'region_id': pd.NA}, index=ratio.index
        )
        w = self.donors.weights(county, donors, self.config)
        keep = (w > 1e-3) & (ratio.index != county)
        if not keep.any():
            return 1.0
        target = float(np.exp(np.average(np.log(ratio[keep]), weights=w[keep])))
        rows = nsi.reindex(values.index).notna() & values.notna()
        if rows.sum() < ANCHOR_MIN_HOMES:
            return 1.0
        current = float(values[rows].sum())
        expected = target * float(nsi.reindex(values.index)[rows].sum())
        if current <= 0:
            return 1.0
        return float(np.clip(expected / current, *ANCHOR_RANGE))

    def _own_labels_too_sparse(
        self, spec: TargetSpec, y: pd.Series, applicable: pd.Series, county: str
    ) -> bool:
        """Whether *county*'s own labels are too sparse to train on.

        True when they cover some but less than MIN_OWN_LABEL_SHARE of
        the rows the target applies to there. A county without labels
        is donor-only anyway. The manufactured-home value model is
        exempt: it trains on listed homes by design, and most homes in
        a county with parks are unlisted (manufactured_home_listing).
        """
        if spec == MANUFACTURED_VALUE:
            return False
        in_county = applicable & self.frame['admin3_id'].eq(county)
        total = int(in_county.sum())
        own = int((in_county & y.notna()).sum())
        return total > 0 and 0 < own < MIN_OWN_LABEL_SHARE * total

    def _fit_predict_county(
        self,
        spec: TargetSpec,
        y: pd.Series,
        filled: pd.DataFrame,
        county: str,
        target_rows: pd.Series,
        exclude: bool,
        config: DonorConfig | None = None,
        drop_own: bool = False,
        patterns: pd.DataFrame | None = None,
        holdout_patterns: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Fit *county*'s model on weighted donors and predict *target_rows*.

        *exclude* hides the county's own labels and masks its parcel value
        inputs (leave-county-out); *drop_own* only hides its labels, for
        a county whose few labels are not a sample of it. *patterns* are
        input rows of genuinely incomplete buildings for the training
        hook (default: the target rows); *holdout_patterns*, given, thin
        the target rows through the held-out hook (evaluation only).
        """
        config = config or self.config
        labeled = self.frame.loc[y.notna(), ['admin3_id', 'region_id']]
        sample = training_sample(
            labeled, county, config, exclude or drop_own, self.seed
        )
        w = self.donors.weights(county, sample, config)
        keep = w > 1e-3
        sample, w = sample[keep], w[keep]
        x = self._x(sample.index, spec, filled)
        xt = self._x(target_rows[target_rows].index, spec, filled)
        if exclude:
            masked = [c for c in PARCEL_VALUE_INPUTS if c in xt.columns]
            xt[masked] = np.nan
        hook = self.extensions.holdout_rows
        if holdout_patterns is not None and hook is not None:
            xt = hook(
                xt, holdout_patterns[xt.columns], np.random.default_rng(self.seed)
            )
        return self._guarded_fit_predict(spec, x, y.loc[sample.index], w, xt, patterns)

    def _gap_pools(
        self, spec: TargetSpec, lacking: pd.Series, county: str, filled: pd.DataFrame
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Input rows of incomplete buildings, split into two halves.

        Used only when an extension hook needs them; None otherwise.
        The split, one half of the patterns for training and the other
        for the held-out test, is Mohammad Askari's (UCLA) design, so
        that no gap pattern both trains and tests.
        """
        if self.extensions.train_rows is None and self.extensions.holdout_rows is None:
            return None, None
        rows = lacking & self.frame['admin3_id'].eq(county)
        if rows.sum() < MIN_GAP_ROWS:
            rows = lacking
        idx = rows[rows].index
        if len(idx) == 0:
            return None, None
        rng = np.random.default_rng(self.seed)
        if len(idx) > 2 * MAX_GAP_ROWS:
            idx = idx[rng.choice(len(idx), 2 * MAX_GAP_ROWS, replace=False)]
        idx = idx[rng.permutation(len(idx))]
        half = len(idx) // 2
        pool = self._x(idx, spec, filled)
        if half == 0:
            return pool, pool
        return pool.iloc[:half], pool.iloc[half:]

    def _guarded_fit_predict(self, spec, x, y, w, xt, patterns=None) -> np.ndarray:
        """Fit and predict, leaving out inputs the target rows lack.

        An input the target rows mostly lack, while the training rows
        mostly carry it, is left out of the model: missing there is
        rare in training and marks unusual rows, so the model would
        read the gap as a property of the buildings. Target rows are
        split by whether they have an NSI point, because buildings
        without one (new outlines, parcel placeholders) lack a whole
        block of evidence at once.

        Measured 2026-10-15 on US-NC-CHO: parcel land value 0 (read as
        missing); labeled rows missing it have a 63% zero structure
        value share. Measured 2026-10-15 on cheer-eastern-nc: none of
        the 213,604 primary buildings missing a construction type has
        NSI or CHEER evidence, and every labeled row has one of them.
        """
        rng = np.random.default_rng(self.seed)
        hook = self.extensions.train_rows
        if hook is not None:
            # The training hook prepares the model for the target rows'
            # gaps itself, so one model fills them all.
            source = xt if patterns is None else patterns[xt.columns]
            x, y, w = hook(x, y, w, source, np.random.default_rng(self.seed + 1))
            empty = [c for c in x.columns if x[c].notna().sum() == 0]
            model = fit_target(spec, x.drop(columns=empty), y, w, self.seed)
            out = predict_target(spec, model, xt.drop(columns=empty), rng)
            return out.astype(float) if spec.kind in {'year', 'value', 'log'} else out
        out = np.empty(len(xt), dtype=object)
        models = {}
        nsi_column = 'log_structure_value_building_nsi'
        has_nsi = (
            xt[nsi_column].notna()
            if nsi_column in xt.columns
            else pd.Series(True, index=xt.index)
        )
        for group in (True, False):
            mask = (has_nsi == group).to_numpy()
            if not mask.any():
                continue
            xg = xt[mask]
            shift = [
                c
                for c in x.columns
                if (xg[c].notna().mean() < 0.5 and x[c].notna().mean() >= 0.8)
                # Empty in this training sample: breaks histogram binning.
                or x[c].notna().sum() == 0
            ]
            # Both groups often leave out the same inputs: fit once.
            key = tuple(shift)
            if key not in models:
                models[key] = fit_target(spec, x.drop(columns=shift), y, w, self.seed)
            out[mask] = predict_target(spec, models[key], xg.drop(columns=shift), rng)
        if spec.kind in {'year', 'value', 'log'}:
            return out.astype(float)
        return out

    def leave_county_out(
        self,
        spec: TargetSpec,
        counties: list[str],
        configs: list[DonorConfig],
        filled: pd.DataFrame,
        cap: int = 20_000,
        log=print,
        thin_heldout: bool = False,
    ) -> pd.DataFrame:
        """Hide each county's labels, predict them from donors only.

        Returns one row per held-out building and configuration with
        the observed and predicted values on the common-year scale
        (values) or in years, plus the NSI value the old pipeline used
        for the same building. The held-out county's parcel value
        inputs are masked, so the test mimics a county without parcel
        values. With *thin_heldout*, the held-out rows are also given
        the input gaps of the county's genuinely incomplete buildings
        through the extensions' held-out hook.

        The county's incomplete buildings (rows the target applies to
        whose label is missing; the whole region's where the county has
        fewer than MIN_GAP_ROWS) are split in two at random: one half
        supplies the training hook's patterns, the other the held-out
        hook's, so no pattern both trains and tests.
        """
        y = self._labels(spec)
        lacking = (
            applies_to(spec, self.frame)
            & labels_for(spec, self.frame, train=False).isna()
        )
        if spec.kind == 'value':
            # Held-out predictions get the delivery's value per m2 bound.
            caps, overall = value_per_area_caps(
                y, filled['area_m2'], self.frame['occupancy_type']
            )
        rows = []
        for county in counties:
            held = y.notna() & self.frame['admin3_id'].eq(county)
            idx = held[held].index
            if len(idx) == 0:
                continue
            if len(idx) > cap:
                rng = np.random.default_rng(self.seed)
                idx = idx[np.sort(rng.choice(len(idx), cap, replace=False))]
            target_rows = pd.Series(False, index=self.frame.index)
            target_rows.loc[idx] = True
            train_pool, test_pool = self._gap_pools(spec, lacking, county, filled)
            for config in configs:
                pred = self._fit_predict_county(
                    spec,
                    y,
                    filled,
                    county,
                    target_rows,
                    exclude=True,
                    config=config,
                    patterns=train_pool,
                    holdout_patterns=test_pool if thin_heldout else None,
                )
                variants = {config.name(): pred}
                if spec.kind == 'value' and self.acs_value is not None:
                    # Scored with and without the price anchor, which
                    # applies to exactly this case: no labels of its own.
                    values = pd.Series(pred, index=idx)
                    factor = self._price_anchor(county, values, y, exclude_own=True)
                    nsi_factor = self._nsi_anchor(county, values, y, exclude_own=True)
                    variants[f'{config.name()}+anchor'] = pred * factor
                    variants[f'{config.name()}+nsi_anchor'] = pred * nsi_factor
                    variants[f'{config.name()}+both_anchors'] = pred * np.sqrt(
                        factor * nsi_factor
                    )
                    log(
                        f'LOCO {county} price anchor {factor:.3f}, '
                        f'NSI anchor {nsi_factor:.3f}'
                    )
                for name, values in variants.items():
                    if spec.kind == 'value':
                        values, _ = clip_to_caps(
                            pd.Series(values, index=idx),
                            pd.Series(True, index=idx),
                            filled.loc[idx, 'area_m2'],
                            self.frame.loc[idx, 'occupancy_type'],
                            caps,
                            overall,
                        )
                        values = values.to_numpy()
                    part = pd.DataFrame(
                        {
                            'admin3_id': county,
                            'census_tract_id': self.frame.loc[idx, 'census_tract_id'],
                            'config': name,
                            'observed': y.loc[idx].to_numpy(),
                            'predicted': values,
                        },
                        index=idx,
                    )
                    rows.append(part)
                log(f'LOCO {spec.column} {county} {config.name()}')
            if spec.kind == 'value':
                nsi = (
                    pd.to_numeric(
                        self.frame.loc[idx, 'structure_value_building_nsi'],
                        errors='coerce',
                    )
                    * self.value_scale.loc[idx]
                )
            else:
                nsi = pd.to_numeric(
                    self.frame.loc[idx, 'year_built_block_median_building_nsi'],
                    errors='coerce',
                )
            rows.append(
                pd.DataFrame(
                    {
                        'admin3_id': county,
                        'census_tract_id': self.frame.loc[idx, 'census_tract_id'],
                        'config': 'nsi',
                        'observed': y.loc[idx].to_numpy(),
                        'predicted': nsi.to_numpy(),
                    },
                    index=idx,
                )
            )
        return pd.concat(rows)

    def leave_county_out_pooled(
        self,
        spec: TargetSpec,
        counties: list[str],
        filled: pd.DataFrame,
        cap: int = 20_000,
        log=print,
        thin_heldout: bool = False,
    ) -> pd.DataFrame:
        """Leave-county-out for a pooled (not per-county) target.

        Fits the region-wide model as the delivery does, on labels
        outside the held-out county, and predicts the county's labeled
        rows, with its parcel value inputs masked and, with
        *thin_heldout*, its incomplete buildings' input gaps imposed,
        as :meth:`leave_county_out` does. Rows carry the observed and
        predicted values, and a 'baseline' configuration: the training
        labels' most common class, or their median.
        """
        y = self._labels(spec)
        rows = applies_to(spec, self.frame)
        lacking = rows & labels_for(spec, self.frame, train=False).isna()
        admin = self.frame['admin3_id']
        out = []
        for county in counties:
            held = y.notna() & rows & admin.eq(county)
            idx = held[held].index
            if len(idx) == 0:
                continue
            if len(idx) > cap:
                rng = np.random.default_rng(self.seed)
                idx = idx[np.sort(rng.choice(len(idx), cap, replace=False))]
            train = self._pooled_sample(y.notna() & rows & admin.ne(county))
            train_pool, test_pool = self._gap_pools(spec, lacking, county, filled)
            x = self._x(train, spec, filled)
            xt = self._x(idx, spec, filled)
            masked = [c for c in PARCEL_VALUE_INPUTS if c in xt.columns]
            xt[masked] = np.nan
            hook = self.extensions.holdout_rows
            if thin_heldout and hook is not None and test_pool is not None:
                xt = hook(xt, test_pool[xt.columns], np.random.default_rng(self.seed))
            pred = self._guarded_fit_predict(
                spec, x, y.loc[train], np.ones(len(train)), xt, train_pool
            )
            labels = y.loc[train]
            if spec.kind == 'class':
                baseline = labels.mode().iloc[0]
            else:
                baseline = pd.to_numeric(labels, errors='coerce').median()
            for name, values in (('model', pred), ('baseline', baseline)):
                out.append(
                    pd.DataFrame(
                        {
                            'admin3_id': county,
                            'config': name,
                            'observed': y.loc[idx].to_numpy(),
                            'predicted': values,
                        },
                        index=idx,
                    )
                )
            log(f'LOCO {spec.column} {county}')
        results = pd.concat(out)
        # One type per column, so the results write to parquet.
        for column in ('observed', 'predicted'):
            results[column] = (
                results[column].astype('string')
                if spec.kind == 'class'
                else pd.to_numeric(results[column], errors='coerce')
            )
        return results


def score_pooled(results: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Score :meth:`Imputer.leave_county_out_pooled` results.

    Classes and counts (both filled by a draw from the predicted class
    probabilities): accuracy, macro F1 (each class counts equally, so a
    model that only predicts the common class scores low), the median
    county accuracy and the median county share gap (half the summed
    absolute differences of class shares). Counts also get the numeric
    scores. Numbers: mean absolute error, the share
    within a tolerance (exact for counts, 25% for area) and the median
    absolute county mean bias.
    """
    from sklearn.metrics import f1_score

    out = []
    for config, grp in results.dropna(subset=['predicted']).groupby('config'):
        row = {'config': config, 'n': len(grp), 'counties': grp['admin3_id'].nunique()}
        if kind in {'class', 'count'}:
            observed = grp['observed'].astype(str)
            predicted = grp['predicted'].astype(str)
            if kind == 'count':
                # '2' and '2.0' are one class.
                observed, predicted = (
                    pd.to_numeric(s, errors='coerce').astype(float).round().astype(str)
                    for s in (observed, predicted)
                )
            hit = observed.eq(predicted)
            # Filled classes are draws, meant to keep each county's
            # class shares right: half the summed share differences.
            shares = pd.concat(
                {
                    'o': observed.groupby(grp['admin3_id']).value_counts(
                        normalize=True
                    ),
                    'p': predicted.groupby(grp['admin3_id']).value_counts(
                        normalize=True
                    ),
                },
                axis=1,
            ).fillna(0)
            share_gap = (shares['o'] - shares['p']).abs().groupby(level=0).sum() / 2
            row.update(
                accuracy=hit.mean(),
                macro_f1=f1_score(observed, predicted, average='macro'),
                county_accuracy_median=hit.groupby(grp['admin3_id']).mean().median(),
                county_share_gap_median=share_gap.median(),
            )
        if kind in {'count', 'log'}:
            observed = pd.to_numeric(grp['observed'], errors='coerce')
            predicted = pd.to_numeric(grp['predicted'], errors='coerce')
            if kind == 'count':
                within = predicted.round().eq(observed)
            else:
                within = (predicted / observed - 1).abs().le(0.25)
            frame = pd.DataFrame(
                {'o': observed, 'p': predicted, 'county': grp['admin3_id']}
            )
            county = frame.groupby('county')[['o', 'p']].mean()
            row.update(
                mae=(predicted - observed).abs().mean(),
                within=within.mean(),
                county_mean_abs_bias_median=(county['p'] - county['o']).abs().median(),
            )
        out.append(row)
    return pd.DataFrame(out).set_index('config')


def score(results: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Score leave-county-out results by configuration.

    Reports the section 5 criteria of the plan: the bias of county and
    tract totals (values) or means (years), then building-level error.
    Rows without a prediction (NSI absent) are dropped per
    configuration, so the NSI row covers only buildings NSI reaches.
    """
    out = []
    for config, grp in results.dropna(subset=['predicted']).groupby('config'):
        err = grp['predicted'] - grp['observed']
        row = {'config': config, 'n': len(grp), 'counties': grp['admin3_id'].nunique()}
        if kind == 'value':
            county = grp.groupby('admin3_id')[['observed', 'predicted']].sum()
            rel = county['predicted'] / county['observed'] - 1
            tract = grp.groupby('census_tract_id')[['observed', 'predicted']].sum()
            tract = tract[tract['observed'] > 0]
            trel = tract['predicted'] / tract['observed'] - 1
            row.update(
                county_total_bias_median=rel.median(),
                county_total_abs_bias_median=rel.abs().median(),
                tract_total_abs_bias_median=trel.abs().median(),
                mae=err.abs().mean(),
                median_ae=err.abs().median(),
                log_rmse=np.sqrt(
                    (
                        (np.log1p(grp['predicted']) - np.log1p(grp['observed'])) ** 2
                    ).mean()
                ),
            )
        else:
            county = grp.groupby('admin3_id')[['observed', 'predicted']].mean()
            tract = grp.groupby('census_tract_id')[['observed', 'predicted']].mean()
            row.update(
                county_mean_bias_median=(
                    county['predicted'] - county['observed']
                ).median(),
                county_mean_abs_bias_median=(county['predicted'] - county['observed'])
                .abs()
                .median(),
                tract_mean_abs_bias_median=(tract['predicted'] - tract['observed'])
                .abs()
                .median(),
                mae=err.abs().mean(),
                within_5=(err.abs() <= 5).mean(),
                era_agreement=(era(grp['predicted']) == era(grp['observed'])).mean(),
            )
        out.append(row)
    return pd.DataFrame(out).set_index('config')


def era(years: pd.Series) -> pd.Series:
    """Coarse construction eras for scoring only (cut years provisional).

    The plan sets the final cut years from NC and TX building-code
    adoption dates, verified against primary sources; until then these
    decade-style bins are a placeholder and are not delivered.
    """
    bins = [0, 1940, 1970, 1990, 2002, 2012, 3000]
    return pd.cut(years, bins=bins, labels=False, right=False)


# Delivery: `export_delivery` fills a bundle's primary columns with the
# functions below, on the frame it is about to write (after withholding),
# so no model sees a value the bundle may not ship.

# Evidence columns the models read; joined to the delivery's first read
# pass. Region, core and 2010 land value inputs are not among them: the
# leave-county-out test showed kilometer weighting without them performs
# as well (plans/building-attribute-imputation.md, section 11).
EVIDENCE_COLUMNS = [
    'structure_value_building_nsi',
    'structure_value_building_nsi_per_area',
    'gross_floor_area_sqft_building_nsi',
    'land_value_parcel',
    'living_area_sqft_parcel',
    'year_built_block_median_building_nsi',
    'n_stories_building_nsi',
    'n_dwellings_nsi',
    'n_buildings_nsi',
    'n_stories_parcel',
    'n_footprints_per_parcel',
    'n_parcels_per_footprint',
    'overlap_fraction_parcel',
    'n_manufactured_homes_per_parcel',
    'has_homestead_exemption_parcel',
    'n_stories_building_cheer',
    'n_stories_brails',
    'occupancy_type_building_nsi',
    'foundation_type_building_nsi',
    'construction_type_building_nsi',
    'foundation_type_building_cheer',
    'construction_type_building_cheer',
    'roof_shape_building_cheer',
    'land_use_class_parcel',
    'geometry_source',
]
ADJUSTED_COLUMNS = ['structure_value_adjusted', 'appraisal_year']
# Suffix of the imputation block's per-column flag: the cell was
# withheld in the bundle the block was fitted for.
WITHHELD_FLAG = '_withheld'
# Token before the imputed marker on every filled cell
# (`<version>+open+imputed`); io.delivery.redaction.OPEN reads it.
OPEN_MARKER = 'open'
UNUSED_LOCATION_INPUTS = (
    'distance_to_core_km',
    'region_type',
    'county_distance_to_core',
    'county_share_metro',
    'log_land_value_2010',
    'county_log_land_value_2010',
)


def delivery_config(spec: dict) -> DonorConfig:
    """Donor weighting from a recipe's `share: delivery: impute:` block."""
    donors = spec.get('donors', {})
    mode = donors.get('mode', 'km')
    if mode == 'region':
        # A delivery has no region graph (orders={}, region_id empty),
        # so every donor would weigh 0.
        raise ValueError("donors: mode 'region' needs a region graph; use 'km'.")
    if 'similarity_bandwidth' in donors:
        # Removed 2026-10-26 (design limit L30): a donor weighted by how
        # alike two counties' buildings are pairs areas by similarity.
        raise ValueError(
            'donors: similarity_bandwidth is no longer supported; donor '
            'weights depend on distance only.'
        )
    return DonorConfig(
        mode=mode,
        bandwidth_km=float(donors.get('bandwidth_km', 100.0)),
        radius_km=float(donors.get('radius_km', 250.0)),
    )


def acs_value_levels(
    acs_county: pd.Series,
    fips_by_county: pd.Series,
    hpi: pd.DataFrame,
    common_year: int,
) -> pd.Series:
    """ACS median home value by admin3_id, moved to *common_year*.

    The ACS 5-year estimate (ACS_PERIOD) pools five survey years, its
    dollars adjusted by the CPI to the period's last year. The anchor
    compares counties through ratios, so a factor common to all of them
    (the CPI step) cancels; each county's own price path does not, and
    the county index moves its value from the period's mean index to
    *common_year*, as appraisal_scales does for roll values.

    Parameters
    ----------
    acs_county : Series
        ACS median home value by 5-digit county FIPS.
    fips_by_county : Series
        County FIPS by admin3_id.
    hpi : DataFrame
        Columns 'county_fips', 'index_year', 'house_price_index'.
    common_year : int
        Year the levels are expressed in.
    """
    table = hpi.assign(fips=hpi['county_fips'].astype(str).str.zfill(5)).pivot_table(
        index='fips', columns='index_year', values='house_price_index'
    )
    years = [y for y in range(ACS_PERIOD[0], ACS_PERIOD[1] + 1) if y in table.columns]
    factor = pd.Series(np.nan, index=table.index)
    if years and common_year in table.columns:
        factor = table[common_year] / table[years].mean(axis=1)
    factor = factor.reindex(fips_by_county.astype(str).to_numpy())
    factor = factor.fillna(factor.median()).fillna(1.0)
    acs = acs_county.reindex(fips_by_county.astype(str).to_numpy())
    return pd.Series(acs.to_numpy() * factor.to_numpy(), index=fips_by_county.index)


def appraisal_scales(
    county_fips: pd.Series,
    appraisal_years: pd.DataFrame,
    hpi: pd.DataFrame,
    common_year: int,
) -> pd.DataFrame:
    """Appraisal year and the factor to *common_year*, per county FIPS.

    Parameters
    ----------
    county_fips : Series
        FIPS codes of the counties in the bundle.
    appraisal_years : DataFrame
        Columns 'county_fips', 'appraisal_year' (the recipe sidecar).
    hpi : DataFrame
        Columns 'county_fips', 'index_year', 'house_price_index'.
    common_year : int
        Year the adjusted values are expressed in.
    """
    fips = pd.Series(sorted(county_fips.dropna().astype(str).unique()))
    years = appraisal_years.assign(
        county_fips=appraisal_years['county_fips'].astype(str).str.zfill(5)
    ).set_index('county_fips')['appraisal_year']
    missing = sorted(set(fips) - set(years.index))
    if missing:
        raise ValueError(
            'No appraisal year recorded for counties '
            + ', '.join(missing)
            + '; add them to the recipe sidecar appraisal-years.csv.'
        )
    index = hpi.rename(
        columns={'index_year': 'year', 'house_price_index': 'hpi'}
    ).assign(fips=hpi['county_fips'].astype(str).str.zfill(5))
    scales = value_scales(index, years.reindex(fips).set_axis(fips), common_year)
    return scales.rename(columns={'year': 'appraisal_year'})


def delivery_frame(pooled: pd.DataFrame) -> pd.DataFrame:
    """Model frame from a delivery's pooled rows (canonical plus evidence)."""
    from pyproj import Transformer

    frame = pooled.copy()
    to_albers = Transformer.from_crs('EPSG:4326', 'EPSG:5070', always_xy=True)
    x, y = to_albers.transform(
        pd.to_numeric(frame['long'], errors='coerce').to_numpy(),
        pd.to_numeric(frame['lat'], errors='coerce').to_numpy(),
    )
    frame['x'], frame['y'] = x, y
    for column in ('region_id', 'region_type'):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype='string')
    for column in ('distance_to_core_km', 'land_value_2010'):
        frame[column] = np.nan
    for column in EVIDENCE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    for column in CENSUS_COLUMNS:
        frame[column] = np.nan
    return frame


def _acs_level(table: pd.DataFrame | None, names, summary_level: str) -> pd.Series:
    """One ACS estimate at one summary level, keyed by the bare GEOID."""
    if table is None:
        return pd.Series(dtype=float)
    name = next((n for n in names if n in table.columns), None)
    if name is None:
        return pd.Series(dtype=float)
    ids = table.index.astype(str)
    # GEO_ID: summary level, four zeros (no component), 'US', GEOID.
    prefix = f'{summary_level}0000US'
    keep = ids.str.startswith(prefix)
    values = pd.to_numeric(table.loc[keep, name], errors='coerce')
    # ACS jam values (suppressed or not computed) are negative.
    return values.where(values > 0).set_axis(ids[keep].str[len(prefix) :])


def census_levels(
    acs: dict[str, pd.DataFrame], gazetteer: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Tract, block-group and place levels from ACS and Gazetteer tables.

    Parameters
    ----------
    acs : dict
        ACS table name (as in ACS_TABLES) to its ingested table, indexed
        by census GEO_ID ('1400000US' plus the tract id, and so on).
    gazetteer : dict
        'tracts' and 'place' Gazetteer tables, indexed by GEOID, with
        'land_area_m2', 'lat' and 'long'.

    Returns
    -------
    dict
        'tract' (by 11-digit id: home value, year built, population
        density per km2, internal point), 'blockgroup' (by 12-digit id:
        home value, income), 'place' (by 7-digit id: population and
        internal point) and 'county' (by 5-digit FIPS: home value, for
        the price anchor).
    """

    def level(key, summary_level):
        table, names = ACS_TABLES[key]
        return _acs_level(acs.get(table), names, summary_level)

    tract = pd.DataFrame(
        {
            'acs_median_home_value_tract': level('home_value', '140'),
            'acs_median_year_built_tract': level('year_built', '140'),
        }
    )
    points = gazetteer.get('tracts')
    if points is not None:
        points = points.set_axis(points.index.astype(str))
        km2 = pd.to_numeric(points['land_area_m2'], errors='coerce') / 1e6
        population = level('population', '140').reindex(points.index)
        tract = tract.reindex(tract.index.union(points.index))
        tract['population_density_tract'] = population / km2.where(km2 > 0)
        tract['lat'] = points['lat']
        tract['long'] = points['long']
    blockgroup = pd.DataFrame(
        {
            'acs_median_home_value_blockgroup': level('home_value', '150'),
            'acs_median_income_blockgroup': level('income', '150'),
        }
    )
    county = pd.DataFrame({'acs_median_home_value_county': level('home_value', '050')})
    place = pd.DataFrame(columns=['population', 'lat', 'long'])
    if gazetteer.get('place') is not None:
        place = gazetteer['place'].set_axis(gazetteer['place'].index.astype(str))
        place = place[['lat', 'long']].assign(
            population=level('population', '160').reindex(place.index)
        )
        place = place.dropna()
    return {'tract': tract, 'blockgroup': blockgroup, 'place': place, 'county': county}


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def nearest_place_km(lat, long, places: pd.DataFrame) -> np.ndarray:
    """Great-circle distance to the nearest of *places*, with no radius."""
    from sklearn.neighbors import BallTree

    lat = np.asarray(lat, float)
    long = np.asarray(long, float)
    out = np.full(len(lat), np.nan)
    ok = np.isfinite(lat) & np.isfinite(long)
    if not len(places) or not ok.any():
        return out
    tree = BallTree(
        np.radians(places[['lat', 'long']].to_numpy(float)), metric='haversine'
    )
    query = np.radians(np.column_stack([lat[ok], long[ok]]))
    out[ok] = tree.query(query, k=1)[0][:, 0] * 6371.0
    return out


def place_gravity(lat, long, places: pd.DataFrame, chunk: int = 256) -> np.ndarray:
    """Sum over every place of population / distance^2, per point.

    Every place counts, however far (design limit L24: no radius), and
    a place closer than MIN_GRAVITY_KM counts at that distance.
    """
    lat = np.asarray(lat, float)
    long = np.asarray(long, float)
    plat = places['lat'].to_numpy(float)
    plong = places['long'].to_numpy(float)
    population = places['population'].to_numpy(float)
    out = np.full(len(lat), np.nan)
    for start in range(0, len(lat), chunk):
        rows = slice(start, start + chunk)
        d = _haversine_km(lat[rows, None], long[rows, None], plat, plong)
        out[rows] = (population / np.maximum(d, MIN_GRAVITY_KM) ** 2).sum(axis=1)
    return out


def attach_census_inputs(frame: pd.DataFrame, census: dict | None) -> None:
    """Write the CENSUS_COLUMNS onto the rows, in place.

    Parameters
    ----------
    frame : DataFrame
        From :func:`delivery_frame`, with 'census_tract_id' (11 digits),
        'census_blockgroup_id' (12), 'lat' and 'long'.
    census : dict or None
        From :func:`census_levels`; None leaves the columns empty, and
        an input empty everywhere enters no model.
    """
    if census is None:
        return
    tract_id = frame['census_tract_id'].astype('string')
    tract = census['tract']
    if len(tract):
        here = tract.reindex(tract_id.dropna().unique())
        for column in tract.columns.intersection(CENSUS_COLUMNS):
            frame[column] = pd.to_numeric(tract_id.map(tract[column]), errors='coerce')
        place = census['place']
        if {'lat', 'long'} <= set(here.columns) and 'population' in place.columns:
            # Gravity at the tract's internal point: cheap, and a tract
            # is the smallest unit its population is known for.
            here = here.dropna(subset=['lat', 'long'])
            gravity = pd.Series(
                place_gravity(here['lat'], here['long'], place), index=here.index
            )
            frame['place_gravity_tract'] = tract_id.map(gravity)
    blockgroup = census['blockgroup']
    if len(blockgroup.columns) and 'census_blockgroup_id' in frame.columns:
        bg_id = frame['census_blockgroup_id'].astype('string')
        for column in blockgroup.columns.intersection(CENSUS_COLUMNS):
            values = bg_id.map(blockgroup[column])
            frame[column] = pd.to_numeric(values, errors='coerce')
    place = census['place']
    if 'population' not in place.columns:
        return
    for label, size in PLACE_SIZES.items():
        towns = place[place['population'] >= size]
        frame[f'distance_to_place_{label}_km'] = nearest_place_km(
            pd.to_numeric(frame['lat'], errors='coerce'),
            pd.to_numeric(frame['long'], errors='coerce'),
            towns,
        )


# Each land-market input: the enrichment column it is read from, the
# frame column it is written to, and the county summary built from it.
LOCATION_INPUTS = {
    'region_type': ('land_value_region_type', 'region_type', 'county_share_metro'),
    'distance_to_core_km': (
        'distance_to_land_value_core_km',
        'distance_to_core_km',
        'county_distance_to_core',
    ),
    'log_land_value_2010': (
        'land_value_per_ha_log_2010',
        'land_value_2010',
        'county_log_land_value_2010',
    ),
}


def build_imputer(
    pooled: pd.DataFrame,
    spec: dict,
    appraisal_years: pd.DataFrame,
    hpi: pd.DataFrame,
    county_area_km2: pd.Series,
    location: pd.DataFrame | None = None,
    census: dict | None = None,
    without: tuple[str, ...] = (),
    extensions: str | None = None,
    log=print,
) -> tuple[Imputer, pd.DataFrame]:
    """The imputer a delivery runs, and its appraisal scales.

    One construction for :func:`impute_delivery` and the leave-county-out
    tool, so the scored model is the shipped one.

    Parameters
    ----------
    pooled, spec, appraisal_years, hpi, county_area_km2, location, census
        See :func:`impute_delivery`.
    without : tuple of str
        Further inputs to leave out of every model (an ablation).
    extensions : str, optional
        'module:attribute' overriding the recipe's `extensions`; '' for
        none.

    Returns
    -------
    imputer : Imputer
    scales : DataFrame
        From :func:`appraisal_scales`.
    """
    common_year = int(spec['common_year'])
    frame = delivery_frame(pooled)
    excluded = set(spec.get('exclude_inputs', ())) | set(without)
    if location is None:
        excluded |= set(UNUSED_LOCATION_INPUTS)
    else:
        location = location.reindex(frame.index)
        for feature, (source, column, county) in LOCATION_INPUTS.items():
            if feature in excluded or source not in location.columns:
                # Never written, so no county summary carries an input
                # left out of the models.
                excluded |= {feature, county}
                continue
            values = location[source]
            frame[column] = (
                values.astype('string')
                if column == 'region_type'
                else pd.to_numeric(values, errors='coerce')
            )
        used = sorted(set(LOCATION_INPUTS) - excluded)
        log(f'location inputs used: {", ".join(used) or "none"}')
    attach_census_inputs(frame, census)
    log(
        'census inputs on '
        f'{frame["acs_median_home_value_tract"].notna().mean():.1%} of rows'
    )
    scales = appraisal_scales(frame['county_fips'], appraisal_years, hpi, common_year)
    fips = frame.groupby('admin3_id')['county_fips'].first().astype(str)
    acs_value = None
    if census is not None and 'county' in census:
        acs_value = acs_value_levels(
            census['county']['acs_median_home_value_county'], fips, hpi, common_year
        )
    imputer = Imputer(
        frame,
        county_area_km2,
        orders={},
        config=delivery_config(spec),
        value_scale=fips.map(scales['scale']),
        # Every draw, sample and fit derives from this one seed, so the
        # recipe's value reproduces a delivery.
        seed=int(spec.get('seed', DEFAULT_SEED)),
        anchor=spec.get('price_anchor', 'acs'),
        exclude_inputs=tuple(sorted(excluded)),
        acs_value=acs_value,
        extensions=load_extensions(
            spec.get('extensions') if extensions is None else extensions
        ),
    )
    return imputer, scales


# A county whose single-family roll median, after the appraisal scaling,
# falls outside this multiple of its ACS median home value (which
# includes land) is flagged: the likeliest cause is a wrong appraisal
# year, which inflates or deflates every label in the county.
# Pender's roll carried its 2026 reappraisal while the sidecar gave
# 2019, a 71% inflation of every Pender label (parcel-sourced median
# 1.60 of ACS, against 0.44 to 1.05 elsewhere).
# Measured 2026-10-24 on cheer-eastern-nc: 0.23 to 0.86 in 36 counties.
ROLL_LEVEL_RANGE = (0.3, 1.3)


def roll_level_check(imputer: Imputer) -> dict:
    """Each county's scaled single-family roll median over its ACS level.

    Returns 'ratio' by admin3_id (counties with ANCHOR_MIN_HOMES labeled
    single-family homes and an ACS level) and 'flagged', those outside
    ROLL_LEVEL_RANGE. Empty without ACS levels.
    """
    if imputer.acs_value is None:
        return {'ratio': {}, 'flagged': {}}
    y = imputer._labels(TARGETS[-1])
    single = imputer.frame['occupancy_type'].astype('string').eq('Single-Family')
    positive = single & y.gt(0).fillna(False)
    grouped = y[positive].groupby(imputer.frame.loc[positive, 'admin3_id'])
    medians = grouped.median()[grouped.size() >= ANCHOR_MIN_HOMES]
    ratio = (medians / imputer.acs_value.reindex(medians.index)).dropna()
    low, high = ROLL_LEVEL_RANGE
    flagged = ratio[(ratio < low) | (ratio > high)]
    return {
        'ratio': {k: round(float(v), 3) for k, v in ratio.items()},
        'flagged': {k: round(float(v), 3) for k, v in flagged.items()},
    }


def impute_delivery(
    pooled: pd.DataFrame,
    spec: dict,
    appraisal_years: pd.DataFrame,
    hpi: pd.DataFrame,
    county_area_km2: pd.Series,
    log=print,
    location: pd.DataFrame | None = None,
    census: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fill a delivery's primary columns; return the filled block and counts.

    Parameters
    ----------
    pooled : DataFrame
        The delivery's rows after withholding: canonical columns, their
        `_source` sidecars, `lat`, `long`, `county_fips`, `admin3_id`
        and the EVIDENCE_COLUMNS.
    spec : dict
        The recipe's `share: delivery: impute:` block ('version',
        'common_year', optional 'donors', 'exclude_inputs',
        'extensions').
    appraisal_years, hpi : DataFrame
        See :func:`appraisal_scales`.
    county_area_km2 : Series
        Land area by admin3_id.
    log : callable
        Progress messages.
    location : DataFrame, optional
        Per-row land-market inputs from the region enrichment
        (LOCATION_INPUTS). An input the recipe excludes is not read at
        all, not even into the county summaries.
        Without it, the models run on the other inputs alone.
    census : dict, optional
        From :func:`census_levels`; the same holds without it.

    Returns
    -------
    block : DataFrame
        For every row: the filled canonical columns, their `_source`
        sidecars (filled cells read '<version>+open+imputed', a structure
        value held to its value per m2 bound
        '<version>+clipped+open+imputed'
        (:func:`value_per_area_caps`); NSI-sourced
        years and values on rows not imputed are emptied).
    counts : dict
        Cells filled per column, cells voided, model version, and the
        appraisal 'scales' by county FIPS that :func:`overlay_delivery`
        applies.
    """
    from openplaces.core.provenance import mark_imputed

    version = spec['version']
    common_year = int(spec['common_year'])
    imputer, scales = build_imputer(
        pooled,
        spec,
        appraisal_years,
        hpi,
        county_area_km2,
        location=location,
        census=census,
        log=log,
    )
    # The frame the models read: after the extensions' repairs, so a
    # value a repair voided ships without the source it no longer has.
    frame = imputer.frame
    level = roll_level_check(imputer)
    if level['flagged']:
        import warnings

        warnings.warn(
            'Roll level far from the ACS home value after the appraisal '
            f'scaling (check appraisal-years.csv): {level["flagged"]}',
            stacklevel=2,
        )
    filled, flags = imputer.run(log=log)
    # `+open`: these models were fit on the delivery's frame after every
    # restricted and share-alike value was withheld from it, so no such
    # value reached a filled cell, and redaction's rule 5 lets the cell
    # stay in a unit whose restricted source it would otherwise follow.
    token = mark_imputed(f'{version}+{OPEN_MARKER}')
    block = pd.DataFrame(index=frame.index)
    for target in TARGETS:
        column = target.column
        source = f'{column}_source'
        block[column] = filled[column]
        sidecar = (
            frame[source].astype('object')
            if source in frame.columns
            else pd.Series(np.nan, index=frame.index, dtype='object')
        )
        sidecar = sidecar.where(~flags[column], token)
        if column == 'structure_value':
            # A value held to its value per m2 bound says so.
            clipped = getattr(imputer, 'clipped', pd.Series(False, index=frame.index))
            sidecar = sidecar.where(
                ~clipped, mark_imputed(f'{version}+clipped+{OPEN_MARKER}')
            )
        block[source] = sidecar
        # Which cells this bundle had withheld before filling them: the
        # only cells a team twin may put its own value back into
        # (overlay_delivery).
        block[f'{column}{WITHHELD_FLAG}'] = (
            frame[source].astype('string').eq('withheld').fillna(False).to_numpy()
            if source in frame.columns
            else False
        )
    for column in ('year_built', 'structure_value'):
        # NSI never reaches a delivery: NSI-sourced cells outside the
        # imputed rows (secondary buildings) are emptied with their source.
        nsi = is_nsi_token(frame[f'{column}_source'].fillna(''))
        drop = nsi & ~flags[column]
        block.loc[drop, [column, f'{column}_source']] = np.nan
    counts = {
        'version': version,
        'common_year': common_year,
        'seed': imputer.seed,
        'price_anchor_mode': imputer.anchor,
        'roll_level_vs_acs': level,
        'filled': {t.column: int(flags[t.column].sum()) for t in TARGETS},
        'voided': imputer.voided,
        'scale_fallback': sorted(scales.index[scales['fallback']].tolist()),
        'park_factor': round(float(getattr(imputer, 'park_factor', 1.0)), 3),
        'value_clipped': int(getattr(imputer, 'clipped', pd.Series(dtype=bool)).sum()),
        'donors_only': imputer.donors_only,
        'price_anchor': {k: round(v, 3) for k, v in imputer.price_anchor.items()},
        # Applied in overlay_delivery to the values the bundle ships, so
        # a team bundle's own values are adjusted too.
        'scales': {
            str(fips): {
                'scale': float(row['scale']),
                'appraisal_year': float(row['appraisal_year']),
            }
            for fips, row in scales.iterrows()
        },
    }
    return block, counts


def overlay_delivery(
    pooled: pd.DataFrame, block: pd.DataFrame, keep_observed: bool, scales: dict
) -> pd.DataFrame:
    """Write an imputation block into a delivery's rows.

    Parameters
    ----------
    pooled : DataFrame
        The rows about to be written, with 'county_fips'.
    block : DataFrame
        From :func:`impute_delivery`, possibly computed for the public
        twin of this bundle.
    scales : dict
        The counts' 'scales' from :func:`impute_delivery`; the
        ADJUSTED_COLUMNS come from them and the merged structure value.
    keep_observed : bool
        True for a team bundle: its own observed (non-NSI) values win
        where the public twin withheld the cell or lacks the row, so the
        team twin keeps values cleared for team sharing that the public
        model never saw. Everywhere else the block wins, including
        observed values the public imputation chose to refill (a
        manufactured home's site-improvement value).

        Measured 2026-10-22 on cheer-eastern-nc.team: letting every own
        value win put 171,165 observed structure values back.
    """
    out = pooled.copy()
    in_block = out.index.isin(block.index)
    block = block.reindex(out.index)
    for column in [t.column for t in TARGETS]:
        source = f'{column}_source'
        if column not in out.columns:
            out[column] = np.nan
        if keep_observed:
            own_source = (
                out[source].astype('object')
                if source in out.columns
                else pd.Series(np.nan, index=out.index, dtype='object')
            )
            own = out[column].notna() & ~is_nsi_token(own_source.fillna(''))
            own &= own_source.astype('string').ne('withheld').fillna(True)
            flag = f'{column}{WITHHELD_FLAG}'
            withheld = (
                block[flag].astype('boolean').fillna(False).astype(bool)
                if flag in block.columns
                else pd.Series(True, index=out.index)
            )
            own &= withheld | ~in_block
        else:
            own = pd.Series(False, index=out.index)
        merged = out[column].astype('object').where(own, block[column])
        # Keep the registry types: numbers stay numeric, classes categorical.
        if pd.api.types.is_numeric_dtype(block[column]):
            out[column] = pd.to_numeric(merged, errors='coerce')
        else:
            out[column] = merged.astype('string').astype('category')
        out[source] = out[source].astype('object') if source in out.columns else np.nan
        out[source] = out[source].where(own, block[source])
    # From the values this bundle ships, a team bundle's own included.
    fips = out['county_fips'].astype(str)
    scale = fips.map({k: v['scale'] for k, v in scales.items()})
    out['structure_value_adjusted'] = (
        pd.to_numeric(out['structure_value'], errors='coerce') * scale
    )
    out['appraisal_year'] = fips.map(
        {k: v['appraisal_year'] for k, v in scales.items()}
    ).astype(float)
    return out


def imputed_cells(frame: pd.DataFrame, version: str) -> dict[str, int]:
    """Cells per target column this imputation *version* filled."""
    return {
        t.column: int(
            frame[f'{t.column}_source']
            .astype('string')
            .str.startswith(f'{version}+')
            .fillna(False)
            .sum()
        )
        for t in TARGETS
        if f'{t.column}_source' in frame.columns
    }
