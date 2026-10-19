"""Static web page of the 3D land and building value scene.

Writes the scene of `show_value_terrain_layer` (parcels extruded by land
value per area, buildings stacked on them by structure value per area)
as static files any plain web host can serve: a page, its script, a
manifest, and gzip-compressed binary files per admin unit. The host runs
no code and holds no database.

Three properties of the page are maintainer's decisions, pinned by
tests:

- It draws only the scene's own layers and the unit boundaries, on a
  plain background: no basemap or tile layer, like every map in
  `openplaces.viz`.
- It has no search, no filter and no control choosing which units to
  show. Every unit of the scene loads at start; a hovered feature shows
  the fields shipped with its own geometry, and nothing is looked up.
- It reads delivery bundles (`export_delivery`), never curated per-unit
  files, so whatever a bundle withholds never reaches the page, and the
  tooltip fields are an explicit allow-list that refuses personal
  columns.

The binary layout, one file per (layer, unit), little-endian:

- `OPVS`, then a uint32 header length and a UTF-8 JSON header naming
  each array (`name`, `dtype`, `offset`, `count`), padded to 8 bytes;
- `xy` (float32): vertex longitude and latitude as offsets from the
  header's `origin`. Offsets keep float32 precise to millimeters within
  a town, and deck.gl draws them directly (`LNGLAT_OFFSETS`), which
  halves the size of float64 coordinates;
- `z` (float32, only in a layer draped on terrain): each vertex's
  height in meters, from which the page extrudes instead of `base`;
- `ring_offsets`, `part_ring_offsets` (uint32): vertex index of each
  ring's start, ring index of each polygon part's start (both with a
  closing entry), and `part_feature` (uint32), the feature of each part,
  so a multipolygon's parts share one feature's values;
- `triangles` (uint32): the parts' top faces as vertex index triples
  into `xy`, computed at build time so the page does no triangulation
  (rings are written outer counterclockwise, holes clockwise);
- per feature: `base` and `height` (float32, meters) and `value`
  (float32), the value per area unit the colors encode, NaN where
  there is none. The page colors from it with the palette the visitor
  picks (`PALETTES`), so switching palettes needs no new files.

The tooltip values go into a second gzip file per (layer, unit), as
columnar JSON. A self-describing JSON-plus-arrays file was chosen over
GeoParquet or GeoArrow because the page then needs one library
(deck.gl's standalone bundle) instead of four that must agree on
versions (parquet-wasm, Apache Arrow, the GeoArrow layers, deck.gl).
"""

import gzip
import json
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import numpy as np
import pandas as pd
import shapely

from openplaces.core.attribute_registry import is_personal_attribute
from openplaces.core.provenance import is_imputed
from openplaces.io.delivery import TEAM_SUFFIX, PersonalColumnError, delivery_paths
from openplaces.io.readers import get_admin
from openplaces.viz.colors import get_diverging_colormap

MAGIC = b'OPVS'
# 2: adds `triangles`, which the page now requires.
FORMAT_VERSION = 2
SOURCE_SUFFIX = '_source'
STATIC_DIR = files('openplaces.viz.web_scene') / 'static'
KNOWN_ISSUES = files('openplaces.viz.web_scene') / 'known_issues.yaml'


def known_issues(admin_ids, path=KNOWN_ISSUES) -> list[dict]:
    """Known issues for a scene's units, for its Sources panel.

    Every page lists the problems we know of in what it shows, so a
    visitor sees that we look for our own mistakes (maintainer,
    2026-10-17). The list is kept by hand in `known_issues.yaml`.

    Parameters
    ----------
    admin_ids : sequence of str
        The scene's level-3 units.
    path : path-like
        The issues file.

    Returns
    -------
    list of dict
        `found` (ISO date) and `text`, newest first; entries for `all`
        units and those naming any of `admin_ids`.
    """
    import yaml

    wanted = set(admin_ids)
    entries = yaml.safe_load(Path(path).read_text(encoding='utf-8')).get('issues') or []
    picked = [
        {'found': str(entry['found']), 'text': ' '.join(str(entry['text']).split())}
        for entry in entries
        if entry['admin_ids'] == 'all' or wanted & set(entry['admin_ids'])
    ]
    return sorted(picked, key=lambda issue: issue['found'], reverse=True)


# deck.gl's standalone bundle, pinned; the page loads nothing else.
DECK_GL_URL = 'https://cdn.jsdelivr.net/npm/deck.gl@9.1.14/dist.min.js'
TOOLTIP_KINDS = ('text', 'money', 'money_per_area', 'year', 'count')
# The page's heading and the smaller line under it (maintainer,
# 2026-10-16).
DEFAULT_TITLE = 'Land & structure value'
# The palette a first visit shows (maintainer, 2026-10-16: Pastel).
DEFAULT_PALETTE = 'pastel'
DEFAULT_SUBTITLE = 'as seen by the tax assessor'


@dataclass(frozen=True)
class TooltipField:
    """One field shown when a feature is hovered.

    Parameters
    ----------
    column : str
        Column of the rendered table. A personal column is refused.
    label : str
        Label shown on the page.
    kind : str
        How the page formats the value: one of `TOOLTIP_KINDS`.
    imputed_from : str, optional
        Provenance column whose `imputed` marker flags the value as
        filled by openplaces; the page then says so beside it.
    """

    column: str
    label: str
    kind: str = 'text'
    imputed_from: str | None = None

    def __post_init__(self):
        if self.kind not in TOOLTIP_KINDS:
            raise ValueError(
                f'TooltipField kind must be one of {TOOLTIP_KINDS}, got {self.kind!r}.'
            )


@dataclass(frozen=True)
class SceneLayer:
    """One extruded layer of the scene.

    Parameters
    ----------
    name : str
        Short id, used in file names (`parcels`, `buildings`).
    label : str
        Name shown in the legend.
    recipe : str
        Curation recipe whose delivery bundle supplies the entities.
    value_column : str
        Value divided by area for height and color.
    area_unit : str
        Display unit of the color scale (`ac`, `sqft`, `ha`, `m2`).
    vmin, vmax : float
        Color-scale bounds, in dollars per `area_unit`.
    cmap : str
        Colormap name (`viz.colors.get_diverging_colormap`; cmasher's
        `cmr.*` names included).
    cmap_range : tuple of float
        The part of the colormap used, as fractions (0-1). Cuts an end
        that would vanish into the page's dark background or wash out
        to white.
    tooltip : tuple of TooltipField
        Fields shown on hover; nothing else of the bundle is written.
    stack_on : str, optional
        Name of an earlier layer this one stands on.
    render : dict
        Further keyword arguments for `show_value_terrain_layer`.
    simplify_m : float, optional
        Simplify each polygon by this many meters before writing it
        (topology-preserving, polygon by polygon). For layers that are
        not a coverage, such as building outlines; a coverage (parcels)
        is simplified inside `show_value_terrain_layer` instead, so
        neighbors keep their shared edges.
    per_dwelling_label : str, optional
        Offer a second view coloring this layer by `value_column` per
        dwelling (`n_dwellings`), under this legend label. In that view
        every feature with a value that is not a home (`occupancy_type`
        outside `RESIDENTIAL_OCCUPANCY`) keeps its value per area on a
        ramp of its own (the palette's `other_cmap`), since it has one
        dwelling or none; homes without a dwelling count and features
        without a value have no value.
    recorded_column : str, optional
        The value as the source recorded it, before openplaces' own
        fills (read from the bundle, or its evidence supplement). The
        assessment view shows it where `value_column` was estimated,
        so a feature with a recorded value never disappears there.
    model_views : tuple of ModelView
        Model land values offered as further colorings of this layer,
        on team pages only (`build_web_scene`); a view whose column is
        missing or empty is left out.
    """

    name: str
    label: str
    recipe: str
    value_column: str
    area_unit: str
    vmin: float
    vmax: float
    cmap: str = 'turbo'
    cmap_range: tuple[float, float] = (0.0, 1.0)
    tooltip: tuple[TooltipField, ...] = ()
    stack_on: str | None = None
    render: dict = field(default_factory=dict)
    simplify_m: float | None = None
    per_dwelling_label: str | None = None
    recorded_column: str | None = None
    model_views: tuple = ()


@dataclass(frozen=True)
class ModelView:
    """One model's land value, a coloring of a team page's lots.

    The page shows the model's prediction per area, scaled by one
    constant per scene and model: the median, over lots with both, of
    the recorded land value per hectare over the predicted one. The
    prediction is a 2010 value; the constant brings it to the scene's
    current assessment level without sale prices or periods (design
    limit L29; `value-viewer-assessment-estimate-toggle.md`, section 5).
    The area of applicability travels with each lot for its tooltip
    only; nothing is hidden or dimmed by it (L4, until the maintainer
    settles it).

    Parameters
    ----------
    key : str
        Short id, the page's view name (`?view=<key>`).
    label : str
        Name on the page's button.
    log_column : str
        Natural log of the predicted land value per hectare.
    aoa_column : str, optional
        Area of applicability index; inside where at most 1.
    scale_column : str
        The recorded value the scale is taken from (`land_value`, as
        the assessor recorded it).
    """

    key: str
    label: str
    log_column: str
    aoa_column: str | None = None
    scale_column: str = 'land_value'


# The PLACES-FMV region-with-neighbors variants cleared for the
# research team (maintainer, 2026-10-18; source
# `US_parcel-placeslab-fmv2026open`), in the order of their buttons.
TEAM_MODEL_VIEWS = tuple(
    ModelView(
        f'model_{variant}',
        label,
        f'land_value_per_ha_log_region_nb_{variant}',
        f'area_of_applicability_region_nb_{variant}',
    )
    for variant, label in (
        ('bare', 'Base'),
        ('people', '+ people'),
        ('buildings', '+ buildings'),
        ('wealth', '+ wealth'),
        ('peoplewealth', '+ people, wealth'),
        ('peoplebuildings', '+ people, buildings'),
        ('popbldwealth', '+ all three'),
    )
)
# Hectares in one display area unit.
HECTARES_PER_UNIT = {'ha': 1.0, 'ac': 0.40468564224, 'm2': 1e-4, 'sqft': 9.290304e-6}

# The models in today's dollars (maintainer, 2026-10-18): a prediction
# (2010 land value in real January 2022 dollars) times CPI-U(October
# 2010) / CPI-U(January 2022), back to 2010 dollars, times the lot's
# county house price index, latest year over 2010. Where a county's
# series ends early, the state's index carries it on to the latest year
# and the page marks the value as an estimate. Both indexes are
# published (FHFA, BLS); nothing here builds one (design limits L7,
# L29).
HOUSE_PRICE_RECIPE = 'US_economy-price-house-fhfa-2026'
# The state index that extends a county series ending early. Without
# it, a short county series stops at its last year.
STATE_HOUSE_PRICE_RECIPE = 'US_economy-price-house-fhfa-2026_state'
CONSUMER_PRICE_RECIPE = 'US_economy-price-consumer-bls-2026'
CONSUMER_PRICE_SERIES = 'CUUR0000SA0'
# The predictions' reference month (2010-10-31; the maintainer's
# choice of base) and the month of their dollars.
MODEL_REFERENCE = (2010, 'M10')
MODEL_DOLLARS = (2022, 'M01')
HOUSE_PRICE_BASE_YEAR = 2010
# Credits the team page's Sources panel shows for its models. FHFA
# requires its notice wherever its data are used; BLS asks to be
# cited; Zillow's standard acknowledgment covers the ZTRAX training
# data.
MODEL_SOURCE_NOTES = (
    {
        'name': 'PLACES-FMV land value models, 2010 (region with neighbors)',
        'credit': 'Data provided by Zillow through the Zillow Transaction and '
        'Assessment Dataset (ZTRAX). More information on accessing the data can '
        'be found at http://www.zillow.com/ztrax. The results and opinions are '
        'those of the author(s) and do not reflect the position of Zillow Group.',
    },
    {
        'name': 'Covariates of the models',
        'credit': 'IPUMS NHGIS, University of Minnesota (www.nhgis.org); '
        'Microsoft Building Footprints (ODbL).',
    },
    {
        'name': 'FHFA House Price Index, counties (all-transactions, annual)',
        'credit': 'This product uses FHFA data but is neither endorsed nor '
        'certified by FHFA.',
    },
    {
        'name': 'Consumer Price Index',
        'credit': 'U.S. Bureau of Labor Statistics, Consumer Price Index for All '
        'Urban Consumers (CPI-U), U.S. city average, all items (CUUR0000SA0).',
    },
)


def dollars_factors(counties, house, consumer, state_house=None) -> dict:
    """Factors that bring a 2010 model to today's dollars, per county.

    Parameters
    ----------
    counties : dict of str to str
        Five-digit county FIPS code to the admin id of its state
        (`US-MA`).
    house : pandas.DataFrame
        The county house price index (`county_fips`, `year`,
        `house_price_index`).
    consumer : pandas.DataFrame
        The consumer price index (`series_id_bls`, `year`,
        `period_bls`, `consumer_price_index`).
    state_house : pandas.DataFrame, optional
        A state house price index (`admin2_id`, `year`,
        `house_price_index`) that extends a county series ending
        before the latest year any county reaches.

    Returns
    -------
    dict
        Per county: `factor`, `to_year` (the index year the factor
        reaches) and `estimated` (True where the state index carried
        the series on). A county without a 2010 value is left out.
    """
    cpi = consumer[consumer['series_id_bls'] == CONSUMER_PRICE_SERIES]

    def month(year, period):
        hit = cpi[(cpi['year'] == year) & (cpi['period_bls'] == period)]
        return float(hit['consumer_price_index'].iloc[0])

    to_2010 = month(*MODEL_REFERENCE) / month(*MODEL_DOLLARS)
    house = house.dropna(subset=['house_price_index'])
    latest = int(house['year'].max())
    out = {}
    for county, state_id in sorted(counties.items()):
        series = house[house['county_fips'] == county].set_index('year')[
            'house_price_index'
        ]
        if HOUSE_PRICE_BASE_YEAR not in series.index:
            continue
        last = int(series.index.max())
        growth = series[last] / series[HOUSE_PRICE_BASE_YEAR]
        estimated = False
        if last < latest and state_house is not None:
            state = (
                state_house[state_house['admin2_id'] == state_id]
                .dropna(subset=['house_price_index'])
                .set_index('year')['house_price_index']
            )
            if {last, latest} <= set(state.index):
                growth *= state[latest] / state[last]
                last, estimated = latest, True
        out[county] = {
            'factor': round(float(to_2010 * growth), 4),
            'to_year': last,
            'estimated': estimated,
        }
    return out


def lot_dollars_factors(gdf) -> tuple[np.ndarray, dict]:
    """Each lot's factor to today's dollars, from its unit's county, and
    the factors per county; NaN and an empty dict where the indexes are
    missing.

    The county is the first five digits of the unit's Census GEOID
    (`admin3_id_admin1`): a county's own, or the county prefix of a New
    England town's. Lots' `census_tract_id` was not used: it is empty
    for whole towns (measured 2026-10-18: all 13,803 Nantucket lots).
    """
    from openplaces.io.readers import get_dataset

    nothing = np.full(len(gdf), np.nan), {}
    if 'admin3_id' not in gdf.columns:
        return nothing
    units = gdf['admin3_id'].astype('string')
    admin = get_admin(sorted(units.dropna().unique()), level=3, geom=False)
    if 'admin3_id' not in admin.columns:
        admin = admin.reset_index()
    county_of = dict(
        zip(
            admin['admin3_id'],
            admin['admin3_id_admin1'].astype(str).str[:5],
            strict=True,
        )
    )
    counties = units.map(county_of)
    states = units.str.split('-').str[:2].str.join('-')
    try:
        house = get_dataset(HOUSE_PRICE_RECIPE)
        consumer = get_dataset(CONSUMER_PRICE_RECIPE)
    except (FileNotFoundError, KeyError, ValueError):
        return nothing
    try:
        state = get_dataset(STATE_HOUSE_PRICE_RECIPE)
    except (FileNotFoundError, KeyError, ValueError):
        state = None
    pairs = pd.DataFrame({'county': counties, 'state': states}).dropna()
    factors = dollars_factors(
        dict(zip(pairs['county'], pairs['state'], strict=True)), house, consumer, state
    )
    lot = counties.map(lambda c: factors.get(c, {}).get('factor', np.nan))
    return lot.to_numpy(dtype=float), factors


# The notebook's scene (notebooks/09_show/03_show_land_and_building_
# values_3d), with buildings without a value drawn flat at their
# parcel's top rather than as floating outlines.
#
# Colors (maintainer's review, 2026-10-15): the page's job is to show
# where the money is, on a dark background. Land is a cool ground that
# recedes (indigo to teal to seafoam) and buildings are heat that ends
# in gold, so the two never share a hue and the costliest buildings
# are the brightest things on the page. Both ramps rise steadily in
# lightness, so their order also reads without color vision. The
# notebook's rainbows (turbo, nipy_spectral) shared every hue between
# land and buildings and ended in white. Land bounds: measured
# 2026-10-15 on boston-inner-core, land value per acre had median
# $3.0M and p99 $51M, so $50K-$50M puts the median mid-ramp; the
# notebook's $250-$10M left nine parcels in ten in its top fifth.
ZERO_VALUE_RGBA = (34, 40, 54, 255)
# Occupancy types (curated footprints' `occupancy_type`) that hold
# homes. The per-dwelling view colors these per dwelling and every
# other type with a value per area on a ramp of its own.
RESIDENTIAL_OCCUPANCY = frozenset(
    {
        'Single-Family',
        'Townhome',
        'Multi-Family',
        'Low-Rise Multi-Family',
        'Mid-Rise Multi-Family',
        'High-Rise Multi-Family',
        'Manufactured Home',
    }
)
# Curated parcels' `land_use_class` values that hold homes; a lot
# counts as residential in the per-dwelling view only with one of these
# and dwellings on it.
RESIDENTIAL_LAND_USE = frozenset(
    {
        'Single-Family',
        'Multi-Family',
        'Condominium',
        'Townhome',
        'Multiple Single-Family',
        'Manufactured Home',
    }
)
# The residential classes whose name says what each unit is, one home:
# a lot of one of these needs no further evidence that it holds homes.
SINGLE_HOME_LAND_USE = RESIDENTIAL_LAND_USE - {'Multi-Family'}
# The page's own class of a building (maintainer, 2026-10-16/17, handoff
# item 1f): the roll's land use class of its lot (`land_use_class_parcel`:
# Condominium, Townhome, Multiple Single-Family, Office, ...), which
# keeps what the CHEER `occupancy_type` folds into "Low-Rise
# Multi-Family"; `occupancy_type` where the lot has no class, and always
# for outbuildings ("Secondary"), since the lot's class describes its
# main building, not a shed.
BUILDING_CLASS = 'building_class'


def building_class(gdf) -> pd.Series:
    """The page's class per building (`BUILDING_CLASS`)."""
    occupancy = gdf['occupancy_type'].astype('string')
    if 'land_use_class_parcel' not in gdf.columns:
        return occupancy
    lot = gdf['land_use_class_parcel'].astype('string')
    use_lot = lot.notna() & ~occupancy.isin(['Secondary']).fillna(False)
    return lot.where(use_lot, occupancy)


# Lots' dwelling count as the roll states it, beside the count taken
# from the homes standing on the lot (`_count_dwellings_from_above`).
ROLL_DWELLINGS = 'n_dwellings_roll'
# A curated column flagging outlines drawn abnormally tall for what is
# mapped there (the sliver-value-allocation work decides which; the
# maintainer's rule of 2026-10-16). The page keeps their height, so
# they are not overlooked, and draws them see-through.
GHOST_COLUMN = 'height_unconfirmed'
VALUE_SCENE_LAYERS = (
    SceneLayer(
        name='parcels',
        label='Land value',
        recipe='US_parcel-openplaces-2026',
        value_column='land_value_imputed',
        area_unit='ac',
        vmin=50_000,
        vmax=50_000_000,
        cmap='cmr.ocean',
        cmap_range=(0.05, 0.85),
        tooltip=(
            TooltipField('address', 'Address'),
            TooltipField('use_group', 'Land use'),
            TooltipField(
                'land_value_imputed',
                'Land value',
                'money',
                imputed_from='land_value_imputed_source',
            ),
            TooltipField(
                'land_value_imputed_per_ac',
                'Land value per acre',
                'money_per_area',
                imputed_from='land_value_imputed_source',
            ),
            # Visitors read "home" (one house, apartment or condo); the
            # data's term stays "dwelling" (maintainer, 2026-10-17).
            TooltipField('n_dwellings', 'Homes', 'count'),
            TooltipField(
                'land_value_imputed_per_dwelling',
                'Land value per home',
                'money',
                imputed_from='land_value_imputed_source',
            ),
            # Shown only when several records share the outline
            # (`merge_identical_outlines`).
            TooltipField('n_records_on_outline', 'Records on this lot', 'count'),
        ),
        # Lots count the dwellings of the homes on them where the roll
        # states none (`_count_dwellings_from_above`).
        per_dwelling_label='Land value per home',
        model_views=TEAM_MODEL_VIEWS,
        render={
            'drop_corridors': True,
            'height_clip_percentile': 99.9999,
            'simplify_tolerance': 0.5,
        },
    ),
    SceneLayer(
        name='buildings',
        label='Building value',
        recipe='US_footprint-openplaces-2026',
        value_column='structure_value',
        area_unit='sqft',
        # Measured 2026-10-15 on boston-inner-core: structure value per
        # sq ft had median $453, p90 $1,251, p99 $3,407; $50-$5,000
        # puts the median near mid-ramp and leaves the brightest gold
        # to the costliest buildings.
        vmin=50,
        vmax=5_000,
        # Sunset's buildings (the page colors from `default_palettes`).
        cmap='plasma',
        cmap_range=(0.3, 1.0),
        stack_on='parcels',
        tooltip=(
            TooltipField('address', 'Address'),
            TooltipField(BUILDING_CLASS, 'Building'),
            TooltipField('year_built', 'Year built', 'year'),
            TooltipField(
                'structure_value',
                'Building value',
                'money',
                imputed_from='structure_value_source',
            ),
            TooltipField(
                'structure_value_per_sqft',
                'Building value per sq ft',
                'money_per_area',
                imputed_from='structure_value_source',
            ),
            TooltipField('n_dwellings', 'Homes', 'count'),
            TooltipField(
                'structure_value_per_dwelling',
                'Building value per home',
                'money',
                imputed_from='structure_value_source',
            ),
            TooltipField('n_records_on_outline', 'Records on this outline', 'count'),
        ),
        per_dwelling_label='Building value per home',
        # The roll's improvement value, apportioned to the outline but
        # without the land estimate subtracted: a condo's whole value
        # (maintainer, 2026-10-16: 131-135 Willow Ave vanished in the
        # assessment view though the roll values it).
        recorded_column='improvement_value_parcel',
        # A building whose height was capped keeps its color: the
        # notebook's translucent magenta marker read as a pink wash
        # over exactly the tallest towers.
        render={
            'height_clip_percentile': 99.999,
            # Opaque: translucent walls blend with whatever was drawn
            # before them, so a wall's shade depended on draw order,
            # i.e. on the row order of the data, and whole blocks
            # showed dark back walls through their fronts (seen
            # 2026-10-16 on Soden St, Cambridge).
            'alpha': 1.0,
            'clipped_fill_rgba': None,
        },
        # Outlines from building structure layers trace every jog of a
        # wall. Measured 2026-10-15 on boston-inner-core: 2.45M vertices
        # for 151k building parts (16.2 per part) against 7.4 per
        # parcel, which a phone could not hold.
        simplify_m=0.5,
    ),
)


def refuse_personal_fields(layers) -> None:
    """Raise if any layer's tooltip names a personal column.

    The same rule as a delivery's `share` block: a personal column
    never leaves the installation, and a scene that asks for one is
    refused rather than written without it.

    Parameters
    ----------
    layers : sequence of SceneLayer

    Raises
    ------
    PersonalColumnError
        If a tooltip field, or the provenance column it reads, is
        personal.
    """
    for layer in layers:
        for tip in layer.tooltip:
            for column in (tip.column, tip.imputed_from):
                if column is None:
                    continue
                attribute = column.removesuffix(SOURCE_SUFFIX)
                if is_personal_attribute(attribute):
                    raise PersonalColumnError(
                        f'Scene layer {layer.name!r} would show the personal '
                        f'column {column!r}. A web scene never carries one.'
                    )


# Names and credit lines the Sources document shows for source ids the
# notices name. The credits are the sources' own requests (MassGIS) or
# what their licence asks (OBM: OpenStreetMap contributors and the OBM
# authors, recorded in footprint-obm-2025.yaml). Other ids show as named.
SOURCE_DISPLAY = {
    'massgis': {
        'name': 'MassGIS',
        'credit': 'MassGIS (Bureau of Geographic Information), Commonwealth of '
        'Massachusetts EOTSS',
    },
    'obm': {
        'name': 'OpenBuildingMap',
        'credit': '© OpenStreetMap contributors; Oostwegel, Schorlemmer, Lingner & '
        'Evaz Zadeh (2025), OpenBuildingMap, doi:10.5880/GFZ.LKUT.2025.002',
    },
    'microsoft': {
        'name': 'Microsoft Building Footprints',
        'credit': 'Microsoft Building Footprints (ODbL 1.0)',
    },
    'fema': {'name': 'FEMA USA Structures'},
    'nconemap': {'name': 'NC OneMap', 'credit': 'NC OneMap (nconemap.gov)'},
}

_SOURCE_LINE = re.compile(r'^\s*(?P<share>\d+(?:\.\d+)?)%\s+(?P<rest>\S.*)$')
_URL_LINE = re.compile(r'^\s+(?P<url>https?://\S+)\s*$')


def notice_sources(text: str) -> dict:
    """The page's Sources document from a bundle's licence notice
    (`io.delivery.terms.format_notice`).

    Reads only the parts a visitor needs: each source with its share,
    licence or credit text and link, and the share-alike obligation.
    The notice's "Recipes" section is never carried over: it names the
    folders the build read recipes from, which are paths on the
    builder's machine (seen 2026-10-17: a user name on the public page).

    Parameters
    ----------
    text : str
        One notice, or several joined (one per delivery region).

    Returns
    -------
    dict
        `sources`: list of {`name`, `share`, `terms`, `url`}, one per
        source and terms, the largest share kept; `share_alike`: list
        of sentences.
    """
    sources: dict[tuple, dict] = {}
    section = None
    last = None
    for line in text.splitlines():
        stripped = line.strip()
        if set(stripped) <= {'-', '='} and stripped:
            continue
        if stripped in {
            'Sources',
            'Recipes',
            'Share-alike',
            'Attribution required',
            'Withheld',
            'Redistribution restricted',
            'Resale restricted',
        }:
            section = stripped
            last = None
            continue
        if stripped.startswith('Data licence and attribution for'):
            section = None
            continue
        if section == 'Sources':
            url = _URL_LINE.match(line)
            if url and last is not None:
                last['url'] = url['url']
                continue
            match = _SOURCE_LINE.match(line)
            if not match:
                continue
            rest = match['rest']
            # "<id> (<recipe>) <terms>" or "<id>   <terms>".
            name, _, terms = rest.partition(' ')
            terms = terms.strip()
            if terms.startswith('('):
                recipe_id, _, terms = terms[1:].partition(')')
                terms = terms.strip()
            key = (name, terms)
            entry = sources.setdefault(
                key, {'name': name, 'share': 0.0, 'terms': terms, 'url': None}
            )
            entry['share'] = max(entry['share'], float(match['share']) / 100)
            last = entry
    return _sources_document(list(sources.values()))


def scene_sources(recipe, geometry_source) -> dict:
    """The Sources document for the features a scene actually draws:
    `io.delivery.terms.bundle_terms` over the scene's own
    `geometry_source`, so each share is that of the scene's outlines
    (the region bundles' notices each state their own region's)."""
    from openplaces.io.delivery.terms import bundle_terms

    terms = bundle_terms(recipe, geometry_source=geometry_source)
    entries = [
        {
            'name': entry['source_id'],
            'share': entry['share'] or 0.0,
            'terms': entry['license'] or 'Terms not recorded',
            'url': entry['terms_url'] or entry['portal_url'],
        }
        for entry in terms['sources']
        if entry['share']
    ]
    return _sources_document(entries)


FILLED_BY_OPENPLACES = 'Filled in by openplaces'


def value_sources(provenance) -> list[dict]:
    """Where a layer's shown values come from, by share of features.

    The outline sources (`scene_sources`) say who drew the shapes; a
    value has its own source, recorded in the value's `_source` column
    (`parcel.massgis`, or `...+imputed` where openplaces filled it in).
    Shown apart, because the outline shares read as the values' own:
    97% of Boston's building outlines are OpenBuildingMap's, while 86%
    of their values are the assessors' via MassGIS (maintainer,
    2026-10-17: "that must be wrong").

    Parameters
    ----------
    provenance : pandas.Series
        The shown value's `_source` column, one entry per feature.

    Returns
    -------
    list of dict
        `name`, `share` and `credit`, largest share first; features
        without a value are not counted.
    """
    tokens = provenance.dropna().astype(str)
    if tokens.empty:
        return []

    def source(token):
        # `parcel.massgis`: a lot's value, from that lot's source.
        source_id = token.split('+')[0].split('.')[-1]
        if source_id in ('', 'parcel'):
            return 'Assessor roll, via the lot (source not recorded)'
        return SOURCE_DISPLAY.get(source_id, {}).get('name', source_id)

    names = tokens.map(source).where(
        ~np.asarray(is_imputed(tokens), dtype=bool), FILLED_BY_OPENPLACES
    )
    shares = names.value_counts(normalize=True)
    credits = {shown['name']: shown.get('credit') for shown in SOURCE_DISPLAY.values()}
    return [
        {'name': name, 'share': float(share), 'credit': credits.get(name)}
        for name, share in shares.items()
    ]


def _sources_document(entries: list[dict]) -> dict:
    """Name, merge (one entry per source shown) and order the sources;
    state share-alike once."""
    merged: dict[str, dict] = {}
    for entry in entries:
        shown = SOURCE_DISPLAY.get(entry['name'], {})
        name = shown.get('name', entry['name'])
        terms = entry['terms']
        # The notice's "Credit requested: ..." is shown as the credit.
        if 'Credit requested:' in terms:
            terms = terms.split('Credit requested:')[0].strip()
        if name in merged:
            # One source behind several recipes (MassGIS outlines and
            # MassGIS lots): one entry, the larger share, the shorter
            # terms text.
            kept = merged[name]
            kept['share'] = max(kept['share'], entry['share'])
            if len(terms) < len(kept['terms']):
                kept['terms'], kept['url'] = terms, entry['url'] or kept['url']
            continue
        merged[name] = {
            'name': name,
            'share': entry['share'],
            'terms': terms,
            'url': entry['url'],
            'credit': shown.get('credit'),
        }
    ordered = sorted(merged.values(), key=lambda e: -e['share'])
    share_alike: list[str] = []
    # Share-alike, stated once from the sources themselves (the notice
    # wraps its sentence over several lines).
    odbl = sum(e['share'] for e in ordered if 'ODbL' in e['terms'])
    if odbl:
        share_alike.append(
            f'{odbl:.0%} of these outlines come from sources under the Open '
            'Database License (ODbL 1.0), a share-alike licence: the data '
            'files of this page are offered under ODbL 1.0 as well.'
        )
    return {'sources': ordered, 'share_alike': share_alike}


def read_evidence_columns(recipe, region, columns) -> pd.DataFrame:
    """Columns of a bundle's evidence supplement, indexed like the bundle
    (empty where the bundle has none or lacks them)."""
    try:
        path = delivery_paths(recipe, region=region).get('evidence')
    except KeyError:
        # A region this recipe does not deliver has no supplement.
        return pd.DataFrame()
    if path is None or not Path(path).exists():
        return pd.DataFrame()
    import pyarrow.parquet as pq

    present = [c for c in columns if c in pq.read_schema(path).names]
    return pd.read_parquet(path, columns=present) if present else pd.DataFrame()


def read_bundle(recipe, region, admin_ids=None) -> gpd.GeoDataFrame:
    """Load a delivery bundle's attributes joined to its polygons.

    Parameters
    ----------
    recipe : str or dict
        Curation recipe that declares `region` under `share: delivery:`.
    region : str
        Region id of the bundle.
    admin_ids : sequence of str, optional
        Keep only these units (`admin3_id`).

    Returns
    -------
    geopandas.GeoDataFrame
        Indexed like the bundle, with its canonical columns and
        `_source` sidecars.
    """
    paths = delivery_paths(recipe, region=region)
    for role in ('canonical', 'geo'):
        if not paths[role].exists():
            raise FileNotFoundError(
                f'The {role} file of the {region!r} bundle of {recipe!r} does '
                f'not exist ({paths[role]}). Run export_delivery first.'
            )
    attributes = pd.read_parquet(paths['canonical'])
    if admin_ids is not None:
        attributes = attributes[attributes['admin3_id'].isin(list(admin_ids))]
    geometry = gpd.read_parquet(paths['geo'], columns=['geometry'])
    gdf = gpd.GeoDataFrame(
        attributes.join(geometry, how='inner'), geometry='geometry', crs=geometry.crs
    )
    if len(gdf) < len(attributes):
        raise ValueError(
            f'{len(attributes) - len(gdf):,} rows of the {region!r} bundle of '
            f'{recipe!r} have no polygon in its _geo file; the bundle is '
            'inconsistent and should be re-exported.'
        )
    return gdf


def _region_members(recipe, region) -> list[str]:
    """Units a recipe's delivery region pools."""
    from openplaces.io.delivery import delivery_members

    return [str(a) for a in delivery_members(recipe, region=region)]


def bundle_notice(recipe, region) -> str:
    """Return the terms notice shipped with a bundle, for the page."""
    return delivery_paths(recipe, region=region)['terms'].read_text(encoding='utf-8')


def _ragged_parts(geometry: np.ndarray):
    """Return coordinates and ring/part offsets of (multi)polygons.

    Polygons and multipolygons are written alike: a polygon is a
    multipolygon of one part, so the page needs one code path.
    """
    include_z = bool(shapely.has_z(geometry).any())
    geom_type, coords, offsets = shapely.to_ragged_array(
        geometry if include_z else shapely.force_2d(geometry), include_z=include_z
    )
    if geom_type == shapely.GeometryType.POLYGON:
        ring_offsets, part_ring_offsets = offsets
        part_feature = np.arange(len(geometry), dtype=np.uint32)
    elif geom_type == shapely.GeometryType.MULTIPOLYGON:
        ring_offsets, part_ring_offsets, feature_part_offsets = offsets
        part_feature = np.repeat(
            np.arange(len(geometry), dtype=np.uint32), np.diff(feature_part_offsets)
        )
    else:
        raise ValueError(f'Expected polygons, got {geom_type!r}.')
    return coords, ring_offsets, part_ring_offsets, part_feature


def _triangulate_parts(coords, ring_offsets, part_ring_offsets) -> np.ndarray:
    """Triangles of every polygon part, as indices into `coords`.

    The page would otherwise triangulate every polygon itself (deck.gl
    runs earcut per polygon) and keep one JavaScript object per part;
    on a phone that ran out of memory and stalled the load. Done here,
    the page hands the arrays to the GPU as they are.

    GEOS's constrained Delaunay triangulation uses only the polygon's
    own vertices, so each triangle corner is found again among the
    part's vertices by its coordinates. A part GEOS cannot triangulate
    (an invalid ring) gets no triangles: its walls still show, its top
    does not.

    Returns
    -------
    numpy.ndarray of uint32
        Flat `[a0, b0, c0, a1, b1, c1, ...]`, global vertex indices.
    """
    ring_offsets = np.asarray(ring_offsets, dtype=np.int64)
    part_ring_offsets = np.asarray(part_ring_offsets, dtype=np.int64)
    n_parts = len(part_ring_offsets) - 1
    if n_parts == 0:
        return np.zeros(0, dtype=np.uint32)
    polygons = shapely.from_ragged_array(
        shapely.GeometryType.POLYGON,
        np.ascontiguousarray(coords[:, :2], dtype=float),
        (ring_offsets, part_ring_offsets),
    )
    try:
        triangulated = shapely.constrained_delaunay_triangles(polygons)
    except shapely.errors.GEOSException:
        triangulated = np.array(
            [_triangulate_one(polygon) for polygon in polygons], dtype=object
        )
    triangles, part = shapely.get_parts(triangulated, return_index=True)
    if not len(triangles):
        return np.zeros(0, dtype=np.uint32)
    corners = shapely.get_coordinates(triangles).reshape(-1, 4, 2)[:, :3, :]

    vertex_start = ring_offsets[part_ring_offsets]
    vertex_part = np.repeat(np.arange(n_parts), np.diff(vertex_start))
    vertices = pd.DataFrame(
        {
            'part': vertex_part,
            'x': coords[:, 0],
            'y': coords[:, 1],
            'vertex': np.arange(len(coords)),
        }
    ).drop_duplicates(['part', 'x', 'y'])
    found = pd.DataFrame(
        {
            'part': np.repeat(part, 3),
            'x': corners[:, :, 0].ravel(),
            'y': corners[:, :, 1].ravel(),
        }
    ).merge(vertices, on=['part', 'x', 'y'], how='left')['vertex']
    index = found.to_numpy(dtype=float).reshape(-1, 3)
    complete = ~np.isnan(index).any(axis=1)
    return index[complete].astype(np.uint32).ravel()


def _triangulate_one(polygon):
    try:
        return shapely.constrained_delaunay_triangles(polygon)
    except shapely.errors.GEOSException:
        return shapely.GeometryCollection()


def encode_scene_file(
    geometry, base, height, value, stack=None, extra=None
) -> tuple[bytes, list]:
    """Encode one unit's features in the binary layout (module docstring).

    Parameters
    ----------
    geometry : array-like of shapely (Multi)Polygon
        In longitude and latitude (EPSG:4326). When any geometry has Z
        (a parcel draped on terrain), the vertices' Z is written as a
        per-vertex `z` array, and the page extrudes from it instead of
        from `base`; a 2D geometry in such a set reads as Z = 0.
    base, height : array-like of float
        Per feature, meters: where the extrusion starts and how tall it is.
    value : array-like of float
        Per feature, the value per area unit the page colors by; NaN
        where there is none.
    stack : array-like of float, optional
        Per feature, the part of `base` that is the extrusion of the
        layer below (a building on its parcel). With that layer
        switched off, the page lowers the feature by it.
    extra : dict of str to array-like, optional
        Further per-feature arrays, written under their names as they
        come (float32 or uint8), e.g. `estimated` (1 where the value is
        openplaces' own fill) or `value_dwelling`.

    Returns
    -------
    bytes
        The uncompressed file.
    list of float
        The origin, `[longitude, latitude]`.
    """
    # Outer rings counterclockwise, holes clockwise: the page tells
    # deck.gl so (`_windingOrder: 'CCW'`) instead of having it check.
    geometry = shapely.orient_polygons(np.asarray(geometry), exterior_cw=False)
    coords, ring_offsets, part_ring_offsets, part_feature = _ragged_parts(geometry)
    triangles = _triangulate_parts(coords, ring_offsets, part_ring_offsets)
    z = coords[:, 2] if coords.shape[1] == 3 else None
    coords = coords[:, :2]
    origin = coords.mean(axis=0) if len(coords) else np.zeros(2)
    # Round the origin so that it reads back exactly from the JSON.
    origin = np.round(origin, 6)
    arrays = {
        'xy': (coords - origin).astype('<f4'),
        **({} if z is None else {'z': np.nan_to_num(z).astype('<f4')}),
        'ring_offsets': np.asarray(ring_offsets, dtype='<u4'),
        'part_ring_offsets': np.asarray(part_ring_offsets, dtype='<u4'),
        'part_feature': np.asarray(part_feature, dtype='<u4'),
        'triangles': np.asarray(triangles, dtype='<u4'),
        'base': np.asarray(base, dtype='<f4'),
        'height': np.asarray(height, dtype='<f4'),
        'value': np.asarray(value, dtype='<f4'),
        **({} if stack is None else {'stack': np.asarray(stack, dtype='<f4')}),
    }
    for name, array in (extra or {}).items():
        array = np.asarray(array)
        arrays[name] = array.astype('u1' if array.dtype.kind in 'bu' else '<f4')
    entries, offset = [], 0
    for name, array in arrays.items():
        entries.append(
            {
                'name': name,
                'dtype': array.dtype.str.lstrip('<|'),
                'offset': offset,
                'count': int(array.size),
            }
        )
        offset += array.nbytes
        offset += -offset % 8
    header = {
        'format': 'openplaces-value-scene',
        'version': FORMAT_VERSION,
        'origin': origin.tolist(),
        'n_features': int(len(geometry)),
        'arrays': entries,
    }
    header_bytes = json.dumps(header).encode('utf-8')
    header_bytes += b' ' * (-(len(MAGIC) + 4 + len(header_bytes)) % 8)
    chunks = [MAGIC, struct.pack('<I', len(header_bytes)), header_bytes]
    for array in arrays.values():
        data = array.tobytes()
        chunks.append(data + b'\0' * (-len(data) % 8))
    return b''.join(chunks), origin.tolist()


def decode_scene_file(data: bytes) -> tuple[dict, dict]:
    """Read a file written by `encode_scene_file` (for tests and checks).

    Returns
    -------
    dict
        The header.
    dict of str to numpy.ndarray
        The arrays by name.
    """
    if data[:4] != MAGIC:
        raise ValueError('Not an openplaces value-scene file.')
    (header_length,) = struct.unpack('<I', data[4:8])
    header = json.loads(data[8 : 8 + header_length])
    start = 8 + header_length
    arrays = {}
    for entry in header['arrays']:
        dtype = np.dtype(entry['dtype']).newbyteorder('<')
        begin = start + entry['offset']
        arrays[entry['name']] = np.frombuffer(
            data, dtype=dtype, count=entry['count'], offset=begin
        )
    return header, arrays


def _tooltip_values(gdf, layer: SceneLayer) -> dict:
    """Columnar tooltip values of *layer*, JSON-ready."""
    values = {}
    for tip in layer.tooltip:
        if tip.column not in gdf.columns:
            raise KeyError(
                f'Scene layer {layer.name!r} shows {tip.column!r}, which the '
                'rendered table lacks.'
            )
        column = gdf[tip.column]
        if tip.kind in ('money', 'money_per_area'):
            column = column.astype(float).round(0 if tip.kind == 'money' else 2)
        elif tip.kind in ('year', 'count'):
            column = column.astype('Int64')
        cells = column.astype(object).where(column.notna(), None).tolist()
        values[tip.column] = cells
        if tip.imputed_from is not None:
            source = (
                gdf[tip.imputed_from]
                if tip.imputed_from in gdf.columns
                else pd.Series(None, index=gdf.index)
            )
            values[f'{tip.column}__imputed'] = is_imputed(source).astype(int).tolist()
    return values


RECORDS_COLUMN = 'n_records_on_outline'


def _count_dwellings_from_above(gdfs: dict, layers) -> None:
    """Give a layer offering a per-dwelling view (`per_dwelling_label`)
    and lacking dwelling counts the dwellings of the homes standing on
    it.

    Lots rarely state their dwellings (measured 2026-10-16: 20% of
    boston-inner-core parcels, none in western-nc), while buildings
    mostly do. A display step: each home (`RESIDENTIAL_OCCUPANCY`) of
    the layer stacked on this one counts toward the feature that
    contains its centroid; a feature's own count wins where it states
    one. Written to `n_dwellings` of the layer's frame, in place.
    """
    for layer in layers:
        if not layer.per_dwelling_label:
            continue
        above = next((u for u in layers if u.stack_on == layer.name), None)
        if above is None:
            continue
        below, homes = gdfs[layer.name], gdfs[above.name]
        if 'n_dwellings' not in homes.columns or 'occupancy_type' not in homes.columns:
            continue
        kind = (
            homes[BUILDING_CLASS]
            if BUILDING_CLASS in homes.columns
            else building_class(homes)
        )
        homes = homes[
            kind.astype('string')
            .isin(RESIDENTIAL_OCCUPANCY | RESIDENTIAL_LAND_USE)
            .fillna(False)
            & (homes['n_dwellings'].fillna(0) > 0)
        ]
        counted = pd.Series(0.0, index=below.index)
        if len(homes) and len(below):
            points = gpd.GeoDataFrame(
                {'n': homes['n_dwellings'].astype(float).to_numpy()},
                geometry=homes.geometry.to_crs(5070).centroid.to_numpy(),
                crs=5070,
            )
            lots = gpd.GeoDataFrame(
                geometry=below.geometry.to_crs(5070).to_numpy(), crs=5070
            )
            hit = gpd.sjoin(points, lots, predicate='within', how='inner')
            sums = hit.groupby('index_right')['n'].sum()
            counted.iloc[sums.index.to_numpy()] = sums.to_numpy()
        own = (
            below['n_dwellings'].astype(float)
            if 'n_dwellings' in below.columns
            else pd.Series(np.nan, index=below.index)
        )
        # The roll's own count, kept apart: it confirms a home where a
        # count from outlines alone does not (`ROLL_DWELLINGS`).
        below[ROLL_DWELLINGS] = own.where(own > 0)
        below['n_dwellings'] = own.where(own > 0, counted.where(counted > 0))


EXEMPT_CLASS = 'Exempt'


def _mark_exempt_ground(gdfs, layers) -> None:
    """Buildings on exempt lots are not homes unless the roll says so.

    A display step, run before `_count_dwellings_from_above`. A building
    takes its class from its lot's land use (`building_class`), so the
    building of an exempt lot whose code reads residential was drawn as
    a home and counted as dwellings (120 Main St, Melrose: a National
    Guard armory; maintainer, 2026-10-17). Each building whose centroid
    lies on a lot with use group "exempt" and no dwellings of the roll's
    own is shown with the class `EXEMPT_CLASS` instead. Public housing
    keeps its homes where the roll counts them. Written to the stacked
    layer's `BUILDING_CLASS`, in place.
    """
    for layer in layers:
        above = next((u for u in layers if u.stack_on == layer.name), None)
        if above is None:
            continue
        lots, buildings = gdfs[layer.name], gdfs[above.name]
        if 'use_group' not in lots.columns or BUILDING_CLASS not in buildings.columns:
            continue
        use = lots['use_group'].astype('string').str.strip().str.lower()
        own = (
            lots['n_dwellings'].astype(float)
            if 'n_dwellings' in lots.columns
            else pd.Series(np.nan, index=lots.index)
        )
        exempt = ((use == 'exempt').fillna(False) & ~(own > 0)).to_numpy()
        if not exempt.any() or buildings.empty:
            continue
        points = gpd.GeoDataFrame(
            geometry=buildings.geometry.to_crs(5070).centroid.to_numpy(), crs=5070
        )
        ground = gpd.GeoDataFrame(
            geometry=lots.geometry[exempt].to_crs(5070).to_numpy(), crs=5070
        )
        hit = gpd.sjoin(points, ground, predicate='within', how='inner')
        on_exempt = np.zeros(len(buildings), dtype=bool)
        on_exempt[hit.index.unique().to_numpy()] = True
        kind = buildings[BUILDING_CLASS].astype('string')
        homes = kind.isin(RESIDENTIAL_OCCUPANCY | RESIDENTIAL_LAND_USE).fillna(False)
        buildings[BUILDING_CLASS] = kind.where(
            ~(homes.to_numpy() & on_exempt), EXEMPT_CLASS
        )


def _without_filters(flags) -> list[str]:
    """rclone flags with every `--filter <rule>` pair removed."""
    out, skip = [], False
    for flag in flags:
        if skip:
            skip = False
        elif flag == '--filter':
            skip = True
        else:
            out.append(flag)
    return out


def _write_content_file(data_dir: Path, stem: str, data: bytes) -> str:
    """Write *data* gzipped as `<stem>.<hash>.bin`; return its page path.

    The name carries a hash of the uncompressed bytes, so a rebuild with
    unchanged data reuses the file: it is neither rewritten here nor
    uploaded again (`deploy_web_scene` copies data with
    `--ignore-existing`). gzip's header timestamp is fixed at 0, so the
    same data also gives the same file. Measured 2026-10-16 on
    boston-inner-core: palette, lighting and control changes left every
    one of 1,589 data files unchanged, yet each publish re-sent them
    all (about 220 MB for the four scenes) under a new build folder.
    """
    import hashlib

    digest = hashlib.sha256(data).hexdigest()[:16]
    name = f'{stem}.{digest}.bin'
    path = data_dir / name
    if not path.exists():
        path.write_bytes(gzip.compress(data, 9, mtime=0))
    return f'data/{name}'


def _read_manifest(output_dir: Path) -> dict | None:
    """The scene manifest in *output_dir*, or None."""
    path = Path(output_dir) / 'scene.json'
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def manifest_files(manifest: dict | None) -> set[str]:
    """Data paths (`data/...`) a scene manifest loads."""
    if not manifest:
        return set()
    return {
        path
        for layer in manifest.get('layers', [])
        for paths in layer.get('files', {}).values()
        for path in paths.values()
    }


def _prune_local_data(output_dir: Path, keep: set[str]) -> None:
    """Delete data files under `output_dir/data` that *keep* omits,
    including folders of builds from before content-named files."""
    data_dir = Path(output_dir) / 'data'
    # A file another program holds (a sync client indexing it) is left
    # for the next build rather than failing this one.
    for path in list(data_dir.rglob('*')):
        if path.is_file() and path.relative_to(output_dir).as_posix() not in keep:
            try:
                path.unlink()
            except OSError:
                continue
    for folder in sorted(
        (p for p in data_dir.rglob('*') if p.is_dir()), key=lambda p: -len(p.parts)
    ):
        try:
            if not any(folder.iterdir()):
                folder.rmdir()
        except OSError:
            continue


LANDMARK_REACH_M = 40.0


def landmark_positions(landmarks, rendered) -> list:
    """Where each landmark's marker stands: on the tallest top drawn
    within `LANDMARK_REACH_M` of its point, so a tower's label sits on
    its roof and an airport's on its ground.

    Parameters
    ----------
    landmarks : sequence of dict
        Each with `name`, `kind`, `longitude`, `latitude`.
    rendered : sequence of TerrainLayer
        The scene's layers as rendered (`show_value_terrain_layer`),
        whose `total_elevation` is each feature's top.

    Returns
    -------
    list of dict
        `name`, `kind`, `position` ([longitude, latitude, top]) and
        `ground`, the base of the bottom layer there, which the page
        takes off when the terrain is switched off.
    """
    out = []
    for mark in landmarks:
        missing = {'name', 'kind', 'longitude', 'latitude'} - set(mark)
        if missing:
            raise ValueError(f'Landmark {mark!r} lacks {sorted(missing)}.')
        point = gpd.GeoSeries(
            [shapely.Point(mark['longitude'], mark['latitude'])], crs=4326
        )
        top, ground = 0.0, 0.0
        for i, layer in enumerate(rendered):
            gdf = layer.gdf
            if gdf.empty:
                continue
            here = point.to_crs(gdf.crs if gdf.crs.is_projected else 5070)
            shapes = gdf.geometry if gdf.crs.is_projected else gdf.geometry.to_crs(5070)
            near = shapes.distance(here.iloc[0]).to_numpy() <= LANDMARK_REACH_M
            if near.any():
                top = max(top, float(np.nanmax(layer.total_elevation[near])))
                if i == 0:
                    base = layer.total_elevation - layer.rendered_elevation
                    ground = float(np.nanmin(base[near]))
        out.append(
            {
                'name': str(mark['name']),
                'kind': str(mark['kind']),
                'position': [
                    round(float(mark['longitude']), 6),
                    round(float(mark['latitude']), 6),
                    round(top, 1),
                ],
                'ground': round(ground, 1),
            }
        )
    return out


def crop_to_area(gdf, area):
    """Keep the features whose center lies inside *area*.

    A display step: features are kept whole, never clipped, so nothing
    is drawn with a shape its source does not have. The center is the
    plain centroid in an equal-area projection (EPSG:5070 within the
    US), not a point moved inside the feature.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Features in any CRS.
    area : shapely geometry
        In longitude and latitude (EPSG:4326).

    Returns
    -------
    geopandas.GeoDataFrame
    """
    if gdf.empty:
        return gdf
    area = shapely.make_valid(area)
    shapely.prepare(area)
    centers = gdf.geometry.to_crs(5070).centroid.to_crs(4326).to_numpy()
    return gdf[
        shapely.contains_xy(area, shapely.get_x(centers), shapely.get_y(centers))
    ]


def merge_identical_outlines(gdf, layer):
    """One feature per outline: records with identical polygons merged.

    Several records can share one outline, e.g. the accounts or units
    of a lot that the parcel layer repeats once per record. Drawn as
    they come, their walls coincide exactly at different heights and
    the GPU cannot order them: they flicker as the camera moves.
    Measured 2026-10-16 on US-NC-DUR: one 46,153 m2 lot outline came
    eleven times. Merged, the feature carries the summed value (the
    lot's whole value over its area, which is what a per-area color
    of that lot means), the first record's other fields, the imputed
    mark if any record's value was imputed, and in `RECORDS_COLUMN`
    how many records it stands for (None when one).

    Grouping by identical geometry is a display step here; design
    limit L2 keeps grouping by identical geometry allowed.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        One layer's rows.
    layer : SceneLayer
        Its value column and tooltip fields.

    Returns
    -------
    geopandas.GeoDataFrame
    """
    gdf = gdf.copy()
    keys = pd.Series(
        shapely.to_wkb(shapely.normalize(gdf.geometry.to_numpy())), index=gdf.index
    )
    counts = keys.map(keys.value_counts())
    gdf[RECORDS_COLUMN] = counts.where(counts > 1).astype('Int64')
    if not (counts > 1).any():
        return gdf
    first = ~keys.duplicated()
    out = gdf[first].copy()
    values = gdf[layer.value_column]
    summed = values.fillna(0).groupby(keys.to_numpy()).sum()
    any_value = values.notna().groupby(keys.to_numpy()).any()
    out[layer.value_column] = (
        keys[first].map(summed).where(keys[first].map(any_value)).to_numpy()
    )
    for column in {t.imputed_from for t in layer.tooltip if t.imputed_from}:
        if column not in gdf.columns:
            continue
        flagged = is_imputed(gdf[column]).groupby(keys.to_numpy()).any()
        merged = keys[first].map(flagged).to_numpy()
        out.loc[merged, column] = 'merged+imputed'
    return out


def layer_cmap(layer) -> mpl.colors.Colormap:
    """The colormap of *layer* (a SceneLayer or PaletteLayer), cut to
    its `cmap_range`. `cmap` is a colormap name or a tuple of color
    stops, interpolated evenly."""
    if isinstance(layer.cmap, tuple | list):
        cmap = mpl.colors.LinearSegmentedColormap.from_list('stops', list(layer.cmap))
    else:
        cmap = get_diverging_colormap(layer.cmap)
    low, high = layer.cmap_range
    if (low, high) == (0.0, 1.0):
        return cmap
    return mpl.colors.ListedColormap(
        cmap(np.linspace(low, high, 256)), name=f'{layer.cmap}_{low:g}_{high:g}'
    )


@dataclass(frozen=True)
class PaletteLayer:
    """How one scene layer is colored under one palette.

    Parameters
    ----------
    cmap : str or tuple of str
        Colormap name (`viz.colors.get_diverging_colormap`), or color
        stops interpolated evenly.
    vmin, vmax : float, optional
        Color-scale bounds, in dollars per the layer's area unit, on a
        log1p scale like `show_value_terrain_layer`'s. None (default)
        takes the scene's own bounds (`scene_bounds`), so one palette
        reads in a city and in a rural county alike.
    cmap_range : tuple of float
        The part of the colormap used (`SceneLayer.cmap_range`).
    alpha : float, optional
        Opacity of the fills under this palette; None keeps the layer's
        own (`SceneLayer.render`).
    line_rgba : tuple of int, optional
        Color of the outlines drawn along every edge of the extruded
        features (deck.gl's wireframe); None draws none.
    line_cmap : str or tuple of str, optional
        Colors the outlines by each feature's value instead, on the
        same scale as the fills (`cmap` then usually holds dark tints
        of the same hues, so the edges carry the color).
    line_alpha : float
        Opacity of the value-colored outlines.
    other_cmap, other_line_cmap : str or tuple of str, optional
        Fill and outline ramps for the non-residential features of a
        layer with a per-dwelling view (`SceneLayer.per_dwelling_label`),
        in hues the layer's main ramp does not use.
    """

    cmap: str
    vmin: float | None = None
    vmax: float | None = None
    cmap_range: tuple[float, float] = (0.0, 1.0)
    alpha: float | None = None
    line_rgba: tuple[int, int, int, int] | None = None
    line_cmap: str | tuple | None = None
    line_alpha: float = 0.9
    other_cmap: str | tuple | None = None
    other_line_cmap: str | tuple | None = None


def _nice(value: float, up: bool) -> float:
    """Round to 1, 2 or 5 times a power of ten, down or up."""
    if not value > 0:
        return 0.0
    power = 10 ** np.floor(np.log10(value))
    steps = (1, 2, 5, 10)
    if up:
        return float(next(s for s in steps if s * power >= value) * power)
    return float(next(s for s in reversed(steps) if s * power <= value) * power)


def scene_bounds(per_area, low_q=0.02, high_q=0.995) -> tuple[float, float]:
    """Color bounds from a scene's own values per area.

    Fixed bounds failed across places. Measured 2026-10-15 on US-NC-HYD:
    with Boston's $50K-$50M per acre, rural land fell almost wholly
    below the lower bound and drew in the ramp's darkest color, next
    to invisible on the dark page. The 2nd and 99.5th percentiles,
    rounded outward to 1-2-5 numbers, keep a scene's spread on the
    ramp and leave its extremes saturated.
    """
    values = np.asarray(per_area, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 1.0, 10.0
    low, high = np.quantile(values, [low_q, high_q])
    low, high = _nice(low, up=False), _nice(high, up=True)
    return (low, high) if high > low else (low, low * 10)


@dataclass(frozen=True)
class Palette:
    """One color option of the page.

    Parameters
    ----------
    id : str
        Short id, used in the page's address (`?palette=<id>`).
    label : str
        Name on the page's palette switch.
    layers : dict of str to PaletteLayer
        Coloring per scene layer name.
    no_value_rgba : tuple of int
        Color of a feature without a value.
    zero_value_rgba : tuple of int
        Color of a feature whose recorded value is zero (an exempt or
        public lot the assessor values at $0), apart from one with no
        value at all (maintainer, 2026-10-17: white).
    background : str
        CSS background of the page.
    boundary_rgba : tuple of int
        Color of the town borders.
    theme : str
        `dark` or `light`: whether the page under the scene is dark or
        light, for marks drawn over it (the pinned tooltip's leader).
    blend : str
        `normal`, or `additive`: translucent fills add their light to
        what is behind them and write no depth, so overlaps glow and
        the result does not depend on the order features are drawn in.
        Translucent fills blended normally do depend on it: walls
        showed their back faces block by block (seen 2026-10-16).
    material : dict, optional
        deck.gl lighting of the extrusions (`ambient`, `diffuse`,
        `shininess`); None keeps deck.gl's own, which shades walls
        well below their color. A light palette needs mostly ambient
        light or its pastels turn gray on the walls.
    """

    id: str
    label: str
    layers: dict
    no_value_rgba: tuple[int, int, int, int] = ZERO_VALUE_RGBA
    zero_value_rgba: tuple[int, int, int, int] = (255, 255, 255, 255)
    background: str = (
        'radial-gradient(ellipse 80% 70% at 50% 45%, #13213d 0%, #0a1222 45%, '
        '#05070d 100%)'
    )
    boundary_rgba: tuple[int, int, int, int] = (235, 240, 255, 90)
    theme: str = 'dark'
    blend: str = 'normal'
    material: dict | None = None


# deck.gl's default lighting weights without its specular highlight:
# the highlight depends on where the camera is, so two walls facing the
# same way shaded differently by their place on the screen (maintainer,
# 2026-10-16: "the direction of the shadows is still inconsistent").
# Matte, a wall's shade depends on the direction it faces alone.
MATTE = {'ambient': 0.35, 'diffuse': 0.6, 'shininess': 32, 'specularColor': [0, 0, 0]}


def default_palettes(layers=VALUE_SCENE_LAYERS) -> list[Palette]:
    """The page's color options (maintainer's choice, 2026-10-16).

    Landscape (the default, first), Sunset, Neon and Pastel. The
    first view stays warm and light at its dearest places: the
    maintainer turned down every variant that bought building-to-land
    contrast with blue or dark grounds, or with a dark, stark top of
    the scale ("looks sad"). Contrast was scored on the color pairs
    that meet on the map (every building against the parcel containing
    it, at both values' places on their scales), by the OKLab distance
    between the two colors. Measured 2026-10-16 on the four published
    scenes (673,329 pairs): 5th percentile (x100) 7.8 for Landscape's
    tops (its sand and terracotta middle is close to its amber
    buildings; walls 17.7), 20.0 for Sunset, 13.9 for Pastel.
    """
    names = {layer.name for layer in layers}
    palettes = [
        # Natural land colors (maintainer, 2026-10-16): green and blue
        # out in the country, through sand, to terracotta and brick red
        # where land is dearest, in muted tones. Not monotone in
        # lightness (sand is its lightest step), so land order reads by
        # hue, as on a relief map. Buildings run dark amber to pale
        # gold, ordered by lightness.
        Palette(
            id='landscape',
            # Shorter on the chip (maintainer, 2026-10-16); the id stays
            # `landscape`, so stored choices and `?palette=` still work.
            label='Earth',
            material=MATTE,
            layers={
                'parcels': PaletteLayer(
                    (
                        '#3f6e8c',
                        '#5f9a8f',
                        '#8db88a',
                        '#d9d3a0',
                        '#e8a77c',
                        '#cf6a55',
                        '#a8443f',
                    ),
                    # Lots without homes, per dwelling view: grays.
                    other_cmap=('#3d4148', '#6a6f78', '#9aa0a8', '#cdd1d6'),
                    # Thin lot lines, each a little darker than its own
                    # lot (maintainer, 2026-10-16): the ramp's stops at
                    # 82% of their brightness.
                    line_cmap=(
                        '#345a73',
                        '#4e7e75',
                        '#749771',
                        '#b2ad83',
                        '#be8966',
                        '#aa5746',
                        '#8a3834',
                    ),
                    line_alpha=0.9,
                ),
                # Non-residential buildings in the per-dwelling view:
                # slate blue to ice, a hue neither the amber buildings
                # nor the sand and terracotta ground use.
                'buildings': PaletteLayer(
                    'cmr.amber',
                    cmap_range=(0.2, 0.95),
                    other_cmap=('#2f3a6b', '#4a63a8', '#7e9bd6', '#c4d6f5'),
                ),
            },
        ),
        # A teal ground under buildings running magenta, orange,
        # yellow: lightness rises with value in both, so the order
        # reads without color vision. The ground starts above the
        # ocean map's near-black, which vanished into the page.
        Palette(
            id='sunset',
            label='Sunset',
            material=MATTE,
            layers={
                'parcels': PaletteLayer(
                    'cmr.ocean',
                    cmap_range=(0.18, 0.88),
                    other_cmap=('#2e3238', '#55595f', '#81858b', '#b3b6ba'),
                ),
                # Non-residential: grays to silver, apart from the
                # magenta-to-yellow homes and the teal ground.
                'buildings': PaletteLayer(
                    'plasma',
                    cmap_range=(0.3, 1.0),
                    other_cmap=('#3b3f4a', '#6b7280', '#a5adba', '#e6e9ef'),
                ),
            },
        ),
        # City lights at night ("nightlights of Tokyo"): neon edges on
        # dark, solid faces. The value sets the hue of a feature's edges
        # at one brightness: land from blue to cyan and aqua, buildings
        # from amber through orange and red to hot pink and magenta at
        # the dearest center. Faces are dark tints of the same hues,
        # opaque and unshaded. The first versions blended see-through
        # fills additively: brightness then followed how many faces
        # overlapped, i.e. density, not value, and stacked violets and
        # magentas all summed to pink (maintainer, 2026-10-16).
        Palette(
            id='neon',
            label='Neon',
            layers={
                # The land speaks through its faces, in very dark hues:
                # black in the outskirts, deep indigo and blue, deep
                # teal at the dearest land. Its outlines are the same
                # hues a little brighter, so lots stay apart without
                # the bright blue lines that competed with the
                # buildings' glow (maintainer, 2026-10-16).
                'parcels': PaletteLayer(
                    (
                        '#000000',
                        '#0b0a26',
                        '#16123f',
                        '#16245a',
                        '#0f3a5e',
                        '#0b4e5a',
                        '#0d5f55',
                    ),
                    line_cmap=(
                        '#141428',
                        '#1d1a48',
                        '#2a2270',
                        '#2a3d90',
                        '#1d5f94',
                        '#167e88',
                        '#1a9884',
                    ),
                    line_alpha=0.85,
                    other_cmap=('#050505', '#111111', '#1c1c1c'),
                    other_line_cmap=('#1a1a1a', '#333333', '#4d4d4d', '#666666'),
                ),
                # A broad run of hues (maintainer, 2026-10-16: wider,
                # with green, "matrix-like", the cheapest darker): a dim
                # dark green at the bottom, bright green, lime, amber,
                # red, hot pink, violet at the top.
                'buildings': PaletteLayer(
                    (
                        '#010803',
                        '#031406',
                        '#0e1404',
                        '#140e02',
                        '#170504',
                        '#16040c',
                        '#0e0416',
                    ),
                    line_cmap=(
                        '#0c4a1c',
                        '#1fd14f',
                        '#c6f03a',
                        '#ffb02e',
                        '#ff4a3d',
                        '#ff2fb0',
                        '#b54dff',
                    ),
                    line_alpha=0.95,
                    # Non-residential: cyan to white edges on the same
                    # dark faces.
                    other_cmap=('#03141a', '#04202a', '#062a36'),
                    other_line_cmap=('#0a5a6e', '#00b8e0', '#7ff0ff', '#f0ffff'),
                ),
            },
            # Faces unshaded and without highlights, so their dark tints
            # stay dark: with `shininess` 0 the specular term is 1 on
            # every face and laid a gray sheen over the whole scene.
            material={
                'ambient': 1.0,
                'diffuse': 0.0,
                'shininess': 32,
                'specularColor': [0, 0, 0],
            },
            no_value_rgba=(14, 14, 30, 255),
            background=(
                'radial-gradient(ellipse 80% 70% at 50% 45%, #0d0820 0%, '
                '#05030d 50%, #000000 100%)'
            ),
            boundary_rgba=(120, 200, 255, 150),
        ),
        # A light, airy page: land from white through sky, mint, butter
        # and peach to pink and lilac, growing more colorful toward the
        # dearest land, with thin gray lot lines; pastel buildings
        # (lavender, periwinkle, sea green, rose, apricot). The dearest
        # places stay light and warm, never dark. The land ramp spans
        # 2.7 times the earlier near-white one (OKLab path 42 vs 15),
        # at some cost to roof contrast (5th percentile 13.9 vs 17.4).
        Palette(
            id='pastel',
            label='Pastel',
            layers={
                'parcels': PaletteLayer(
                    # Starts at a faint blue-gray, not white, so the
                    # cheapest land stays visibly a value beside "No
                    # value recorded" (maintainer, 2026-10-17).
                    (
                        '#eef1f7',
                        '#e6f4ff',
                        '#cfeee8',
                        '#e9f5c4',
                        '#fff0b0',
                        '#ffd3b0',
                        '#ffb8c9',
                        '#e7b0ef',
                    ),
                    line_rgba=(150, 150, 165, 70),
                    # Lots without homes, per dwelling view: grays, from
                    # about the "no value" gray of the Fells' lakes
                    # down; the earlier near-whites hid such land
                    # (0 S Border Rd, Medford; maintainer, 2026-10-17).
                    other_cmap=('#e2dfd9', '#ccc8c1', '#b4afa7', '#99948c'),
                ),
                # Non-residential: warm grays, quieter than the pastel
                # homes. The top tier of the homes' ramp ends in
                # a pastel yellow (maintainer, 2026-10-16).
                'buildings': PaletteLayer(
                    ('#a99fdc', '#8eb0e6', '#7fc6bb', '#ea8fa8', '#ffb866', '#ffe066'),
                    # Leaf greens, light to deep: the warm grays looked
                    # drab beside the pastel homes (maintainer,
                    # 2026-10-17). Darker and yellower than the homes'
                    # sea green (#7fc6bb), which a pale sage matched
                    # (5 Maxwells Grn looked non-residential).
                    other_cmap=('#b5dd8f', '#74b94f', '#3f8a35', '#1d5a26'),
                ),
            },
            # A warm gray clearly darker than the faint low end of the
            # land ramp, so "no value" and "little value" differ.
            no_value_rgba=(212, 208, 201, 255),
            background=(
                'radial-gradient(ellipse 80% 70% at 50% 45%, #ffffff 0%, '
                '#fbfaf7 60%, #f3f1ec 100%)'
            ),
            boundary_rgba=(90, 96, 120, 120),
            theme='light',
            # Mostly ambient: walls a little darker than roofs, not gray.
            material={
                'ambient': 0.75,
                'diffuse': 0.35,
                'shininess': 32,
                'specularColor': [20, 20, 20],
            },
        ),
    ]
    return [p for p in palettes if set(p.layers) == names]


def palette_manifest(palette: Palette, layers, bounds=None, stops: int = 256) -> dict:
    """A palette as the page reads it: per layer a 256-step lookup
    table (also drawn as the legend) and the bounds it spans.

    Parameters
    ----------
    bounds : dict of str to tuple of float, optional
        The scene's own bounds per layer (`scene_bounds`), used where
        the palette sets none.
    """
    alphas = {layer.name: layer.render.get('alpha', 1.0) for layer in layers}
    defaults = {layer.name: (layer.vmin, layer.vmax) for layer in layers}
    bounds = bounds or {}
    out = {}
    for name, spec in palette.layers.items():
        cmap = layer_cmap(spec)
        rgb = (cmap(np.linspace(0, 1, stops))[:, :3] * 255).round().astype(int)
        auto = bounds.get(name, defaults.get(name, (1.0, 10.0)))
        out[name] = {
            'vmin': spec.vmin if spec.vmin is not None else auto[0],
            'vmax': spec.vmax if spec.vmax is not None else auto[1],
            'alpha': round(
                255 * (spec.alpha if spec.alpha is not None else alphas.get(name, 1.0))
            ),
            'lut': [mpl.colors.to_hex(c / 255) for c in rgb],
            'scale': 'log1p',
            'line': list(spec.line_rgba) if spec.line_rgba else None,
        }

        def ramp(cmap):
            rgb = layer_cmap(PaletteLayer(cmap))(np.linspace(0, 1, stops))[:, :3]
            return [mpl.colors.to_hex(c) for c in rgb]

        if spec.line_cmap is not None:
            out[name]['line_lut'] = ramp(spec.line_cmap)
            out[name]['line_alpha'] = round(255 * spec.line_alpha)
        if spec.other_cmap is not None:
            out[name]['other_lut'] = ramp(spec.other_cmap)
        if spec.other_line_cmap is not None:
            out[name]['other_line_lut'] = ramp(spec.other_line_cmap)
    return {
        'id': palette.id,
        'label': palette.label,
        'layers': out,
        'no_value': list(palette.no_value_rgba),
        'zero_value': list(palette.zero_value_rgba),
        'background': palette.background,
        'boundary': list(palette.boundary_rgba),
        'theme': palette.theme,
        'blend': palette.blend,
        'material': palette.material,
    }


def _admin_names(admin_ids) -> pd.DataFrame:
    """Names and boundaries of the scene's units (level 3)."""
    admin = get_admin(list(admin_ids), level=3, geom=True)
    if 'admin3_id' not in admin.columns:
        admin = admin.reset_index()
    return admin[['admin3_id', 'name', 'geometry']].to_crs(4326)


def lake_polygons(admin_ids) -> gpd.GeoSeries:
    """Lakes, ponds and reservoirs around the scene's units, from NHD.

    NHD High Resolution waterbodies (`US_water-waterbody-usgs-nhdhr2023`)
    of feature types Lake/Pond (390) and Reservoir (436), read within
    the units' bounding box. Rivers are not among them: NHD draws wide
    rivers as areas of their own (NHDArea), so a border along the
    Charles or the Mystic stays drawn. A state whose file is missing
    contributes nothing.

    Parameters
    ----------
    admin_ids : sequence of str
        The scene's level-3 units.

    Returns
    -------
    geopandas.GeoSeries
        Lake polygons in EPSG:4326.
    """
    from openplaces.core.schema import AdminId
    from openplaces.recipe import get_output_path, get_recipe_by_id

    admin_ids = list(admin_ids)
    states = sorted({str(AdminId(a).truncate_to_level(2)) for a in admin_ids})
    units = get_admin(states, level=3, geom=True)
    if 'admin3_id' not in units.columns:
        units = units.reset_index()
    units = units[units['admin3_id'].isin(admin_ids)]
    bounds = units.to_crs(4326).total_bounds
    recipe = get_recipe_by_id('US_water-waterbody-usgs-nhdhr2023')
    parts = []
    for state in states:
        attributes_path = get_output_path(recipe, state)
        geo_path = get_output_path(recipe, state, geo=True)
        if not (attributes_path.exists() and geo_path.exists()):
            continue
        types = pd.read_parquet(
            attributes_path, columns=['feature_type_nhd', '_join_id']
        )
        keys = types.loc[types['feature_type_nhd'].isin(LAKE_TYPES), '_join_id']
        geo = gpd.read_parquet(geo_path, bbox=tuple(bounds))
        parts.append(geo.loc[geo['_join_id'].isin(keys), 'geometry'].to_crs(4326))
    if not parts:
        return gpd.GeoSeries([], crs=4326)
    return gpd.GeoSeries(pd.concat(parts, ignore_index=True), crs=4326)


# NHD feature types that cut a town border (`lake_polygons`).
LAKE_TYPES = (390, 436)


def land_borders(
    admin_ids, land=None, lakes=None, tolerance_m=5.0, open_water_m=300.0
) -> list:
    """Border lines the scene's units share with another unit, on land.

    A unit's whole outline also runs along the coast, which reads on
    the page as a line drawn in the sea. Only the stretches where a
    unit meets a neighbor (inside the scene or not) are kept, so the
    coastline drops out. A stretch through a lake is left out too
    (Spy Pond, the Mystic Lakes), while one along a road or a river is
    drawn (maintainer, 2026-10-17): an earlier rule kept only stretches
    within 25 m of a parcel and broke the Arlington and Belmont border
    where it follows Route 2. Census units also extend into open water
    and meet there (Boston and its harbor neighbors), so a stretch more
    than `open_water_m` from any parcel is left out as well. Neighbors
    are all units at the same level under the same parent (level 2); a
    border with a unit across a state line is therefore not drawn.

    Parameters
    ----------
    admin_ids : sequence of str
        The scene's level-3 units.
    land : geopandas.GeoDataFrame, optional
        Parcels marking land. Without it, no stretch counts as open
        water.
    lakes : geopandas.GeoSeries, optional
        Lake polygons (`lake_polygons`). Without them, no stretch is
        cut as a lake.
    tolerance_m : float
        Two outlines closer than this count as one border (the
        sources' shared edges rarely coincide exactly).
    open_water_m : float
        How far from every parcel a stretch must lie to count as open
        water; well beyond a highway's or a river's width.

    Returns
    -------
    list of shapely.LineString
        In longitude and latitude, each shared border once.
    """
    from openplaces.core.schema import AdminId

    admin_ids = list(admin_ids)
    parents = sorted({str(AdminId(a).truncate_to_level(2)) for a in admin_ids})
    units = get_admin(parents, level=3, geom=True)
    if 'admin3_id' not in units.columns:
        units = units.reset_index()
    units = units[['admin3_id', 'geometry']]
    units = units.to_crs(units.estimate_utm_crs())
    scene = units[units['admin3_id'].isin(admin_ids)]
    seen, lines = set(), []
    for row in scene.itertuples():
        outline = row.geometry.boundary
        near = units[
            (units['admin3_id'] != row.admin3_id)
            & units.intersects(row.geometry.buffer(tolerance_m))
        ]
        for other in near.itertuples():
            pair = frozenset((row.admin3_id, other.admin3_id))
            if pair in seen:
                continue
            seen.add(pair)
            shared = outline.intersection(other.geometry.buffer(tolerance_m))
            shared = shapely.line_merge(shared) if not shared.is_empty else shared
            lines.extend(
                part
                for part in shapely.get_parts(shared)
                if part.geom_type == 'LineString' and part.length > tolerance_m
            )
    if not lines:
        return []
    borders = gpd.GeoSeries(lines, crs=units.crs)
    has_land = land is not None and len(land)
    has_lakes = lakes is not None and len(lakes)
    if has_land or has_lakes:
        # Each border is cut into short pieces, and a piece is kept
        # unless its midpoint lies in a lake or no parcel lies within
        # `open_water_m` of it (a nearest-neighbor query). Parcels are
        # never merged into a land shape: that would treat them as one
        # unit, which this display step has no need for.
        parcels = water = None
        if has_land:
            parcels = gpd.GeoSeries(land.geometry.to_crs(units.crs).to_numpy())
        if has_lakes:
            water = gpd.GeoSeries(lakes.to_crs(units.crs).to_numpy())
        step = 25.0
        kept = []
        for line in borders:
            pieces = shapely.segmentize(line, step)
            coords = np.asarray(pieces.coords)
            segments = shapely.linestrings(np.stack([coords[:-1], coords[1:]], axis=1))
            middles = shapely.line_interpolate_point(segments, 0.5, normalized=True)
            on_land = np.ones(len(segments), dtype=bool)
            if parcels is not None:
                hit, _ = parcels.sindex.nearest(
                    middles, max_distance=open_water_m, return_all=False
                )
                on_land[:] = False
                on_land[hit] = True
            if water is not None:
                wet, _ = water.sindex.query(middles, predicate='within')
                on_land[wet] = False
            # Runs of consecutive pieces on land become one line.
            run = []
            for segment, keep in zip(segments, on_land, strict=True):
                if keep:
                    run.append(segment)
                elif run:
                    kept.append(shapely.line_merge(shapely.union_all(run)))
                    run = []
            if run:
                kept.append(shapely.line_merge(shapely.union_all(run)))
        borders = gpd.GeoSeries(
            [
                part
                for geom in kept
                for part in shapely.get_parts(geom)
                if part.geom_type == 'LineString' and part.length > step
            ],
            crs=units.crs,
        )
    return list(borders.to_crs(4326))


def border_paths(lines, land) -> list:
    """Border lines as page paths, set on the ground of the scene.

    A border has no height of its own. On a draped scene each vertex
    takes the ground height (`z`) of the nearest parcel vertex, so the
    line runs roughly flush with the bottoms of the parcels beside it
    rather than at sea level beneath them; on a flat scene `z` is 0.
    The page draws borders with depth testing, so parcels and
    buildings in front of a border hide it.

    Parameters
    ----------
    lines : sequence of shapely.LineString
        In longitude and latitude (`land_borders`).
    land : geopandas.GeoDataFrame
        The scene's bottom layer as rendered (draped parcels carry Z).

    Returns
    -------
    list of list
        Per line, `[longitude, latitude, z]` per vertex.
    """
    from scipy.spatial import cKDTree

    if not lines:
        return []
    ground = land.geometry.to_crs(4326).to_numpy()
    tree = None
    if len(ground) and shapely.has_z(ground).any():
        vertices = shapely.get_coordinates(ground, include_z=True)
        vertices = vertices[np.isfinite(vertices[:, 2])]
        # Degrees scaled to about equal meters for the nearest query.
        scale = np.cos(np.radians(vertices[:, 1].mean()))
        tree = cKDTree(np.column_stack([vertices[:, 0] * scale, vertices[:, 1]]))
    paths = []
    for line in lines:
        xy = np.asarray(line.coords)[:, :2]
        z = np.zeros(len(xy))
        if tree is not None:
            _, nearest = tree.query(np.column_stack([xy[:, 0] * scale, xy[:, 1]]))
            z = vertices[nearest, 2]
        paths.append(np.column_stack([np.round(xy, 6), np.round(z, 1)]).tolist())
    return paths


def tour_foci(
    gdf,
    value_column,
    names,
    n_foci=6,
    cell_m=400.0,
    min_separation_m=1500.0,
    min_parcels=20,
) -> list[dict]:
    """Points of greatest value density, for the page's opening flyover.

    The tour circles places a visitor recognizes as centers. Rather
    than naming them (which would put a place's geography in code),
    they are read off the data: the scene is cut into square cells,
    each cell's value is summed over its area, and the densest cells
    are kept. Every unit contributes its densest cell, so each town
    center is visited; further cells, at least `min_separation_m` from
    any kept one, fill up to `n_foci`. The densest cell of all comes
    first, so the tour lands on the main downtown before anything
    else, and the rest follow in nearest-neighbor order, so the flights
    between them stay short.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The rendered bottom layer (parcels), with `admin3_id`.
    value_column : str
        Value to sum per cell. The bottom layer's value (land) picks
        a district of costly land, which need not be the center a
        visitor knows; so `total_value` (land and buildings) is used
        whenever the table has it, and this column otherwise.
        Measured 2026-10-15 on boston-inner-core: land value alone put
        the first stop in Kendall Square, total value in downtown
        Boston.
    names : dict of str to str
        Unit names by `admin3_id`, for labels.
    n_foci : int
        Most points to return.
    cell_m : float
        Cell size in meters; about a few city blocks.
    min_separation_m : float
        Least distance between two points. Measured 2026-10-15 on
        boston-inner-core: at 2500 m, Kendall Square (2.4 km from
        downtown Boston) was refused and Cambridge's stop fell on its
        edge.
    min_parcels : int
        Least parcels in a cell. A center is many parcels; without
        this, one vast parcel (an airport, a rail yard) wins its cell.

    Returns
    -------
    list of dict
        `name`, `longitude`, `latitude`, tour order first to last.
    """
    if gdf.empty or n_foci < 1:
        return []
    # Cells in meters: the layer's own CRS when it is projected (the
    # bundles are), its UTM zone otherwise.
    projected = (
        gdf
        if gdf.crs is not None and gdf.crs.is_projected
        else gdf.to_crs(gdf.estimate_utm_crs())
    )
    points = projected.geometry.centroid
    cells = pd.DataFrame(
        {
            'i': np.floor(points.x / cell_m).astype(int),
            'j': np.floor(points.y / cell_m).astype(int),
            'value': gdf[
                'total_value' if 'total_value' in gdf.columns else value_column
            ]
            .fillna(0)
            .to_numpy(dtype=float),
            'admin3_id': gdf['admin3_id'].to_numpy(),
        }
    )
    summed = (
        cells.groupby(['i', 'j'])
        .agg(
            value=('value', 'sum'),
            n=('value', 'size'),
            admin3_id=('admin3_id', lambda s: s.mode()[0]),
        )
        .reset_index()
        .query('n >= @min_parcels')
        .sort_values('value', ascending=False)
    )
    if summed.empty:
        return []
    summed['x'] = (summed['i'] + 0.5) * cell_m
    summed['y'] = (summed['j'] + 0.5) * cell_m

    best_per_unit = summed.drop_duplicates('admin3_id')
    chosen = []

    def far_enough(row):
        return all(
            np.hypot(row.x - c.x, row.y - c.y) >= min_separation_m for c in chosen
        )

    for row in best_per_unit.itertuples():
        if len(chosen) < n_foci and far_enough(row):
            chosen.append(row)
    for row in summed.itertuples():
        if len(chosen) >= n_foci:
            break
        if far_enough(row):
            chosen.append(row)

    # Densest first, then always the nearest one not yet visited.
    chosen.sort(key=lambda r: -r.value)
    ordered = [chosen.pop(0)]
    while chosen:
        last = ordered[-1]
        nearest = min(chosen, key=lambda r: np.hypot(r.x - last.x, r.y - last.y))
        chosen.remove(nearest)
        ordered.append(nearest)

    lonlat = gpd.GeoSeries(
        gpd.points_from_xy([r.x for r in ordered], [r.y for r in ordered]),
        crs=projected.crs,
    ).to_crs(4326)
    return [
        {
            'name': names.get(row.admin3_id, row.admin3_id),
            # The page loads this unit first when it is the first stop.
            'admin3_id': row.admin3_id,
            'longitude': round(point.x, 6),
            'latitude': round(point.y, 6),
        }
        for row, point in zip(ordered, lonlat, strict=True)
    ]


# Least ratio of the two neighborhoods' typical value per home for the
# tour to say aloud that costly and modest homes stand close together
# (maintainer, 2026-10-18: "only where that claim is true"). Measured
# 2026-10-18 on the published scenes: 2.9 (Asheville) to 12.6
# (Greensboro); at 3, Asheville's stop stays silent.
CONTRAST_CLAIM_RATIO = 3.0


def contrast_focus(
    gdf,
    column,
    cell_m=1000.0,
    radius_m=3000.0,
    min_homes=200,
    shown=None,
    near=None,
    near_m=4000.0,
) -> dict | None:
    """The point where homes' value per dwelling differs most nearby.

    The opening tour switches to the per-dwelling view once, over the
    sharpest contrast between costly and inexpensive homes it can show
    in one view (maintainer, 2026-10-17). The scene is cut into square
    cells; each cell's homes give a median value per dwelling (on a log
    scale, so a ratio counts the same at any price level); of all pairs
    of cells within `radius_m` of each other, the pair whose medians
    differ most wins, and the tour circles the point halfway between
    them. Medians over at least `min_homes` homes keep one odd lot from
    deciding it.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        A rendered layer with a per-dwelling value column, NaN where a
        feature is not a home.
    column : str
        The per-dwelling value column.
    cell_m : float
        Cell size in meters; about a neighborhood. Measured 2026-10-17
        on boston-inner-core: 500 m cells put the stop between Bay State
        Road and the Fenway's edge, two blocks of institutional and
        student housing, not two neighborhoods.
    radius_m : float
        Farthest apart two cells may be; about what one view shows.
    min_homes : int
        Least homes in a cell, so that a cell is a neighborhood's homes
        and not one complex.
    shown : str, optional
        The per-dwelling column the page colors the homes by at that
        stop (land value per home, while buildings are hidden). Its
        median in each of the two cells is returned, so the page can
        name the colors the visitor sees there.
    near : tuple of float, optional
        (longitude, latitude) the tour comes from (downtown). The pair
        whose midpoint lies within `near_m` of it is preferred, so the
        tour reaches the contrast within seconds (maintainer,
        2026-10-18); without such a pair, the sharpest anywhere.
    near_m : float
        How far from `near` a preferred midpoint may lie.

    Returns
    -------
    dict or None
        `longitude`, `latitude`, `ratio` (the higher median over the
        lower) and `km` between the two cells; `high` and `low`, each
        [longitude, latitude, median]; with `shown`, `high_shown` and
        `low_shown`; and `claim`, whether the page may say that costly
        and modest homes stand close (`ratio` at least
        `CONTRAST_CLAIM_RATIO` and, with `shown`, the costly cell also
        colored higher); with `near`, `near` (whether the pair lies
        within `near_m`). None where no two cells qualify.
    """
    if gdf.empty or column not in gdf.columns:
        return None
    values = gdf[column].to_numpy(dtype=float)
    homes = np.isfinite(values) & (values > 0)
    if not homes.any():
        return None
    projected = (
        gdf[homes]
        if gdf.crs is not None and gdf.crs.is_projected
        else gdf[homes].to_crs(gdf[homes].estimate_utm_crs())
    )
    points = projected.geometry.centroid
    home_cells = pd.DataFrame(
        {
            'i': np.floor(points.x / cell_m).astype(int),
            'j': np.floor(points.y / cell_m).astype(int),
            'log_value': np.log10(values[homes]),
        }
    )
    cells = (
        home_cells.groupby(['i', 'j'])['log_value']
        .agg(['median', 'size'])
        .reset_index()
    )
    cells = cells[cells['size'] >= min_homes]
    if len(cells) < 2:
        return None
    x = ((cells['i'] + 0.5) * cell_m).to_numpy()
    y = ((cells['j'] + 0.5) * cell_m).to_numpy()
    median = cells['median'].to_numpy()
    # All pairs at once: a few thousand cells make a matrix of a few
    # million entries.
    close = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) <= radius_m
    gap = np.where(close, np.abs(median[:, None] - median[None, :]), -1.0)
    reached = None
    if near is not None:
        # Pairs whose midpoint lies near where the tour comes from.
        origin = (
            gpd.GeoSeries([shapely.Point(*near)], crs=4326)
            .to_crs(projected.crs)
            .iloc[0]
        )
        midpoint_m = np.hypot(
            (x[:, None] + x[None, :]) / 2 - origin.x,
            (y[:, None] + y[None, :]) / 2 - origin.y,
        )
        reached = bool(((midpoint_m <= near_m) & (gap >= 0)).any())
        if reached:
            gap = np.where(midpoint_m <= near_m, gap, -1.0)
    a, b = np.unravel_index(np.argmax(gap), gap.shape)
    if median[a] < median[b]:
        a, b = b, a
    points = gpd.GeoSeries(
        gpd.points_from_xy(
            [(x[a] + x[b]) / 2, x[a], x[b]], [(y[a] + y[b]) / 2, y[a], y[b]]
        ),
        crs=projected.crs,
    ).to_crs(4326)
    middle, high, low = points
    ratio = float(10 ** gap[a, b])
    out = {
        'longitude': round(middle.x, 6),
        'latitude': round(middle.y, 6),
        'ratio': round(ratio, 1),
        'km': round(float(np.hypot(x[a] - x[b], y[a] - y[b])) / 1000, 1),
        # The two cells and their medians.
        'high': [round(high.x, 5), round(high.y, 5), round(float(10 ** median[a]))],
        'low': [round(low.x, 5), round(low.y, 5), round(float(10 ** median[b]))],
    }
    claim = ratio >= CONTRAST_CLAIM_RATIO
    if shown is not None and shown in gdf.columns:
        colored = gdf[shown].to_numpy(dtype=float)[homes]

        def cell_median(k):
            inside = (home_cells['i'].to_numpy() == cells['i'].iloc[k]) & (
                home_cells['j'].to_numpy() == cells['j'].iloc[k]
            )
            return float(np.nanmedian(colored[inside]))

        out['high_shown'] = round(cell_median(a))
        out['low_shown'] = round(cell_median(b))
        # The colors must say what the words say.
        claim = claim and out['high_shown'] > out['low_shown']
    out['claim'] = bool(claim)
    if reached is not None:
        out['near'] = reached
    return out


def layer_region(recipe, region) -> str:
    """The region a layer reads: a team region where its recipe ships
    one, else that region's public bundle.

    A team page's footprints have no team twin when no source of them
    is cleared for team sharing; they then come from the public bundle,
    which withholds as much or more.
    """
    from openplaces.io.delivery import delivery_regions

    region = str(region)
    if not region.endswith(TEAM_SUFFIX):
        return region
    declared = {str(spec.get('region_id')) for spec in delivery_regions(recipe)}
    return region if region in declared else region.removesuffix(TEAM_SUFFIX)


def model_view_arrays(
    gdf, layer, homes=None, home_view=None, dollars=None
) -> tuple[dict, dict]:
    """Per-feature arrays and manifest entries of a layer's model views.

    Each view's scale is the median, over the lots with both, of the
    recorded land value per hectare (`ModelView.scale_column`, the
    assessor's own figure, never openplaces' fills; maintainer,
    2026-10-18) over the predicted one: one number per page and view.
    With `homes`, each model also gets a per-home view (maintainer,
    2026-10-18): the lot's scaled model land value over its homes,
    on the same lots and scale as the layer's own per-home view.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The rendered layer.
    layer : SceneLayer
        Its `model_views` and `area_unit`.
    homes : numpy.ndarray, optional
        Dwellings per feature where the layer's per-home view counts it
        as homes, NaN elsewhere.
    home_view : dict, optional
        The layer's per-home view (`vmin`, `vmax`, `other`), whose scale
        the models' per-home views share.
    dollars : numpy.ndarray, optional
        Each lot's factor from the prediction to today's dollars
        (`lot_dollars_factors`). With it, each model also gets views in
        today's dollars (`<key>__2026`, `<key>__2026__home`) beside the
        ones scaled to the assessments; a lot without a factor has no
        value there.

    Returns
    -------
    tuple of (dict, dict)
        Arrays for the scene file (`value_<view>` for every view;
        `aoa_<key>`, 2 inside the area of applicability, 1 outside, 0
        unknown) and views for the manifest (`label`, `unit`, `model`,
        `scale` or `dollars`; a per-home view also `per_home` and
        `base`, the per-area view of the same scale). A model without
        predictions, or without a recorded value to scale it on, is
        left out.
    """
    metric = (
        gdf.geometry
        if gdf.crs.is_projected
        else gdf.geometry.to_crs(gdf.estimate_utm_crs())
    )
    hectares = metric.area.to_numpy() / 1e4
    arrays, views = {}, {}
    for view in layer.model_views:
        if view.log_column not in gdf.columns or view.scale_column not in gdf.columns:
            continue
        recorded = gdf[view.scale_column].to_numpy(dtype=float)
        land_per_ha = np.where(hectares > 0, recorded / hectares, np.nan)
        predicted = np.exp(gdf[view.log_column].to_numpy(dtype=float))
        both = np.isfinite(predicted) & (predicted > 0) & (land_per_ha > 0)
        if not both.any():
            continue
        scale = float(np.median(land_per_ha[both] / predicted[both]))
        if view.aoa_column and view.aoa_column in gdf.columns:
            aoa = gdf[view.aoa_column].to_numpy(dtype=float)
            arrays[f'aoa_{view.key}'] = np.where(
                np.isfinite(aoa), np.where(aoa <= 1, 2, 1), 0
            ).astype(np.uint8)
        # Per hectare, scaled to the assessments; and in today's dollars.
        scalings = [(view.key, predicted * scale, {'scale': round(scale, 3)})]
        if dollars is not None and np.isfinite(dollars).any():
            scalings.append(
                (f'{view.key}__2026', predicted * dollars, {'dollars': True})
            )
        for key, per_ha, note in scalings:
            arrays[f'value_{key}'] = per_ha * HECTARES_PER_UNIT[layer.area_unit]
            views[key] = {
                'label': view.label,
                'unit': layer.area_unit,
                'model': True,
                **note,
            }
            if homes is not None and home_view is not None:
                arrays[f'value_{key}__home'] = np.where(
                    homes > 0, per_ha * hectares / homes, np.nan
                )
                views[f'{key}__home'] = {
                    **{
                        k: home_view[k]
                        for k in ('vmin', 'vmax', 'other')
                        if k in home_view
                    },
                    'label': view.label,
                    'unit': 'dwelling',
                    'model': True,
                    'per_home': True,
                    'base': key,
                    **note,
                }
    return arrays, views


def residential_lots(lots) -> np.ndarray:
    """Which lots the per-home view treats as homes.

    Residential only where two sources agree, the roll's land use and
    dwellings (its own count, or homes standing on it). A park, river
    bank or public campus holding a stray residential outline stays
    non-residential (maintainer, 2026-10-16: the Esplanade, the Fens).

    A third check: the roll states dwellings itself, or its use group
    is residential or mixed, or the class names one home per unit
    (`SINGLE_HOME_LAND_USE`) on a lot that is not exempt. A land use of
    "Multi-Family" on an exempt or unclassed lot with one outline
    counted as a home is not enough (125 Arborway, the Arnold
    Arboretum: 157 acres, one counted dwelling; maintainer,
    2026-10-16). Public housing keeps its roll's own count; a
    homeowners' association's common land fails the use-group test.
    Rolls that keep their own use codes in `use_group` ("SFR GOLF",
    "PATIO HOME") and state no dwellings failed the test on every
    house before the class could confirm itself (Brunswick County NC;
    maintainer, 2026-10-18).

    Parameters
    ----------
    lots : pandas.DataFrame
        `n_dwellings`, and where present `land_use_class`, `use_group`
        and the roll's own count (`ROLL_DWELLINGS`).

    Returns
    -------
    numpy.ndarray of bool
    """
    empty = pd.Series(pd.NA, index=lots.index, dtype='string')
    dwellings = lots['n_dwellings'].to_numpy(dtype=float)
    land_use = (
        lots['land_use_class'].astype('string') if 'land_use_class' in lots else empty
    )
    roll_count = (
        lots[ROLL_DWELLINGS].to_numpy(dtype=float)
        if ROLL_DWELLINGS in lots
        else np.full(len(lots), np.nan)
    )
    use = (
        lots['use_group'].astype('string').str.strip().str.lower()
        if 'use_group' in lots
        else empty
    )
    use_says_home = (
        (use.str.startswith('res') | use.str.contains('mixed')).fillna(False).to_numpy()
    )
    exempt = use.str.startswith('exempt').fillna(False).to_numpy()
    own_class = land_use.isin(SINGLE_HOME_LAND_USE).fillna(False).to_numpy()
    confirmed = (roll_count > 0) | use_says_home | (own_class & ~exempt)
    return (
        (dwellings > 0)
        & land_use.isin(RESIDENTIAL_LAND_USE).fillna(False).to_numpy()
        & confirmed
    )


def downtown_stop(downtown, admin) -> dict | None:
    """The tour's downtown stop for the manifest, with the unit holding
    it (`admin3_id`), which the page then loads first; None without
    one."""
    if downtown is None:
        return None
    point = shapely.Point(float(downtown[0]), float(downtown[1]))
    units = admin.to_crs(4326) if admin.crs is not None else admin
    holding = units.loc[units.contains(point), 'admin3_id']
    return {
        'longitude': round(point.x, 6),
        'latitude': round(point.y, 6),
        'admin3_id': str(holding.iloc[0]) if len(holding) else None,
    }


def scene_audience(region) -> str:
    """'team' when any region read is a team bundle, else 'public'.

    Derived from the regions rather than passed as a flag, so a scene
    that read team data cannot be marked public by mistake.
    """
    regions = [] if region is None else [region] if isinstance(region, str) else region
    return 'team' if any(str(r).endswith(TEAM_SUFFIX) for r in regions) else 'public'


def build_web_scene(
    region,
    admin_ids,
    output_dir,
    layers=VALUE_SCENE_LAYERS,
    title=None,
    build_id=None,
    elevation_recipe=None,
    terrain_exaggeration=1.0,
    n_foci=6,
    palettes=None,
    crop=None,
    local_only=False,
    landmarks=None,
    phone_scene=None,
    downtown=None,
    verbose=True,
) -> Path:
    """Write the static web scene of a region's delivery bundles.

    Parameters
    ----------
    region : str or list of str
        Region id(s); each layer's recipe must deliver them, and each
        unit is read from the region that pools it. None builds a
        **local preview** from the curated per-unit files instead: no
        delivery withholding applies, so the manifest is marked
        `preview` and `deploy_web_scene` refuses to publish it. A team
        region (`<region>.team`) marks the manifest `audience: team`,
        and `deploy_web_scene` then publishes it only behind a password.
    admin_ids : sequence of str
        Units (`admin3_id`) of the region to include. All of them load
        when the page opens.
    output_dir : str or Path
        Directory to write the site into. Existing files of an earlier
        build stay, so a deploy can keep serving them until the new
        page replaces the old.
    layers : sequence of SceneLayer
        The scene, bottom layer first. Defaults to parcels by land value
        with buildings by structure value on top.
    title : str, optional
        Page title. Defaults to `DEFAULT_TITLE`.
    build_id : str, optional
        Name of this build's data folder. Defaults to the UTC time.
    elevation_recipe : str, optional
        DEM recipe (e.g. `US_land-elevation-usgs-3dep`) to ground the
        scene in terrain. The bottom layer is draped (one polygon per
        feature, each vertex at its own ground height; the notebook's
        mesh mode falls back to the same at city scale), and stacked
        layers stand flat on it. None (default) keeps flat ground.
    terrain_exaggeration : float
        Multiplier on ground relief only, never on value height.
    n_foci : int
        Most points the opening flyover circles (`tour_foci`).
    palettes : sequence of Palette, optional
        Color options the page offers, the first one shown by default.
        Defaults to `default_palettes(layers)`.
    crop : shapely geometry, optional
        An area in longitude and latitude (EPSG:4326), e.g. a land
        market region: only features whose center lies inside it are
        drawn. Features are kept whole, never cut, so a lot on the edge
        is in or out by where its center falls. `admin_ids` then names
        the units to read, which may reach beyond the area.
    local_only : bool
        Mark the scene as never to be published (`deploy_web_scene`
        refuses it), e.g. when `crop` comes from a source that may not
        be redistributed.
    landmarks : sequence of dict, optional
        Named places drawn as labeled markers with a short tooltip:
        each with `name`, `kind` (a few words, e.g. "Airport"),
        `longitude` and `latitude`. A landmark is a fixed, editorial
        label, not a search result: the page offers no way to look one
        up. Its marker stands on the top of the tallest feature drawn
        within 40 m (`landmark_positions`).
    phone_scene : str, optional
        Relative address of a lighter page of the same place (e.g.
        `../boston-phone/`, a smaller circle). A phone-sized or
        low-memory device opening this page goes there instead, unless
        the address carries `full=1`. A phone held the full Boston scene
        poorly or crashed (maintainer's Pixel 9, 2026-10-17).
    downtown : tuple of float, optional
        (longitude, latitude) of the place's downtown, an editorial
        point: the tour's first stop, where it rests longest and hides
        the buildings (maintainer, 2026-10-18). None starts the tour at
        its densest stop instead.
    verbose : bool

    Returns
    -------
    Path
        The output directory.
    """
    # The import is here, not at module level, because lonboard (which
    # terrain imports) belongs to the optional viz-fast extra.
    from openplaces.viz.terrain import show_value_terrain_layer

    refuse_personal_fields(layers)
    palettes = list(palettes) if palettes else default_palettes(layers)
    admin_ids = list(admin_ids)
    output_dir = Path(output_dir)
    build_id = build_id or datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    # One flat data folder, files named by their content
    # (`_write_content_file`): a rebuild that changes no data writes and
    # uploads no data.
    data_dir = output_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    previous_manifest = _read_manifest(output_dir)

    admin = _admin_names(admin_ids)
    names = dict(zip(admin['admin3_id'], admin['name'], strict=True))

    # One datum for every layer, or they slide against each other
    # (`viz.elevation.get_elevation_datum`).
    ground = {}
    if elevation_recipe is not None:
        from openplaces.viz.elevation import get_elevation_datum

        ground = {
            'elevation_recipe': elevation_recipe,
            'elevation_datum': get_elevation_datum(
                admin, elevation_recipe, admin_id_column='admin3_id'
            ),
            'terrain_exaggeration': terrain_exaggeration,
        }

    rendered = {}
    # Per county, the factor of the models' views in today's dollars.
    model_dollars = {}
    estimated_by_layer = {}
    source_docs = {}
    bounds = {}
    manifest_layers = []
    notices = {}
    gdfs = {}
    for layer in layers:
        if region is None:
            from openplaces.io.readers import get_entities

            gdf = get_entities(layer.recipe, admin_id=admin_ids, geom=True)
            notices[layer.recipe] = 'Local preview from curated files; not published.'
        else:
            # Several regions: each unit is read from the bundle of the
            # region it belongs to. An entity on a boundary between two
            # regions can sit in both bundles; the first copy is kept.
            regions = [
                layer_region(layer.recipe, name)
                for name in ([region] if isinstance(region, str) else region)
            ]
            parts = []
            for name in regions:
                members = set(_region_members(layer.recipe, name))
                wanted = [a for a in admin_ids if a in members]
                if wanted:
                    part = read_bundle(layer.recipe, name, wanted)
                    if layer.recorded_column and layer.recorded_column not in part:
                        extra = read_evidence_columns(
                            layer.recipe, name, [layer.recorded_column]
                        )
                        part = part.join(extra, how='left')
                    parts.append(part)
            if not parts:
                raise ValueError(f'No unit of {admin_ids} is in {regions}.')
            gdf = pd.concat(parts)
            gdf = gdf[~gdf.index.duplicated(keep='first')]
            notices[layer.recipe] = '\n\n'.join(
                bundle_notice(layer.recipe, name) for name in regions
            )
        if crop is not None:
            gdf = crop_to_area(gdf, crop)
            if verbose:
                print(f'{layer.name}: {len(gdf):,} features inside the crop')
        if 'occupancy_type' in gdf.columns:
            gdf = gdf.assign(**{BUILDING_CLASS: building_class(gdf)})
        gdfs[layer.name] = merge_identical_outlines(gdf, layer)
    _mark_exempt_ground(gdfs, layers)
    _count_dwellings_from_above(gdfs, layers)

    for layer in layers:
        gdf = gdfs[layer.name]
        stack_on = rendered[layer.stack_on] if layer.stack_on else None
        terrain = show_value_terrain_layer(
            gdf,
            None,
            value_column=layer.value_column,
            area_unit=layer.area_unit,
            vmin=layer.vmin,
            vmax=layer.vmax,
            cmap=layer_cmap(layer),
            stack_on=stack_on,
            elevation_mode='drape' if ground and stack_on is None else 'flat',
            missing_value='render',
            silent=not verbose,
            **ground,
            **layer.render,
        )
        rendered[layer.name] = terrain
        # The Sources document: from the drawn outlines' own sources
        # where the layer records them, else from the bundle notices.
        source_docs[layer.name] = (
            scene_sources(layer.recipe, gdf['geometry_source'])
            if 'geometry_source' in gdf.columns and region is not None
            else notice_sources(notices.get(layer.recipe, ''))
        )
        provenance = f'{layer.value_column}{SOURCE_SUFFIX}'
        if provenance in gdf.columns:
            source_docs[layer.name]['values'] = value_sources(gdf[provenance])
        out = terrain.gdf
        base = terrain.total_elevation - terrain.rendered_elevation
        geometry = out.geometry
        if layer.simplify_m:
            metric = geometry.crs is not None and geometry.crs.is_projected
            work = geometry if metric else geometry.to_crs(out.estimate_utm_crs())
            work = gpd.GeoSeries(
                shapely.simplify(
                    work.to_numpy(), layer.simplify_m, preserve_topology=True
                ),
                index=work.index,
                crs=work.crs,
            )
            geometry = work
        geometry = geometry.to_crs(4326).to_numpy()
        raw_value = out[layer.value_column].to_numpy(dtype=float)
        views = {}
        extra = {}
        if layer.per_dwelling_label:
            # The value per dwelling: a display ratio of the feature's
            # own two fields, none where it has no dwellings.
            dwellings = out['n_dwellings'].to_numpy(dtype=float)
            has = (dwellings > 0) & (raw_value > 0)
            per_dwelling = np.full(len(raw_value), np.nan)
            per_dwelling[has] = raw_value[has] / dwellings[has]
            if BUILDING_CLASS in out.columns:
                occupancy = out[BUILDING_CLASS].astype('string')
                residential = occupancy.isin(
                    RESIDENTIAL_OCCUPANCY | RESIDENTIAL_LAND_USE
                ).to_numpy()
                known = occupancy.notna().to_numpy()
            else:
                residential = residential_lots(out)
                known = np.ones(len(out), dtype=bool)
            # Everything that is not a home and carries a value, accessory
            # types included: campus buildings classed "Secondary" on a
            # one-lot university carry $3M-$146M each (Harvard Yard,
            # seen 2026-10-16) and drew as "no value" when every
            # accessory type did.
            other = ~residential & known & (raw_value > 0)
            # Homes per dwelling only; everything else drawn otherwise.
            per_dwelling[~residential] = np.nan
            out[f'{layer.value_column}_per_dwelling'] = per_dwelling
            extra['value_dwelling'] = per_dwelling
            extra['nonresidential'] = other.astype(np.uint8)
            low, high = scene_bounds(per_dwelling)
            # The same bounds as the per-area view, so a building's
            # color means the same value in both. Bounds of the
            # non-residential values alone ran from $1 to $10,000 per
            # sq ft in Boston (2026-10-16), most of it washed out.
            per_area_all = out[f'{layer.value_column}_per_{layer.area_unit}'].to_numpy(
                dtype=float
            )
            other_low, other_high = scene_bounds(
                np.where(raw_value > 0, per_area_all, np.nan)
            )
            views['dwelling'] = {
                'label': layer.per_dwelling_label,
                'unit': 'dwelling',
                'vmin': low,
                'vmax': high,
                # Non-residential buildings, per area, own ramp.
                'other': {
                    # Short enough for one legend line with its unit.
                    'label': 'Non-residential '
                    + ('buildings' if layer.name == 'buildings' else 'land'),
                    'unit': layer.area_unit,
                    'vmin': other_low,
                    'vmax': other_high,
                },
            }
        if layer.model_views and scene_audience(region) == 'team':
            homes = None
            if 'dwelling' in views:
                # The lots the per-home view counts as homes.
                homes = np.where(
                    np.isfinite(
                        out[f'{layer.value_column}_per_dwelling'].to_numpy(dtype=float)
                    ),
                    out['n_dwellings'].to_numpy(dtype=float),
                    np.nan,
                )
            dollars, model_dollars = lot_dollars_factors(out)
            arrays, model_views = model_view_arrays(
                out,
                layer,
                homes=homes,
                home_view=views.get('dwelling'),
                dollars=dollars,
            )
            extra.update(arrays)
            views.update(model_views)
        # Which values openplaces filled itself (the `imputed` mark in
        # the value's `_source`); the page can draw them as no value.
        source = f'{layer.value_column}_source'
        estimated = (
            is_imputed(out[source]).to_numpy()
            if source in out.columns
            else np.zeros(len(out), dtype=bool)
        )
        extra['estimated'] = estimated.astype(np.uint8)
        if layer.recorded_column and layer.recorded_column in out.columns:
            # Per area as the source recorded it, where the shown value
            # was estimated; the same area as the shown value's ratio.
            recorded = out[layer.recorded_column].to_numpy(dtype=float)
            shown_per_area = out[
                f'{layer.value_column}_per_{layer.area_unit}'
            ].to_numpy(dtype=float)
            area = np.where(raw_value > 0, raw_value / shown_per_area, np.nan)
            extra['value_assessed'] = np.where(
                estimated & (recorded > 0) & (area > 0), recorded / area, np.nan
            )
            # A feature with a recorded value is not an estimate there.
            estimated = estimated & ~(recorded > 0)
            extra['estimated'] = estimated.astype(np.uint8)
        estimated_by_layer[layer.name] = estimated
        if layer.stack_on and estimated_by_layer.get(layer.stack_on) is not None:
            # Whether the feature below (whose extrusion lifts this one)
            # was estimated: in the assessment view that feature drops
            # to the ground, and this one with it. The same largest-
            # overlap match that set the lift (`stack_elevation`).
            from openplaces.viz.terrain import _match_largest_overlap

            below = rendered[layer.stack_on]
            extra['on_estimated'] = (
                _match_largest_overlap(
                    out,
                    below.gdf,
                    estimated_by_layer[layer.stack_on].astype(float),
                )
                > 0.5
            ).astype(np.uint8)
        # Outlines curate flags as drawn too tall for what is mapped
        # there (`GHOST_COLUMN`): the page draws them see-through at
        # full height, behind a switch of their own.
        ghost = (
            out[GHOST_COLUMN].fillna(False).astype(bool).to_numpy()
            if GHOST_COLUMN in out.columns
            else np.zeros(len(out), dtype=bool)
        )
        if ghost.any():
            extra['ghost'] = ghost.astype(np.uint8)
        values = _tooltip_values(out, layer)
        # No value: a dark slate that recedes into the page, not the
        # renderer's mid gray, which read as a dull patch in the scene.
        per_area = out[f'{layer.value_column}_per_{layer.area_unit}'].to_numpy(
            dtype=float
        )
        # Missing stays NaN ("no value"); a recorded zero is shipped as
        # 0, which the page draws in its own color (`zero_value_rgba`):
        # a public lot the assessor values at $0 is a fact, not a gap.
        recorded = out[layer.value_column].to_numpy(dtype=float)
        per_area = np.where(
            np.isnan(recorded), np.nan, np.where(recorded == 0, 0.0, per_area)
        )
        bounds[layer.name] = scene_bounds(np.where(per_area > 0, per_area, np.nan))
        files_by_unit = {}
        for admin_id in admin_ids:
            rows = (out['admin3_id'] == admin_id).to_numpy()
            if not rows.any():
                continue
            data, _ = encode_scene_file(
                geometry[rows],
                base[rows],
                terrain.rendered_elevation[rows],
                per_area[rows],
                stack=terrain.stack_elevation[rows] if layer.stack_on else None,
                extra={name: array[rows] for name, array in extra.items()},
            )
            stem = f'{layer.name}_{admin_id}'
            unit_values = {
                key: [v for v, keep in zip(column, rows, strict=True) if keep]
                for key, column in values.items()
            }
            files_by_unit[admin_id] = {
                'geometry': _write_content_file(data_dir, stem, data),
                'tooltip': _write_content_file(
                    data_dir,
                    f'{stem}_tooltip',
                    json.dumps(unit_values).encode('utf-8'),
                ),
            }
            if verbose:
                print(f'{stem}: {int(rows.sum()):,} features')
        manifest_layers.append(
            {
                'name': layer.name,
                'unit': layer.area_unit,
                'label': layer.label,
                'tooltip': [
                    {
                        'column': tip.column,
                        'label': tip.label,
                        'kind': tip.kind,
                        'imputed': tip.imputed_from is not None,
                    }
                    for tip in layer.tooltip
                ],
                'files': files_by_unit,
                # Other colorings of this layer (`per_dwelling_label`).
                'views': views,
                'n_estimated': int(estimated.sum()),
                'n_ghosts': int(ghost.sum()),
            }
        )

    land = rendered[layers[0].name].gdf
    borders = land_borders(admin_ids, land=land, lakes=lake_polygons(admin_ids))
    if crop is not None:
        # No border runs past the area's edge into ground the scene
        # does not draw.
        borders = [
            part
            for line in borders
            for part in shapely.get_parts(line.intersection(crop))
            if part.geom_type == 'LineString'
        ]
    boundaries = [{'path': path} for path in border_paths(borders, land)]
    marks = landmark_positions(
        landmarks or [], [rendered[layer.name] for layer in layers]
    )
    if verbose:
        for mark in marks:
            print(f'landmark {mark["name"]}: top {mark["position"][2]:.0f} m')

    center = admin.union_all().centroid
    bottom = layers[0]
    foci = tour_foci(
        rendered[bottom.name].gdf,
        bottom.value_column,
        names,
        n_foci=n_foci,
    )
    # One stop of the tour shows the per-dwelling view, where homes'
    # values differ most within one view (`contrast_focus`).
    contrast = None
    if any(layer.per_dwelling_label for layer in layers):
        lots = rendered[bottom.name].gdf
        per_dwelling = f'{bottom.value_column}_per_dwelling'
        # What the page colors the lots by at the stop.
        shown = lots[per_dwelling] if per_dwelling in lots.columns else None
        # A home's whole value (land and building) per dwelling where
        # the lot has both; land alone mostly measures density: on
        # 2026-10-17 it put Boston's stop at the Fenway, where a condo
        # unit's share of land is small whatever the unit is worth.
        if {'total_value', 'n_dwellings', per_dwelling} <= set(lots.columns):
            homes = np.isfinite(lots[per_dwelling].to_numpy(dtype=float))
            total = lots['total_value'].to_numpy(dtype=float) / lots[
                'n_dwellings'
            ].to_numpy(dtype=float)
            lots = lots[['geometry']].assign(
                value_per_dwelling=np.where(homes, total, np.nan)
            )
            per_dwelling = 'value_per_dwelling'
        if shown is not None:
            lots = lots.assign(shown_per_dwelling=shown.to_numpy(dtype=float))
        contrast = contrast_focus(
            lots,
            per_dwelling,
            shown='shown_per_dwelling' if shown is not None else None,
            near=downtown,
        )
        if verbose and contrast:
            print(f'per-dwelling tour stop: {contrast}')
    manifest = {
        'title': title or DEFAULT_TITLE,
        'subtitle': DEFAULT_SUBTITLE,
        'build_id': build_id,
        'built': datetime.now(UTC).isoformat(timespec='seconds'),
        'units': [names.get(a, a) for a in admin_ids],
        'view': {
            'longitude': round(center.x, 6),
            'latitude': round(center.y, 6),
            'zoom': 12,
            'pitch': 45,
            'bearing': 0,
        },
        'layers': manifest_layers,
        'boundaries': boundaries,
        # Per unit [west, south, east, north]: a device short of memory
        # loads only the units in view (viewer.js `constrained`).
        'unit_bounds': {
            row.admin3_id: [round(v, 5) for v in row.geometry.bounds]
            for row in admin.itertuples()
        },
        # A document, not the raw notices: those carry the builder's
        # recipe folders (`notice_sources`).
        'sources': [
            {'layer': layer.label, **source_docs[layer.name]} for layer in layers
        ],
        'preview': region is None,
        'local_only': bool(local_only),
        'audience': scene_audience(region),
        'landmarks': marks,
        'phone_scene': phone_scene,
        'known_issues': known_issues(admin_ids),
        'deck_gl_url': DECK_GL_URL,
        'palettes': [palette_manifest(p, layers, bounds) for p in palettes],
        # Shown first unless the visitor chose another before.
        'default_palette': (
            DEFAULT_PALETTE
            if any(p.id == DEFAULT_PALETTE for p in palettes)
            else palettes[0].id
        ),
        'tour': foci,
        'tour_contrast': contrast,
        'tour_downtown': downtown_stop(downtown, admin),
        # Team pages with models: the factors to today's dollars per
        # county, and the models' credits for the Sources panel.
        'model_dollars': model_dollars,
        'model_sources': (
            list(MODEL_SOURCE_NOTES)
            if any(
                v.get('model')
                for layer in manifest_layers
                for v in layer['views'].values()
            )
            else []
        ),
        'terrain': {
            'elevation_recipe': elevation_recipe,
            'terrain_exaggeration': terrain_exaggeration if ground else None,
        },
    }
    (output_dir / 'scene.json').write_text(json.dumps(manifest), encoding='utf-8')
    # Keep this build's data and the previous build's (a page opened
    # before a deploy still loads it); delete the rest.
    _prune_local_data(
        output_dir, manifest_files(manifest) | manifest_files(previous_manifest)
    )
    for item in STATIC_DIR.iterdir():
        if item.name in ('landing.html', 'methods.html'):
            # Site pages beside the scenes, written by `landing.py`.
            continue
        target = '.htaccess' if item.name == 'htaccess' else item.name
        (output_dir / target).write_bytes(item.read_bytes())
    if verbose:
        size = sum((output_dir / p).stat().st_size for p in manifest_files(manifest))
        size /= 2**20
        print(f'Wrote {output_dir} (build {build_id}, {size:.1f} MB of data)')
    return output_dir


# The deploy follows the maintainer's other sites on the same web host
# (the LakeSense dashboard's web/deploy.py, the PLACES-FMV docs): the
# target comes from the environment or an env file kept outside the
# repository, and rclone reaches the host over SFTP through a remote
# defined only in environment variables, so no rclone.conf entry is
# needed.
DEPLOY_PREFIX = 'OPENPLACES_WEB_DEPLOY_'
DEPLOY_REQUIRED = ('HOST', 'USER', 'PATH')
# Builds whose data stay online: this one and the one before, so a
# page opened before the switch keeps loading its own files.
KEEP_BUILDS = 2
# Shared hosting refuses connections once too many SSH sessions are
# open (Bluehost's cPHulk), and rclone's SFTP backend opens one per
# transfer and checker. Files are compared on size and modification
# time: `--checksum` would make the server run md5sum on every file.
# cPanel's AutoSSL writes certificate challenges to /.well-known/, which
# a deploy must never touch.
RCLONE_SFTP_FLAGS = (
    '--filter',
    '- /.well-known/**',
    '--transfers',
    '2',
    '--checkers',
    '2',
    '--tpslimit',
    '4',
    '--retries',
    '3',
    '--low-level-retries',
    '10',
)
# cPHulk refuses SSH for a while after a burst of connections; every
# step of the deploy is safe to repeat after a pause.
RETRY_TRIES = 4
RETRY_WAIT_S = 60


def default_deploy_env_file() -> Path:
    """`web-deploy.env` beside the user's `credentials.yaml`."""
    from openplaces.config import cfg

    return cfg.credentials_path.parent / 'web-deploy.env'


def load_deploy_settings(env_file=None) -> dict:
    """Read the web host's deploy settings.

    Variables already in the environment win over the file, so a CI
    secret beats a stale local copy.

    Parameters
    ----------
    env_file : str or Path, optional
        `KEY=value` lines. Defaults to `default_deploy_env_file`, in the
        user's openplaces config folder, outside every repository.
        Keys, all prefixed `OPENPLACES_WEB_DEPLOY_`:
        `HOST`, `USER` and `PATH` (absolute path of the site's folder on
        the host) are required; `KEY` (private key file),
        `KNOWN_HOSTS` (pinned host key file) and `URL` (public address)
        are optional, and so is `TEAM_PASSWD`: the absolute path, on the
        host, of the password file guarding team pages
        (`deploy_web_scene`). The maintainer creates that file on the
        host (cPanel's "Directory Privacy"), so no password passes
        through openplaces.

    Returns
    -------
    dict
        `host`, `user`, `path`, `key`, `known_hosts`, `url`,
        `team_passwd`.
    """
    import os

    env_file = Path(env_file) if env_file else default_deploy_env_file()
    values = {}
    if env_file.exists():
        for raw in env_file.read_text(encoding='utf-8').splitlines():
            line = raw.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip().strip('"\'')
    values.update({k: v for k, v in os.environ.items() if k.startswith(DEPLOY_PREFIX)})

    missing = [
        DEPLOY_PREFIX + k for k in DEPLOY_REQUIRED if not values.get(DEPLOY_PREFIX + k)
    ]
    if missing:
        raise ValueError(
            f'Missing deploy settings {", ".join(missing)}. Set them in the '
            f'environment or in {env_file}, e.g.\n'
            f'  {DEPLOY_PREFIX}HOST=<the host your other sites deploy to>\n'
            f'  {DEPLOY_PREFIX}USER=<ssh user>\n'
            f'  {DEPLOY_PREFIX}PATH=/home/<user>/public_html/<folder>\n'
            f'  {DEPLOY_PREFIX}KEY=C:/Users/<you>/.ssh/<deploy key>\n'
            f'  {DEPLOY_PREFIX}KNOWN_HOSTS=C:/Users/<you>/.ssh/<known hosts>\n'
            f'  {DEPLOY_PREFIX}URL=https://<domain>/<folder>'
        )

    def _file(name):
        value = values.get(DEPLOY_PREFIX + name, '')
        if not value:
            return ''
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f'{DEPLOY_PREFIX}{name} names no file: {value}')
        return str(path)

    return {
        'host': values[DEPLOY_PREFIX + 'HOST'],
        'user': values[DEPLOY_PREFIX + 'USER'],
        'path': values[DEPLOY_PREFIX + 'PATH'].rstrip('/'),
        'key': _file('KEY'),
        'known_hosts': _file('KNOWN_HOSTS'),
        'url': values.get(DEPLOY_PREFIX + 'URL', '').rstrip('/'),
        'team_passwd': values.get(DEPLOY_PREFIX + 'TEAM_PASSWD', ''),
    }


def team_htaccess(passwd_path) -> str:
    """The `.htaccess` of a team page: a password, then the usual rules.

    Apache's basic authentication in the page folder covers its `data/`
    folder too, so no file of a team page is served without the
    password. Crawlers are asked not to index what they cannot open
    anyway.

    Parameters
    ----------
    passwd_path : str
        Absolute path of the password file on the host.

    Returns
    -------
    str
        The file's text.
    """
    if not passwd_path.startswith('/') or re.search(r'["\s]', passwd_path):
        raise ValueError(
            f'{DEPLOY_PREFIX}TEAM_PASSWD must be an absolute path on the host '
            f'without spaces or quotes, got {passwd_path!r}.'
        )
    guard = (
        '# A team page: readable only with the team password.\n'
        'AuthType Basic\n'
        'AuthName "openplaces team page"\n'
        f'AuthUserFile "{passwd_path}"\n'
        'Require valid-user\n'
        '<IfModule mod_headers.c>\n'
        '  Header set X-Robots-Tag "noindex, nofollow"\n'
        '</IfModule>\n\n'
    )
    return guard + (STATIC_DIR / 'htaccess').read_text(encoding='utf-8')


def _rclone_env(settings) -> dict:
    """Environment defining the `bh:` SFTP remote for one rclone call."""
    import os

    env = os.environ.copy()
    env['RCLONE_CONFIG_BH_TYPE'] = 'sftp'
    env['RCLONE_CONFIG_BH_HOST'] = settings['host']
    env['RCLONE_CONFIG_BH_USER'] = settings['user']
    # Skip rclone's interactive shell probe on shared hosting.
    env['RCLONE_CONFIG_BH_SHELL_TYPE'] = 'unix'
    if settings['key']:
        env['RCLONE_CONFIG_BH_KEY_FILE'] = settings['key']
    if settings['known_hosts']:
        path = Path(settings['known_hosts'])
        env['RCLONE_CONFIG_BH_KNOWN_HOSTS_FILE'] = str(path)
        algorithms = _known_hosts_algorithms(path)
        if algorithms:
            env['RCLONE_CONFIG_BH_HOST_KEY_ALGORITHMS'] = ' '.join(algorithms)
    return env


def _known_hosts_algorithms(path: Path) -> list[str]:
    """Host key algorithms a known_hosts file pins.

    rclone otherwise negotiates whichever algorithm the server offers
    first, which can differ from the pinned key's and then fails the
    host check. An `ssh-rsa` key is also offered under its SHA-2
    signature names, which current servers require.
    """
    algorithms = []
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return []
    for line in lines:
        if line.startswith(('#', '@')):
            continue
        fields = line.split()
        if len(fields) >= 3 and fields[1] not in algorithms:
            algorithms.append(fields[1])
    if 'ssh-rsa' in algorithms:
        i = algorithms.index('ssh-rsa')
        algorithms[i : i + 1] = ['rsa-sha2-512', 'rsa-sha2-256', 'ssh-rsa']
    return algorithms


def deploy_web_scene(
    output_dir, env_file=None, go=False, keep_builds=KEEP_BUILDS, subpath=None
):
    """Upload a built scene to the web host with `rclone` over SFTP.

    Nothing is written unless `go` is True: the default is a dry run
    that lists what would be added, changed or deleted.

    Three phases, so a visitor never sees a half-updated site:

    1. data: files the host lacks are added to `data/`. Data files are
       named by their content (`_write_content_file`), so one already
       online is never sent again, and nothing a page already loads
       changes;
    2. switch: the page, its script, the manifest and `.htaccess`,
       which point at the new files;
    3. clean up: data files that neither this build nor the build it
       replaced loads are deleted (with `keep_builds` of 1, only this
       build's are kept), so a page opened just before the switch still
       finds its files.

    A team scene (`audience: team`, built from a `<region>.team` bundle)
    is published only behind a password: its `.htaccess`
    (`team_htaccess`) goes up before anything else, and the deploy
    refuses when `TEAM_PASSWD` is not set or names no file on the host.

    Parameters
    ----------
    output_dir : str or Path
        Directory written by `build_web_scene`.
    env_file : str or Path, optional
        Deploy settings (`load_deploy_settings`).
    go : bool
        Upload for real.
    keep_builds : int
        Builds whose data stay online, this one included.
    subpath : str, optional
        Folder under the configured path, e.g. a place's slug beside the
        landing page (`/values/boston/`).

    Returns
    -------
    str
        The public address, or the remote path when no URL is set.
    """
    output_dir = Path(output_dir)
    if not (output_dir / 'index.html').exists():
        raise FileNotFoundError(f'No built scene in {output_dir}.')
    manifest = json.loads((output_dir / 'scene.json').read_text(encoding='utf-8'))
    build_id = manifest['build_id']
    if manifest.get('preview'):
        raise ValueError(
            f'{output_dir} is a local preview built from curated files, '
            'without the delivery withholding; it is never published. '
            'Build from a delivery region to publish.'
        )
    if manifest.get('local_only'):
        raise ValueError(
            f'{output_dir} was built for local use only (`local_only`, e.g. '
            'cropped to an area that may not be redistributed); it is never '
            'published.'
        )
    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    if subpath:
        subpath = subpath.strip('/')
        settings = {
            **settings,
            'path': f'{settings["path"]}/{subpath}',
            'url': f'{settings["url"]}/{subpath}' if settings['url'] else '',
        }
    target = f'bh:{settings["path"]}'
    dry = [] if go else ['--dry-run']
    print(f'Target: {settings["user"]}@{settings["host"]}:{settings["path"]}')
    if settings['url']:
        print(f'URL   : {settings["url"]}/')
    print(f'Mode  : {"PUBLISH" if go else "dry run"}')

    team = manifest.get('audience') == 'team'
    if team:
        passwd = settings['team_passwd']
        if not passwd:
            raise ValueError(
                f'{output_dir} is a team page and is published only behind a '
                f'password. Create the password file on the host and name it '
                f'in {DEPLOY_PREFIX}TEAM_PASSWD.'
            )
        # Written before the check, so a bad path fails without a call
        # to the host.
        (output_dir / '.htaccess').write_text(team_htaccess(passwd), encoding='utf-8')
        if not _rclone_output(env, 'lsf', f'bh:{passwd}'):
            raise FileNotFoundError(
                f'{DEPLOY_PREFIX}TEAM_PASSWD names no file on the host '
                f'({passwd}); the team page was not published.'
            )
        print('\nPhase 0: password (.htaccess) before any file of the page')
        _rclone(
            env,
            'copyto',
            str(output_dir / '.htaccess'),
            f'{target}/.htaccess',
            *_without_filters(RCLONE_SFTP_FLAGS),
            *dry,
        )
    elif 'AuthUserFile' in (output_dir / '.htaccess').read_text(encoding='utf-8'):
        raise ValueError(
            f'{output_dir} is a public page but its .htaccess asks for a '
            'password; rebuild it.'
        )

    # The build currently online, read before the switch, stays.
    previous = _rclone_output(env, 'cat', f'{target}/scene.json')
    try:
        previous_manifest = json.loads(previous) if previous else None
    except ValueError:
        previous_manifest = None
    wanted = manifest_files(manifest)

    print(f'\nPhase 1: data of build {build_id} ({len(wanted)} files)')
    with tempfile.TemporaryDirectory() as tmp:
        # Only this build's files, each sent once ever: a name already
        # on the host holds the same bytes.
        listing = Path(tmp) / 'files.txt'
        listing.write_text(
            '\n'.join(sorted(p.removeprefix('data/') for p in wanted)) + '\n',
            encoding='utf-8',
        )
        _rclone(
            env,
            'copy',
            str(output_dir / 'data'),
            f'{target}/data',
            '--files-from',
            str(listing),
            '--ignore-existing',
            # rclone refuses `--files-from` beside other filters; the
            # list names only data files, so the `/.well-known/` guard
            # of the shared flags has nothing to protect here.
            *_without_filters(RCLONE_SFTP_FLAGS),
            *dry,
        )
    print('\nPhase 2: switch (index.html, viewer.js, scene.json, .htaccess)')
    _rclone(
        env,
        'copy',
        str(output_dir),
        target,
        '--max-depth',
        '1',
        *RCLONE_SFTP_FLAGS,
        '--filter',
        '+ /{index.html,viewer.js,scene.json,.htaccess}',
        '--filter',
        '- **',
        *dry,
    )

    keep = set(wanted)
    if keep_builds > 1:
        keep |= manifest_files(previous_manifest)
    online = _rclone_output(
        env,
        'lsf',
        '--files-only',
        '--recursive',
        f'{target}/data',
        '--checkers',
        '2',
        '--tpslimit',
        '4',
    )
    stale = sorted(
        name for name in (online or '').split() if name and f'data/{name}' not in keep
    )
    print(f'\nPhase 3: clean up ({len(stale)} file(s) no page loads)')
    if stale:
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / 'stale.txt'
            listing.write_text('\n'.join(stale) + '\n', encoding='utf-8')
            _rclone(
                env,
                'delete',
                f'{target}/data',
                '--files-from',
                str(listing),
                '--checkers',
                '2',
                '--tpslimit',
                '4',
                *dry,
            )
        # Folders of builds from before content-named files, now empty.
        _rclone(
            env,
            'rmdirs',
            f'{target}/data',
            '--leave-root',
            '--checkers',
            '2',
            '--tpslimit',
            '4',
            *dry,
        )
    address = settings['url'] or settings['path']
    print(f'\n{"Published" if go else "Dry run complete"}: {address}/')
    return address


def deploy_landing(output_dir, env_file=None, go=False):
    """Upload a landing page (`landing.build_landing`) to the configured
    path: its page, card images and list of places only; the scenes in
    the folders beside it are never touched."""
    output_dir = Path(output_dir)
    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    dry = [] if go else ['--dry-run']
    print(f'Target: {settings["user"]}@{settings["host"]}:{settings["path"]}')
    _rclone(
        env,
        'copy',
        str(output_dir),
        f'bh:{settings["path"]}',
        *RCLONE_SFTP_FLAGS,
        '--filter',
        '+ /index.html',
        '--filter',
        '+ /thumbs/**',
        '--filter',
        '+ /places.json',
        '--filter',
        '+ /methods.html',
        '--filter',
        '- **',
        *dry,
    )
    address = settings['url'] or settings['path']
    print(f'{"Published" if go else "Dry run complete"}: {address}/')


def deploy_redirect(old_path, new_url, env_file=None, go=False):
    """Point an old address at a new one: write an `.htaccess` with a
    permanent redirect into `old_path` on the host. Files already there
    stay (unreachable behind the redirect) until removed by hand."""
    import tempfile

    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    dry = [] if go else ['--dry-run']
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / '.htaccess').write_text(
            '# Moved; written by openplaces.viz.web_scene.deploy_redirect.\n'
            f'RedirectMatch 301 ^.*$ {new_url}\n',
            encoding='utf-8',
        )
        _rclone(env, 'copy', tmp, f'bh:{old_path}', *RCLONE_SFTP_FLAGS, *dry)
    print(f'{"Redirected" if go else "Dry run complete"}: {old_path} -> {new_url}')


def _rclone_bin() -> str:
    found = shutil.which('rclone')
    if found is None:
        raise RuntimeError(
            'rclone is not on the PATH; it is part of the openplaces conda environment.'
        )
    return found


def _rclone(env, *args) -> None:
    """Run rclone, retrying after a pause while the host refuses SSH."""
    import time

    command = [_rclone_bin(), *args]
    for attempt in range(1, RETRY_TRIES + 1):
        if subprocess.run(command, env=env, check=False).returncode == 0:
            return
        if attempt < RETRY_TRIES:
            print(
                f'  rclone {args[0]} failed (the host may be limiting SSH); '
                f'retrying in {RETRY_WAIT_S} s ({attempt}/{RETRY_TRIES - 1})'
            )
            time.sleep(RETRY_WAIT_S)
    raise RuntimeError(f'`rclone {args[0]}` failed {RETRY_TRIES} times.')


def _rclone_output(env, *args) -> str | None:
    """Output of a read-only rclone call, or None if it fails."""
    completed = subprocess.run(
        [_rclone_bin(), *args], env=env, capture_output=True, text=True, check=False
    )
    return completed.stdout if completed.returncode == 0 else None
