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
from dataclasses import dataclass, field, replace
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
# The openplaces logo as the pages' icons (maintainer, 2026-10-19): the
# browser tab's and a phone's home-screen size, made from
# Admin/openplaces/Visualization/openplaces_icon_transparent.png.
ICON_FILES = ('icon-32.png', 'icon-180.png')
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
    imputed_columns : tuple of str
        Columns taken from a region's imputed twin bundle
        (`<region>.imputed`, beside the regular one) where the model
        filled them (`IMPUTATION_TOKEN` in their `_source`); every other
        cell stays the regular bundle's (`overlay_imputed_values`).
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
    imputed_columns: tuple = ()


# The building imputation (maintainer, 2026-10-19: shown on the public
# pages, marked as imputed): its bundles sit beside the regular ones as
# `<region>.imputed`, written by branch `building-imputation`, which is
# not merged into main; its filled cells carry this token.
IMPUTED_SUFFIX = '.imputed'
IMPUTATION_TOKEN = 'building-imputation'
IMPUTATION_NOTE_NAME = 'Imputed building values (openplaces building imputation)'
IMPUTATION_NOTE_TEXT = (
    'Buildings the assessor gives no usable value are filled by a model and '
    'marked "estimated"; "Tax assessor" shows only recorded values. '
    'Manufactured homes the roll does not list get a model of their own per '
    'county, trained on the listed ones.'
)
# The model code's line in an imputed bundle's notice; the commit is read
# from there, never written here, so the page names the code that made
# the values it shows.
MODEL_CODE_LINE = re.compile(r'Model code:[^.]*?commit [0-9a-f]{7,40}\.')


def imputation_note(model_code) -> dict:
    """The Sources note on imputed values, with the model code lines the
    imputed bundles' notices state."""
    credit = ' '.join([IMPUTATION_NOTE_TEXT, *sorted(set(model_code))])
    return {'name': IMPUTATION_NOTE_NAME, 'credit': credit}


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
# They carry no area of applicability: PLACES-FMV computed it over
# building footprint covariates (design limit L4), so no bundle ships
# it (maintainer, 2026-10-24).
TEAM_MODEL_VIEWS = tuple(
    ModelView(
        f'model_{variant}',
        label,
        f'land_value_per_ha_log_region_nb_{variant}',
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


@dataclass(frozen=True)
class ModelFiles:
    """Land value models handed over as files, for a team page.

    The valuation work (openplaces-valuation) fits the models and hands
    one parquet per county, keyed by the bundle's `parcel_id`, with a
    JSON sidecar beside it (same stem, `.json`). The page joins the
    columns onto the lots by that key, so no recipe or bundle changes
    (maintainer, 2026-10-19, for the BEA meeting). Their predictions
    are already in one year's dollars (`year`), so no price index is
    applied to them: the "dollars" view is the prediction itself.

    Parameters
    ----------
    table : pandas.DataFrame
        Per lot, indexed by `parcel_id`: each view's log column and its
        area of applicability column.
    views : tuple of ModelView
        In the order of their buttons.
    year : int
        The year of the predictions' dollars.
    basis : str
        The price basis in words, for the Sources panel.
    credits : tuple of dict
        `name` and `credit` entries for the Sources panel.
    data_credits : tuple of dict
        The sidecars' own `credits`: the data the files were made from
        (a lots file's outlines and values), whatever their names.
    training_sales : dict
        Per view label, (admin id, sales it was trained on) per unit.
    """

    table: pd.DataFrame
    views: tuple
    year: int
    basis: str = ''
    credits: tuple = ()
    data_credits: tuple = ()
    training_sales: dict = field(default_factory=dict)


def read_model_files(paths) -> ModelFiles | None:
    """The models of `paths` (parquets with JSON sidecars), pooled.

    The sidecar holds `year` or `predict_year` (the predictions'
    dollars), `basis` or `price_basis`, `views` (each `key`, `label`,
    `log_column` and optionally `aoa_column`, `credit` and
    `n_training_sales`) and optionally `credits` (each `name`,
    `credit`). Files that do not exist are skipped; None when none do.
    """
    tables, views, credits, data, years, bases = [], {}, [], [], set(), set()
    training = {}
    for path in map(Path, paths):
        if not path.exists():
            continue
        side = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        table = pd.read_parquet(path)
        if 'parcel_id' in table.columns:
            table = table.set_index('parcel_id')
        tables.append(table)
        # The valuation work writes `predict_year` and `price_basis`.
        years.add(int(side.get('year', side.get('predict_year'))))
        basis = str(side.get('basis', side.get('price_basis', '')))
        # The columns are logs; the page shows the values.
        bases.add(basis.replace(' (natural log)', ''))
        for view in side['views']:
            if view['log_column'] not in table.columns:
                continue
            views.setdefault(
                view['key'],
                ModelView(
                    view['key'],
                    view['label'],
                    view['log_column'],
                    view.get('aoa_column'),
                ),
            )
            if view.get('credit'):
                credits.append({'name': view['label'], 'credit': view['credit']})
            if view.get('n_training_sales'):
                # Per view, the unit and its count; the page names the
                # units (`model_credits`).
                training.setdefault(view['label'], []).append(
                    (side.get('admin_id', path.stem), int(view['n_training_sales']))
                )
        credits += list(side.get('credits') or [])
        data += list(side.get('credits') or [])
    if not tables:
        return None
    if len(years) > 1:
        raise ValueError(f'Model files in different dollars: {sorted(years)}.')
    table = pd.concat(tables)
    unique = {(c['name'], c['credit']): c for c in credits}
    data_unique = {(c['name'], c['credit']): c for c in data}
    return ModelFiles(
        table=table[~table.index.duplicated(keep='first')],
        views=tuple(views.values()),
        year=years.pop(),
        basis='; '.join(sorted(b for b in bases if b)),
        credits=tuple(unique.values()),
        data_credits=tuple(data_unique.values()),
        training_sales=training,
    )


def model_credits(model_files, unit_names) -> list[dict]:
    """The Sources panel's model credits: each model's citation, with
    the sales it was trained on per unit named in one line ("Trained on
    sales: District of Columbia 5,798; Prince George's 15,358"). A count
    of training sales, never a summary of the models' values."""
    out = []
    for credit in model_files.credits:
        sales = model_files.training_sales.get(credit['name'])
        if sales:
            counts = '; '.join(f'{unit_names.get(a, a)} {n:,}' for a, n in sales)
            credit = {
                **credit,
                'credit': f'{credit["credit"]}. Trained on sales: {counts}.',
            }
        out.append(credit)
    return out


def lots_from_model_file(path, layer) -> tuple[gpd.GeoDataFrame, SceneLayer]:
    """Lots and their layer from a model file that carries the outlines.

    For a county openplaces has no bundle of (Prince George's County
    MD, maintainer 2026-10-19): the valuation work hands one GeoParquet
    of lots with the source's land values and the models' columns. The
    land value shown is the recorded one (nothing is filled), and
    without `n_dwellings` the layer has no per-home view. A personal
    column in the file is refused, whether or not the page would show
    it.

    Parameters
    ----------
    path : str or Path, or a list of them
        The GeoParquet(s), one per unit, each with `parcel_id`,
        `admin3_id` and `land_value` (Prince George's and Washington DC
        on one page; maintainer, 2026-10-19).
    layer : SceneLayer
        The lots' layer of `VALUE_SCENE_LAYERS`.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, SceneLayer)
    """
    gdf = _read_given_files(path, 'parcel_id')
    if layer.value_column not in gdf.columns:
        gdf[layer.value_column] = gdf['land_value']
    return _fit_given_layer(gdf, layer)


def _read_given_files(path, key) -> gpd.GeoDataFrame:
    """Handed-over GeoParquets pooled in longitude and latitude, indexed
    by `key`; refused when one carries a personal column."""
    paths = [path] if isinstance(path, str | Path) else list(path)
    parts = []
    for one in paths:
        part = gpd.read_parquet(one)
        personal = [c for c in part.columns if is_personal_attribute(c)]
        if personal:
            raise PersonalColumnError(f'{one} carries personal columns: {personal}.')
        parts.append(part.to_crs(4326))
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    return gdf.set_index(key) if key in gdf.columns else gdf


def _fit_given_layer(gdf, layer) -> tuple[gpd.GeoDataFrame, SceneLayer]:
    """A handed-over frame and its layer, fitted to each other: no
    per-home view without dwellings, and missing tooltip columns
    present but empty."""
    if 'n_dwellings' not in gdf.columns:
        dropped = {'n_dwellings', f'{layer.value_column}_per_dwelling'}
        layer = replace(
            layer,
            per_dwelling_label=None,
            tooltip=tuple(t for t in layer.tooltip if t.column not in dropped),
        )
    for tip in layer.tooltip:
        if tip.column not in gdf.columns and tip.column != 'n_records_on_outline':
            gdf[tip.column] = pd.NA
    return gdf, layer


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
# Words of a roll's use group (lowercased) naming a use that is not
# homes, and words naming homes; a use group with the first and none
# of the second keeps the class from confirming a lot
# (`residential_lots`). Farm and forest words are not among the first:
# a farmhouse is a home (maintainer, 2026-10-19), and farmland with no
# building on the roll fails the `NO_BUILDING_VALUE` test instead.
NON_RESIDENTIAL_USE = r'commercial|industrial|open.?space|vacant|exempt'
# Words of a farm or forest lot's use group. Such a lot's land is drawn
# per acre, not per home, unless its use group also names a home (a
# homesite): one farmhouse would otherwise carry the whole farm's land
# (maintainer, 2026-10-19). The farmhouse stays a home on the building
# layer, which classes buildings on their own.
FARM_USE = r'use value|agricultur|forest|farm'
RESIDENTIAL_USE = (
    r'resid|family|home|dwell|housing|condo|mobile|manufactured'
    r'|\bmfg\b|\bmh\b|\bres\b|\bsfr\b'
)
# An improvement value at or below this is no building in the roll's
# eyes (Massachusetts records $200 on exempt open land).
NO_BUILDING_VALUE = 1_000.0
# Public owners of protected open space (`protected_open_space_owner_type`)
# and the least share of a lot under such land that makes it no home
# lot where the roll counts no dwellings on it (`residential_lots`): a
# reservation's gatehouse lot or a city park whose only class came
# from a building model (maintainer, 2026-10-24: Chestnut Hill
# Reservation, Ramsay Park). Apartment lots on a beach reservation keep
# their roll's counts and stay homes.
PUBLIC_OPEN_SPACE_OWNERS = frozenset({'Municipal', 'State', 'Federal', 'County'})
PUBLIC_OPEN_SPACE_SHARE = 0.9
# Words of a roll's use subgroup (`use_subgroup`) naming residential
# land rather than homes ("developable residential land"). Such a lot
# with no building on the roll is not a home lot, though its use group
# names a home (maintainer, 2026-10-24).
RESIDENTIAL_LAND_SUBGROUP = r'residential land'
# Least dwellings a public-housing subgroup implies; an apartment
# subgroup states its own ("apartments over 8 units", "apartments 4-8
# units", `implied_min_homes`).
PUBLIC_HOUSING_MIN_HOMES = 4
# Evidence columns of the lots read for the tests above.
LOT_EVIDENCE_COLUMNS = ('use_subgroup', 'protected_open_space_owner_type')
# Lots smaller than this (ha) that the roll calls vacant keep their
# class: a new home on a lot the roll still lists as vacant.
NEW_HOME_MAX_HA = 0.5
# The page's own class of a building (maintainer, 2026-10-16/17, handoff
# item 1f): the roll's land use class of its lot (`land_use_class_parcel`:
# Condominium, Townhome, Multiple Single-Family, Office, ...), which
# keeps what the CHEER `occupancy_type` folds into "Low-Rise
# Multi-Family"; `occupancy_type` where the lot has no class, and always
# for outbuildings ("Secondary"), since the lot's class describes its
# main building, not a shed.
BUILDING_CLASS = 'building_class'
# The type view (maintainer, 2026-10-19): one list of types for lots
# and buildings, so a house and its lot share a color. Lots take their
# roll's land use class; buildings their `BUILDING_CLASS` (the lot's
# class for a main building, CHEER's occupancy type otherwise). Colors
# start from the QGIS footprint style (single-family cyan, manufactured
# home orange, multi-family purple, outbuildings pale), the lots' own
# QGIS style being a different scheme. Entry 0 is the unknown type.
TYPE_CATEGORIES = (
    {'key': 'unknown', 'label': 'Unknown', 'color': '#969696', 'values': ()},
    {
        'key': 'single_family',
        'label': 'Single-family',
        'color': '#30bfe2',
        'values': ('Single-Family', 'Multiple Single-Family'),
    },
    {
        'key': 'townhome',
        'label': 'Townhome',
        'color': '#4f6fd9',
        'values': ('Townhome',),
    },
    {
        'key': 'multi_family',
        'label': 'Multi-family',
        'color': '#8260d4',
        'values': (
            'Multi-Family',
            'Low-Rise Multi-Family',
            'Mid-Rise Multi-Family',
            'High-Rise Multi-Family',
        ),
    },
    {
        'key': 'condominium',
        'label': 'Condominium',
        'color': '#cc60d4',
        'values': ('Condominium',),
    },
    {
        'key': 'manufactured_home',
        'label': 'Manufactured home',
        'color': '#fcab60',
        'values': ('Manufactured Home', 'Manufactured Home Park', 'RV Park'),
    },
    {
        'key': 'commercial',
        'label': 'Commercial',
        'color': '#e6cf3c',
        'values': (
            'Commercial',
            'Retail',
            'Office',
            'Professional Technical Services',
            'Medical Office',
            'Personal & Repair Services',
            'Hotel',
            'Bank',
            'Entertainment/Recreation',
            'Theater',
            'Garage',
        ),
    },
    {
        'key': 'industrial',
        'label': 'Industrial',
        'color': '#a0785a',
        'values': (
            'Industrial',
            'Light Industrial',
            'Heavy Industrial',
            'Wholesale',
            'Food/Drug/Chemical',
            'Metals/Minerals processing',
            'High Technology',
            'Construction',
        ),
    },
    {
        'key': 'institutional',
        'label': 'Institutional',
        'color': '#3fa596',
        'values': (
            'Institutional',
            'Church',
            'Average School',
            'College/University',
            'Government Services',
            'Hospital',
            'Nursing Home',
            'Emergency Response',
            'Institutional Dormitory',
        ),
    },
    {
        'key': 'open_land',
        'label': 'Open land',
        'color': '#8fbf6a',
        'values': ('Vacant', 'Mostly Vacant', 'Agricultural', 'Recreation'),
    },
    {
        'key': 'outbuilding',
        'label': 'Outbuilding',
        'color': '#c4cdca',
        'values': ('Secondary',),
    },
)


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
        imputed_columns=('structure_value',),
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
    'nsi': {'name': 'National Structure Inventory (USACE)'},
    'vgin': {'name': 'Virginia Geographic Information Network (VGIN)'},
    # Fairfax VA's assessor roll, shown on team pages only (its terms
    # are for internal use; maintainer's approval, 2026-10-19).
    'fairfaxcounty': {'name': 'Fairfax County assessor'},
    # Alexandria VA's real estate accounts, team pages only (cleared
    # for team sharing, 2026-10-24).
    'alexgis': {'name': 'City of Alexandria (AlexGIS)'},
    # Arlington VA's assessments, on restricted pages only (personal
    # use, `RESTRICTED_TIER`).
    'arlington': {'name': 'Arlington County (Data API)'},
    'nconemap': {'name': 'NC OneMap', 'credit': 'NC OneMap (nconemap.gov)'},
    # The acknowledgment Maryland's licence asks of derived data
    # (US-MD_parcel-mdimap-2026), and DC's CC BY 4.0 attribution
    # (US-DC_property-dcgovitspe-2026).
    'mdimap': {
        'name': 'Maryland Department of Planning (MD iMAP)',
        'credit': 'State of Maryland / Maryland Department of Planning',
    },
    'dcgovitspe': {
        'name': 'DC Office of Tax and Revenue (ITSPE)',
        'credit': 'District of Columbia, Office of the Chief Technology Officer and '
        'Office of Tax and Revenue (CC BY 4.0)',
    },
    # DC's lot outlines (US-DC_parcel-dcgovrecordlots-2026,
    # US-DC_parcel-dcgovtaxlots-2026), CC BY 4.0 like its roll.
    'dcgovrecordlots': {
        'name': 'DC record lots (DC GIS)',
        'credit': 'District of Columbia, Office of the Chief Technology Officer '
        '(CC BY 4.0)',
    },
    'dcgovtaxlots': {
        'name': 'DC tax lots (DC GIS)',
        'credit': 'District of Columbia, Office of the Chief Technology Officer '
        '(CC BY 4.0)',
    },
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
# A withheld cell's sidecar token (`io.delivery.redaction.WITHHELD`),
# and how the Sources panel names it.
WITHHELD_TOKEN = 'withheld'
WITHHELD_BY_TERMS = "Withheld: the source's terms do not allow sharing it"
# A unit whose bundle brings no value at all (no roll, or one not
# cleared for the page).
NO_VALUES_IN_BUNDLE = "No values in this page's data"


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
        # A cell the delivery emptied because its source's terms do
        # not allow sharing it (`io.delivery.redaction.WITHHELD`).
        if token == WITHHELD_TOKEN:
            return WITHHELD_BY_TERMS
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
        {
            'name': name,
            'share': float(share),
            'credit': credits.get(name),
            **({'withheld': True} if name == WITHHELD_BY_TERMS else {}),
        }
        for name, share in shares.items()
    ]


def value_sources_by_unit(provenance, units, unit_names) -> list[dict]:
    """`value_sources` for each unit of a scene, so a county whose
    source is withheld (Fairfax VA's building values, 2026-10-24) shows
    apart from its neighbors instead of disappearing into a scene-wide
    share (maintainer's decision, via the coordinator).

    Parameters
    ----------
    provenance : pandas.Series
        The shown value's `_source` column.
    units : pandas.Series
        Each feature's `admin3_id`, aligned with `provenance`.
    unit_names : dict
        `admin3_id` to the unit's name.

    Returns
    -------
    list of dict
        `unit` (its name) and `values` (as `value_sources`), in the
        order of the names given. A unit whose features carry no value
        at all says so (`NO_VALUES_IN_BUNDLE`, marked `missing`), so its
        empty lots do not read as unvalued land; a unit without
        features is left out.
    """
    out = []
    for unit, name in unit_names.items():
        rows = (units == unit).to_numpy()
        if not rows.any():
            continue
        sources = value_sources(provenance[rows]) or [
            {'name': NO_VALUES_IN_BUNDLE, 'share': 1.0, 'credit': None, 'missing': True}
        ]
        out.append({'unit': name, 'values': sources})
    return out


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


# Lots smaller than this (m2) whose value per area tops the page's
# `LAND_GHOST_QUANTILE` `LAND_GHOST_FACTOR` times over are drawn as
# ghosts (`mark_land_ghosts`).
LAND_GHOST_MAX_M2 = 100.0
LAND_GHOST_QUANTILE = 0.999
LAND_GHOST_FACTOR = 10.0


def mark_land_ghosts(gdfs, layers) -> None:
    """Flag tiny lots carrying a value their area cannot hold.

    A display step, the land counterpart of the buildings' height check
    (`GHOST_COLUMN`), for every layer whose recipe runs none. A lot's
    height is its value per area, so a sliver or an air-rights outline
    carrying a whole account's value rises as a needle: a 23 m2 sliver
    in Port Arthur held the $14.1M of a 402-acre lot, and Charlotte's
    16 m2 air-rights outlines came to $1-2 billion per acre. A lot under
    `LAND_GHOST_MAX_M2` whose value per area is more than
    `LAND_GHOST_FACTOR` times the page's own `LAND_GHOST_QUANTILE` of
    lots with a value is written as a ghost; its value is unchanged.
    The factor spares the costliest real land: Beacon Hill row-house
    lots of about 70 m2 come to $64M per acre, at Boston's p99.9 and
    far below the errors (20 to 3,000 times Charlotte's p99.9). Large
    lots are never flagged: a high value over a large area is a real,
    if rare, price.

    Written to the layer's `GHOST_COLUMN`, in place.
    """
    for layer in layers:
        gdf = gdfs[layer.name]
        if (
            gdf.empty
            or layer.value_column not in gdf.columns
            or ghost_rule(layer.recipe) is not None
        ):
            continue
        area = gdf.geometry.to_crs(5070).area.to_numpy()
        value = gdf[layer.value_column].to_numpy(dtype=float)
        has = (value > 0) & (area > 0)
        if not has.any():
            continue
        per_area = np.full(len(gdf), np.nan)
        per_area[has] = value[has] / area[has]
        cut = LAND_GHOST_FACTOR * np.quantile(per_area[has], LAND_GHOST_QUANTILE)
        ghost = has & (area < LAND_GHOST_MAX_M2) & (per_area > cut)
        gdf[GHOST_COLUMN] = pd.array(ghost, dtype='boolean')


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

    The same holds on a lot of a residential class with no roll count
    whose record says it holds no home: no building on the roll (and
    no use group naming a home), public open space, or residential land
    (maintainer, 2026-10-24: a monument and cemetery chapels drawn as
    homes, their lots' codes missing from the use-code table). A class
    merely unconfirmed does not do it, nor does a farm: where a roll
    states neither use groups nor counts, its houses stay homes, and a
    farmhouse stays a home.
    Measured 2026-10-24 on boston: 457 outlines no longer drawn as homes
    """
    for layer in layers:
        above = next((u for u in layers if u.stack_on == layer.name), None)
        if above is None:
            continue
        lots, buildings = gdfs[layer.name], gdfs[above.name]
        if BUILDING_CLASS not in buildings.columns:
            continue
        empty = pd.Series(pd.NA, index=lots.index, dtype='string')
        use = (
            lots['use_group'].astype('string').str.strip().str.lower()
            if 'use_group' in lots.columns
            else empty
        )
        own = (
            lots['n_dwellings'].astype(float)
            if 'n_dwellings' in lots.columns
            else pd.Series(np.nan, index=lots.index)
        )
        exempt = ((use == 'exempt').fillna(False) & ~(own > 0)).to_numpy()
        if layer.per_dwelling_label:
            tests = _home_lot_tests(lots.assign(**{ROLL_DWELLINGS: own.where(own > 0)}))
            rejected = tests['home_class'] & (
                tests['no_building'] | tests['public_land'] | tests['bare_land']
            )
            exempt = exempt | (rejected & ~(own > 0).to_numpy())
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
    files = {
        path
        for layer in manifest.get('layers', [])
        for paths in layer.get('files', {}).values()
        for path in paths.values()
    }
    if manifest.get('people'):
        files.add(manifest['people']['file'])
    return files


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


def scene_bounds(per_area, spread=2.0) -> tuple[float, float]:
    """Color bounds from a scene's own values (the local scale).

    Fixed bounds failed across places. Measured 2026-10-15 on US-NC-HYD:
    with Boston's $50K-$50M per acre, rural land fell almost wholly
    below the lower bound and drew in the ramp's darkest color, next
    to invisible on the dark page. The ramp therefore follows the
    place's own distribution (maintainer, 2026-10-18: its mean and
    standard deviation): on a log scale, the mean plus and minus
    `spread` standard deviations, never wider than the values
    themselves, rounded outward to 1-2-5 numbers. A few extreme lots
    then saturate the ends instead of stretching the ramp. The page
    also offers fixed nationwide bounds (`NATIONAL_BOUNDS`).
    """
    values = np.asarray(per_area, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values):
        return 1.0, 10.0
    logs = np.log10(values)
    mean, sd = logs.mean(), logs.std()
    low = 10 ** max(mean - spread * sd, logs.min())
    high = 10 ** min(mean + spread * sd, logs.max())
    low, high = _nice(low, up=False), _nice(high, up=True)
    return (low, high) if high > low else (low, low * 10)


# Fixed color bounds, the same for every place, per display unit
# (maintainer, 2026-10-18: a nationwide gradient beside the local
# one). Land from rural acreage to the dearest downtowns; buildings
# from sheds to towers; per home, land or building, from a modest
# share to an estate. Measured 2026-10-15 on boston-inner-core: land
# per acre had p99 $51M and structure value per sq ft p99 $3,407.
NATIONAL_BOUNDS = {
    'ac': (1_000.0, 100_000_000.0),
    'ha': (2_000.0, 200_000_000.0),
    'sqft': (10.0, 10_000.0),
    'm2': (100.0, 100_000.0),
    'dwelling': (10_000.0, 10_000_000.0),
}


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

    Pastel (the default, first; maintainer, 2026-10-20), then
    Landscape (shown as "Earth"), Sunset and Neon. Landscape was the
    first default; its view stays warm and light at its dearest places: the
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
            # A lot valued at $0 in black, not the default white, which
            # glared out of the night view (maintainer, 2026-10-24).
            zero_value_rgba=(0, 0, 0, 255),
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
    # The default (Pastel) first, on the chips too (maintainer,
    # 2026-10-20); the others keep their order.
    palettes.sort(key=lambda p: p.id != DEFAULT_PALETTE)
    # A scene of fewer layers (lots alone, `lots_from_model_file`) keeps
    # each palette's colors for the layers it has.
    return [
        replace(p, layers={k: v for k, v in p.layers.items() if k in names})
        for p in palettes
        if names <= set(p.layers)
    ]


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
    return _nhd_polygons(admin_ids, 'US_water-waterbody-usgs-nhdhr2023', LAKE_TYPES)


def _nhd_polygons(admin_ids, recipe_id, feature_types, bounds=None) -> gpd.GeoSeries:
    """NHD polygons of `feature_types` from `recipe_id`'s state files,
    within `bounds` (longitude and latitude; by default the units'
    bounding box) and cut to it, in EPSG:4326. A state whose file is
    missing contributes nothing."""
    from openplaces.core.schema import AdminId
    from openplaces.recipe import get_output_path, get_recipe_by_id

    admin_ids = list(admin_ids)
    states = sorted({str(AdminId(a).truncate_to_level(2)) for a in admin_ids})
    if bounds is None:
        units = get_admin(states, level=3, geom=True)
        if 'admin3_id' not in units.columns:
            units = units.reset_index()
        units = units[units['admin3_id'].isin(admin_ids)]
        bounds = units.to_crs(4326).total_bounds
    recipe = get_recipe_by_id(recipe_id)
    parts = []
    for state in states:
        attributes_path = get_output_path(recipe, state)
        geo_path = get_output_path(recipe, state, geo=True)
        if not (attributes_path.exists() and geo_path.exists()):
            continue
        types = pd.read_parquet(
            attributes_path, columns=['feature_type_nhd', '_join_id']
        )
        keys = types.loc[types['feature_type_nhd'].isin(feature_types), '_join_id']
        geo = gpd.read_parquet(geo_path, bbox=tuple(bounds))
        parts.append(geo.loc[geo['_join_id'].isin(keys), 'geometry'].to_crs(4326))
    if not parts:
        return gpd.GeoSeries([], crs=4326)
    # The sea is one polygon per state: only the stretch in view is kept.
    shapes = shapely.clip_by_rect(
        pd.concat(parts, ignore_index=True).to_numpy(), *bounds
    )
    shapes = shapes[~shapely.is_empty(shapes)]
    return gpd.GeoSeries(shapes, crs=4326)


# NHD feature types that cut a town border (`lake_polygons`).
LAKE_TYPES = (390, 436)
# NHDArea feature types drawn as water under the lots (`water_polygons`):
# the sea and ocean, and wide rivers. Foreshore (364), the land between
# the tides, is not among them.
SEA_RIVER_TYPES = (445, 460)


def water_polygons(admin_ids, bounds) -> gpd.GeoSeries:
    """Open water within `bounds` (longitude and latitude) in the
    units' states, from NHD: lakes and reservoirs (`LAKE_TYPES`), and
    the sea and wide rivers of NHDArea (`SEA_RIVER_TYPES`), in
    EPSG:4326."""
    lakes = _nhd_polygons(
        admin_ids, 'US_water-waterbody-usgs-nhdhr2023', LAKE_TYPES, bounds
    )
    seas = _nhd_polygons(
        admin_ids, 'US_water-waterbody-usgs-nhdhr2023_area', SEA_RIVER_TYPES, bounds
    )
    return gpd.GeoSeries(pd.concat([lakes, seas], ignore_index=True), crs=4326)


# The land part of a lot is drawn in this column (`mask_water`).
LAND_SHAPE = '_land_shape'
# Margin (m) kept as land around a building standing over mapped water,
# so a pier's building keeps the ground it stands on.
PIER_MARGIN_M = 2.0
# A lot keeps less than this share of its area as land, or less than
# `MASKED_MIN_M2`, and it is water: not drawn at all.
MASKED_MIN_SHARE = 0.01
MASKED_MIN_M2 = 5.0


def mask_water(lots, water, buildings=None) -> gpd.GeoDataFrame:
    """Draw only the land of each lot (maintainer, 2026-10-24: lots
    reaching into the Charles River and the harbor).

    A display step, and the one place a lot is drawn with a shape its
    source does not have: rolls draw many waterfront lots out to a
    harbor line or a river's middle, and drawn whole they cover the
    water. The lot's value and its value per area stay those of the
    whole lot (the rendering reads its area before the mask applies),
    so a lot's height and color do not change; only its outline stops
    at the shore. Ground under a building is kept within
    `PIER_MARGIN_M`, so piers keep their buildings' footing. A lot that
    keeps less than `MASKED_MIN_SHARE` of its area, or under
    `MASKED_MIN_M2`, is water and is left out. NHD's shorelines are
    mapped at about 1:24,000.
    Measured 2026-10-24 on boston: 116 all-water lots left out

    Parameters
    ----------
    lots : geopandas.GeoDataFrame
        The bottom layer's features.
    water : geopandas.GeoSeries
        Open water (`water_polygons`).
    buildings : geopandas.GeoDataFrame, optional
        The layer stacked on the lots.

    Returns
    -------
    geopandas.GeoDataFrame
        The lots, wet ones' land shapes in `LAND_SHAPE` (WKB in
        EPSG:4326; None where a lot is all land), all-water lots
        dropped.
    """
    lots = lots.copy()
    lots[LAND_SHAPE] = None
    if lots.empty or water.empty:
        return lots
    metric = lots.geometry.to_crs(5070)
    wet = water.to_crs(5070)
    wet = gpd.GeoSeries(shapely.make_valid(wet.to_numpy()), crs=5070)
    hit_water, hit_lot = metric.sindex.query(wet.to_numpy(), predicate='intersects')
    if not len(hit_lot):
        return lots
    water_shape = shapely.union_all(wet.to_numpy()[np.unique(hit_water)])
    if buildings is not None and not buildings.empty:
        footing = buildings.geometry.to_crs(5070)
        near = footing.sindex.query(water_shape, predicate='intersects')
        if len(near):
            kept = shapely.buffer(footing.to_numpy()[near], PIER_MARGIN_M)
            water_shape = shapely.difference(water_shape, shapely.union_all(kept))
    rows = np.unique(hit_lot)
    shapes = shapely.make_valid(metric.to_numpy()[rows])
    xmin, ymin, xmax, ymax = shapely.bounds(shapes).T
    # Each lot meets only the water within its own box, a small piece of
    # a coastline that may run for kilometers.
    local_water = shapely.intersection(water_shape, shapely.box(xmin, ymin, xmax, ymax))
    land = shapely.difference(shapes, local_water)
    # Polygons only: a difference can leave slivers as lines or points.
    mixed = shapely.get_type_id(land) == 7
    if mixed.any():
        land[mixed] = [
            shapely.union_all(
                [p for p in shapely.get_parts(g) if p.geom_type.endswith('Polygon')]
            )
            for g in land[mixed]
        ]
    whole = shapely.area(shapes)
    left = shapely.area(land)
    drop = (left < MASKED_MIN_M2) | (left < MASKED_MIN_SHARE * whole)
    changed = ~drop & (left < whole - 0.5)
    # As WKB: the rendering hands the frame to Arrow, which takes bytes
    # but no shapes in a plain column.
    column = np.full(len(lots), None, dtype=object)
    column[rows[changed]] = shapely.to_wkb(
        gpd.GeoSeries(land[changed], crs=5070).to_crs(4326).to_numpy()
    )
    lots[LAND_SHAPE] = column
    keep = np.ones(len(lots), dtype=bool)
    keep[rows[drop]] = False
    return lots[keep]


# The people cloud (`race_dot_cloud`): the 2020 Census groups of race and
# Hispanic origin that the Racial Dot Map (Dustin Cable, University of
# Virginia, 2013) drew, in its colors. Every group but Hispanic or
# Latino counts non-Hispanic people only, so each person is in one
# group. `column` names the count in the census dataset.
PEOPLE_GROUPS = (
    {'key': 'white', 'label': 'White', 'color': '#73b2ff', 'column': 'n_people_white'},
    {'key': 'black', 'label': 'Black', 'color': '#9fd400', 'column': 'n_people_black'},
    {'key': 'asian', 'label': 'Asian', 'color': '#ff4a4a', 'column': 'n_people_asian'},
    {
        'key': 'hispanic',
        'label': 'Hispanic or Latino',
        'color': '#ffaa00',
        'column': 'n_people_hispanic',
    },
    {
        'key': 'other',
        'label': 'Other or two or more races',
        'color': '#b08060',
        'column': 'n_people_other',
    },
)
# Census block counts (P.L. 94-171, table P2) and the 2020 tabulation
# blocks they key on (`census_block_geoid`, GEOID20).
PEOPLE_RECIPE = 'US_population-census-2020pl'
PEOPLE_BLOCK_RECIPE = 'US_tile-census-2025_block'
PEOPLE_KEY = 'census_block_geoid'
# People per dot as built; the page may show every second or fourth.
PEOPLE_PER_DOT = 5
# The cloud's height over a lot's top: a gap, then a band the dots
# spread through at random (m).
PEOPLE_GAP_M = 25.0
PEOPLE_BAND_M = 50.0
# The page fades the cloud out closer than this zoom, where single
# houses can be told apart and a dot could read as one household.
PEOPLE_MAX_ZOOM = 16.0
# Pages that may carry the cloud: team pages only, until counsel has
# seen the fair-housing question of race shown beside home values on a
# public page (maintainer, 2026-10-24).
PEOPLE_AUDIENCES = frozenset({'team'})
PEOPLE_NOTE = {
    'name': 'People (dot cloud)',
    'credit': (
        'U.S. Census Bureau, 2020 Census Redistricting Data (P.L. 94-171), '
        'table P2, by census block; public domain. One dot per {per_dot} '
        'people, placed at random over the homes of its block, never on a '
        "single house. Block counts carry the Bureau's disclosure-avoidance "
        'noise. After the Racial Dot Map (Dustin Cable, University of '
        'Virginia, 2013).'
    ),
}


def read_people_blocks(admin_ids, bounds) -> gpd.GeoDataFrame | None:
    """Census blocks within `bounds` (longitude and latitude) in the
    units' states, with their people per group (`PEOPLE_GROUPS`), or
    None where the census counts are not on this machine."""
    from openplaces.core.schema import AdminId
    from openplaces.recipe import get_output_path, get_recipe_by_id

    states = sorted({str(AdminId(a).truncate_to_level(2)) for a in admin_ids})
    try:
        people_recipe = get_recipe_by_id(PEOPLE_RECIPE)
    except OSError:
        # No such recipe yet on this checkout.
        return None
    block_recipe = get_recipe_by_id(PEOPLE_BLOCK_RECIPE)
    columns = [PEOPLE_KEY] + [g['column'] for g in PEOPLE_GROUPS]
    parts = []
    for state in states:
        counts_path = get_output_path(people_recipe, state)
        attributes_path = get_output_path(block_recipe, state)
        geo_path = get_output_path(block_recipe, state, geo=True)
        if not all(p.exists() for p in (counts_path, attributes_path, geo_path)):
            continue
        counts = pd.read_parquet(counts_path, columns=columns)
        keys = pd.read_parquet(attributes_path, columns=[PEOPLE_KEY, '_join_id'])
        geo = gpd.read_parquet(geo_path, bbox=tuple(bounds))
        blocks = geo.merge(keys, on='_join_id').merge(counts, on=PEOPLE_KEY)
        parts.append(blocks[columns + ['geometry']].to_crs(4326))
    if not parts:
        return None
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def _random_points(area, n, rng) -> np.ndarray:
    """`n` points drawn uniformly inside polygon `area`, as an (n, 2)
    array, by rejection from its bounding box."""
    xmin, ymin, xmax, ymax = area.bounds
    shapely.prepare(area)
    fill = max(area.area / max((xmax - xmin) * (ymax - ymin), 1e-9), 0.01)
    found = []
    left = n
    while left > 0:
        size = int(left / fill * 1.3) + 16
        x = rng.uniform(xmin, xmax, size)
        y = rng.uniform(ymin, ymax, size)
        inside = shapely.contains_xy(area, x, y)
        found.append(np.column_stack([x[inside], y[inside]])[:left])
        left -= len(found[-1])
    return np.concatenate(found)


def race_dot_cloud(blocks, homes, per_dot=PEOPLE_PER_DOT, seed=0) -> pd.DataFrame:
    """One dot per `per_dot` people of each census block, placed at
    random over the block's homes (maintainer, 2026-10-24: a cloud
    above residential lots, after the Racial Dot Map).

    A dot is drawn uniformly within the union of the block's
    residential lots, cut to the block, so it belongs to no lot,
    building or household: the census counts people per block, not per
    house. A block with people and no residential lot spreads its dots
    over the whole block, so counts stay whole. Each group's count is
    divided by `per_dot` and rounded at random (the remainder becomes
    one more dot with that probability), so small groups in small
    blocks are neither dropped nor inflated on average. The generator
    is seeded by the block's id, so a rebuild draws the same cloud.
    The dots come out shuffled: any first part of them is an even
    sample of the whole, which the page uses to thin the cloud.

    Parameters
    ----------
    blocks : geopandas.GeoDataFrame
        Census blocks (`PEOPLE_KEY`) with a count per group
        (`PEOPLE_GROUPS` columns).
    homes : geopandas.GeoSeries
        Residential lots, as drawn (land shapes).
    per_dot : int
        People per dot.
    seed : int
        Added to every block's seed.

    Returns
    -------
    pandas.DataFrame
        `longitude`, `latitude` and `group` (the index into
        `PEOPLE_GROUPS`, uint8); no id of any kind.
    """
    columns = [g['column'] for g in PEOPLE_GROUPS]
    empty = pd.DataFrame(
        {
            'longitude': np.array([], dtype=float),
            'latitude': np.array([], dtype=float),
            'group': np.array([], dtype=np.uint8),
        }
    )
    if blocks.empty:
        return empty
    metric = blocks.to_crs(5070).reset_index(drop=True)
    home_shapes = (
        homes.to_crs(5070).to_numpy() if len(homes) else np.array([], dtype=object)
    )
    by_block = {}
    if len(home_shapes):
        tree = shapely.STRtree(home_shapes)
        hit_block, hit_home = tree.query(
            metric.geometry.to_numpy(), predicate='intersects'
        )
        for b, h in zip(hit_block, hit_home, strict=True):
            by_block.setdefault(b, []).append(h)
    counts = metric[columns].fillna(0).to_numpy(dtype=float)
    points, groups = [], []
    for i, block in enumerate(metric.geometry.to_numpy()):
        if counts[i].sum() <= 0 or block is None or block.is_empty:
            continue
        key = str(metric[PEOPLE_KEY].iloc[i])
        block_seed = int(key) if key.isdigit() else i
        rng = np.random.default_rng([seed, block_seed])
        share = counts[i] / per_dot
        n = (np.floor(share) + (rng.random(len(share)) < share % 1)).astype(int)
        if n.sum() == 0:
            continue
        area = None
        if i in by_block:
            area = shapely.intersection(
                shapely.union_all(home_shapes[by_block[i]]), shapely.make_valid(block)
            )
            area = shapely.union_all(
                [p for p in shapely.get_parts(area) if p.geom_type.endswith('Polygon')]
            )
        if area is None or area.is_empty or area.area < 1.0:
            area = shapely.make_valid(block)
        points.append(_random_points(area, int(n.sum()), rng))
        groups.append(np.repeat(np.arange(len(n), dtype=np.uint8), n))
    if not points:
        return empty
    xy = np.concatenate(points)
    group = np.concatenate(groups)
    order = np.random.default_rng(seed).permutation(len(xy))
    lonlat = gpd.GeoSeries(gpd.points_from_xy(xy[order, 0], xy[order, 1]), crs=5070)
    lonlat = lonlat.to_crs(4326)
    return pd.DataFrame(
        {
            'longitude': lonlat.x.to_numpy(),
            'latitude': lonlat.y.to_numpy(),
            'group': group[order],
        }
    )


def people_heights(dots, lots, tops, homes, seed=0) -> np.ndarray:
    """Each dot's height: over the top of what stands under it (the lot,
    or the tallest building on that lot), plus `PEOPLE_GAP_M` and a
    random lift within `PEOPLE_BAND_M`, so the dots hang as one cloud
    over the homes. A dot over no lot takes the median top of the
    homes."""
    rng = np.random.default_rng(seed)
    lift = PEOPLE_GAP_M + rng.random(len(dots)) * PEOPLE_BAND_M
    if not len(dots):
        return lift
    shapes = lots.geometry.to_crs(4326).to_numpy()
    tree = shapely.STRtree(shapes)
    points = shapely.points(dots['longitude'].to_numpy(), dots['latitude'].to_numpy())
    hit_dot, hit_lot = tree.query(points, predicate='within')
    base = np.full(len(dots), np.nan)
    base[hit_dot] = tops[hit_lot]
    fallback = np.nanmedian(tops[homes]) if homes.any() else 0.0
    return np.where(np.isfinite(base), base, fallback) + lift


def _people_cloud(admin_ids, crop, layers, rendered, homes, data_dir) -> dict | None:
    """The people cloud's manifest entry, its file written, or None
    where the census counts are missing or nobody lives in view."""
    lot_layer = rendered[layers[0].name]
    lots = lot_layer.gdf
    blocks = read_people_blocks(admin_ids, lots.to_crs(4326).total_bounds)
    if blocks is None or blocks.empty:
        return None
    if crop is not None:
        blocks = crop_to_area(blocks, crop)
    shapes = lots.geometry.to_crs(4326).to_numpy().copy()
    if LAND_SHAPE in lots.columns:
        land = lots[LAND_SHAPE].to_numpy()
        wet = np.array([shape is not None for shape in land], dtype=bool)
        shapes[wet] = shapely.from_wkb(land[wet].tolist())
    dots = race_dot_cloud(blocks, gpd.GeoSeries(shapes[homes], crs=4326))
    if dots.empty:
        return None
    # A lot's top, raised to its tallest building's top.
    tops = np.asarray(lot_layer.total_elevation, dtype=float).copy()
    above = next((u for u in layers if u.stack_on == layers[0].name), None)
    if above is not None and above.name in rendered:
        buildings = rendered[above.name]
        centers = buildings.gdf.geometry.to_crs(5070).centroid.to_crs(4326)
        tree = shapely.STRtree(lots.geometry.to_crs(4326).to_numpy())
        hit_building, hit_lot = tree.query(centers.to_numpy(), predicate='within')
        np.maximum.at(
            tops,
            hit_lot,
            np.asarray(buildings.total_elevation, dtype=float)[hit_building],
        )
    heights = people_heights(dots, lots, tops, homes)
    data, origin = encode_people_file(dots, heights)
    return {
        'file': _write_content_file(data_dir, 'people', data),
        'count': len(dots),
        'origin': [round(origin[0], 7), round(origin[1], 7)],
        'per_dot': PEOPLE_PER_DOT,
        'max_zoom': PEOPLE_MAX_ZOOM,
        'groups': [{k: g[k] for k in ('key', 'label', 'color')} for g in PEOPLE_GROUPS],
    }


def encode_people_file(dots, heights) -> tuple[bytes, list]:
    """The cloud as the page reads it: float32 longitude and latitude
    offsets from an origin, float32 heights (m), then uint8 groups;
    returns the bytes and the origin [longitude, latitude]."""
    origin = [float(dots['longitude'].mean()), float(dots['latitude'].mean())]
    lon = (dots['longitude'].to_numpy() - origin[0]).astype('<f4')
    lat = (dots['latitude'].to_numpy() - origin[1]).astype('<f4')
    data = (
        lon.tobytes()
        + lat.tobytes()
        + np.asarray(heights, dtype='<f4').tobytes()
        + dots['group'].to_numpy(dtype=np.uint8).tobytes()
    )
    return data, origin


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


def type_codes(classes) -> np.ndarray:
    """Each feature's position in `TYPE_CATEGORIES` (0: unknown), from a
    lot's land use class or a building's class (`BUILDING_CLASS`)."""
    lookup = {
        value: code
        for code, category in enumerate(TYPE_CATEGORIES)
        for value in category['values']
    }
    return classes.astype('string').map(lookup).fillna(0).astype(np.uint8).to_numpy()


def lot_total_value(gdf, value_column) -> np.ndarray:
    """Each lot's whole value, land and buildings, for choosing tour
    stops: the roll's `total_value` where it has one, else land plus
    improvement value where both are known, else `value_column`.

    A roll may leave `total_value` empty throughout. Measured 2026-10-18
    on US-MA-NTC: 0 of 13,803 lots carry it, against 97% with a land
    value; the tour then summed nothing and its stops fell arbitrarily
    on the island's west end.
    """

    def column(name):
        if name not in gdf.columns:
            return np.full(len(gdf), np.nan)
        return gdf[name].to_numpy(dtype=float)

    total = column('total_value')
    parts = column('land_value') + column('improvement_value')
    total = np.where(total > 0, total, parts)
    return np.where(total > 0, total, column(value_column))


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
        visitor knows; so the lot's whole value (`lot_total_value`) is
        used wherever it is known, and this column otherwise.
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
            'value': np.nan_to_num(lot_total_value(gdf, value_column)),
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


def overlay_imputed_values(part, recipe, region, columns) -> tuple[pd.DataFrame, int]:
    """Take the model's values from a region's imputed twin bundle.

    The twin (`<region>.imputed`) sits beside the regular bundle and is
    read alongside it, never copied over it. Only cells whose `_source`
    carries `IMPUTATION_TOKEN` are taken, with their source, so each is
    marked imputed on the page; every other cell keeps the regular
    bundle's value.

    Parameters
    ----------
    part : pandas.DataFrame
        Rows read from the regular bundle, indexed like it.
    recipe : str
        The layer's recipe.
    region : str
        The regular region id.
    columns : sequence of str
        Columns to take where the model filled them.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The rows, and how many cells came from the model (0 without a
        twin).
    """
    if not columns:
        return part, 0
    try:
        canonical = delivery_paths(recipe, region=region)['canonical']
    except (KeyError, ValueError):
        # A region without a delivery path has no twin either.
        return part, 0
    twin = imputed_twin_dir(canonical, region)
    if twin is None:
        return part, 0
    twin = twin / canonical.name
    if not twin.exists():
        return part, 0
    wanted = [c for c in columns if c in part.columns]
    read = wanted + [f'{c}{SOURCE_SUFFIX}' for c in wanted]
    imputed = pd.read_parquet(twin, columns=read).reindex(part.index)
    part = part.copy()
    taken = 0
    taken_rows = {}
    for column in wanted:
        source = f'{column}{SOURCE_SUFFIX}'
        filled = (
            imputed[source]
            .astype('string')
            .str.contains(IMPUTATION_TOKEN)
            .fillna(False)
            & imputed[column].notna()
        )
        part.loc[filled, column] = imputed.loc[filled, column]
        if source in part.columns:
            part[source] = part[source].astype('string')
            part.loc[filled, source] = imputed.loc[filled, source].astype('string')
        else:
            part[source] = imputed[source].astype('string').where(filled)
        taken += int(filled.sum())
        taken_rows[column] = filled
    part = flag_model_heights(part, recipe, taken_rows)
    return part, taken


def fill_withheld_values(part, recipe, units) -> tuple[pd.DataFrame, int]:
    """Fill the cells a bundle withheld in `units` from the curated
    per-unit files, for a restricted page (`audience: restricted`).

    Only cells the bundle marks withheld (`WITHHELD_TOKEN` in their
    `_source`) and only in the units named are taken, with the curated
    file's own source, so the Sources panel names the roll. The units
    are listed by a person in `places.yaml` (`restricted_units`) after
    reading the source's terms, never derived: another source's
    withheld cells (a licence that forbids any display) stay withheld.
    The bundle itself is never changed.

    Parameters
    ----------
    part : pandas.DataFrame
        Rows read from a bundle, indexed by entity id like the curated
        files.
    recipe : str
        The layer's recipe.
    units : sequence of str
        Units (`admin3_id`) whose withheld cells are filled.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The rows, and how many cells were filled.
    """
    from openplaces.io.readers import get_entities

    units = [u for u in units if u in set(part['admin3_id'])]
    sources = [
        c
        for c in part.columns
        if c.endswith(SOURCE_SUFFIX)
        and (part[c].astype('string') == WITHHELD_TOKEN).any()
    ]
    if not units or not sources:
        return part, 0
    curated = get_entities(recipe, admin_id=units, geom=False)
    curated = curated.reindex(part.index)
    in_units = part['admin3_id'].isin(units)
    part = part.copy()
    filled = 0
    for source in sources:
        column = source[: -len(SOURCE_SUFFIX)]
        if column not in part.columns or column not in curated.columns:
            continue
        take = (
            in_units
            & (part[source].astype('string') == WITHHELD_TOKEN).fillna(False)
            & curated[column].notna()
        )
        if isinstance(part[column].dtype, pd.CategoricalDtype):
            # The roll's own labels may be categories the bundle lacks.
            part[column] = part[column].astype(object)
        part.loc[take, column] = curated.loc[take, column]
        part[source] = part[source].astype('string')
        if source in curated.columns:
            part.loc[take, source] = curated.loc[take, source].astype('string')
        filled += int(take.sum())
    return part, filled


def ghost_rule(recipe) -> dict | None:
    """The parameters of a recipe's `flag_unconfirmed_height` step, or
    None where its pipeline has none (the parcel recipe)."""
    from openplaces.recipe import get_recipe_by_id

    spec = get_recipe_by_id(recipe) if isinstance(recipe, str) else recipe
    step = next(
        (
            s
            for s in (spec or {}).get('pipeline', [])
            if s.get('step') == 'flag_unconfirmed_height'
        ),
        None,
    )
    return None if step is None else {k: v for k, v in step.items() if k != 'step'}


def flag_model_heights(part, recipe, taken_rows) -> pd.DataFrame:
    """Judge the model's values by the same height check as recorded ones.

    Curate flags an outline whose value per area its floors do not
    explain (`GHOST_COLUMN`, `io.curator.evidence.flag_unconfirmed_height`)
    before the imputed twin exists, so a cell the model filled carries
    no flag of its own: the page drew $15M to $456M on houses of 13 to
    213 m2 at full height.
    Measured 2026-10-19 on cheer-eastern-nc: 196 imputed values over $20M

    The check reruns here with the recipe's own step and parameters,
    over the whole part so that a lot's other outlines count, and only
    the rows whose value came from the model take its answer.

    Parameters
    ----------
    part : pandas.DataFrame
        Rows of one bundle, the model's values already overlaid.
    recipe : str or dict
        The layer's curation recipe.
    taken_rows : dict of str to pandas.Series of bool
        Per column, the rows whose value came from the model.

    Returns
    -------
    pandas.DataFrame
        *part*, with `GHOST_COLUMN` rewritten on those rows.
    """
    from types import SimpleNamespace

    from openplaces.io.curator.evidence import flag_unconfirmed_height

    rule = ghost_rule(recipe)
    if rule is None:
        return part
    rule = {**rule, 'output_column': GHOST_COLUMN}
    value_column = rule.get('value_column', 'structure_value')
    filled = taken_rows.get(value_column)
    if filled is None or not filled.any():
        return part
    probe = SimpleNamespace(curated=part.copy(), verbose=False)
    flag_unconfirmed_height(probe, **rule)
    judged = probe.curated[GHOST_COLUMN].astype('boolean')
    current = (
        part[GHOST_COLUMN].astype('boolean')
        if GHOST_COLUMN in part.columns
        else pd.Series(pd.NA, index=part.index, dtype='boolean')
    )
    part[GHOST_COLUMN] = current.where(~filled, judged)
    return part


def imputed_twin_dir(bundle_file, region):
    """The folder of a region's imputed twin bundle, or None.

    A region's bundle sits in a folder of its own name, its twin beside
    it (`<region>.imputed`); a state's only region sits in the state's
    folder itself, its twin then inside it (measured 2026-10-19:
    cheer-coastal-tx, whose regular bundle is the US-TX file).
    """
    folder = Path(bundle_file).parent
    for candidate in (
        folder.parent / f'{folder.name}{IMPUTED_SUFFIX}',
        folder / f'{region}{IMPUTED_SUFFIX}',
    ):
        if candidate.is_dir():
            return candidate
    return None


def imputed_model_code(recipe, region) -> list[str]:
    """The model code lines (`MODEL_CODE_LINE`) of a region's imputed
    twin bundle's notice; empty without a twin or such a line."""
    try:
        terms = delivery_paths(recipe, region=region)['terms']
    except (KeyError, ValueError):
        return []
    twin = imputed_twin_dir(terms, region)
    notice = twin / terms.name if twin is not None else None
    if notice is None or not notice.exists():
        return []
    text = ' '.join(notice.read_text(encoding='utf-8').split())
    return MODEL_CODE_LINE.findall(text)


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

    Model values are shown per feature only: no page shows a mean,
    median or total of a model's values over a named area (tract, ZIP,
    town, region, the page itself), nor any summary of a model's
    per-home values (legal review, 2026-10-22; design-limits register,
    US8676680B2). The scale above is applied, never displayed.
    Aggregates of the assessor's recorded values are not affected.

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

    The class alone confirms nothing where the roll's own record says
    otherwise (2026-10-19), unless its use group also names a home
    (`RESIDENTIAL_USE`): the use group names a non-residential use
    (`NON_RESIDENTIAL_USE`: industrial land with two outlines seen from
    above), or the roll records no building (an improvement value of
    at most `NO_BUILDING_VALUE`: "Use Value | Forestry & Related", 51
    acres of New Hanover County land valued $19.4M; a 109-acre
    reservoir in Boston with no use group, called Single-Family by NSI
    alone and drawn at $52M per home). A farm or forest lot
    (`FARM_USE`) is never a home lot, even with a farmhouse on it: the
    farmhouse is a home, but its lot's whole land value would be
    counted to it, so the land is drawn per acre and the farmhouse
    shows as a home on the building layer. A use group naming a
    homesite keeps it a home lot. Codes a roll states in place of words
    (Wisconsin's property classes) are not read. A lot under
    `NEW_HOME_MAX_HA` the roll calls vacant keeps its class: a new home
    the roll has not caught up with. A manufactured home's land record
    often carries no building, the home being personal property; its
    use group names it, so the test spares it, as it spares a park of
    them that the roll files as commercial. A class drawn from building
    models alone was tried as a third test and dropped: the Nantucket
    bundle states neither use group nor improvement value for many
    houses.
    Measured 2026-10-19 on nantucket-ma: 150 houses lost to that test

    Two more tests apply where the roll counts no dwellings itself
    (maintainer, 2026-10-24): a lot at least `PUBLIC_OPEN_SPACE_SHARE`
    under protected open space of a public owner
    (`PUBLIC_OPEN_SPACE_OWNERS`) is no home lot, and neither is a lot
    whose use subgroup names residential land
    (`RESIDENTIAL_LAND_SUBGROUP`) with no building on the roll, though
    its use group names a home.
    Measured 2026-10-24 on boston: 62 open-space lots, 244 land lots

    Parameters
    ----------
    lots : pandas.DataFrame
        `n_dwellings`, and where present `land_use_class`, `use_group`,
        `use_subgroup`, `improvement_value`, `area_ha`,
        `protected_open_space_share`, `protected_open_space_owner_type`
        and the roll's own count (`ROLL_DWELLINGS`).

    Returns
    -------
    numpy.ndarray of bool
    """
    tests = _home_lot_tests(lots)
    return (
        (lots['n_dwellings'].to_numpy(dtype=float) > 0)
        & tests['home_class']
        & tests['confirmed']
        & ~tests['farm']
        & ~tests['public_land']
        & ~tests['bare_land']
    )


def _home_lot_tests(lots) -> dict:
    """The tests of `residential_lots`, each an array of bool per lot,
    apart from the dwellings: `home_class`, `confirmed`, `farm`,
    `public_land`, `bare_land` and `no_building`, and `roll_count`
    (floats)."""
    empty = pd.Series(pd.NA, index=lots.index, dtype='string')
    dwellings = (
        lots['n_dwellings'].to_numpy(dtype=float)
        if 'n_dwellings' in lots
        else np.full(len(lots), np.nan)
    )
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
    names_home = use.str.contains(RESIDENTIAL_USE).fillna(False).to_numpy()
    small = (
        (pd.to_numeric(lots['area_ha'], errors='coerce') < NEW_HOME_MAX_HA)
        .fillna(False)
        .to_numpy()
        if 'area_ha' in lots
        else np.zeros(len(lots), dtype=bool)
    )
    new_home = small & use.str.contains('vacant|open.?space').fillna(False).to_numpy()
    # A park of manufactured homes is a business to many rolls.
    park = land_use.eq('Manufactured Home').fillna(False).to_numpy() & (dwellings >= 2)
    use_says_other = (
        use.str.contains(NON_RESIDENTIAL_USE).fillna(False).to_numpy()
        & ~names_home
        & ~new_home
        & ~park
    )
    no_building = (
        (pd.to_numeric(lots['improvement_value'], errors='coerce') <= NO_BUILDING_VALUE)
        .fillna(False)
        .to_numpy()
        & ~names_home
        & ~new_home
        & ~park
        if 'improvement_value' in lots
        else np.zeros(len(lots), dtype=bool)
    )
    own_class = land_use.isin(SINGLE_HOME_LAND_USE).fillna(False).to_numpy() & ~(
        exempt | use_says_other | no_building
    )
    confirmed = (roll_count > 0) | use_says_home | own_class
    farm = use.str.contains(FARM_USE).fillna(False).to_numpy() & ~names_home
    counted_by_roll = roll_count > 0
    owner = (
        lots['protected_open_space_owner_type'].astype('string')
        if 'protected_open_space_owner_type' in lots
        else empty
    )
    open_share = (
        pd.to_numeric(lots['protected_open_space_share'], errors='coerce').to_numpy()
        if 'protected_open_space_share' in lots
        else np.full(len(lots), np.nan)
    )
    public_land = (
        owner.isin(PUBLIC_OPEN_SPACE_OWNERS).fillna(False).to_numpy()
        & (open_share >= PUBLIC_OPEN_SPACE_SHARE)
        & ~counted_by_roll
    )
    subgroup = (
        lots['use_subgroup'].astype('string').str.lower()
        if 'use_subgroup' in lots
        else empty
    )
    without_building = (
        (pd.to_numeric(lots['improvement_value'], errors='coerce') <= NO_BUILDING_VALUE)
        .fillna(False)
        .to_numpy()
        if 'improvement_value' in lots
        else np.zeros(len(lots), dtype=bool)
    )
    bare_land = (
        subgroup.str.contains(RESIDENTIAL_LAND_SUBGROUP).fillna(False).to_numpy()
        & without_building
        & ~counted_by_roll
    )
    return {
        'home_class': land_use.isin(RESIDENTIAL_LAND_USE).fillna(False).to_numpy(),
        'confirmed': confirmed,
        'farm': farm,
        'public_land': public_land,
        'bare_land': bare_land,
        'no_building': no_building,
        'roll_count': roll_count,
    }


def implied_min_homes(lots) -> np.ndarray:
    """The least dwellings a lot's use subgroup implies, NaN where it
    implies none.

    "Apartments over 8 units" implies 9, "apartments 4-8 units" 4, and
    public housing `PUBLIC_HOUSING_MIN_HOMES`. Read from the subgroup's
    own words, so a roll naming its classes this way needs no list.
    """
    if 'use_subgroup' not in lots:
        return np.full(len(lots), np.nan)
    subgroup = lots['use_subgroup'].astype('string').str.lower()
    over = pd.to_numeric(
        subgroup.str.extract(r'over (\d+) units?', expand=False), errors='coerce'
    )
    span = pd.to_numeric(
        subgroup.str.extract(r'(\d+)\s*-\s*\d+ units?', expand=False), errors='coerce'
    )
    least = (over + 1).fillna(span)
    housing = subgroup.str.contains('public housing').fillna(False)
    least = least.where(~housing | least.notna(), float(PUBLIC_HOUSING_MIN_HOMES))
    return least.to_numpy(dtype=float)


def undercounted_lots(lots) -> np.ndarray:
    """Lots counting fewer dwellings than their use subgroup implies
    (`implied_min_homes`).

    Their count is wrong, not their homes: an apartment complex of
    twenty buildings whose roll states 1 to 8 units, or whose count
    came from one address point per building. The per-home view shows
    them as homes of unknown count, with no value per home
    (maintainer, 2026-10-24).
    Measured 2026-10-24 on boston: 1,313 lots
    """
    least = implied_min_homes(lots)
    dwellings = lots['n_dwellings'].to_numpy(dtype=float)
    return np.isfinite(least) & (np.nan_to_num(dwellings) < least)


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
    place_title=None,
    model_files=None,
    lots_file=None,
    models_public=False,
    notes=None,
    land_mask=True,
    people_cloud=True,
    preview_team=False,
    restricted_units=None,
    restricted_page=None,
    unrestricted_page=None,
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
    place_title : str, optional
        The place's name ("Boston area"), shown at the top of the page
        for a few seconds as it opens (maintainer, 2026-10-18).
    model_files : ModelFiles, optional
        Land value models handed over as files (`read_model_files`):
        joined onto the lots by `parcel_id`, they replace the layer's
        own `model_views`, and they make the page a team page
        (`audience: team`), published only behind a password.
    lots_file : str or Path, or a list of them, optional
        Model files carrying the lots' outlines, one per unit, for
        units with no bundle (`lots_from_model_file`): the scene is then
        their lots
        alone, read from that file, and `region` is not read. Needs
        `model_files`; the sidecar's credits are the Sources.
    models_public : bool
        Draw `model_files`' views on a public page instead of making it
        a team page: only for models whose training data's terms allow
        publishing their predictions (the BEA greeting page, maintainer
        2026-10-23: DC, Prince George's and Montgomery). Never applies
        to the layer's own `model_views` (PLACES-FMV), which stay on
        team pages.
    notes : sequence of dict, optional
        Further `name` and `credit` notes for the Sources panel, e.g.
        which units of the circle are left out and why.
    land_mask : bool
        Draw only the land of lots that reach into open water mapped by
        NHD (`mask_water`); their values are unchanged. Where a state's
        NHD files are missing, its lots are drawn whole.
    people_cloud : bool
        Add the people cloud (`race_dot_cloud`) where the census block
        counts are on this machine and the page's audience is in
        `PEOPLE_AUDIENCES` (team pages only); the page shows it behind
        a switch, off by default.
    preview_team : bool
        Mark the page a team page though it reads public bundles, so a
        local preview can show the people cloud before any team region
        is declared. Allowed only with `local_only`, so such a page is
        never published.
    restricted_units : sequence of str, optional
        Units whose withheld cells are filled from the curated files
        (`fill_withheld_values`). The page is then a restricted page
        (`audience: restricted`), which `deploy_web_scene` publishes
        only behind the restricted tokens (`RESTRICTED_TIER`).
    restricted_page : str, optional
        Relative address of this page's restricted twin
        (`../<slug>-restricted/`). The page offers a switch to it, shown
        only to a visitor the twin's gate lets in.
    unrestricted_page : str, optional
        On a restricted page, the address of the page it is the twin
        of, where its switch leads back.
    verbose : bool

    Returns
    -------
    Path
        The output directory.
    """
    # The import is here, not at module level, because lonboard (which
    # terrain imports) belongs to the optional viz-fast extra.
    from openplaces.viz.terrain import show_value_terrain_layer

    # Frames handed over as files, by layer name, instead of bundles.
    given = {}
    given_credits = {}
    if lots_file is not None:
        if model_files is None:
            raise ValueError('A lots file comes with its model files.')
        given['parcels'], lots_layer = lots_from_model_file(lots_file, layers[0])
        given_credits['parcels'] = list(model_files.data_credits)
        layers = (lots_layer,)
    refuse_personal_fields(layers)
    # Model files make a team page whatever bundles it reads.
    if models_public and model_files is None:
        raise ValueError('Public model views come only from handed-over model files.')
    if preview_team and not local_only:
        raise ValueError('A team preview of public bundles is local only.')
    audience = (
        'team'
        if preview_team or (model_files is not None and not models_public)
        else scene_audience(region)
    )
    restricted_units = list(restricted_units or [])
    if restricted_units:
        audience = RESTRICTED_AUDIENCE
    # Cells filled from the curated files (`fill_withheld_values`).
    restricted_cells = 0
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
    # Per layer offering a per-home view: which features are homes.
    home_lots = {}
    # Cells the building imputation supplied (`overlay_imputed_values`).
    imputed_cells = 0
    # The model code lines their bundles' notices state.
    model_code = []
    # Per county, the factor of the models' views in today's dollars.
    model_dollars = {}
    estimated_by_layer = {}
    source_docs = {}
    bounds = {}
    manifest_layers = []
    # The types any layer draws, and lots per type: the legend lists
    # the types drawn, those on most lots first.
    type_drawn = np.zeros(len(TYPE_CATEGORIES), dtype=bool)
    type_count = np.zeros(len(TYPE_CATEGORIES), dtype=np.int64)
    notices = {}
    gdfs = {}
    for layer in layers:
        if layer.name in given:
            gdf = given[layer.name][given[layer.name]['admin3_id'].isin(admin_ids)]
            notices[layer.recipe] = ''
        elif region is None:
            from openplaces.io.readers import get_entities

            gdf = get_entities(layer.recipe, admin_id=admin_ids, geom=True)
            notices[layer.recipe] = 'Local preview from curated files; not published.'
        else:
            # Several regions: each unit is read from the bundle of the
            # region it belongs to. An entity on a boundary between two
            # regions can sit in both bundles; the first copy is kept.
            # A layer reads only the regions its own recipe ships: the
            # parcels and the buildings of one place may come from
            # differently named regions (Houston: parcels in
            # `houston-tx`, buildings in `cheer-coastal-tx`).
            regions = [
                layer_region(layer.recipe, name)
                for name in ([region] if isinstance(region, str) else region)
            ]
            parts = []
            used = []
            for name in regions:
                try:
                    members = set(_region_members(layer.recipe, name))
                except KeyError:
                    # A region of the other layer's recipe.
                    continue
                wanted = [a for a in admin_ids if a in members]
                if wanted:
                    part = read_bundle(layer.recipe, name, wanted)
                    part, taken = overlay_imputed_values(
                        part, layer.recipe, name, layer.imputed_columns
                    )
                    imputed_cells += taken
                    part, released = fill_withheld_values(
                        part, layer.recipe, restricted_units
                    )
                    restricted_cells += released
                    if taken:
                        model_code += imputed_model_code(layer.recipe, name)
                    if layer.recorded_column and layer.recorded_column not in part:
                        extra = read_evidence_columns(
                            layer.recipe, name, [layer.recorded_column]
                        )
                        part = part.join(extra, how='left')
                    if layer.per_dwelling_label and layer.stack_on is None:
                        # The lots' subgroup and open-space owner, for
                        # `residential_lots`.
                        wanted_evidence = [
                            c for c in LOT_EVIDENCE_COLUMNS if c not in part
                        ]
                        extra = read_evidence_columns(
                            layer.recipe, name, wanted_evidence
                        )
                        part = part.join(extra, how='left')
                    parts.append(part)
                    used.append(name)
            if not parts:
                raise ValueError(f'No unit of {admin_ids} is in {regions}.')
            gdf = pd.concat(parts)
            gdf = gdf[~gdf.index.duplicated(keep='first')]
            notices[layer.recipe] = '\n\n'.join(
                bundle_notice(layer.recipe, name) for name in used
            )
        if model_files is not None and layer.model_views and layer.name not in given:
            # Before outlines merge, while each row is still one lot.
            gdf = gdf.join(model_files.table, how='left')
            if verbose:
                found = gdf[model_files.views[0].log_column].notna().sum()
                print(f'{layer.name}: models for {found:,} of {len(gdf):,} lots')
        if crop is not None:
            gdf = crop_to_area(gdf, crop)
            if verbose:
                print(f'{layer.name}: {len(gdf):,} features inside the crop')
        if 'occupancy_type' in gdf.columns:
            gdf = gdf.assign(**{BUILDING_CLASS: building_class(gdf)})
        gdfs[layer.name] = merge_identical_outlines(gdf, layer)
    if land_mask:
        bottom = layers[0]
        above = next((u for u in layers if u.stack_on == bottom.name), None)
        lots = gdfs[bottom.name]
        if not lots.empty:
            gdfs[bottom.name] = mask_water(
                lots,
                water_polygons(admin_ids, lots.to_crs(4326).total_bounds),
                gdfs[above.name] if above is not None else None,
            )
    _mark_exempt_ground(gdfs, layers)
    _count_dwellings_from_above(gdfs, layers)
    mark_land_ghosts(gdfs, layers)

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
        if layer.name in given:
            # The files' own sources, as their sidecars credit them.
            source_docs[layer.name] = {
                'sources': [
                    {'name': c['name'], 'share': 1.0, 'terms': c['credit'], 'url': None}
                    for c in given_credits.get(layer.name, [])
                ],
                'share_alike': [],
            }
        provenance = f'{layer.value_column}{SOURCE_SUFFIX}'
        if provenance in gdf.columns:
            source_docs[layer.name]['values'] = value_sources(gdf[provenance])
            # Per unit too, where a scene spans several: one county's
            # source can differ, or be withheld.
            if len(names) > 1 and 'admin3_id' in gdf.columns:
                source_docs[layer.name]['values_by_unit'] = value_sources_by_unit(
                    gdf[provenance], gdf['admin3_id'], names
                )
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
        if LAND_SHAPE in out.columns:
            # The land part of a lot reaching into water (`mask_water`).
            land_shapes = out[LAND_SHAPE].to_numpy()
            wet = np.array([shape is not None for shape in land_shapes], dtype=bool)
            geometry = geometry.copy()
            geometry[wet] = shapely.from_wkb(land_shapes[wet].tolist())
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
                # The lots the people cloud floats over, homes of
                # unknown count included (`race_dot_cloud`).
                home_lots[layer.name] = residential.copy()
                known = np.ones(len(out), dtype=bool)
                # Homes of unknown count: no value per home, and not
                # drawn as non-residential either (`undercounted_lots`).
                unknown_count = residential & undercounted_lots(out)
                residential = residential & ~unknown_count
                known = ~unknown_count
                # The tooltip's "Homes": the roll's own count, or the
                # count from outlines on a lot shown as homes; never an
                # outline count on a lot that is not (a monument on a
                # cemetery lot read "Homes 1"; maintainer, 2026-10-24).
                roll = (
                    out[ROLL_DWELLINGS].to_numpy(dtype=float)
                    if ROLL_DWELLINGS in out
                    else np.full(len(out), np.nan)
                )
                out['n_dwellings'] = out['n_dwellings'].where(residential | (roll > 0))
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
        # The type view: one category per feature, from the same list
        # for lots and buildings (`TYPE_CATEGORIES`).
        type_column = (
            BUILDING_CLASS if BUILDING_CLASS in out.columns else 'land_use_class'
        )
        if type_column in out.columns:
            codes = type_codes(out[type_column])
            extra['type_code'] = codes
            n_types = len(TYPE_CATEGORIES)
            type_drawn |= np.bincount(codes, minlength=n_types) > 0
            if layer is layers[0]:
                # Lots per type, for the legend's order: by building,
                # outbuildings ranked second in Wilmington, and by
                # area the order lost the most common homes
                # (maintainer, 2026-10-19).
                type_count += np.bincount(codes, minlength=n_types)
            views['type'] = {'label': 'Type', 'unit': 'type', 'categorical': True}
        if layer.model_views and (audience in GATED_AUDIENCES or models_public):
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
            if model_files is not None:
                # Already in one year's dollars: no index applies.
                dollars = np.ones(len(out))
                model_dollars = {
                    'models': {
                        'factor': 1.0,
                        'to_year': model_files.year,
                        'estimated': False,
                    }
                }
                shown_layer = replace(layer, model_views=model_files.views)
            else:
                dollars, model_dollars = lot_dollars_factors(out)
                shown_layer = layer
            arrays, model_views = model_view_arrays(
                out,
                shown_layer,
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
                # Features drawn; for the lots, the page's parcel count,
                # which orders the site's places (`site.landing`).
                'n_features': int(len(out)),
                'n_ghosts': int(ghost.sum()),
            }
        )

    people = None
    if people_cloud and audience in PEOPLE_AUDIENCES and layers[0].name in home_lots:
        people = _people_cloud(
            admin_ids, crop, layers, rendered, home_lots[layers[0].name], data_dir
        )
    if people is not None:
        notes = [
            *(notes or []),
            {
                'name': PEOPLE_NOTE['name'],
                'credit': PEOPLE_NOTE['credit'].format(per_dot=people['per_dot']),
            },
        ]
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

    # The page opens, and the tour starts, over the circle's center
    # (maintainer, 2026-10-19); a place of whole units over theirs. The
    # units a circle touches reach far beyond it, so their centroid sat
    # off to one side (Boston: the middle of the towns, not BU).
    center = crop.centroid if crop is not None else admin.union_all().centroid
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
        if {'n_dwellings', per_dwelling} <= set(lots.columns):
            homes = np.isfinite(lots[per_dwelling].to_numpy(dtype=float))
            total = lot_total_value(lots, bottom.value_column) / lots[
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
        'preview': region is None and not given,
        'local_only': bool(local_only),
        'audience': audience,
        'restricted_page': restricted_page,
        'unrestricted_page': unrestricted_page,
        'restricted_cells': restricted_cells,
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
        # The people cloud (`race_dot_cloud`), or None.
        'people': people,
        'tour_downtown': downtown_stop(downtown, admin),
        # The nationwide color bounds per display unit, beside the
        # scene's own (local) ones in the palettes and views.
        'national_bounds': {unit: list(b) for unit, b in NATIONAL_BOUNDS.items()},
        'place_title': place_title,
        # The type view's categories, indexed by each feature's
        # type_code; the same list for lots and buildings.
        'types': [
            {
                **{k: c[k] for k in ('key', 'label', 'color')},
                'present': bool(type_drawn[code]),
                # Lots of the type.
                'count': int(type_count[code]),
            }
            for code, c in enumerate(TYPE_CATEGORIES)
        ],
        # Team pages with models: the factors to today's dollars per
        # county, and the models' credits for the Sources panel.
        'model_dollars': model_dollars,
        # Notes on values a model filled, for the Sources panel.
        'value_notes': [imputation_note(model_code)] if imputed_cells else [],
        # Notes on what the page shows (`notes`), e.g. units left out.
        'page_notes': [{'name': n['name'], 'credit': n['credit']} for n in notes or []],
        # The year the models describe: PLACES-FMV's 2010, or none for
        # models handed over in today's dollars (`model_files`).
        'model_year': None if model_files is not None else MODEL_REFERENCE[0],
        'model_basis': model_files.basis if model_files is not None else '',
        'model_sources': (
            (
                model_credits(model_files, names)
                if model_files
                else list(MODEL_SOURCE_NOTES)
            )
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
        if item.name in ('landing.html', 'methods.html', ACCESS_PAGE):
            # Site pages beside the scenes, written by `landing.py` and
            # `deploy_landing`.
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
        through openplaces. `ACCESS_TOKENS` names the local file of
        access tokens (`read_access_tokens`; by default
        `ACCESS_TOKENS_FILE` beside the settings file); every deploy
        needs tokens in it.
        `ACCESS_CONTACT` is the email address the access page names
        (`access_page`); the landing page's deploy needs it.

    Returns
    -------
    dict
        `host`, `user`, `path`, `key`, `known_hosts`, `url`,
        `team_passwd`, `access_tokens`, `access_contact`.
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
        # Not `_file`: `tokens add` creates the file on first use.
        'access_tokens': values.get(DEPLOY_PREFIX + 'ACCESS_TOKENS')
        or str(env_file.parent / ACCESS_TOKENS_FILE),
        'access_contact': values.get(DEPLOY_PREFIX + 'ACCESS_CONTACT', ''),
    }


# The site is shared by invitation (maintainer, 2026-10-23): a request
# passes with a listed token, in the address (`?t=`) or in the cookie
# the first such request sets; any other is sent to `ACCESS_PAGE`. One
# token per person, so one person's access can be withdrawn alone.
ACCESS_COOKIE = 'op_access'
# Thirty days, in minutes (mod_rewrite's unit for a cookie's life).
ACCESS_COOKIE_MINUTES = 43_200
ACCESS_PAGE = 'access.html'
# The local token file's default name, beside the deploy settings.
ACCESS_TOKENS_FILE = 'value-access-tokens.tsv'
# Random URL-safe tokens (`secrets.token_urlsafe`): letters, digits,
# `-` and `_`, which a regular expression takes as they are.
ACCESS_TOKEN_PATTERN = re.compile(r'^[A-Za-z0-9_-]{16,64}$')
# A token's tier (the token file's third column) that also opens the
# restricted pages (`audience: restricted`): values a bundle withholds,
# shown under terms that allow personal use only (Arlington,
# `plans/findings/northern-virginia-roll-terms.md`). Personal use means
# one holder, the maintainer (2026-10-25); a second token would be
# distribution, which needs the source's written permission first.
RESTRICTED_TIER = 'restricted'
MAX_RESTRICTED_TOKENS = 1
RESTRICTED_AUDIENCE = 'restricted'
# Pages published only behind a gate beyond the site's token gate.
GATED_AUDIENCES = frozenset({'team', RESTRICTED_AUDIENCE})
# A restricted twin's folder: `<slug>-restricted`.
RESTRICTED_SUFFIX = '-restricted'


def read_access_tokens(path, tier=None) -> list[tuple[str, str]]:
    """The access tokens of the local token file, as (token, name).

    One `token<TAB>name[<TAB>tier]` per line; blank lines and `#`
    comments are skipped. The names stay on this machine: only tokens
    go to the host (`access_htaccess`).

    Parameters
    ----------
    path : str or Path
        The token file.
    tier : str, optional
        Only the tokens of this tier (`RESTRICTED_TIER`); None reads
        every token.

    Raises
    ------
    ValueError
        If the file is not named, holds no token (of the tier), a line
        is not a valid token, or more tokens are restricted than
        `MAX_RESTRICTED_TOKENS`.
    """
    if not path:
        raise ValueError(
            f'No access tokens: name the token file in {DEPLOY_PREFIX}ACCESS_TOKENS '
            'and add one with `python -m openplaces.viz.web_scene.tokens add <name>`. '
            'No page of the value site is published without the token gate.'
        )
    path = Path(path).expanduser()
    tokens = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            token, _, rest = line.partition('\t')
            name, _, token_tier = rest.partition('\t')
            token = token.strip()
            if not ACCESS_TOKEN_PATTERN.match(token):
                raise ValueError(f'{path}: not an access token: {token[:8]}...')
            if tier is None or token_tier.strip() == tier:
                tokens.append((token, name.strip()))
    if tier == RESTRICTED_TIER and len(tokens) > MAX_RESTRICTED_TOKENS:
        raise ValueError(
            f'{path} holds {len(tokens)} restricted tokens; at most '
            f'{MAX_RESTRICTED_TOKENS} until the source allows sharing in writing.'
        )
    if not tokens:
        what = f'{tier} access token' if tier else 'access token'
        raise ValueError(f'{path} holds no {what}; the site would let nobody in.')
    return tokens


def access_htaccess(tokens, url_path, host) -> str:
    """The token gate that heads every `.htaccess` of the value site.

    A request whose `t=` parameter or `ACCESS_COOKIE` matches a token
    passes; one with the token in its address also sets the cookie,
    so the link need not be kept. Every other request goes to
    `ACCESS_PAGE` (302); that page and the site's icons pass always.
    Written into the site's root and into each page folder, so no
    page, script or data file is served without it.

    Parameters
    ----------
    tokens : sequence of str
        Valid tokens (`read_access_tokens`).
    url_path : str
        The site's path on its host, e.g. `/values`.
    host : str
        The site's host name, the cookie's domain (`placeslab.org`).

    Returns
    -------
    str
    """
    tokens = list(tokens)
    if not tokens or not all(ACCESS_TOKEN_PATTERN.match(t) for t in tokens):
        raise ValueError('The token gate needs valid access tokens.')
    if not re.fullmatch(r'[A-Za-z0-9.-]+', host):
        raise ValueError(f'Not a host name: {host!r}.')
    url_path = '/' + url_path.strip('/')
    alternatives = '|'.join(tokens)
    exempt = '|'.join(re.escape(f) for f in (ACCESS_PAGE, *ICON_FILES))
    cookie = f'{ACCESS_COOKIE}=(?:{alternatives})'
    return (
        '# Shared by invitation: a listed access token, in the address\n'
        '# (?t=) or in the cookie that address sets, or the access page.\n'
        'RewriteEngine On\n'
        f'RewriteCond %{{QUERY_STRING}} (?:^|&)t=({alternatives})(?:&|$)\n'
        f'RewriteRule ^ - [CO={ACCESS_COOKIE}:%1:{host}:'
        f'{ACCESS_COOKIE_MINUTES}:{url_path}:secure:httponly]\n'
        f'RewriteCond %{{REQUEST_URI}} !(?:^|/)(?:{exempt})$\n'
        f'RewriteCond %{{QUERY_STRING}} !(?:^|&)t=(?:{alternatives})(?:&|$)\n'
        f'RewriteCond %{{HTTP_COOKIE}} !(?:^|;\\s*){cookie}(?:;|$)\n'
        # QSD: a refused token does not travel on to the access page.
        f'RewriteRule ^ {url_path}/{ACCESS_PAGE} [R=302,L,QSD]\n'
        '<IfModule mod_headers.c>\n'
        '  Header set X-Robots-Tag "noindex, nofollow"\n'
        '</IfModule>\n\n'
    )


def access_page(settings) -> str:
    """The page a visitor without a token sees, with the contact address
    of the deploy settings (`ACCESS_CONTACT`), which never enters the
    repository."""
    import html

    contact = settings.get('access_contact', '')
    if not re.fullmatch(r'[^@\s<>"]+@[^@\s<>"]+\.[A-Za-z]{2,}', contact):
        raise ValueError(
            f'Set {DEPLOY_PREFIX}ACCESS_CONTACT to the email address the access '
            'page names for requests.'
        )
    page = (STATIC_DIR / ACCESS_PAGE).read_text(encoding='utf-8')
    return page.replace('{{CONTACT}}', html.escape(contact))


def site_htaccess(settings, team=False, restricted=False) -> str:
    """A page folder's or the site root's `.htaccess`: the token gate,
    the team password where `team`, then the usual rules.

    A `restricted` page's gate lets in only the restricted tokens
    (`RESTRICTED_TIER`) and asks for no password: the switch to it on
    its twin probes it in the background, and a password prompt would
    then open for every team member.
    """
    from urllib.parse import urlparse

    if not settings['url']:
        raise ValueError(f'{DEPLOY_PREFIX}URL is needed to place the token gate.')
    tier = RESTRICTED_TIER if restricted else None
    tokens = [t for t, _ in read_access_tokens(settings['access_tokens'], tier)]
    if restricted:
        url = urlparse(settings['url'])
        gate = access_htaccess(tokens, url.path, url.hostname or '')
        return gate + (STATIC_DIR / 'htaccess').read_text(encoding='utf-8')
    url = urlparse(settings['url'])
    gate = access_htaccess(tokens, url.path, url.hostname or '')
    if team:
        return gate + team_htaccess(settings['team_passwd'])
    return gate + (STATIC_DIR / 'htaccess').read_text(encoding='utf-8')


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
        # Apache's own 401 page: the host's error page (/401.shtml) does
        # not exist, and the WordPress rules above the site hand the
        # missing page to WordPress, which answers 404, so browsers
        # never asked for the password (seen 2026-10-19 on every team
        # page).
        'ErrorDocument 401 default\n'
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

    Every page goes up behind the site's token gate (`site_htaccess`):
    its `.htaccess` goes up before anything else, and the deploy
    refuses without access tokens (`ACCESS_TOKENS`). A team scene
    (`audience: team`) also asks for the team password (`team_htaccess`),
    and the deploy refuses when `TEAM_PASSWD` is not set or names no
    file on the host.

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
    restricted = manifest.get('audience') == RESTRICTED_AUDIENCE
    # The folder's name tells `site access` which gate it gets, so a
    # restricted page goes only to a `-restricted` folder.
    if restricted != bool(subpath and subpath.endswith(RESTRICTED_SUFFIX)):
        raise ValueError(
            f'{output_dir}: a restricted page is published only to a folder '
            f'ending in {RESTRICTED_SUFFIX!r}, and only a restricted page there '
            f'(subpath {subpath!r}).'
        )
    if team:
        passwd = settings['team_passwd']
        if not passwd:
            raise ValueError(
                f'{output_dir} is a team page and is published only behind a '
                f'password. Create the password file on the host and name it '
                f'in {DEPLOY_PREFIX}TEAM_PASSWD.'
            )
    # The token gate (and a team page's password) before any file of
    # the page. Written before the host check, so a bad setting fails
    # without a call to the host.
    (output_dir / '.htaccess').write_text(
        site_htaccess(settings, team, restricted), encoding='utf-8'
    )
    if team and not _rclone_output(env, 'lsf', f'bh:{settings["team_passwd"]}'):
        raise FileNotFoundError(
            f'{DEPLOY_PREFIX}TEAM_PASSWD names no file on the host '
            f'({settings["team_passwd"]}); the team page was not published.'
        )
    print('\nPhase 0: access (.htaccess) before any file of the page')
    _rclone(
        env,
        'copyto',
        str(output_dir / '.htaccess'),
        f'{target}/.htaccess',
        *_without_filters(RCLONE_SFTP_FLAGS),
        *dry,
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
    print('\nPhase 2: switch (index.html, viewer.js, scene.json, .htaccess, icons)')
    _rclone(
        env,
        'copy',
        str(output_dir),
        target,
        '--max-depth',
        '1',
        *RCLONE_SFTP_FLAGS,
        '--filter',
        '+ /{index.html,viewer.js,scene.json,.htaccess,' + ','.join(ICON_FILES) + '}',
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
    path: its page, card images, list of places and the access page,
    after the site root's token gate (`site_htaccess`); the scenes in
    the folders beside it are never touched."""
    output_dir = Path(output_dir)
    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    dry = [] if go else ['--dry-run']
    print(f'Target: {settings["user"]}@{settings["host"]}:{settings["path"]}')
    (output_dir / '.htaccess').write_text(site_htaccess(settings), encoding='utf-8')
    (output_dir / ACCESS_PAGE).write_text(access_page(settings), encoding='utf-8')
    _rclone(
        env,
        'copyto',
        str(output_dir / '.htaccess'),
        f'bh:{settings["path"]}/.htaccess',
        *_without_filters(RCLONE_SFTP_FLAGS),
        *dry,
    )
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
        f'+ /{ACCESS_PAGE}',
        '--filter',
        '+ /{' + ','.join(ICON_FILES) + '}',
        '--filter',
        '- **',
        *dry,
    )
    address = settings['url'] or settings['path']
    print(f'{"Published" if go else "Dry run complete"}: {address}/')


def deploy_access(site_dir, env_file=None, go=False) -> list[str]:
    """Upload the token gate alone: the `.htaccess` of the site root and
    of every page folder online, so an added or withdrawn token takes
    effect without a rebuild (`tokens add` / `tokens revoke`).

    The folders are those on the host (a page published from another
    machine included); a team page's folder keeps its password, read
    from its local manifest, and one with no local build is a team
    page if its `.htaccess` online asks for a password.

    Parameters
    ----------
    site_dir : str or Path
        The site's local folder (`site.site_root`).
    env_file : str or Path, optional
    go : bool
        Upload for real.

    Returns
    -------
    list of str
        The folders whose gate was written, the root as ''.
    """
    site_dir = Path(site_dir)
    settings = load_deploy_settings(env_file)
    env = _rclone_env(settings)
    dry = [] if go else ['--dry-run']
    root = f'bh:{settings["path"]}'
    # Three connections in all: the host blocks SSH for a while after a
    # burst, and one connection per folder ran into that after ten
    # folders (2026-10-23), leaving half the site gated.
    online = _rclone_output(env, 'lsf', '--dirs-only', root) or ''
    folders = [
        '',
        *sorted(n.rstrip('/') for n in online.split() if n and n != 'thumbs/'),
    ]
    written = []
    with tempfile.TemporaryDirectory() as tmp:
        # The `.htaccess` files online, all in one copy.
        current_dir = Path(tmp) / 'current'
        _rclone(
            env,
            'copy',
            root,
            str(current_dir),
            *_without_filters(RCLONE_SFTP_FLAGS),
            '--max-depth',
            '2',
            '--filter',
            '+ /*/.htaccess',
            '--filter',
            '- **',
        )
        upload = Path(tmp) / 'upload'
        for folder in folders:
            manifest = site_dir / folder / 'scene.json'
            current = current_dir / folder / '.htaccess'
            current = current.read_text(encoding='utf-8') if current.exists() else ''
            if folder and not manifest.exists() and 'RedirectMatch' in current:
                # An old address pointing at a page (`deploy_redirect`),
                # whose target is gated: left as it is.
                print(f'{folder}: redirect, left as it is')
                continue
            audience = (
                json.loads(manifest.read_text(encoding='utf-8')).get('audience')
                if manifest.exists()
                else None
            )
            # By its name as well as its manifest: a restricted folder
            # this machine has no build of must never get the general
            # gate, which every token passes.
            restricted = audience == RESTRICTED_AUDIENCE or str(folder).endswith(
                RESTRICTED_SUFFIX
            )
            if manifest.exists():
                team = audience == 'team'
            else:
                team = bool(folder) and 'AuthUserFile' in current
            target = upload / folder / '.htaccess'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                site_htaccess(settings, team and not restricted, restricted),
                encoding='utf-8',
            )
            kind = (
                'restricted tokens only'
                if restricted
                else 'token gate + team password'
                if team
                else 'token gate'
            )
            print(f'{folder or "(root)"}: {kind}')
            written.append(folder)
        # Every gate in one copy, nothing else.
        _rclone(
            env,
            'copy',
            str(upload),
            root,
            *_without_filters(RCLONE_SFTP_FLAGS),
            '--max-depth',
            '2',
            '--filter',
            '+ /.htaccess',
            '--filter',
            '+ /*/.htaccess',
            '--filter',
            '- **',
            *dry,
        )
    print(f'{"Updated" if go else "Dry run complete"}: {len(written)} folders')
    return written


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
