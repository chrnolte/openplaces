"""Divide each tax-roll account's values over the lots it is drawn on.

A roll states one land value, one improvement value, one total and one
dwelling count per account (one property spine row). Where the parcel
layer draws that account as several lots, earlier stages put a value
on each lot by three different rules: the parcel geospine copies the
account's whole value onto every lot sharing its key, a harmonize rule
moves units onto lots, and the link table's `share` divides by area.
`divide_account_values` replaces all three for the columns it names,
in the parcel curate: it decides every cell of a lot an account
reaches from the account's own statement, divided over the account's
lots, so the lots of an account add up to what the roll states for it
(plans/account-values-one-deciding-step.md, packages V3 and V5).

The rule per account, over its links to curated lots:

- land and other value follow lot area (the link `share`, renormalized
  over the lots the curate kept);
- improvement and dwellings follow the floor area of the building
  pieces standing on each lot (each piece counts for the lot it stands
  on); an account none of whose lots holds a piece divides them by lot
  area too;
- a stated total is kept and divided in proportion to each lot's
  decided parts; an account stating parts and no total gets the sum of
  its parts per lot, and an account stating a total and no parts
  divides it by lot area;
- a value the account does not state, which the parcel layer repeats
  on each of the account's lots, is taken once and divided by the same
  rules over the lots on which the layer states it (`layer_columns`).

Design limits checked: L2 (lots relate through the assessor's account
id in the link table, never by owner), L6 (one roll states each value;
the layer's figures and the roll's total stay separate columns from
separate sources, no weights between them), L17 and L28 (no link is
added, scored or removed: shared-key links are read as any other),
L22 (no source column is overwritten: the decided columns are curate
outputs with their own source labels), L25 (a value is divided only
over the lots for which the source states it for that account; a lot
where it states none receives none from a sibling lot), L27 (no
contiguity or neighbor rule: a lot's weight reads only the building
pieces inside it).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from openplaces.core.schema import is_synthetic_geometry
from openplaces.io.curator import CurateState, _register
from openplaces.io.curator.aggregation import (
    ROW_SOURCE_COLUMN,
    _row_source_tokens,
    _union_tokens,
    mask_units_rows,
    read_curate_link,
    units_only_links,
)
from openplaces.io.curator.estimators.stories import (
    FLOOR_HEIGHT_M,
    _first_in_range,
    round_half_up,
    stories_from_height,
)
from openplaces.io.curator.provenance import (
    mark_imputed,
    record_sources,
    source_column,
)
from openplaces.io.readers import describe_recipe, get_entities

#: Working column names of the per-link frame.
ACCOUNT = 'account'
LOT = 'lot'

#: Below this many dollars (or dwellings, for counts) a lot sum and the
#: account's value are taken to agree in the conservation report.
VALUE_TOLERANCE = 1.0
COUNT_TOLERANCE = 0.01

#: Default report file, written per unit through `openplaces.path`.
REPORT_FILENAME = 'account-values-conservation.csv'


def area_shares(
    account: pd.Series, lot: pd.Series, share: pd.Series, lot_area: pd.Series
) -> pd.Series:
    """Each link's share of its account by lot area.

    The link table's `share` where every link of the account carries
    one, else the curated lot's area; either way renormalized over the
    account's links given here (the lots the curate kept), so the shares
    of an account sum to one. The link table's shares sum to one over
    the geospine's lots, and the curate drops slivers and placeholders:
    read as stored, those lots would keep part of the account's value.
    An account whose lots have no area divides equally.
    """
    share = pd.to_numeric(share, errors='coerce').astype(float)
    area = pd.to_numeric(lot.map(lot_area), errors='coerce').astype(float)
    any_missing = share.isna().groupby(account).transform('any')
    basis = share.where(~any_missing, area).where(lambda b: b >= 0)
    total = basis.fillna(0).groupby(account).transform('sum')
    n = account.map(account.value_counts()).astype(float)
    usable = basis.notna().groupby(account).transform('all') & (total > 0)
    return (basis / total.where(total > 0)).where(usable, 1.0 / n)


def lot_piece_floor_area(
    overlaps: pd.DataFrame, stories: pd.Series, min_piece_m2: float
) -> pd.Series:
    """Per lot, the floor area of the building pieces standing on it.

    *overlaps* holds one row per building and lot it overlaps
    (`building`, `lot`, `overlap_m2`). Each piece counts for the lot it
    stands on, with its area times its building's story count (one
    story where the count is missing); a piece under *min_piece_m2* is
    ignored. Returns a Series indexed by lot.
    """
    area = pd.to_numeric(overlaps['overlap_m2'], errors='coerce').fillna(0.0)
    keep = (area >= min_piece_m2) & (area > 0)
    count = pd.to_numeric(overlaps['building'].map(stories), errors='coerce')
    count = count.where(count > 0, 1.0)
    return (area * count)[keep].groupby(overlaps.loc[keep, LOT]).sum()


def account_floor_area(
    links: pd.DataFrame,
    overlaps: pd.DataFrame,
    stories: pd.Series,
    min_piece_m2: float,
    cluster: pd.Series | None = None,
) -> tuple[pd.Series, dict]:
    """Floor area weighing each of an account's lots.

    A lot weighs the floor area of the building pieces standing on it
    (`lot_piece_floor_area`), whichever building or account a piece
    belongs to otherwise. Why pieces, and not whole buildings judged
    by a threshold: the maintainer's labels of borderline cases
    (2026-11-18) follow the parcel lines, so the part of a building on
    a lot belongs to that lot's account (a duplex split by a lot line,
    a large building on two lots, one outline drawn over separate
    buildings). Proportional weights have no cliff where a building
    flips from counting fully to not at all, and the rule never reads
    who owns a lot (design limit L2): only the account's link and the
    geometry of the pieces.

    A condominium cluster outline (*cluster*) weighs only for an
    account none of whose lots holds a piece of another drawn
    building. A cluster outline is drawn around a block of units and is
    often not one building (case 07 of the labels); beside drawn
    footprints it would compete with them, and alone it is the best
    outline the account has. Stand-in outlines are not in *overlaps*.

    Parameters
    ----------
    links : pandas.DataFrame
        One row per account and lot (`account`, `lot`).
    overlaps : pandas.DataFrame
        One row per drawn building and lot it overlaps (`building`,
        `lot`, `overlap_m2`).
    stories : pandas.Series
        Story count per building.
    min_piece_m2 : float
        Pieces smaller than this are ignored (noise of the overlay).
    cluster : pandas.Series, optional
        True per building for a condominium cluster outline.

    Returns
    -------
    tuple of (pandas.Series, dict)
        Floor area per row of *links*, and counts for the report.
    """
    area = pd.to_numeric(overlaps['overlap_m2'], errors='coerce').fillna(0.0)
    is_cluster = (
        overlaps['building'].map(cluster).fillna(False).astype(bool)
        if cluster is not None
        else pd.Series(False, index=overlaps.index)
    )
    drawn = overlaps[~is_cluster.to_numpy()]
    clusters = overlaps[is_cluster.to_numpy()]
    lot_drawn = lot_piece_floor_area(drawn, stories, min_piece_m2)
    lot_cluster = lot_piece_floor_area(clusters, stories, min_piece_m2)
    on_drawn = links[LOT].map(lot_drawn).fillna(0.0)
    on_cluster = links[LOT].map(lot_cluster).fillna(0.0)
    has_drawn = (on_drawn > 0).groupby(links[ACCOUNT]).transform('any')
    floor = on_drawn.where(has_drawn, on_cluster).astype(float)
    weighed = overlaps.loc[
        (area >= min_piece_m2).to_numpy() & overlaps[LOT].isin(set(links[LOT])),
        'building',
    ].unique()
    known = pd.to_numeric(stories.reindex(weighed), errors='coerce')
    stats = {
        'buildings_weighed': int(len(weighed)),
        'buildings_without_stories': int((~(known > 0)).sum()),
        'pieces_under_floor': int(((area > 0) & (area < min_piece_m2)).sum()),
        'accounts_cluster_set_aside': int(
            ((on_cluster > 0) & has_drawn).groupby(links[ACCOUNT]).any().sum()
        ),
    }
    return floor, stats


def _account_sum(values: pd.Series, account: pd.Series) -> pd.Series:
    """Per row, the sum of *values* over its account; missing where the
    account has no value at all."""
    stated = values.notna().groupby(account).transform('sum')
    return values.fillna(0).groupby(account).transform('sum').where(stated > 0)


def _weights(share: pd.Series, floor: pd.Series, account: pd.Series) -> pd.Series:
    """Floor area share where the account has any floor area, else *share*."""
    total = floor.groupby(account).transform('sum')
    return (floor / total.where(total > 0)).where(total > 0, share)


def divide(
    frame: pd.DataFrame,
    *,
    by_area: list[str],
    by_floor_area: list[str],
    total: str | None,
    parts: list[str],
    layer: pd.DataFrame | None = None,
    layer_columns: list[str] = (),
    layer_tolerance: float = VALUE_TOLERANCE,
) -> tuple[pd.DataFrame, dict]:
    """What each account puts on each of its lots.

    Parameters
    ----------
    frame : pandas.DataFrame
        One row per account and lot: `account`, `lot`, `share` (its
        area share, `area_shares`), `floor` (its floor area,
        `account_floor_area`) and the account's stated value of every
        column, missing where the account does not state it.
    by_area, by_floor_area : list of str
        Columns divided by lot area and by building floor area.
    total : str or None
        The total column.
    parts : list of str
        The columns a total is made of.
    layer : pandas.DataFrame, optional
        Per lot, the values the parcel layer states (*layer_columns*).
    layer_columns : list of str
        Columns where a value the account does not state is taken from
        the layer, once per account (see Notes).
    layer_tolerance : float
        Largest difference between the layer's values on an account's
        lots for them to read as one repeated value.

    Returns
    -------
    tuple of (pandas.DataFrame, dict)
        Per row of *frame*, each column's contribution (missing where
        the row decides nothing), with `{total}_computed` (a sum of
        parts) and `{column}_layer` (taken from the layer) flags, and
        counts for the report.

    Notes
    -----
    A layer value is taken where the account states no value of the
    column, it reaches two or more lots that no other account reaches,
    and the layer states the same value (within *layer_tolerance*) on
    every one of them: a statewide layer that repeats an account's
    value on each of its polygons. It is divided over those lots only,
    renormalizing their shares; a lot of the account where the layer
    states nothing receives nothing (design limit L25). Values that
    differ between the lots are each lot's own and are left as they
    are.
    """
    account = frame[ACCOUNT]
    share = frame['share'].astype(float)
    floor = frame['floor'].astype(float)
    weight = _weights(share, floor, account)
    out = pd.DataFrame({ACCOUNT: account, LOT: frame[LOT]}, index=frame.index)
    stats = {}
    for column in by_area:
        out[column] = share * frame[column]
    for column in by_floor_area:
        out[column] = weight * frame[column]
    for column in list(by_area) + list(by_floor_area):
        out[f'{column}_layer'] = False

    if layer is not None and len(layer_columns):
        accounts_on_lot = frame.groupby(LOT)[ACCOUNT].transform('nunique')
        for column in layer_columns:
            if column not in layer.columns or column not in out.columns:
                continue
            states = frame[column].notna().groupby(account).transform('any')
            value = pd.to_numeric(frame[LOT].map(layer[column]), errors='coerce')
            candidate = ~states & value.notna() & (accounts_on_lot == 1)
            grouped = value.where(candidate).groupby(account)
            n = grouped.transform('count')
            spread = grouped.transform('max') - grouped.transform('min')
            taken = candidate & (n >= 2) & (spread <= layer_tolerance)
            if not taken.any():
                stats[f'{column}_layer_accounts'] = 0
                continue
            once = value.where(taken).groupby(account).transform('mean')
            sub_share = share.where(taken)
            sub_share = sub_share / sub_share.groupby(account).transform('sum')
            if column in by_floor_area:
                sub_weight = _weights(sub_share, floor.where(taken, 0.0), account)
                part = sub_weight * once
            else:
                part = sub_share * once
            out[column] = out[column].where(~taken, part)
            out[f'{column}_layer'] = taken
            stats[f'{column}_layer_accounts'] = int(account[taken].nunique())

    if total:
        own_parts = [c for c in parts if c in out.columns]
        stated = out[own_parts].where(frame[own_parts].notna())
        lot_parts = stated.sum(axis=1, min_count=1)
        account_parts = _account_sum(lot_parts, account)
        proportion = lot_parts / account_parts.where(account_parts != 0)
        stated_total = pd.to_numeric(frame[total], errors='coerce')
        divided = stated_total * proportion.where(proportion.notna(), share)
        computed = stated_total.isna() & lot_parts.notna()
        out[total] = divided.where(stated_total.notna(), lot_parts)
        out[f'{total}_computed'] = computed
    return out, stats


def lot_values(contributions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Per lot, the sum of what its accounts put on it, column by column.

    A lot no account decides a column for is missing from that column.
    """
    grouped = contributions.groupby(LOT)
    return pd.DataFrame(
        {column: grouped[column].sum(min_count=1) for column in columns}
    )


def conservation(
    contributions: pd.DataFrame,
    accounts: pd.DataFrame,
    columns: list[str],
    masked: pd.Series,
) -> pd.DataFrame:
    """Per column, accounts whose lots do not add up to their value.

    An account is checked where it states the column and no row of it
    was left out by the stacked-units rule (*masked*, per account). The
    tolerance is a dollar, or a hundredth of a dwelling for counts.
    """
    rows = []
    sums = contributions.groupby(ACCOUNT)
    for column in columns:
        if column not in accounts.columns or column not in contributions.columns:
            continue
        stated = pd.to_numeric(accounts[column], errors='coerce')
        checked = stated.notna() & ~masked.reindex(stated.index, fill_value=False)
        checked &= stated.index.isin(contributions[ACCOUNT].unique())
        total = sums[column].sum(min_count=1).reindex(stated.index)
        deviation = (total.fillna(0) - stated).abs()[checked]
        tolerance = COUNT_TOLERANCE if column.startswith('n_') else VALUE_TOLERANCE
        rows.append(
            {
                'column': column,
                'accounts_checked': int(checked.sum()),
                'accounts_failing': int((deviation > tolerance).sum()),
                'largest_deviation': float(deviation.max()) if len(deviation) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def identity_failures(
    lots: pd.DataFrame, parts: list[str], total: str, tolerance: float = 1.0
) -> tuple[int, int]:
    """Lots with every part and the total, and those whose parts do not
    add up to the total (a reported check, never enforced)."""
    needed = [c for c in parts if c != 'other_value'] + [total]
    if any(c not in lots.columns for c in needed):
        return 0, 0
    present = lots[needed].notna().all(axis=1)
    summed = lots[[c for c in parts if c in lots.columns]].fillna(0).sum(axis=1)
    failing = present & ((summed - lots[total]).abs() > tolerance)
    return int(present.sum()), int(failing.sum())


def _read_evidence(state, spec: dict, buildings: pd.DataFrame, floor_height_m):
    """One story evidence column per building (see `building_stories`)."""
    from openplaces.io import read_parquet
    from openplaces.recipe import get_output_path, get_recipe_by_id

    column = spec.get('column')
    height = spec.get('height_column')
    if 'recipe_id' not in spec:
        if column not in buildings.columns:
            return None
        return pd.to_numeric(buildings[column], errors='coerce')
    path = get_output_path(
        get_recipe_by_id(spec['recipe_id']),
        state.admin_id,
        entity_recipe_id=spec.get('entity_recipe') or state.entity_recipe,
    )
    if not path.exists():
        return None
    evidence = read_parquet(path)
    key = spec.get('key')
    if key and key in evidence.columns:
        evidence = evidence.set_index(key)
    evidence = evidence[~evidence.index.duplicated()]
    if height and height in evidence.columns:
        values = stories_from_height(evidence[height], floor_height_m)
    elif column and column in evidence.columns:
        values = pd.to_numeric(evidence[column], errors='coerce')
    else:
        return None
    values.index = values.index.astype(str)
    lookup = buildings[key].astype(str) if key else buildings.index.astype(str)
    return pd.Series(lookup, index=buildings.index).map(values)


def building_stories(state, buildings: pd.DataFrame, stories: dict | None) -> pd.Series:
    """Story count per building from the evidence the parcel curate can read.

    The building curate's `decide_stories` sets the count a delivered
    building carries, but it runs after this curate and reads it (its
    roll gap fill reads the lot's count), so this step cannot read that
    decision. It applies the same steps 1 to 3 to the same evidence:
    the first value within *valid_range* of the ordered evidence, a
    measured height converted at *floor_height_m*, rounded half up.
    Only the roll gap fill (step 4) is missing, and a building without
    a count weighs as one story.

    *stories* holds `evidence` (a list; each entry names a `column` on
    the building rows, or a `recipe_id` with `entity_recipe`, `key` and
    a `column` or `height_column`), `floor_height_m` and `valid_range`.
    """
    spec = dict(stories or {})
    floor_height_m = float(spec.get('floor_height_m', FLOOR_HEIGHT_M))
    low, high = spec.get('valid_range', (1, 110))
    frame = pd.DataFrame(index=buildings.index)
    for position, entry in enumerate(spec.get('evidence') or []):
        values = _read_evidence(state, dict(entry), buildings, floor_height_m)
        if values is not None:
            frame[f'evidence_{position}'] = values.astype(float)
    if frame.empty or not len(frame.columns):
        return pd.Series(np.nan, index=buildings.index)
    value, _ = _first_in_range(frame, list(frame.columns), float(low), float(high))
    return round_half_up(value)


def _building_overlaps(state, building_recipe_id: str) -> pd.DataFrame | None:
    """The building geospine's overlay of buildings on lots, link-labeled
    pairs only (`building`, `lot`, `overlap_m2`), or None."""
    from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
    from openplaces.io.harmonizer.links import _resolve_reference_recipe

    resolved, _ = _resolve_reference_recipe(None, 'parcel', state.admin_id)
    if resolved is None:
        return None
    owner = get_link_owner_recipe_id(building_recipe_id)
    path = get_entity_link_path(owner, resolved, state.admin_id)
    if not path.exists():
        return None
    pairs = pd.read_parquet(path)
    if 'link' in pairs.columns:
        pairs = pairs[pairs['link'].notna()]
    id_column = 'building_id' if 'building_id' in pairs.columns else pairs.columns[0]
    return pd.DataFrame(
        {
            'building': pairs[id_column].astype(str).to_numpy(),
            LOT: pairs['parcel_id'].astype(str).to_numpy(),
            'overlap_m2': pd.to_numeric(
                pairs['area_intersection_m2'], errors='coerce'
            ).to_numpy(),
        }
    )


def _story_columns(stories: dict | None) -> list[str]:
    """Columns the story evidence reads off the building rows."""
    names = []
    for entry in (stories or {}).get('evidence') or []:
        if 'recipe_id' in entry:
            if entry.get('key'):
                names.append(entry['key'])
        elif entry.get('column'):
            names.append(entry['column'])
    return names


def _value_columns(arguments) -> list[str]:
    """Every column the step decides."""
    names = list(arguments.get('by_area') or []) + list(
        arguments.get('by_floor_area') or []
    )
    if arguments.get('total'):
        names.append(arguments['total'])
    return names


def _building_reads(arguments) -> list[str]:
    return ['geometry_source', *_story_columns(arguments.get('stories'))]


@_register(
    'divide_account_values',
    phase='gather',
    reads=['{area_column}', '{layer_columns}', '{layer_columns}_source'],
    writes=[_value_columns, lambda a: [source_column(c) for c in _value_columns(a)]],
    decides=[_value_columns],
    reads_from={
        'recipe_id': [
            _value_columns,
            lambda a: [source_column(c) for c in _value_columns(a)],
            ROW_SOURCE_COLUMN,
        ],
        'building_recipe_id': [_building_reads],
    },
)
def divide_account_values(
    state: CurateState,
    recipe_id: str,
    building_recipe_id: str | None = None,
    by_area: list[str] = ('land_value', 'other_value'),
    by_floor_area: list[str] = ('improvement_value', 'n_dwellings'),
    total: str | None = 'total_value',
    layer_columns: list[str] = (),
    min_piece_m2: float = 10.0,
    cluster_sources: list[str] = ('condo_cluster',),
    stories: dict | None = None,
    area_column: str = 'area_ha',
    report: str | None = REPORT_FILENAME,
) -> CurateState:
    """Decide each lot's account values by dividing every account over its lots.

    The one step that decides the account values on a lot. For every
    account of *recipe_id* (the property spine) linked to curated lots,
    including links found on a shared key, it divides the account's
    stated values over its lots:

    1. *by_area* columns (land, other value) by lot area share: the
       link's `share`, renormalized over the curated lots.
    2. *by_floor_area* columns (improvement, dwellings) by the floor
       area standing on each lot: every drawn building piece on the
       lot, its area times its story count, whatever building or
       account it belongs to otherwise (`account_floor_area`); pieces
       under *min_piece_m2* ignored. A condominium cluster outline
       weighs only where no other drawn piece stands on the account's
       lots. A parcel-shaped stand-in outline never weighs: it was
       drawn because a lot had a value and no outline, so it would
       weigh the value that made it.
    3. Where no lot of the account has weight, *by_floor_area* columns
       by lot area as well.
    4. *total*: a stated total divided in proportion to each lot's
       decided parts; the sum of the parts where the account states no
       total; by lot area where it states no part.
    5. *layer_columns*: a value the account does not state and the
       parcel layer repeats on its lots, taken once (see `divide`).

    A lot's value is the sum of what its accounts put on it, and every
    cell a stating account reaches is replaced, whatever an earlier
    step left there: the parcel geospine's whole copies, the link-share
    sums of `aggregate_from_entities` (which fill only empty cells and
    stay for units without a link table). A stated zero stays zero and
    a column no account of the lot states keeps the lot's own value,
    missing or the layer's. Stacked-unit rows follow the rule
    `aggregate_from_entities` applies (`mask_units_rows`).

    `{column}_source` names the sources that stated the value: each
    account's own cell label, or its row's source. A divided value is
    the source's statement divided, not an estimate, so it carries no
    `imputed` marker (`provenance`). A total computed as the sum of
    parts where the roll states none is a cell no source filled, and is
    marked imputed. A value taken from the layer keeps the lot's label.

    Writes a per-unit report (*report*, under the reports root):
    conservation per column (accounts whose lots do not add up to the
    account's value, which by construction should be none) and the
    per-lot identity land + improvement + other = total, reported and
    never enforced.

    Parameters
    ----------
    state : CurateState
        The curation state; the curated entity is the parcel.
    recipe_id : str
        The accounts' recipe (`US_property-spine-2026`), read through
        its link table to the curated parcels.
    building_recipe_id : str, optional
        The building spine (`US_building-spine-2026`): its rows give a
        building's area, geometry source and story evidence, and its
        geospine's parcel overlay the building-lot overlaps. None: every
        account divides by lot area.
    by_area, by_floor_area : list of str
        Columns divided by lot area and by building floor area.
    total : str or None
        The total column (default `total_value`).
    layer_columns : list of str
        Columns where the parcel layer's repeated value is taken once
        per account (default none).
    min_piece_m2 : float
        Building pieces smaller than this, in square meters, carry no
        weight (default 10): overlay noise along lot lines. A floor of
        0, 5 or 10 m2 barely changes a lot; a floor relative to the
        building (a piece under 2% of it) fails on cluster outlines,
        whose unit lots each hold a small part of a large outline.
        Measured 2026-11-18 on US-FL-SJ: to lot area, 2% floor 1,195 accounts, 10 m2 5.
    cluster_sources : list of str
        First parts of `geometry_source` that mark a condominium
        cluster outline (default `condo_cluster`).
    stories : dict, optional
        Story evidence (`building_stories`). None: every building weighs
        as one story.
    area_column : str
        Lot area in hectares (default `area_ha`).
    report : str or None
        Report file name; None writes none.
    """
    curated = state.curated
    by_area = [c for c in by_area]
    by_floor_area = [c for c in by_floor_area]
    columns = by_area + by_floor_area + ([total] if total else [])
    try:
        described = set(describe_recipe(recipe_id, state.admin_id).index)
    except (FileNotFoundError, OSError, ValueError) as exc:
        warnings.warn(
            f'divide_account_values: could not read {recipe_id} for '
            f'{state.admin_id}: {exc}. Skipping.',
            stacklevel=2,
        )
        return state
    present = [c for c in columns if c in described]
    if not present:
        return state
    links, path = read_curate_link(state, recipe_id)
    if links is None:
        if state.verbose:
            print(f'  divide_account_values: no link to {recipe_id}; skipped.')
        return state
    sidecars = [source_column(c) for c in present if source_column(c) in described]
    row_source = ROW_SOURCE_COLUMN if ROW_SOURCE_COLUMN in described else None
    accounts = get_entities(
        recipe_id,
        state.admin_id,
        columns=[*present, *sidecars, *([row_source] if row_source else [])],
    )
    if accounts is None or len(accounts) == 0:
        return state
    accounts.index = pd.Index(accounts.index).astype(str)
    accounts = accounts[~accounts.index.duplicated()]
    for column in columns:
        if column in accounts.columns:
            accounts[column] = pd.to_numeric(accounts[column], errors='coerce')
        else:
            accounts[column] = np.nan

    account_id, lot_id = links.columns[0], links.columns[1]
    if curated.index.name == account_id:
        account_id, lot_id = lot_id, account_id
    lot_index = pd.Index(curated.index).astype(str)
    frame = pd.DataFrame(
        {
            ACCOUNT: links[account_id].astype(str).to_numpy(),
            LOT: links[lot_id].astype(str).to_numpy(),
            'share': pd.to_numeric(links['share'], errors='coerce').to_numpy()
            if 'share' in links.columns
            else np.nan,
            'units': units_only_links(links),
        }
    )
    frame = frame[frame[ACCOUNT].isin(accounts.index) & frame[LOT].isin(lot_index)]
    # One row per account and lot; a pair two passes found is a units
    # pair only if every pass found it through the units.
    frame = (
        frame.groupby([ACCOUNT, LOT], sort=False)
        .agg(share=('share', 'first'), units=('units', 'all'))
        .reset_index()
    )
    if frame.empty:
        return state
    stated = accounts.loc[frame[ACCOUNT], columns].reset_index(drop=True)
    values = stated.copy()
    left_out = mask_units_rows(values, frame[LOT], frame['units'], columns)
    # Accounts the units rule took a value from: their lots need not
    # add up to their value, so the conservation check leaves them out.
    masked = (values.isna() & stated.notna()).any(axis=1) | left_out
    masked_accounts = masked.groupby(frame[ACCOUNT].to_numpy()).any()
    keep = ~left_out.to_numpy()
    frame = frame[keep].reset_index(drop=True)
    values = values[keep].reset_index(drop=True)
    frame = pd.concat([frame, values], axis=1)

    lot_area_ha = pd.to_numeric(curated[area_column], errors='coerce')
    lot_area_ha.index = lot_index
    frame['share'] = area_shares(
        frame[ACCOUNT], frame[LOT], frame['share'], lot_area_ha
    )

    floor_stats = {}
    frame['floor'] = 0.0
    if building_recipe_id and by_floor_area:
        frame['floor'], floor_stats = _floor_area(
            state, frame, building_recipe_id, stories, min_piece_m2, cluster_sources
        )

    layer = None
    layer_columns = [c for c in layer_columns if c in curated.columns]
    if layer_columns:
        layer = curated[layer_columns].apply(pd.to_numeric, errors='coerce')
        layer.index = lot_index
    # A count is not a part of the total.
    parts = by_area + [c for c in by_floor_area if not c.startswith('n_')]
    contributions, layer_stats = divide(
        frame,
        by_area=by_area,
        by_floor_area=by_floor_area,
        total=total,
        parts=parts,
        layer=layer,
        layer_columns=layer_columns,
    )

    decided = lot_values(contributions, columns)
    tokens = _lot_tokens(
        frame, accounts, contributions, columns, parts, curated, lot_index
    )
    position = pd.Series(np.arange(len(curated)), index=lot_index)
    for column in columns:
        lots = decided[column].dropna().index
        if not len(lots):
            continue
        rows = position.reindex(lots).to_numpy()
        target = (
            pd.to_numeric(curated[column], errors='coerce').astype(float)
            if column in curated.columns
            else pd.Series(np.nan, index=curated.index)
        )
        target.iloc[rows] = decided.loc[lots, column].to_numpy()
        curated[column] = target
        mask = pd.Series(False, index=curated.index)
        mask.iloc[rows] = True
        token = pd.Series(pd.NA, index=curated.index, dtype=object)
        token.iloc[rows] = tokens[column].reindex(lots).to_numpy()
        record_sources(curated, column, token, mask=mask, clear_missing=True)
    state.curated = curated

    _report(
        state,
        report,
        contributions,
        accounts,
        columns,
        masked_accounts,
        decided,
        parts,
        total,
        floor_stats,
        layer_stats,
        frame,
    )
    return state


def _floor_area(
    state, frame, building_recipe_id, stories, min_piece_m2, cluster_sources
):
    """Floor area per row of *frame* from the building spine and overlay."""
    overlaps = _building_overlaps(state, building_recipe_id)
    if overlaps is None or overlaps.empty:
        if state.verbose:
            print('  divide_account_values: no building overlay; lot area only.')
        return pd.Series(0.0, index=frame.index), {}
    described = set(describe_recipe(building_recipe_id, state.admin_id).index)
    wanted = [
        c
        for c in dict.fromkeys(['geometry_source', *_story_columns(stories)])
        if c in described
    ]
    buildings = get_entities(building_recipe_id, state.admin_id, columns=wanted)
    buildings.index = pd.Index(buildings.index).astype(str)
    buildings = buildings[~buildings.index.duplicated()]
    stand_in = is_synthetic_geometry(buildings, {'entity_type': 'building'})
    drawn = buildings.index[~stand_in.to_numpy()]
    overlaps = overlaps[overlaps['building'].isin(drawn)]
    source = (
        buildings['geometry_source'].astype('string').str.split('.').str[0]
        if 'geometry_source' in buildings.columns
        else pd.Series(pd.NA, index=buildings.index, dtype='string')
    )
    cluster = source.isin(list(cluster_sources or ())).fillna(False)
    count = building_stories(state, buildings.loc[drawn], stories)
    floor, stats = account_floor_area(frame, overlaps, count, min_piece_m2, cluster)
    stats['stand_in_buildings'] = int(stand_in.sum())
    return floor, stats


def _account_tokens(accounts: pd.DataFrame, column: str) -> pd.Series:
    """Per account, the source token of its *column*: the cell's own
    label where set, else the row's source, reduced to source ids."""
    side = source_column(column)
    own = (
        accounts[side].astype('string')
        if side in accounts.columns
        else pd.Series(pd.NA, index=accounts.index, dtype='string')
    )
    if ROW_SOURCE_COLUMN in accounts.columns:
        own = own.where(own.notna(), accounts[ROW_SOURCE_COLUMN].astype('string'))
    return _row_source_tokens(own)


def _lot_tokens(frame, accounts, contributions, columns, parts, curated, lot_index):
    """Per column, the provenance token of each decided lot.

    A lot names every source whose statement reached it. A layer value
    keeps the lot's own label. A total computed as the sum of parts
    names the parts' sources and carries the imputed marker.
    """
    rows = {c: frame[ACCOUNT].map(_account_tokens(accounts, c)) for c in columns}
    result = {}
    for column in columns:
        side = source_column(column)
        token = rows[column].astype(object)
        flag = f'{column}_layer'
        if flag in contributions.columns and contributions[flag].any():
            lot_label = (
                curated[side].astype(object).set_axis(lot_index)
                if side in curated.columns
                else pd.Series(pd.NA, index=lot_index, dtype=object)
            )
            taken = contributions[flag].to_numpy(dtype=bool)
            token[taken] = frame.loc[taken, LOT].map(lot_label).to_numpy()
        computed = f'{column}_computed'
        is_computed = (
            contributions[computed].to_numpy(dtype=bool)
            if computed in contributions.columns
            else np.zeros(len(frame), dtype=bool)
        )
        if is_computed.any():
            stated = pd.concat(
                [rows[c][is_computed].astype(object) for c in parts if c in rows],
                axis=1,
            )
            joined = stated.apply(
                lambda r: '+'.join(dict.fromkeys(t for t in r.dropna())) or None,
                axis=1,
            )
            token[is_computed] = joined.to_numpy()
        values = pd.DataFrame(
            {column: contributions[column], side: token}, index=frame.index
        )
        lot_tokens = _union_tokens(values, frame[LOT], column, side)
        if is_computed.any():
            marked = (
                pd.Series(is_computed, index=frame.index)
                .groupby(frame[LOT])
                .any()
                .reindex(lot_tokens.index, fill_value=False)
            )
            lot_tokens = lot_tokens.where(~marked, lot_tokens.map(mark_imputed))
        result[column] = lot_tokens
    return result


def _report(
    state,
    report,
    contributions,
    accounts,
    columns,
    masked_accounts,
    decided,
    parts,
    total,
    floor_stats,
    layer_stats,
    frame,
):
    """Write the conservation report and print a summary."""
    table = conservation(contributions, accounts, columns, masked_accounts)
    n_lots = frame.groupby(ACCOUNT)[LOT].nunique()
    present, failing = (
        identity_failures(state.curated, parts, total) if total else (0, 0)
    )
    extra = {
        'accounts': int(len(n_lots)),
        'accounts_on_several_lots': int((n_lots >= 2).sum()),
        'accounts_by_floor_area': int(
            (frame['floor'].groupby(frame[ACCOUNT]).sum() > 0).sum()
        ),
        'lots_decided': int(decided.notna().any(axis=1).sum()),
        'lots_with_parts_and_total': present,
        'lots_failing_identity': failing,
        **floor_stats,
        **layer_stats,
    }
    summary = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {'column': name, 'accounts_checked': value}
                    for name, value in extra.items()
                ]
            ),
        ],
        ignore_index=True,
    )
    if state.verbose:
        print(f'  divide_account_values: {extra}')
        for row in table.itertuples():
            print(
                f'    {row.column}: {row.accounts_failing:,} of '
                f'{row.accounts_checked:,} accounts do not add up'
            )
    if not report:
        return
    try:
        from openplaces.path import reports_path

        report_path = reports_path(state.admin_id, filename=report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(report_path, index=False)
    except Exception as exception:
        warnings.warn(
            f'divide_account_values: report failed: {exception}', stacklevel=3
        )
