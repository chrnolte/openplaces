"""Harmonize step writing id-based n:m link tables between entities.

A property reaches its parcel through one key column today
(parcel_id_local on the property row), and one column holds one parcel.
Three measured cases do not fit: several properties on one lot whose
units carry their own account keys, one property on several lots (260
accounts in Galveston County TX), and later a deed conveying several
parcels. A link table holds all three; a key column holds none of them.

The spatial joins already persist n:m link sidecars
(links.link_to_reference, geo.link.get_entity_link_path). This module
adds the same kind of table for a relationship found by ids instead of
by an overlay, at the same canonical path, so a reader needs one
mechanism for both.

Why exact keys and a fixed label, and nothing more: a link row records
which rule found the pair (link_method). It is not a strength score,
nothing is tuned from the links made, and no later pass removes a link
an earlier one wrote. Those three absences are deliberate (AGENTS.md,
"Patent risk", shape 4) and should stay that way.
"""

from __future__ import annotations

import json
import warnings

import pandas as pd

from openplaces.core.schema import ENTITY_LINK_ORDER
from openplaces.geo.link import get_entity_link_path, get_link_owner_recipe_id
from openplaces.io import to_parquet
from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    get_output_path,
    get_recipe_by_id,
    get_recipe_id,
    get_save_admin_level,
)

# The only columns a link table may hold. Rolls carry owner names in
# some counties; a link table is two ids and the labels below, so it
# can never carry a person. A test pins this list. `share_basis` is
# added by `apportion_shares`, after the pairs are found.
LINK_COLUMNS_TAIL = ('link_method', 'link_source', 'share')
LINK_COLUMNS_WRITTEN = (*LINK_COLUMNS_TAIL, 'share_basis')
ENTITY_LINK_METADATA_KEY = 'openplaces:entity_link'


# Appended to link_method where link_by_id would leave the rows
# unjoined: the key is on more rows than its degenerate-key cutoff, on
# either side, and is not the one spared case (many finer rows on
# exactly one coarser row: a large stack on one lot, as in New Hanover
# County NC's parks and condominium complexes). What is left is the
# many-to-many case, where summing every finer row onto every coarser
# row put $7.5 billion on each of 2,443 Brazoria County parcel rows. A
# link table need not drop such pairs: it keeps them and says which rule
# found them. A reader that sums values leaves this method out; a
# reader that counts or locates can use it. The plain method's rows are
# the pairs link_by_id joins, less keys overused on the coarser side
# only, which link_by_id does not judge (one roll row in Galveston
# County TX, on a key carried by 418 parcel rows).
SHARED_KEY_SUFFIX = '_shared_key'


def _overused_keys(key: pd.Series) -> set:
    """Key values on more rows than link_by_id's degenerate-key cutoff."""
    from openplaces.io.harmonizer.links import (
        DEGENERATE_KEY_MAX_SHARE,
        DEGENERATE_KEY_MIN_ROWS,
    )

    counts = key.dropna().value_counts()
    cutoff = max(DEGENERATE_KEY_MIN_ROWS, DEGENERATE_KEY_MAX_SHARE * len(key))
    return set(counts[counts > cutoff].index)


def _entity_type_of(recipe) -> str:
    entity = recipe.get('entity')
    if entity is None:
        raise ValueError(
            f'link_entities_by_id: {get_recipe_id(recipe)} declares no entity.'
        )
    return str(entity.entity_type)


def build_id_links(
    finer: pd.DataFrame,
    coarser: pd.DataFrame,
    finer_key: str,
    coarser_key: str,
    finer_id: str,
    coarser_id: str,
    link_method: str,
    source_column: str | None = 'source',
) -> pd.DataFrame:
    """Pair the rows of two tables whose key columns hold the same value.

    Every pair is kept: a finer row whose key sits on several coarser
    rows yields one link row per coarser row, and the reverse. A row
    whose key is missing, or is a placeholder rather than an identifier,
    yields none.

    Parameters
    ----------
    finer, coarser : pandas.DataFrame
        The two entity tables, each indexed by its entity id.
    finer_key, coarser_key : str
        The key column on each side.
    finer_id, coarser_id : str
        Names of the two id columns in the result.
    link_method : str
        Fixed label naming the rule that found the pairs.
    source_column : str, optional
        Column of *finer* naming the source each row came from. It
        becomes `link_source`, and placeholder keys are judged within
        each source, as the key join this table mirrors judges them
        (`links.link_by_id` reads one source at a time).

    Returns
    -------
    pandas.DataFrame
        Columns `finer_id`, `coarser_id`, `link_method`, `link_source`,
        `share` (empty: undivided), sorted by both ids.
    """
    from openplaces.io.harmonizer.links import _placeholder_key_mask

    columns = [finer_id, coarser_id, *LINK_COLUMNS_TAIL]
    if finer_key not in finer.columns or coarser_key not in coarser.columns:
        return pd.DataFrame(columns=columns)

    has_source = source_column is not None and source_column in finer.columns
    left = pd.DataFrame(
        {
            finer_id: finer.index,
            'key': finer[finer_key].astype('string').to_numpy(),
            'link_source': (
                finer[source_column].astype('string').to_numpy()
                if has_source
                else pd.NA
            ),
        }
    )
    left['link_source'] = left['link_source'].astype('string')
    right = pd.DataFrame(
        {
            coarser_id: coarser.index,
            'key': coarser[coarser_key].astype('string').to_numpy(),
        }
    )
    # A key on more rows than link_by_id's cutoff, judged within each
    # source and over all of its rows, as link_by_id judges it (it
    # reads one source at a time). link_by_id does not judge the
    # spine's side at all; here a key on that many coarser rows is
    # flagged the same way, since it cannot name one of them.
    left = left.assign(shared=False)
    groups = (
        left.groupby('link_source', dropna=False, sort=False)
        if has_source
        else [(None, left)]
    )
    for _, group in groups:
        overused = _overused_keys(group['key'])
        left.loc[group.index[group['key'].isin(overused)], 'shared'] = True
    right_overused = _overused_keys(right['key'])
    # link_by_id spares an overused key that exactly one coarser row
    # carries: a large stack on one lot, where a sum lands once.
    on_one = right['key'].dropna().value_counts()
    on_one = set(on_one[on_one == 1].index)
    left.loc[left['key'].isin(on_one), 'shared'] = False

    # Blank and all-zero keys are never identifiers: no link.
    left = left[~_placeholder_key_mask(left['key'])]
    right = right[~_placeholder_key_mask(right['key'])]

    links = left.merge(right, on='key', how='inner')
    is_shared = links['shared'] | links['key'].isin(right_overused)
    links['link_method'] = link_method
    links.loc[is_shared, 'link_method'] = f'{link_method}{SHARED_KEY_SUFFIX}'
    links['share'] = pd.Series(pd.NA, index=links.index, dtype='Float64')
    links['link_method'] = links['link_method'].astype('string')
    links['link_source'] = links['link_source'].astype('string')
    return (
        links[columns]
        .sort_values([finer_id, coarser_id], kind='stable')
        .reset_index(drop=True)
    )


STACKED_UNITS_METHOD = 'stacked_units'
STACKED_UNITS_CROSSWALK_METHOD = 'stacked_units_crosswalk'


def stacked_unit_links(
    links: pd.DataFrame,
    properties: pd.DataFrame,
    parcels: pd.DataFrame,
    pairs: pd.DataFrame,
    property_key: str,
    parcel_key: str,
    property_id: str,
    parcel_id: str,
) -> pd.DataFrame:
    """Add the links that only the ingest-time split knows.

    On a stacked lot the parcel row carries the lot's key and each unit
    keeps its own, so a unit's key names no parcel row and the key pass
    finds nothing. The split recorded which unit sits on which lot
    (`io.stacked_units`, column `lot_id_local`), and two more exact
    passes use that:

    - `stacked_units`: a property row that names its lot links to the
      parcel row carrying that lot key.
    - `stacked_units_crosswalk`: any other property row whose own key
      is a split unit's key (a tax roll's row for the same unit) links
      to that unit's lot. A key the split saw on two lots links to both,
      which is how an account drawn on several lots is held.

    Pairs an earlier pass already found are kept under its method.
    Exact lookups on keys the sources issued; nothing is scored and no
    earlier link is removed or replaced.

    Parameters
    ----------
    links : pandas.DataFrame
        Links found by the key pass (`build_id_links`).
    properties, parcels : pandas.DataFrame
        The two entity tables, indexed by their ids.
    pairs : pandas.DataFrame
        Columns `unit_key` and `lot_key`: every unit-to-lot pair the
        split recorded for this admin unit.
    property_key, parcel_key : str
        The matching-key column on each side.
    property_id, parcel_id : str
        Names of the id columns in *links*.
    """
    from openplaces.io.stacked_units import LOT_LINK_KEY

    lots = pd.DataFrame(
        {
            parcel_id: parcels.index,
            'lot_key': parcels[parcel_key].astype('string').to_numpy(),
        }
    ).dropna(subset=['lot_key'])
    source = (
        properties['source'].astype('string').to_numpy()
        if 'source' in properties.columns
        else pd.NA
    )
    added = []
    if LOT_LINK_KEY in properties.columns:
        named = pd.DataFrame(
            {
                property_id: properties.index,
                'lot_key': properties[LOT_LINK_KEY].astype('string').to_numpy(),
                'link_source': source,
            }
        ).dropna(subset=['lot_key'])
        named = named.merge(lots, on='lot_key')
        named['link_method'] = STACKED_UNITS_METHOD
        added.append(named)
    if len(pairs) and property_key in properties.columns:
        keyed = pd.DataFrame(
            {
                property_id: properties.index,
                'unit_key': properties[property_key].astype('string').to_numpy(),
                'link_source': source,
            }
        ).dropna(subset=['unit_key'])
        crossed = keyed.merge(pairs.drop_duplicates(), on='unit_key').merge(
            lots, on='lot_key'
        )
        crossed['link_method'] = STACKED_UNITS_CROSSWALK_METHOD
        added.append(crossed)
    if not added:
        return links
    columns = list(links.columns)
    more = pd.concat(added, ignore_index=True)
    more['share'] = pd.Series(pd.NA, index=more.index, dtype='Float64')
    more = more.astype({'link_method': 'string', 'link_source': 'string'})
    out = pd.concat([links, more[columns]], ignore_index=True)
    out = out.drop_duplicates(subset=[property_id, parcel_id], keep='first')
    return out.sort_values([property_id, parcel_id], kind='stable').reset_index(
        drop=True
    )


def apportion_shares(
    links: pd.DataFrame, coarser: pd.DataFrame, area_column: str = 'area_ha'
) -> pd.DataFrame:
    """Divide a finer entity that sits on several coarser ones.

    A property on two lots would have its value summed onto both. Its
    links get a `share` proportional to each lot's area, and
    `share_basis` says so; every other link keeps an empty share, read
    as undivided. Area conserves the total and needs nothing new. It is
    wrong for improvements, which stand on one lot, and is recorded as
    the basis so that a better one can replace it (the derived
    property-to-footprint link). The case is rare: 260 accounts in
    Galveston County TX.
    """
    links = links.copy()
    links['share_basis'] = pd.Series(pd.NA, index=links.index, dtype='string')
    if links.empty or area_column not in coarser.columns:
        return links
    finer_id, coarser_id = links.columns[0], links.columns[1]
    several = links[finer_id].duplicated(keep=False)
    if not several.any():
        return links
    area = pd.to_numeric(
        links.loc[several, coarser_id].map(coarser[area_column]), errors='coerce'
    )
    total = area.groupby(links.loc[several, finer_id]).transform('sum')
    n = several.groupby(links[finer_id]).transform('sum')[several]
    by_area = area.notna() & total.gt(0)
    # A lot with no area cannot be weighed: fall back to an equal split
    # for that property, and say so.
    whole = by_area.groupby(links.loc[several, finer_id]).transform('all')
    share = (area / total).where(whole, 1.0 / n)
    links.loc[several, 'share'] = share.astype('Float64')
    links.loc[several, 'share_basis'] = pd.Series(
        ['area' if w else 'equal' for w in whole], index=whole.index, dtype='string'
    )
    return links


def read_entity_link(path) -> pd.DataFrame | None:
    """Read a link table if it still describes the files it was built from.

    Returns None when the file is absent, carries no footer, or one of
    its two inputs has changed since (size and mtime, or the content
    hash where only the mtime moved): a reader then falls back to what
    it did before the link existed, and says so.
    """
    from openplaces.io.aggregate import read_file_metadata
    from openplaces.io.cleanup import _resolve_relative
    from openplaces.io.harmonizer.links import _hash_file

    if path is None or not path.exists():
        return None
    raw = read_file_metadata(path).get(ENTITY_LINK_METADATA_KEY)
    if raw is None:
        return None
    try:
        stored = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for entry in stored.get('sources') or []:
        try:
            source = _resolve_relative(entry.get('path'))
            if not source.exists():
                return None
            stat = source.stat()
            if stat.st_size != entry.get('size'):
                return None
            if round(stat.st_mtime, 3) != entry.get('mtime'):
                digest = entry.get('sha256')
                if not digest or _hash_file(source) != digest:
                    return None
        except OSError:
            return None
    return pd.read_parquet(path)


def load_unit_lot_pairs(admin_id) -> pd.DataFrame:
    """Every unit-to-lot pair the ingest-time split recorded for a unit.

    Read from the implicit stacked-units layer of each parcel recipe
    covering *admin_id*, not from the property spine: merging an account
    drawn on two lots into one property row keeps one lot there, and the
    split's own table keeps both.
    """
    from openplaces.io.stacked_units import LOT_LINK_KEY
    from openplaces.recipe import find_additional_layer_recipes

    frames = []
    for match in find_additional_layer_recipes('property', admin_id):
        if not match.get('stacked_units_layer'):
            continue
        try:
            units = get_entities(
                match['recipe_id'], admin_id, layer=match['layer'], missing='ignore'
            )
        except (FileNotFoundError, OSError, KeyError, ValueError):
            continue
        if units is None or LOT_LINK_KEY not in units.columns:
            continue
        if 'parcel_id_local' not in units.columns:
            continue
        frames.append(
            pd.DataFrame(
                {
                    'unit_key': units['parcel_id_local'].astype('string').to_numpy(),
                    'lot_key': units[LOT_LINK_KEY].astype('string').to_numpy(),
                }
            ).dropna()
        )
    if not frames:
        return pd.DataFrame({'unit_key': [], 'lot_key': []}, dtype='string')
    pairs = pd.concat(frames, ignore_index=True).drop_duplicates()
    # A unit carrying its lot's own key is already found by the key pass.
    return pairs[pairs['unit_key'] != pairs['lot_key']]


def _add_stacked_unit_links(
    links, properties, parcels, property_key, parcel_key, property_id, parcel_id, state
):
    pairs = load_unit_lot_pairs(state.admin_id)
    out = stacked_unit_links(
        links,
        properties,
        parcels,
        pairs,
        property_key,
        parcel_key,
        property_id,
        parcel_id,
    )
    if state.verbose and len(out) != len(links):
        print(
            f'  link_entities_by_id: {len(out) - len(links):,d} links from the '
            f'stacked-units split ({len(pairs):,d} unit-to-lot pairs)'
        )
    return out


def _input_stamp(recipe_id: str, admin_id) -> dict:
    """Size and mtime of one input's output file, for the link footer."""
    from openplaces.io.cleanup import _relative_posix

    recipe = get_recipe_by_id(recipe_id)
    unit = (
        admin_id.truncate_to_level(get_save_admin_level(recipe))
        if admin_id is not None
        else None
    )
    path = get_output_path(recipe, admin_id=unit)
    entry = {'path': _relative_posix(path), 'size': None, 'mtime': None}
    if path.exists():
        stat = path.stat()
        entry['size'] = stat.st_size
        entry['mtime'] = round(stat.st_mtime, 3)
    return entry


@_register('link_entities_by_id')
def link_entities_by_id(
    state: HarmonizeState,
    recipe_id: str,
    spine_key: str = 'parcel_id_local',
    ref_key: str = 'parcel_id_local',
    link_method: str | None = None,
) -> HarmonizeState:
    """Write the n:m link table between the spine and another entity.

    Pairs the spine's rows with the rows of *recipe_id* wherever the two
    key columns hold the same value, and saves the pairs at the canonical
    link path (`geo.link.get_entity_link_path`), beside the finer
    entity's output. The spine itself is not changed.

    The step runs in the attribute recipe of the coarser entity (the
    parcel spine for property-to-parcel) rather than in the finer
    entity's own recipe, because the property spine is built before the
    parcel geospine, which reads it: the parcel rows a property links to
    do not exist yet when the property spine runs. Here both exist, and
    a rerun costs no geometry.

    The finer entity's id is its row index. The property spine's index
    is a row number, so a link is only valid against the exact property
    file it was built from; the footer records that file's size, mtime
    and content hash, and a reader must check them
    (`links._fingerprints_match`).

    Parameters
    ----------
    recipe_id : str
        Recipe of the other entity (e.g. `US_property-spine-2026`).
    spine_key, ref_key : str
        Key columns on the spine and on the other entity. Only an exact
        match on the standardized key is supported by design: the
        punctuation-free fallback keys are shared by hundreds of parcels
        in some counties, which a key join survives (fill_only) and an
        n:m table does not.
    link_method : str, optional
        Label written to every link row. Defaults to *ref_key*.
    """
    from openplaces.io.harmonizer.links import _with_source_hashes

    if state.spine is None:
        warnings.warn('link_entities_by_id: spine is None; skipping.')
        return state

    own_id = get_recipe_id(state.recipe)
    # Link sidecars are named after the recipe that mints the rows: the
    # geospine when the spine is split, the recipe itself when not.
    own_owner_id = get_link_owner_recipe_id(state.recipe)
    other = get_recipe_by_id(recipe_id)
    path = get_entity_link_path(recipe_id, own_owner_id, state.admin_id)

    try:
        ref = get_entities(recipe_id, state.admin_id)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        ref = None
    if ref is None:
        ref = pd.DataFrame()

    # Which side is finer decides the column order, as it decides the
    # path; the spine is the coarser one in the property-to-parcel case.
    ref_type = _entity_type_of(other)
    spine_type = _entity_type_of(state.recipe)
    for entity_type in (ref_type, spine_type):
        if entity_type not in ENTITY_LINK_ORDER:
            raise ValueError(
                f'link_entities_by_id: {entity_type!r} has no place in '
                'ENTITY_LINK_ORDER, so its link path is undefined.'
            )
    if ref_type == spine_type:
        raise ValueError(
            'link_entities_by_id links two different entity types; both '
            f'sides are {ref_type!r}.'
        )
    ref_is_finer = ENTITY_LINK_ORDER.index(ref_type) > ENTITY_LINK_ORDER.index(
        spine_type
    )
    ref_id = f'{ref_type}_id'
    spine_id = f'{spine_type}_id'
    if ref_is_finer:
        links = build_id_links(
            ref,
            state.spine,
            ref_key,
            spine_key,
            ref_id,
            spine_id,
            link_method or ref_key,
        )
    else:
        links = build_id_links(
            state.spine,
            ref,
            spine_key,
            ref_key,
            spine_id,
            ref_id,
            link_method or spine_key,
        )

    if ref_is_finer and ref_type == 'property' and spine_type == 'parcel':
        links = _add_stacked_unit_links(
            links, ref, state.spine, ref_key, spine_key, ref_id, spine_id, state
        )
    links = apportion_shares(links, state.spine if ref_is_finer else ref)

    fingerprint = {
        'format': 'id-1',
        'writer_recipe_id': own_id,
        'finer_recipe_id': recipe_id if ref_is_finer else own_owner_id,
        'coarser_recipe_id': own_owner_id if ref_is_finer else recipe_id,
        'admin_id': str(state.admin_id) if state.admin_id is not None else None,
        'step_config': {
            'spine_key': spine_key,
            'ref_key': ref_key,
            'link_method': link_method,
        },
        'sources': [
            _input_stamp(recipe_id, state.admin_id),
            _input_stamp(own_owner_id, state.admin_id),
        ],
    }
    to_parquet(
        links,
        path,
        file_metadata={
            ENTITY_LINK_METADATA_KEY: json.dumps(_with_source_hashes(fingerprint))
        },
    )
    if state.verbose:
        n_finer = links.iloc[:, 0].nunique()
        n_multi = int((links.groupby(links.columns[0]).size() > 1).sum())
        print(
            f'  link_entities_by_id: {len(links):,d} links, {n_finer:,d} of '
            f'{len(ref):,d} {recipe_id} rows linked, {n_multi:,d} to more '
            f'than one; wrote {path.name}'
        )
    return state
