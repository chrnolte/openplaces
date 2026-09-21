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
# can never carry a person. A test pins this list.
LINK_COLUMNS_TAIL = ('link_method', 'link_source', 'share')
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
