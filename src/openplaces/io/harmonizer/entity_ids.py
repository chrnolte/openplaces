"""Harmonize step minting stable entity ids for a non-spatial spine.

A spatial entity gets its id from its geometry. A property has none, and
`union_spine_sources` numbers its rows, so a property's id was its
position in one particular file: it changed with every rebuild, and
nothing could refer to a property across runs, sources or entities.

The id minted here is the number the assessor issued for the account,
prefixed with the admin unit that scopes it: `US-TX-VIC_000123`. Of the
106 property sources contributing rows on 2026-09-20, 104 are assessor
rolls or statewide republications of them, so the issuer is left unnamed;
a source that is not the assessor would need `{admin}_{namespace}:{id}`.

Two sources describing the same properties therefore produce the same
ids, and their rows merge into one (measured on Boston: MassGIS's
assessment layer and the city's own table share 179,865 of 180,445
account numbers as published). Without the merge, adding a county's own
table beside a statewide roll doubles the county's properties.

Why the account number and not a hash or a sequence under the parcel: a
content hash changes when an address is corrected and matches nothing in
a deed; `{parcel}.{n}` cannot name a property on two lots or on none, and
would make the property spine wait for the parcel spine, which reads it.
The matching is exact, on a normalized issued number. No fuzzy tier, no
score, nothing learned, and no row is unmerged afterwards (AGENTS.md,
"Patent risk", shape 4).
"""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register
from openplaces.io.stacked_units import LOT_LINK_KEY, STACKED_UNITS_LABEL_SUFFIX
from openplaces.recipe import get_recipe_by_id

# Tried in order when a recipe names no `entity_id` column. The raw
# assessor columns only: `parcel_id_local` and the other standardized
# keys name the lot, which is the one thing a property id must not do.
DEFAULT_ID_COLUMNS = {
    'property': ('property_id_assessor', 'property_id_admin2', 'parcel_id_assessor'),
}

# The account column is chosen, not tested against a threshold: of the
# candidates a source carries, the one on which the fewest rows share
# their number (`choose_id_column`). New Hanover County NC's account
# number repeats on no row and its PIN on 19,519 of 115,027, so the
# choice needs no cutoff, and a recipe's `entity_id` overrides it. One
# bound remains, as a statement about the source rather than a tuning
# knob: a source whose best column still repeats on most rows issues no
# account number at all, and its rows are named by their content (the
# units split off North Carolina's statewide parcel layer, every one of
# which carries its lot's number).
NO_ACCOUNT_NUMBER_SHARE = 0.5

# A source holding several roll years (Florida's DOR roll: 24 yearly
# rolls per county, kept on purpose as a panel for temporal joins)
# describes every property once per year. The spine takes the latest
# roll of the unit being built, not the latest row per account, which
# would revive every account retired since 2002 with its last values.
VINTAGE_COLUMN = 'tax_year'


def select_latest_vintage(
    rows: pd.DataFrame, column: str = VINTAGE_COLUMN
) -> tuple[pd.DataFrame, int]:
    """Keep the rows of the latest roll year, where a source has several.

    Returns the rows kept and the number dropped. A source with one
    year, or none, passes through untouched.
    """
    if column not in rows.columns:
        return rows, 0
    year = pd.to_numeric(rows[column], errors='coerce')
    if year.nunique(dropna=True) <= 1:
        return rows, 0
    keep = year.eq(year.max())
    return rows[keep], int((~keep).sum())


def choose_id_column(rows: pd.DataFrame, candidates) -> str | None:
    """The candidate on which the fewest rows share their number.

    Ties go to the earlier candidate. A candidate that is absent or
    empty is skipped.
    """
    best, best_repeats = None, None
    for column in candidates:
        if column not in rows.columns:
            continue
        issued = normalize_issued_id(rows[column])
        if issued.notna().sum() == 0:
            continue
        repeats = int(issued.dropna().duplicated(keep=False).sum())
        repeats += int(issued.isna().sum())
        if best_repeats is None or repeats < best_repeats:
            best, best_repeats = column, repeats
    return best


def normalize_issued_id(values: pd.Series) -> pd.Series:
    """Reduce an issued number to what two publications of it agree on.

    Folds case and turns every run of separators into one hyphen, so
    `12_a  3`, `12-A-3` and `12 A 3` converge. The separators' positions
    are kept, unlike in `geo.ids.add_parcel_id_alnum`: an id is not a
    fallback match key, and in a map-block-lot number the positions
    carry meaning. Measured on Somerville MA, 2026-09-20: dropping
    separators made 426 of 19,013 account numbers collide (`12_A_1_2`
    with `12_A_12`) where 24 repeat as published. The cost is that a
    source publishing the same number with no separators at all does
    not converge; Boston's two sources needed no help.

    Blank and all-zero values become missing: they are placeholders,
    not numbers anyone issued.
    """
    text = (
        values.astype('string')
        .str.upper()
        .str.replace(r'[^A-Z0-9]+', '-', regex=True)
        .str.strip('-')
    )
    usable = text.notna() & text.str.replace('-', '').str.strip('0').ne('')
    return text.where(usable)


def _content_hash(rows: pd.DataFrame) -> pd.Series:
    """Order-independent fingerprint of each row's values."""
    return pd.util.hash_pandas_object(rows.astype('string'), index=False)


def mint_ids(
    rows: pd.DataFrame,
    admin_id: str,
    id_column: str | None,
    label: str,
    repeats_by_content: bool = False,
    caller: str = 'assign_entity_ids',
    advice: str = "Name a better column in the recipe's `entity_id` if it has one.",
) -> tuple[pd.Series, dict]:
    """Ids for one source's rows within one admin unit.

    Parameters
    ----------
    rows : pandas.DataFrame
        The source's rows.
    admin_id : str
        Admin unit that scopes the issued numbers.
    id_column : str or None
        Column holding the issued account number, or None when the
        source has none.
    label : str
        The source's label, used only for rows with no issued number.
    repeats_by_content : bool, optional
        Name the rows whose number repeats by their content instead of
        suffixing them, for units split off a parcel table. Any source
        whose number repeats on most rows (`NO_ACCOUNT_NUMBER_SHARE`)
        is treated the same way, with a warning.

    Returns
    -------
    ids : pandas.Series
        One id per row, aligned to *rows*. Unique except across exact
        duplicate rows, which share one.
    report : dict
        `n_without_number`, `n_repeated`, `n_exact_duplicates`, and
        `n_coarse_numbers` where repeating rows were named by content.
    """
    hashes = _content_hash(rows.drop(columns='source', errors='ignore'))
    issued = (
        normalize_issued_id(rows[id_column])
        if id_column is not None
        else pd.Series(pd.NA, index=rows.index, dtype='string')
    )
    ids = (f'{admin_id}_' + issued).astype('string')
    # No issued number: the row still needs a name that survives a
    # rebuild, and its content is all there is to name it by.
    missing = issued.isna()
    if missing.any():
        named = hashes[missing].map('{:016x}'.format).astype('string')
        ids[missing] = f'{admin_id}_{label}:' + named

    # A later copy of an identical row is a duplicate, not a second
    # account; only differing rows under one number count as repeats.
    exact = pd.DataFrame({'id': ids, 'hash': hashes}).duplicated(keep='first')
    distinct_repeats = ~exact & ~missing
    distinct_repeats &= ids.where(~exact).duplicated(keep=False)
    copy_of = ids.astype('object') + '|' + hashes.astype('string').astype('object')
    share = distinct_repeats.sum() / max(len(rows), 1)
    no_account_number = share > NO_ACCOUNT_NUMBER_SHARE
    if no_account_number:
        warnings.warn(
            f'{caller}: {distinct_repeats.sum():,d} of {len(rows):,d} '
            f'{label} rows ({share:.0%}) share their {id_column!r} with a '
            'different row, so the source issues no account number; those '
            f'rows are named by their content. {advice}',
            stacklevel=2,
        )
    if distinct_repeats.any() and (repeats_by_content or no_account_number):
        # A number on several differing rows does not name one of them:
        # units split off a parcel table that repeats the lot's number
        # (North Carolina's statewide layer, on every stack), or a
        # source with no account number at all. `_2`, `_3` would read
        # as accounts and let an arbitrary one merge with a roll row.
        # Only the rows whose number repeats: a number carried by one
        # row is still the account, and is what lets that unit merge
        # with its tax roll row (Galveston County TX: 5,265 of 5,577
        # unit numbers are roll accounts, and naming every unit by
        # content kept all of them apart from the roll).
        named = hashes[distinct_repeats].map('{:016x}'.format).astype('string')
        ids[distinct_repeats] = f'{admin_id}_{label}:' + named
        ids = ids.groupby(copy_of, sort=False).transform('first')
        return ids, {
            'n_without_number': int(missing.sum()),
            'n_repeated': 0,
            'n_exact_duplicates': int(exact.sum()),
            'n_coarse_numbers': int(distinct_repeats.sum()),
        }
    if distinct_repeats.any():
        # Deterministic: ordered by content, not by position in the file.
        order = (
            pd.DataFrame({'id': ids, 'hash': hashes})[distinct_repeats]
            .sort_values(['id', 'hash'], kind='stable')
            .groupby('id')
            .cumcount()
        )
        later = order[order > 0]
        ids.loc[later.index] = (
            ids.loc[later.index] + '_' + (later + 1).astype('string')
        ).astype('string')
        # An identical copy follows its original, whichever id that got.
        ids = ids.groupby(copy_of, sort=False).transform('first')
    report = {
        'n_without_number': int(missing.sum()),
        'n_repeated': int(distinct_repeats.sum()),
        'n_exact_duplicates': int(exact.sum()),
    }
    return ids, report


def merge_on_id(rows: pd.DataFrame, ids: pd.Series, id_name: str) -> pd.DataFrame:
    """One row per id: the first source's values, later ones filling gaps.

    *rows* must be in priority order (most specific source first, as
    `union_spine_sources` loads them). `source` becomes the `+`-joined
    labels of the sources that described the row, in that order.
    """
    rows = rows.copy()
    rows[id_name] = ids.to_numpy()
    if not rows[id_name].duplicated().any():
        return rows.set_index(id_name)
    grouped = rows.groupby(id_name, sort=False, dropna=False)
    # GroupBy.first takes the first non-missing value per column.
    merged = grouped.first()
    if 'source' in rows.columns:
        merged['source'] = grouped['source'].agg(
            lambda labels: '+'.join(dict.fromkeys(labels.astype('string')))
        )
    return merged


def _declared_id_column(recipe_id: str, layer: str | None) -> str | None:
    recipe = get_recipe_by_id(recipe_id)
    if layer is None:
        return recipe.get('entity_id')
    for spec in recipe.get('additional_layers') or []:
        entity = spec.get('entity') if isinstance(spec, dict) else None
        if entity is not None and str(entity.entity_type) == layer:
            return spec.get('entity_id')
    return None


@_register('assign_entity_ids')
def assign_entity_ids(state: HarmonizeState) -> HarmonizeState:
    """Index a unioned spine by stable ids and merge rows sharing one.

    For each source `union_spine_sources` loaded, the id column is the
    source recipe's `entity_id` (or that of its `additional_layers`
    entry), else the first of `DEFAULT_ID_COLUMNS` the source carries.
    The spine's index becomes `{entity type}_id`.

    Runs after the supplements join, which selects a roll's rows by
    their single `source` label; merged rows carry several.
    """
    if state.spine is None or state.spine.empty:
        return state
    entity = state.recipe.get('entity')
    entity_type = str(entity.entity_type) if entity is not None else 'entity'
    id_name = f'{entity_type}_id'
    admin = str(state.admin_id)
    origins = state.metadata.get('spine_source_origins') or {}
    defaults = DEFAULT_ID_COLUMNS.get(entity_type, ())

    spine = state.spine
    labels = (
        spine['source'].astype('string')
        if 'source' in spine.columns
        else pd.Series('source', index=spine.index, dtype='string')
    )
    # A source holding several roll years keeps its latest only. The
    # rows go before any id is minted, so that a panel of yearly rolls
    # is never mistaken for a column that names the wrong thing.
    kept = []
    for label, part in spine.groupby(labels, sort=False):
        latest, n_dropped = select_latest_vintage(part)
        kept.append(latest.index)
        if n_dropped:
            print(
                f'  assign_entity_ids: {label}: kept the latest '
                f'{VINTAGE_COLUMN} ({len(latest):,d} rows), dropped '
                f'{n_dropped:,d} rows of earlier rolls'
            )
    keep_index = kept[0].append(kept[1:]) if kept else spine.index
    if len(keep_index) != len(spine):
        spine = spine.loc[spine.index.isin(keep_index)]
        labels = labels.loc[spine.index]

    ids = pd.Series(pd.NA, index=spine.index, dtype='string')
    for label, part in spine.groupby(labels, sort=False):
        part = part.dropna(axis=1, how='all')
        recipe_id, layer = origins.get(label, (None, None))
        column = _declared_id_column(recipe_id, layer) if recipe_id else None
        if column is not None and column not in part.columns:
            raise ValueError(
                f'assign_entity_ids: {recipe_id} names entity_id {column!r}, '
                f'which its output for {admin} does not carry.'
            )
        if column is None:
            column = choose_id_column(part, defaults)
        if column is None:
            warnings.warn(
                f'assign_entity_ids: {label} carries no issued number for '
                f'{admin}; its rows are named by their content.'
            )
        is_units = str(label).endswith(STACKED_UNITS_LABEL_SUFFIX)
        if is_units:
            # One account drawn on two lots arrives as two rows that
            # differ in their lot alone: one property, so the lot stays
            # out of what makes a row distinct. Both lots are kept by
            # the link table, which reads the split's own pairs.
            part = part.drop(columns=[LOT_LINK_KEY, 'geo_id'], errors='ignore')
        part_ids, report = mint_ids(
            part, admin, column, label, repeats_by_content=is_units
        )
        ids.loc[part.index] = part_ids
        n_coarse = report.get('n_coarse_numbers', 0)
        if state.verbose or report['n_repeated'] or n_coarse:
            print(
                f'  assign_entity_ids: {label} on {column}: '
                f'{report["n_repeated"]:,d} rows repeat a number, '
                f'{n_coarse:,d} named by content for it, '
                f'{report["n_exact_duplicates"]:,d} exact duplicates, '
                f'{report["n_without_number"]:,d} without a number'
            )

    n_before = len(spine)
    # Exact duplicates within a source share an id and collapse here too.
    state.spine = merge_on_id(spine, ids, id_name)
    if state.verbose or len(state.spine) != n_before:
        print(
            f'  assign_entity_ids: {n_before:,d} rows, '
            f'{len(state.spine):,d} {entity_type} ids'
        )
    return state
