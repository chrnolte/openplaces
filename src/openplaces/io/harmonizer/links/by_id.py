"""Join a reference to the spine on an exact key (link_by_id): address
keys, count columns, degenerate-key guards, the stacked-units move
and the step itself.
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd

from openplaces.core.attribute_registry import (
    get_agg_func,
    get_attributes,
)
from openplaces.geo.ids import (
    PARCEL_ID_ALNUM_KEYS,
    PARCEL_ID_RELINK_THRESHOLD,
    add_parcel_id_alnum,
)
from openplaces.io.aggregate import _agg_func_for
from openplaces.io.harmonizer import (
    _PROVENANCE_SUFFIX,
    HarmonizeState,
    _register,
    restrict_to_admin_by_name,
)
from openplaces.io.harmonizer.links._shared import (
    _LINK_KEY_COLUMNS,
    DEFAULT_LINK_KEY,
    PARCEL_ID_LOCAL_KEYS,
)
from openplaces.io.harmonizer.links.combine import (
    _write_prioritized,
)
from openplaces.io.harmonizer.links.discovery import (
    _apply_remap_csvs,
    _discover_link_sources,
    _select_supplements,
)
from openplaces.io.readers import get_entities
from openplaces.recipe import (
    resolve_attribute_name,
    source_id_from_recipe_id,
)

_REF_ADDRESS_COLUMNS = (
    ('street_column', 'address_street'),
    ('number_column', 'address_number'),
)


def _derive_address_key(frame, admin_id, spec, key):
    """Derive one address matching key for a join, without keeping it.

    Parameters
    ----------
    frame : pandas.DataFrame
        Spine or reference table.
    admin_id : str or AdminId
        Admin unit being processed.
    spec : dict
        Keyword arguments of
        :func:`~openplaces.io.harmonizer.addresses.add_address_id_local`.
    key : str
        Which derived column to return (the plain or city-inclusive key).

    Returns
    -------
    pandas.Series or None
        The key aligned to *frame*, or None when *frame* lacks the street
        or number column or *key* is not one of the derived columns.
    """
    from openplaces.io.harmonizer.addresses import add_address_id_local

    spec = dict(spec)
    needed = [spec.get(k, d) for k, d in _REF_ADDRESS_COLUMNS]
    if not all(c in frame.columns for c in needed):
        return None
    optional = [spec.get(k, d) for k, d in _ADDRESS_SCOPE_COLUMNS]
    columns = needed + [c for c in optional if c and c in frame.columns]
    keyed = add_address_id_local(
        frame[list(dict.fromkeys(columns))].copy(), admin_id, **spec
    )
    return keyed[key] if key in keyed.columns else None


#: Optional scope columns of add_address_id_local and their defaults.
_ADDRESS_SCOPE_COLUMNS = (('city_column', 'city'), ('admin4_column', 'admin4_id'))


#: Count columns already written during this harmonize run, so a
#: run's first source replaces whatever a restored spine carried and
#: every later source adds to it (see :func:`_accumulate_count`).
_COUNT_COLUMNS_KEY = '_link_count_columns'


def _accumulate_count(
    state: HarmonizeState,
    spine: gpd.GeoDataFrame,
    name: str,
    counts: pd.Series,
) -> None:
    """Add *counts* into ``spine[name]``, once per source in this run.

    A count is a tally of contributing reference records, not a competing
    estimate of one quantity, so two sources are summed rather than
    resolved against each other: under auto-discovery the last matched
    source used to overwrite the column outright, and every earlier
    source's records vanished from it. :func:`_write_prioritized` is the
    wrong tool here for a mechanical reason too, since a count column is
    dense by construction (an unmatched row is a real 0, never null), so
    its coverage rule would always take the newest source whole.

    The first write of a run replaces the column, so a spine restored
    with a stale count does not accumulate on top of it; later writes add.

    Parameters
    ----------
    state : HarmonizeState
        Run state; its metadata records which columns this run has
        already written.
    spine : geopandas.GeoDataFrame
        Spine to write into, mutated in place.
    name : str
        Count column name.
    counts : pandas.Series
        Per-spine-row record count for this source, aligned to *spine*.
    """
    written = state.metadata.setdefault(_COUNT_COLUMNS_KEY, set())
    values = counts.fillna(0).astype('int64')
    if name in spine.columns and name in written:
        spine[name] = (
            pd.to_numeric(spine[name], errors='coerce').fillna(0).astype('int64')
            + values
        )
    else:
        spine[name] = values
    written.add(name)


# A join key value carried by this many rows, or by this share of the
# reference, is a placeholder rather than an identifier. Measured on the
# 2026 coastal-Texas and eastern-NC sources: a healthy file's most common
# `parcel_id_local` covers 5 rows (Kleberg) to 49 (Pender, a real
# multi-unit building), while Brazoria's most common value is '0' on
# 24,241 rows and its runner-up covers 2,443. The floor keeps a genuine
# condominium's units together; the share scales with file size.
DEGENERATE_KEY_MIN_ROWS = 100
DEGENERATE_KEY_MAX_SHARE = 0.001


def _placeholder_key_mask(key: pd.Series) -> pd.Series:
    """Values that cannot be identifiers whatever their frequency.

    Empty strings and all-zero codes ('0', '000', '000-00-0000'): assessors
    write these where a parcel number is unknown, and every row carrying one
    is a different parcel.
    """
    text = key.astype('string').str.strip()
    blank = text.isna() | text.eq('')
    zeros = text.str.replace(r'[^0-9A-Za-z]', '', regex=True).str.fullmatch('0+')
    return blank | zeros.fillna(False)


def _neutralize_degenerate_keys(
    ref: pd.DataFrame,
    ref_key: str,
    recipe_id: str | None = None,
    spine_key: pd.Series | None = None,
) -> pd.DataFrame:
    """Blank out join-key values that are placeholders, not identifiers.

    A key shared by thousands of rows makes every mode of
    :func:`link_by_id` wrong, and silently: 'attributes' picks one row
    arbitrarily for all of them, 'count' reports the whole group's size on
    each, and 'aggregate' **sums** their value columns and writes that sum
    onto every one. Measured consequence before this guard: a quarter-acre
    Brazoria County lot carrying a $7.5 billion `total_value`, and 152
    eastern-NC footprints holding 62% of the region's entire delivered
    parcel value.

    Setting the key to missing (rather than dropping the rows) lets each
    mode's existing `dropna(subset=[ref_key])` skip them, so an unmatched
    parcel simply gains no reference attributes -- the same outcome as a
    parcel the reference never mentioned.

    A key on many reference rows is spared when *spine_key* is given and
    exactly one spine row carries it. That is a large stack, not a
    placeholder, and the harm above cannot arise: it came from the
    product, every one of a key's reference rows summed onto every one of
    its spine rows (Brazoria's $7.5 billion key sits on 2,443 reference
    rows **and** 2,443 parcel rows, and is as well formed as any other,
    so its shape could not have told it apart). With one spine row the
    sum lands once, where it belongs. Measured 2026-09-20 on New Hanover
    County NC: 16 roll keys over the cutoff, 116 to 327 accounts each, 15
    of them on exactly one parcel of 0.6 to 10 ha (parks and condominium
    complexes), 2,853 roll rows that were being left off their parcel.

    Returns *ref* unchanged when nothing is degenerate.
    """
    if ref_key not in ref.columns or ref.empty:
        return ref
    key = ref[ref_key]
    bad = _placeholder_key_mask(key)

    counts = key.astype('string').value_counts()
    cutoff = max(DEGENERATE_KEY_MIN_ROWS, DEGENERATE_KEY_MAX_SHARE * len(ref))
    overused = set(counts[counts > cutoff].index)
    if overused and spine_key is not None:
        on_spine = spine_key.dropna().astype('string').value_counts()
        overused = {k for k in overused if on_spine.get(k, 0) != 1}
    if overused:
        bad = bad | key.astype('string').isin(overused)

    # Only rows that actually carried a value are a change worth reporting;
    # an already-missing key was never going to join.
    bad = bad & key.notna()
    if not bad.any():
        return ref
    ref = ref.copy()
    ref.loc[bad, ref_key] = pd.NA
    where = f' in {recipe_id}' if recipe_id else ''
    sample = ', '.join(repr(v) for v in sorted(overused)[:3])
    warnings.warn(
        f'link_by_id: {int(bad.sum()):,} of {len(ref):,} reference rows'
        f'{where} carry a degenerate {ref_key!r} (a placeholder, or a '
        f'value shared by more than {cutoff:,.0f} rows'
        + (f'; e.g. {sample}' if overused else '')
        + '). They are left unjoined rather than aggregated together.',
        stacklevel=2,
    )
    return ref


def _warn_if_duplicate_key(
    key_series: pd.Series,
    key_name: str,
    context: str,
    is_own_identity_key: bool = True,
) -> None:
    """Warn when *key_series* has duplicate values.

    openplaces indices (e.g. ``parcel_id``, a geo_id) are unique by design.
    Joining on a column that isn't -- ``parcel_id_local`` is the common case --
    is fine as long as it's *known* to be a many-to-one key (and, for
    'aggregate'/'count' modes, is actually aggregated); this makes that
    non-uniqueness visible instead of a silent assumption.

    *is_own_identity_key* silences the warning when *key_name* is a foreign
    key borrowed from another entity (e.g. ``parcel_id_local`` on a
    transaction spine) rather than the checked side's own identity -- there,
    duplicates are structurally expected (many transactions share one
    parcel) rather than a same-entity collision.
    """
    if not is_own_identity_key:
        return
    valid = key_series.dropna()
    dup_mask = valid.duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        n_dup_values = int(valid[dup_mask].nunique())
        warnings.warn(
            f'link_by_id ({context}): {key_name!r} is not unique -- '
            f'{n_dup_rows:,d} rows share a value with at least one other row, '
            f'across {n_dup_values:,d} duplicated values.'
        )


def _columns_as_pairs(
    columns: list[str] | dict[str, str] | None,
) -> list[tuple[str, str]]:
    """Normalize *columns* to ``(ref_column, output_name)`` pairs.

    A plain list keeps the reference column's own name as the output name; a
    dict (``{ref_column: output_name}``) renames on the way in, e.g. joining a
    transaction reference's ``price``/``recorded_date`` onto the canonical
    ``last_sale_price``/``last_sale_date`` parcel attributes.
    """
    if not columns:
        return []
    if isinstance(columns, dict):
        return list(columns.items())
    return [(c, c) for c in columns]


def _upstream_tokens(ref_indexed, column: str, skey: pd.Series, token: str | None):
    """Per-row provenance for a joined column, original source preferred.

    A reference that records where its own values came from is asked
    first: its `{column}_source` is carried through the join so the
    spine keeps the name of the source that *originally* supplied the
    value, not the name of the table it was read out of. Where the
    reference has no sidecar, or none for a given row, the recipe's own
    token stands in.

    This is what makes the sidecar usable by
    :func:`openplaces.io.redaction.withhold`, which withholds a
    restricted source's values cell by cell by matching that name. A
    value taken from a parcel spine used to read `spine`, hiding the
    county source that supplied it, so a restricted county reaching a
    sale through the spine could not be withheld per cell.
    """
    if token is None:
        return None
    sidecar = f'{column}{_PROVENANCE_SUFFIX}'
    if sidecar not in ref_indexed.columns:
        return token
    upstream = skey.map(ref_indexed[sidecar])
    return upstream.where(upstream.notna(), token)


def _warn_if_link_underperforms(
    matched: int,
    total: int,
    spine_key: str,
    recipe_id: str,
    admin_id,
    fill_only: bool = False,
) -> None:
    """Flag a `parcel_id_local` join that resolved too few of the spine.

    The bundled conversions in `geo/parcel_id_links.csv` were measured
    once, on one vintage of one pair of sources. A newly ingested source
    for the same county can format its ids differently, and the join then
    quietly returns few rows rather than failing -- there is no error to
    catch, only a thin result. So the achieved share is compared against
    `PARCEL_ID_RELINK_THRESHOLD` and a shortfall is named, with the
    command that re-derives the rule from the data actually in hand.

    Only keys carrying a `parcel_id_local` value are checked: others are
    not derived by a conversion this repository can re-fit. That includes
    `parcel_link_key`, which *is* that key for all but the rows naming a
    stacked unit. Checking it is not optional housekeeping: Lake County
    FL fell from 96.5% to nothing when its bundled conversion stopped
    fitting, and said so to no one, because the spine had by then moved
    off the literal column name this guard used to require.
    """
    if spine_key not in PARCEL_ID_LOCAL_KEYS or not total:
        return
    # A `fill_only` pass exists to reach the rows an earlier pass could
    # not, so matching a minority is its job, not a symptom. The
    # transaction spine's property link reaches only sales of stacked
    # units: warning on it fired for 30 Florida counties at once and
    # buried the passes that genuinely misfit (Pasco's parcel join at
    # 62.4%, Volusia's at 60.5%).
    if fill_only:
        return
    achieved = matched / total
    if achieved >= PARCEL_ID_RELINK_THRESHOLD:
        return
    warnings.warn(
        f'link_by_id: {recipe_id} matched only {achieved:.1%} of '
        f'{total:,d} spine rows for {admin_id} on {spine_key} '
        f'(under {PARCEL_ID_RELINK_THRESHOLD:.0%}). The bundled conversion '
        'may not fit this source. Re-derive it with '
        '`openplaces.geo.build_parcel_id_links.recheck_parcel_id_links('
        f'[{str(admin_id)!r}])`.',
        stacklevel=3,
    )


def _move_units_to_lots(
    ref: pd.DataFrame,
    ref_key: str,
    spine_key: pd.Series,
    pairs: pd.DataFrame,
    spine: pd.DataFrame,
    area_column: str = 'area_ha',
) -> tuple[pd.DataFrame, int, int]:
    """Re-key reference rows that name a split unit onto the unit's lot.

    On a stacked lot the parcel row carries the lot's key, so a tax roll
    row keyed on a unit's own number finds no spine row. *pairs* is what
    the ingest-time split recorded (`unit_key`, `lot_key`), and a row
    whose key is a unit's, and is on no spine row, takes the lot's.

    An account the split saw on several lots becomes one row per lot.
    Its land (every additive column whose name says land) is divided by
    lot area, which is what land is. Every other additive column goes
    whole to the largest lot and is left empty on the others: a building
    stands on one lot, and dividing an improvement value by land area
    would put part of a house on a vacant lot. `total_value` therefore
    overstates the largest lot by the other lots' share of the land;
    the property-to-footprint link is what will place improvements
    properly. Exact lookups on issued keys; nothing is scored.

    Returns the re-keyed frame, the number of rows moved, and how many
    of them sit on several lots.
    """
    own = ref[ref_key].astype('string')
    known = set(spine_key.dropna())
    pairs = pairs.drop_duplicates()
    pairs = pairs[pairs['unit_key'].isin(set(own.dropna()) - known)]
    if pairs.empty:
        return ref, 0, 0
    n_lots = pairs.groupby('unit_key')['lot_key'].transform('size')
    single = pairs[n_lots == 1].set_index('unit_key')['lot_key']
    several = pairs[n_lots > 1]

    ref = ref.copy()
    target = own.map(single)
    ref[ref_key] = own.where(target.isna(), target)
    n_moved = int(target.notna().sum())
    if several.empty:
        return ref, n_moved, 0

    area = None
    if area_column in spine.columns:
        area = (
            pd.to_numeric(spine[area_column], errors='coerce').groupby(spine_key).sum()
        )
    several = several.assign(
        lot_area=several['lot_key'].map(area) if area is not None else float('nan')
    )
    total = several.groupby('unit_key')['lot_area'].transform('sum')
    size = several.groupby('unit_key')['lot_key'].transform('size')
    weighable = (
        several['lot_area'].notna().groupby(several['unit_key']).transform('all')
    )
    several['share'] = (several['lot_area'] / total).where(
        weighable & total.gt(0), 1.0 / size
    )
    rank = several.groupby('unit_key')['share'].rank(method='first', ascending=False)
    several['largest'] = rank.eq(1)

    divided = ref[own.isin(set(several['unit_key']))]
    expanded = divided.assign(unit_key=own[divided.index]).merge(
        several[['unit_key', 'lot_key', 'share', 'largest']], on='unit_key'
    )
    expanded[ref_key] = expanded['lot_key']
    for column in divided.columns:
        if column == ref_key or get_agg_func(resolve_attribute_name(column)) != 'sum':
            continue
        values = pd.to_numeric(expanded[column], errors='coerce')
        if 'land' in column:
            expanded[column] = values * expanded['share']
        else:
            expanded[column] = values.where(expanded['largest'])
    expanded = expanded.drop(columns=['unit_key', 'lot_key', 'share', 'largest'])
    ref = pd.concat([ref.drop(index=divided.index), expanded], ignore_index=True)
    return ref, n_moved + len(divided), len(divided)


@_register('link_by_id')
def link_by_id(
    state: HarmonizeState,
    recipe_id: str | None = None,
    auto_discover: bool = False,
    entity_type: str = 'parcel',
    mode: str = 'attributes',
    spine_key: str = DEFAULT_LINK_KEY,
    ref_key: str = DEFAULT_LINK_KEY,
    columns: list[str] | dict[str, str] | None = None,
    aggregation_function: dict[str, str] | None = None,
    suffix: str | None = None,
    count_as: str | bool | None = None,
    flag_as: str | None = 'is_transacted',
    layer: str | None = None,
    ref_sort_by: str | None = None,
    ref_sort_ascending: bool = True,
    track_provenance: list[str] | None = None,
    fill_only: bool = False,
    supplements_only: bool = False,
    ref_address_key: dict | None = None,
    spine_address_key: dict | None = None,
    _protect_own_columns: set[str] | None = None,
    _supplement_of: str | None = None,
    _unit_lot_pairs: pd.DataFrame | None = None,
) -> HarmonizeState:
    """Link a reference entity to the spine by a precomputed id key (non-spatial).

    Joins on the standardized matching key (``parcel_id_local``) that data
    ingestion already computed on both sides, so no re-conversion happens here.

    Column priority (``'attributes'`` and ``'aggregate'`` modes): when a
    column is already on the spine (from an earlier call, e.g. an earlier
    *auto_discover* match), the new source only overwrites it outright if the
    new source covers a majority of spine rows for that column; otherwise it
    only fills the existing column's gaps (see :func:`_write_prioritized`).
    Combined with *auto_discover*'s least-specific-to-most-specific join
    order, this makes the most admin-specific source the default winner for
    each column, without letting a sparse more-specific source blank out a
    more complete less-specific one.

    Parameters
    ----------
    recipe_id : str, optional
        Reference entity recipe (e.g. an assessment roll or a transaction
        table). Required unless *auto_discover* is set.
    auto_discover : bool
        When set, ignore *recipe_id* and instead discover every ingest
        recipe of *entity_type* (plus any bundled ``additional_layers``)
        whose admin scope covers the admin unit being processed
        (:func:`_discover_link_sources`), and recurse into this function once
        per match. Each match always joins via ``mode='aggregate'`` (a
        correct generalization of ``'attributes'`` for 1:1 data too) on its
        resolved key (``parcel_id_local`` for a standalone roll; a layer's
        own ``layer_key`` for a bundled ``additional_layers`` entry), with
        *columns* defaulting to the attribute registry's canonical columns
        for that entity type (:func:`~openplaces.core.attribute_registry.
        get_attributes`) — pass an explicit *columns* to override this
        default for every discovered match. *suffix* and *count_as* are
        forwarded to each discovered match unchanged (e.g. a fixed
        ``count_as='n_transactions'`` for every auto-discovered
        ``entity_type='transaction'`` source). Any ``*-remap.csv`` crosswalk
        found beside a matched source is applied automatically (see
        :func:`_apply_remap_csvs`). A standalone match that is also one of
        the spine's own geometry sources has its ``resolve_spine``
        ``keep_columns`` protected row-by-row (via ``_protect_own_columns``,
        internal-only) rather than dropped from the join outright: only the
        rows where *this* source's own geometry actually won
        (``geometry_source`` equals its label) are shielded, since those
        already carry a correct, un-aggregated value from ``resolve_spine``
        directly -- re-deriving one via this aggregate join would pool
        values across every row sharing a non-unique local key. Every other
        row (geometry won by a *different* source) is free to receive this
        source's value through the ordinary null-aware
        :func:`_write_prioritized` gap-fill/overwrite, so a more
        admin-specific source's richer keep_columns data can still fill a
        gap left by whichever source won geometry there. Falls back to the
        old, coarser "drop the column from this match entirely" guard when
        the spine has no ``geometry_source`` column at all (e.g. a
        :func:`union_spine_sources`-built, non-spatial spine).
    entity_type : str
        Entity type to discover when *auto_discover* is set (default
        ``'parcel'``).
    mode : {'attributes', 'count', 'aggregate'}
        ``'attributes'`` joins *columns* from the reference onto the spine
        (1:1 on the key, keeping the first reference row per key). ``'count'``
        aggregates a 1:many reference into a per-spine count (*count_as*) and a
        boolean presence flag (*flag_as*) — used to track which parcels have
        been transacted. ``'aggregate'`` reduces a 1:many reference onto the
        spine by grouping on the key and applying each column's attribute-
        registry aggregation (e.g. sum land_value/n_dwellings, max year_built),
        or *aggregation_function*'s override where one is given for that
        column, falling back to the first non-null value for columns without
        a usable rule; it also emits a per-key record count. Use it when several
        reference rows share one spine key, such as MassGIS L3_ASSESS condominium
        records stacked on one parcel polygon.
    spine_key, ref_key : str
        Key columns on the spine and reference (default ``parcel_id_local``).
        Unlike an entity's own index (unique by design), this key is not
        guaranteed unique; see :func:`_warn_if_duplicate_key`, which warns
        when either side turns out to have duplicates, so that risk is
        visible rather than silently assumed away.
    columns : list of str or dict of {str: str}, optional
        Reference columns to attach in ``'attributes'``/``'aggregate'`` mode.
        A dict renames each reference column to its value on write (e.g.
        ``{'price': 'last_sale_price', 'recorded_date': 'last_sale_date'}``),
        so the registry aggregation lookup and ``_write_prioritized`` gap-fill
        apply to the actual canonical output name rather than the reference's
        own column name.
    aggregation_function : dict of {str: str}, optional
        ``'aggregate'`` mode only. Per-output-column override of the
        attribute-registry aggregation, keyed by the same post-rename
        canonical output name as *columns*'s dict form (e.g.
        ``{'year_built': 'min'}``). Exists because the registry default is a
        property-level default (``year_built`` is ``'mean'``, i.e. one value
        per property in most sources), while a reference where several rows
        share a key for a structural reason -- e.g. one row per building
        component in a PACS ``APPRAISAL_IMPROVEMENT_DETAIL`` roll -- needs
        that key's *min* (the original structure's year) instead.
        Changing the registry default would
        corrupt every other recipe's one-row-per-property case, so the
        override lives here, per recipe, instead. A column absent from the
        dict keeps the registry default. Resolved through the same
        :func:`~openplaces.table._agg_func_for` alias lookup as the
        registry path, so ``'join_nonnull'`` works here too, not only the
        plain pandas reducer names.

        In ``auto_discover`` mode this explicit override still applies to
        every discovered match, so prefer letting the *reference recipe*
        declare its own top-level ``aggregation_function`` key instead
        (:func:`_discover_link_sources` reads it): that scopes the
        override to just that one recipe's columns, leaving every sibling
        match's registry default untouched. Pass this parameter in
        ``auto_discover`` mode only when the override genuinely belongs to
        the caller, not to one specific source.
    suffix : str, optional
        Suffix appended to attached column names (``'attributes'``/
        ``'aggregate'`` mode).
    count_as, flag_as : str, optional
        Output column names in ``'count'`` mode (default ``'n_transactions'``/
        ``'is_transacted'`` when unset). In ``'aggregate'`` mode, *count_as*
        names the per-key record count column (default ``'n_records_per_key'``
        when unset — deliberately not ``'n_transactions'`` unless requested,
        since ``'aggregate'`` is also used for non-transaction references like
        MassGIS condo unit stacks); pass ``flag_as=None`` to skip the
        ``'count'``-mode presence flag when it isn't needed (e.g. it's exactly
        ``count_as > 0`` and not worth persisting). A count column written
        by more than one source in a run (every auto-discovered match
        shares one *count_as*) accumulates across them rather than being
        overwritten by the last; see :func:`_accumulate_count`. In
        ``'aggregate'`` mode, ``count_as=False`` writes no count column at
        all, for a join that should add only its attributes.
    layer : str, optional
        Secondary layer (entity type or full entity string) of an
        ``additional_layers`` entity to load from *recipe_id*, e.g. the
        ``property`` assessor table bundled inside a MassGIS parcel recipe.
    ref_sort_by : str, optional
        Reference column to sort by before ``'aggregate'``/``'attributes'``
        mode picks a row per key. The registry's ``'first'``/``'last'``
        aggregation (and ``'attributes'`` mode's own duplicate-key pick) take
        whichever row happens to come first in *ref*'s existing order, which
        is not necessarily meaningful order (e.g. a raw transaction table is
        not guaranteed sorted by date) -- set this to make ``'first'`` mean
        "most recent" for a column like ``last_sale_price``/``last_sale_date``
        derived from a ``recorded_date`` reference.
    ref_sort_ascending : bool, default True
        Sort direction for *ref_sort_by* (``False`` so ``'first'`` picks the
        most recent row when sorting by a date column).
    fill_only : bool, optional
        Never overwrite a value the spine already holds; only fill gaps
        (default False). Set this on a pass whose key is lossier than the
        one before it. The default write rule overwrites outright once a
        reference covers half the spine, and a lossy key is most
        dangerous exactly where it matches best: in Carteret County NC
        97% of parcels share their punctuation-free key with others, in
        groups of up to 335, so the second pass replaced every parcel's
        own improvement value with one arbitrary row's, and the county's
        total came out 31 times its source (measured 2026-09-09).
    track_provenance : list of str, optional
        Output column names (post-rename, post-suffix base names) to record
        per-cell source provenance for via :func:`_write_prioritized`'s
        ``provenance_token`` (see there) -- writes a ``{column}_source``
        sidecar for exactly the columns named here, not every joined
        attribute. In ``auto_discover`` mode, forwarded unchanged to every
        discovered match, each stamping its own ``source_id`` (see
        :func:`~openplaces.recipe.source_id_from_recipe_id`) as the token.
    supplements_only : bool, optional
        ``auto_discover`` mode only. Join only the discovered recipes that
        declare ``supplements:`` for one of this spine's own sources
        (``state.metadata['spine_source_recipe_ids']``), never a roll or an
        unrelated table (default False). This is how a property spine
        receives its rolls' detail tables as columns on the properties they
        describe, without re-joining the rolls onto themselves.

        A supplement declaring ``supplements_key: <column>`` joins on that
        column on both sides, whatever *spine_key*/*ref_key* say, and only
        onto spine rows loaded from the roll it names (the spine's
        ``source`` label is the roll's source id or recipe id). Such a
        key relates the table to its roll alone, so a discovery without
        *supplements_only* skips it rather than joining it onto another
        entity on a key that entity does not share. A missing key column
        on either side raises instead of warning: the recipes promise the
        column (see :func:`~openplaces.recipe.get_supplements_key`), so
        its absence means an output written before the recipe changed.
    ref_address_key : dict, optional
        Derive the address matching key on the reference before joining,
        with the arguments of
        :func:`~openplaces.io.harmonizer.addresses.derive_address_id_local`
        (e.g. ``{street_column: street, number_column: street_no,
        admin4_column: null, output_column: address_id_local_county}``),
        so a reference that was never harmonized is keyed with exactly the
        normalization the spine's own key used. *ref_key* must name one of
        the derived columns. Exists for sources whose rows are not the
        entity of any spine (building permits: one row per permit, not per
        property), which therefore never pass through a harmonized table
        that would key them.
    spine_address_key : dict, optional
        The same derivation on the spine, used for this join only and not
        written to the spine, so a key a single link needs (an address
        key without the town scope a permit source cannot supply) does
        not become a persisted column. Ignored where the spine already
        carries *spine_key*.
    """
    if auto_discover:
        # A standalone roll that is also one of the spine's own geometry
        # sources (state.metadata['spine_source_recipe_ids'], set by
        # resolve_spine) would otherwise re-derive its keep_columns
        # attributes (e.g. use_group/use_subgroup) by aggregating across
        # every spine row sharing its join key -- overwriting an
        # already-correct per-geometry value with one pooled from unrelated
        # rows. Those columns are already on the spine directly from the
        # same source's own row wherever its geometry won; protect exactly
        # those rows (see _protect_own_columns below) rather than dropping
        # the column from the whole match, so this same source can still
        # fill a keep_columns gap on a row a *different* source's geometry
        # occupies.
        spine_source_ids = state.metadata.get('spine_source_recipe_ids', set())
        spine_keep_columns = state.metadata.get('spine_keep_columns', set())
        has_geometry_source = (
            state.spine is not None and 'geometry_source' in state.spine.columns
        )
        matches = _discover_link_sources(state, entity_type)
        if supplements_only:
            matches = _select_supplements(matches, spine_source_ids)
        for match in matches:
            keyed = match.get('supplements_key')
            if keyed and not supplements_only:
                # Its key names rows of its roll, not parcels or any
                # other entity: joining it here would match on a column
                # the spine does not share, or on one that means
                # something else there. Its attributes reach other
                # entities in curate, from the property spine.
                continue
            # The registry default lists the matching keys too, and
            # copying a matched source's key over the spine's own is
            # never what a link means: the key is how the rows met.
            # Pender County NC (geospine of 2026-09-08) lost its county
            # key on 99.8% of parcels to one placeholder value this way,
            # when a punctuation-free fallback pass matched a statewide
            # layer whose assessor id is '0' on every row. An explicit
            # *columns* list is left as the caller wrote it.
            match_columns = columns or [
                c
                for c in get_attributes(match['layer'] or entity_type).index
                if c not in _LINK_KEY_COLUMNS
            ]
            protect_columns: set[str] | None = None
            if match['layer'] is None and match['recipe_id'] in spine_source_ids:
                keep_overlap = {c for c in match_columns if c in spine_keep_columns}
                if keep_overlap:
                    if has_geometry_source:
                        protect_columns = keep_overlap
                    else:
                        # No geometry_source to key row-level protection on
                        # (e.g. a union_spine_sources-built non-spatial
                        # spine) -- fall back to the coarser column drop
                        # rather than risk the pooled-duplicate-key
                        # corruption this guard exists to prevent.
                        match_columns = [
                            c for c in match_columns if c not in keep_overlap
                        ]
                        if not match_columns:
                            continue
            # A match's own declared override (e.g. the improvement-detail
            # sibling's year_built: min) wins over the caller's for the
            # columns it names, but never reaches a sibling match with no
            # such declaration -- see _discover_link_sources.
            match_aggregation_function = {
                **(aggregation_function or {}),
                **(match['aggregation_function'] or {}),
            }
            # Auto-discovery normally picks the key per match. An
            # explicit key from the caller overrides it, which is what
            # lets a second pass re-run the same discovery on the
            # punctuation-free fallback key. A supplements_key is the
            # one column relating a supplement to its roll, so no
            # caller key replaces it.
            unit_lot_pairs = None
            if keyed:
                match_spine_key = match_ref_key = keyed
            elif match.get('stacked_units_layer'):
                # Units split off a parcel table name their lot in the
                # layer's key and keep their own `parcel_id_local`, so
                # the pair of columns differs by side. A caller's
                # fallback key is built from the unit's own id and
                # cannot name a lot: such a pass skips this layer.
                if spine_key != DEFAULT_LINK_KEY or ref_key != DEFAULT_LINK_KEY:
                    continue
                match_spine_key, match_ref_key = DEFAULT_LINK_KEY, match['key']
            else:
                match_spine_key = (
                    spine_key if spine_key != DEFAULT_LINK_KEY else match['key']
                )
                match_ref_key = ref_key if ref_key != DEFAULT_LINK_KEY else match['key']
                if (
                    entity_type == 'property'
                    and match_spine_key == DEFAULT_LINK_KEY
                    and match_ref_key == DEFAULT_LINK_KEY
                ):
                    # A roll keyed on a unit's own number finds no parcel
                    # row on a stacked lot, whose row carries the lot's
                    # key. The split's unit-to-lot pairs carry it there.
                    from openplaces.io.harmonizer.entity_links import (
                        load_unit_lot_pairs,
                    )

                    unit_lot_pairs = load_unit_lot_pairs(state.admin_id)
            state = link_by_id(
                state,
                recipe_id=match['recipe_id'],
                mode='aggregate',
                spine_key=match_spine_key,
                ref_key=match_ref_key,
                columns=match_columns,
                aggregation_function=match_aggregation_function or None,
                suffix=suffix,
                count_as=count_as,
                layer=match['layer'],
                ref_sort_by=ref_sort_by,
                ref_sort_ascending=ref_sort_ascending,
                track_provenance=track_provenance,
                fill_only=fill_only,
                _protect_own_columns=protect_columns,
                _supplement_of=match['supplements'] if keyed else None,
                _unit_lot_pairs=unit_lot_pairs,
            )
            state = _apply_remap_csvs(state, match['recipe_id'])
        return state

    if recipe_id is None:
        warnings.warn('link_by_id: no recipe_id and auto_discover is False; skipping.')
        return state

    if state.spine is None:
        warnings.warn('link_by_id: spine is None; skipping.')
        return state
    # The punctuation-free fallback key is derived on demand rather than
    # required from ingest. It is a pure function of id columns both sides
    # already carry, so deriving it here makes the fallback work against
    # everything already on disk instead of forcing a re-ingest of every
    # parcel source in the country.
    if spine_key in PARCEL_ID_ALNUM_KEYS:
        # Recomputed, never inherited. resolve_spine would otherwise carry
        # a copy from whichever source won the geometry, which for exactly
        # the counties this fallback exists to serve is the source that
        # has no usable id -- Pender arrived with 49 of 55,101 filled.
        state.spine = add_parcel_id_alnum(state.spine, key=spine_key)
    if _supplement_of and spine_key not in state.spine.columns:
        raise ValueError(
            f'link_by_id: {recipe_id} joins {_supplement_of} on its '
            f'supplements_key {spine_key!r}, but the spine has no such '
            f'column. Re-ingest {_supplement_of}: its output predates the key.'
        )
    derived_spine_key = None
    if spine_address_key and spine_key not in state.spine.columns:
        derived_spine_key = _derive_address_key(
            state.spine, state.admin_id, spine_address_key, spine_key
        )
    if spine_key not in state.spine.columns and derived_spine_key is None:
        warnings.warn(
            f'link_by_id: spine has no {spine_key!r}; skipping {recipe_id}. '
            'Was the spine source ingested with a parcel_id_local directive '
            'and resolve_spine keep_columns?'
        )
        return state

    try:
        ref = get_entities(recipe_id, state.admin_id, layer=layer)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        # The reference is not available for this admin (e.g. a source-specific
        # roll listed in a shared pipeline that does not apply here, including
        # one scoped to a different state) or has not been ingested; skip.
        if state.verbose:
            print(f'  link_by_id: no {recipe_id} for {state.admin_id}; skipping.')
        return state
    if ref is not None and layer is None:
        # A reference scoped coarser than state.admin_id with no matching
        # admin-id column (e.g. a statewide transaction table keyed only by a
        # free-text county name) comes back unfiltered from get_entities --
        # restrict it here so a per-county aggregate isn't silently pooled
        # across the whole state.
        ref = restrict_to_admin_by_name(ref, recipe_id, state.admin_id)
    if ref is not None and ref_address_key:
        keyed = _derive_address_key(ref, state.admin_id, ref_address_key, ref_key)
        if keyed is not None:
            ref = ref.assign(**{ref_key: keyed})
    if ref is not None and ref_key in PARCEL_ID_ALNUM_KEYS:
        ref = add_parcel_id_alnum(ref, key=ref_key)
    if _supplement_of and ref is not None and ref_key not in ref.columns:
        raise ValueError(
            f'link_by_id: {recipe_id} declares supplements_key {ref_key!r}, '
            f'but its output has no such column. Re-ingest {recipe_id}: '
            'its output predates the key.'
        )
    if ref is None or ref_key not in ref.columns:
        if ref_key in PARCEL_ID_ALNUM_KEYS:
            # A derived fallback key simply cannot be built for a source
            # that carries none of its input columns -- a county roll has
            # no statewide PIN column, and that is the normal case rather
            # than a misconfigured recipe. Silent, or every extra pass
            # warns once per source in every county.
            if state.verbose:
                print(f'  link_by_id: {recipe_id} cannot derive {ref_key!r}; skipping.')
        else:
            warnings.warn(
                f'link_by_id: reference {recipe_id} has no {ref_key!r}; skipping.'
            )
        return state
    if ref_sort_by and ref_sort_by in ref.columns:
        ref = ref.sort_values(ref_sort_by, ascending=ref_sort_ascending, kind='stable')

    spine = state.spine
    skey = (
        derived_spine_key if derived_spine_key is not None else spine[spine_key]
    ).astype('string')
    if _supplement_of and 'source' in spine.columns:
        # A supplements_key is a source's own id, unique only among its
        # roll's rows; another source unioned onto the same spine may
        # carry the same value for a different property. Only the
        # roll's rows take part (union_spine_sources labels them with
        # the roll's source id, an explicit source with its recipe id).
        own_labels = {_supplement_of, source_id_from_recipe_id(_supplement_of)}
        skey = skey.where(spine['source'].astype('string').isin(own_labels))
    # Before any mode reads the key: a placeholder shared by thousands of
    # rows is not an identifier, and every mode below would silently treat
    # it as one (see _neutralize_degenerate_keys).
    if _unit_lot_pairs is not None and len(_unit_lot_pairs):
        ref, n_moved, n_divided = _move_units_to_lots(
            ref, ref_key, skey, _unit_lot_pairs, state.spine
        )
        if state.verbose and n_moved:
            print(
                f'  link_by_id: {n_moved:,d} {recipe_id} rows keyed on a '
                f'stacked unit moved to their lot ({n_divided:,d} of them '
                'across several lots)'
            )
    ref = _neutralize_degenerate_keys(ref, ref_key, recipe_id, spine_key=skey)
    rkey = ref[ref_key].astype('string')
    spine_entity = state.recipe.get('entity')
    spine_entity_type = (
        str(spine_entity.entity_type) if spine_entity is not None else None
    )
    is_own_key = spine_entity_type is None or spine_key.startswith(
        f'{spine_entity_type}_id'
    )
    _warn_if_duplicate_key(skey, spine_key, 'spine key', is_own_identity_key=is_own_key)

    if mode == 'attributes':
        pairs = [(c, o) for c, o in _columns_as_pairs(columns) if c in ref.columns]
        # 'attributes' keeps one arbitrary row per key (no aggregation, unlike
        # 'aggregate'/'count') -- a duplicate ref_key here is silently resolved
        # by drop_duplicates below, so flag it before that happens.
        _warn_if_duplicate_key(rkey, ref_key, 'attributes reference key')
        ref_unique = ref.dropna(subset=[ref_key]).drop_duplicates(ref_key).copy()
        ref_unique.index = ref_unique[ref_key].astype('string')
        provenance_cols = set(track_provenance or [])
        token = source_id_from_recipe_id(recipe_id) if provenance_cols else None
        for col, out_name in pairs:
            name = f'{out_name}{suffix}' if suffix else out_name
            ref_series = ref_unique[col]
            mapper = ref_series.to_dict() if ref_series.empty else ref_series
            _write_prioritized(
                spine,
                name,
                skey.map(mapper),
                # A lossy key must only fill what a precise one left.
                majority_coverage=float('inf') if fill_only else 0.5,
                provenance_token=(
                    _upstream_tokens(ref_unique, col, skey, token)
                    if out_name in provenance_cols
                    else None
                ),
            )
        matched = int(skey.isin(set(rkey.dropna())).sum())
        if state.verbose:
            print(
                f'  Link by id (attributes): {matched:,d}/{len(spine):,d} spine '
                f'rows matched {recipe_id} ({len(pairs)} columns)'
            )
        _warn_if_link_underperforms(
            matched, len(spine), spine_key, recipe_id, state.admin_id, fill_only
        )
    elif mode == 'count':
        count_as = count_as or 'n_transactions'
        counts = rkey.dropna().value_counts()
        mapper = counts.to_dict() if counts.empty else counts
        _accumulate_count(state, spine, count_as, skey.map(mapper))
        if flag_as:
            spine[flag_as] = spine[count_as] > 0
        linked = int((spine[count_as] > 0).sum())
        if state.verbose:
            print(
                f'  Link by id (count): {linked:,d}/'
                f'{len(spine):,d} spine rows linked to {recipe_id} ({count_as})'
            )
        # A count link is a different question: a parcel with no
        # transaction is a real zero, not a failed join, so a low share
        # says nothing about the conversion and nothing is flagged here.
    elif mode == 'aggregate':
        pairs = [
            (c, o)
            for c, o in _columns_as_pairs(columns)
            if c in ref.columns and c != ref_key
        ]
        ref_valid = ref.dropna(subset=[ref_key]).copy()
        ref_valid[ref_key] = ref_valid[ref_key].astype('string')
        grouped = ref_valid.groupby(ref_key, sort=False)

        # Registry-driven reduction (sum values/dwellings, mean year, etc.);
        # columns without a usable registry rule fall back to the first value.
        # Looked up by the *output* name -- the canonical slot being filled --
        # not the reference's own column name, so a rename (e.g. price ->
        # last_sale_price) still resolves the right aggregation. 'join_nonnull'
        # (e.g. address, use_group) is not a pandas groupby function on its
        # own -- routed through _agg_func_for (the same helper aggregate_rows
        # uses) to concatenate every distinct non-null value instead of
        # silently degrading to 'first' (an arbitrary row's value, discarding
        # every other row's -- the original multi-property-per-parcel
        # collapse bug). 'address' gets its own joiner there
        # (join_nonnull_addresses) rather than the generic one: a condo/
        # apartment building's per-unit property records typically differ
        # only by an APT/UNIT/# suffix, and the plain joiner's ' + '-
        # concatenation of every unit's full address corrupts downstream
        # address parsing (no parser can split a multi-address blob back
        # into one street/number). 'use_group' and other join_nonnull
        # columns keep the plain joiner -- their concatenated values are
        # consumed directly, never re-parsed.
        reducible = {
            'sum',
            'mean',
            'max',
            'min',
            'first',
            'last',
            'median',
            'join_nonnull',
        }
        own_geometry_mask = None
        if _protect_own_columns and 'geometry_source' in spine.columns:
            own_label = source_id_from_recipe_id(recipe_id)
            own_geometry_mask = spine['geometry_source'].astype('string') == own_label
        provenance_cols = set(track_provenance or [])
        token = source_id_from_recipe_id(recipe_id) if provenance_cols else None
        for col, out_name in pairs:
            canonical_name = resolve_attribute_name(out_name)
            fname = (aggregation_function or {}).get(out_name) or get_agg_func(
                canonical_name
            )
            func = (
                _agg_func_for(canonical_name, fname) if fname in reducible else 'first'
            )
            name = f'{out_name}{suffix}' if suffix else out_name
            col_series = ref_valid[col]
            # A registry-numeric column can still arrive here as pandas
            # 'string'/object dtype (e.g. a fixed-width ingest, which never
            # casts a mapped column's dtype -- see the PACS improvement-
            # detail recipe). 'sum'/'mean'/'median' on a string column
            # crashes outright; 'min'/'max' does not, but silently compares
            # lexicographically instead of numerically (e.g. '12' < '5'),
            # which is worse -- no crash to notice it by. Coerce first for
            # either failure mode.
            if fname in (
                'sum',
                'mean',
                'median',
                'min',
                'max',
            ) and not pd.api.types.is_numeric_dtype(col_series):
                grouped_col = pd.to_numeric(col_series, errors='coerce').groupby(
                    ref_valid[ref_key], sort=False
                )
            else:
                grouped_col = grouped[col]
            if fname == 'sum':
                # A group whose values are all missing has an unknown
                # total, not a total of zero: a property whose every bath
                # row failed to parse must not read as zero baths.
                # min_count=1 matches transform._aggregate_cols.
                agg_series = grouped_col.sum(min_count=1)
            else:
                agg_series = grouped_col.agg(func)
            mapper = agg_series.to_dict() if agg_series.empty else agg_series
            new_vals = skey.map(mapper)
            if fname == 'sum':
                # A group total stamped onto every spine row sharing the
                # key multiplies it by the number of rows: in Carteret
                # County NC 97% of parcels share their punctuation-free
                # key (groups of up to 335), and the county's improvement
                # value came out 31 times its source. A sum belongs to one
                # spine row; where the key cannot say which, assign none.
                # Only where a sum happened, though: a reference key held
                # by one row was not summed, and its value is that row's,
                # so every spine row that names it may carry it. That is
                # the transaction spine's case, several sales of one
                # parcel each carrying the parcel's value (Lake County
                # FL, 2026-09-12: 572,120 of 597,666 sales withheld
                # before this distinction, every parcel value lost).
                group_sizes = grouped.size()
                summed = skey.map(group_sizes).fillna(0) > 1
                shared = skey.duplicated(keep=False) & skey.notna() & summed
                if shared.any():
                    warnings.warn(
                        f'link_by_id (aggregate): {name!r} is a sum over '
                        f'{ref_key!r}, and {int(shared.sum()):,} spine rows '
                        f'share their {spine_key!r} with others; the sum is '
                        'not assigned to them, since stamping it on each '
                        'would count it once per row.',
                        stacklevel=2,
                    )
                    new_vals = new_vals.mask(shared)
            if (
                _protect_own_columns
                and out_name in _protect_own_columns
                and own_geometry_mask is not None
            ):
                new_vals = new_vals.mask(own_geometry_mask)
            _write_prioritized(
                spine,
                name,
                new_vals,
                # A lossy key must only fill what a precise one left.
                majority_coverage=float('inf') if fill_only else 0.5,
                # The recipe's own name, not an upstream sidecar, unlike
                # the 'attributes' branch. Two reasons, and they agree:
                # this is the auto-discovered path, so `recipe_id` is
                # already the county source that supplied the value and
                # there is nothing more original to reach; and a value
                # here may be an aggregate over several reference rows,
                # which can disagree about their own provenance, so one
                # token per spine row would be a guess.
                provenance_token=token if out_name in provenance_cols else None,
            )
        # count_as=False asks for the attributes alone: a spine receiving
        # a detail table's columns has no use for a count of its rows,
        # and an unneeded column is redundancy in harmonize.
        if count_as is False:
            count_col = None
        else:
            count_col = count_as or 'n_records_per_key'
            gsize = grouped.size()
            mapper = gsize.to_dict() if gsize.empty else gsize
            _accumulate_count(state, spine, count_col, skey.map(mapper))
        # Counted whether or not anyone is watching: this is the mode
        # auto-discovery uses, so a spine whose only parcel link is an
        # auto-discovered one is exactly the case that most needs the
        # guard below. Computing it only under `verbose` is how Holmes
        # County FL rebuilt at 6.8% matched without a word.
        matched = int(skey.isin(set(rkey.dropna())).sum())
        if state.verbose:
            print(
                f'  Link by id (aggregate): {matched:,d}/{len(spine):,d} spine '
                f'rows matched {recipe_id} '
                f'({len(pairs)} columns, {count_col or "no count"})'
            )
        _warn_if_link_underperforms(
            matched, len(spine), spine_key, recipe_id, state.admin_id, fill_only
        )
    else:
        raise ValueError(
            f'link_by_id: unknown mode {mode!r}; expected '
            "'attributes', 'count', or 'aggregate'."
        )

    state.spine = spine
    return state
