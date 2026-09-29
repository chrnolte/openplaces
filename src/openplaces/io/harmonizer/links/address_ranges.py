"""Link by address range (link_address_ranges)."""

from __future__ import annotations

import warnings

import pandas as pd

from openplaces.core.schema import (
    AdminId,
)
from openplaces.io.harmonizer import (
    HarmonizeState,
    _register,
    restrict_to_admin_by_name,
)
from openplaces.io.harmonizer.links.by_id import (
    _columns_as_pairs,
)
from openplaces.io.harmonizer.links.combine import (
    _write_prioritized,
)
from openplaces.io.readers import get_entities


@_register('link_address_ranges')
def link_address_ranges(
    state: HarmonizeState,
    recipe_id: str,
    columns: list[str] | dict[str, str] | None = None,
    street_column: str = 'address_street',
    number_column: str = 'address_number',
    admin4_column: str = 'admin4_id',
    suffix: str | None = None,
) -> HarmonizeState:
    """Resolve multi-unit address ranges left unmatched by :func:`link_by_id`.

    A raw address number like ``'704-706'`` (a standard multi-unit/multi-
    family deed notation, preserved by
    :func:`~openplaces.geo.address.normalize_address_components` rather than
    squashed into ``'704706'``) never matches a parcel reference's own
    single-number ``address_id_local`` key via the ordinary
    :func:`link_by_id` steps -- no real parcel has that combined number. This
    step splits the range (:func:`~openplaces.geo.address.split_number_range`)
    and tries each individual number against *recipe_id*'s own
    ``admin4|street|number`` key components (the same construction
    :func:`~openplaces.io.harmonizer.addresses.derive_address_id_local` uses)
    instead. It also handles the mirror case: a *reference* row whose own
    number is a range (e.g. a parcel recorded as ``'20-22 Main St'``) while
    the spine lists a single plain number that's one of the two halves --
    for spine rows :func:`link_by_id` left unmatched, each reference range
    is likewise registered under both of its individual numbers.

    A multi-family range normally corresponds to one parcel: if exactly one
    distinct reference row resolves (from either side's range, or a plain
    match), link it (gap-fill only, via :func:`_write_prioritized`, so an
    already-linked row is never touched). If more than one distinct
    reference row resolves -- ambiguous, and not expected in the normal case
    -- the row is left unmatched rather than guessed at; every such row this
    call finds is reported in a single warning with a small sample, since
    arbitrarily picking one parcel over the other would be wrong until the
    ambiguity is looked at deliberately.

    Parameters mirror :func:`link_by_id`'s naming for the columns it shares
    (*street_column*/*number_column*/*admin4_column* also match
    :func:`~openplaces.io.harmonizer.addresses.derive_address_id_local`'s
    defaults, so a recipe using those defaults on both sides needs none of
    them repeated here).
    """
    if state.spine is None or number_column not in state.spine.columns:
        return state

    from openplaces.geo.address import canonicalize_for_match, split_number_range

    spine = state.spine
    split = (
        spine[number_column]
        .astype('string')
        .map(lambda v: split_number_range(v) if pd.notna(v) else None)
    )
    is_range = split.notna()

    try:
        ref = get_entities(recipe_id, state.admin_id)
    except (FileNotFoundError, OSError, KeyError, ValueError):
        if state.verbose:
            print(
                f'  link_address_ranges: no {recipe_id} for {state.admin_id}; skipping.'
            )
        return state
    if ref is not None:
        ref = restrict_to_admin_by_name(ref, recipe_id, state.admin_id)
    if (
        ref is None
        or street_column not in ref.columns
        or number_column not in ref.columns
    ):
        return state

    admin_str = str(state.admin_id) if state.admin_id else ''
    admin_levels = AdminId(admin_str).levels if admin_str else ()
    admin1_id = admin_levels[0] if admin_levels else None

    def _base_key(frame):
        street = frame[street_column].astype('string').fillna('')
        canon_map = {
            s: canonicalize_for_match(s, admin1_id) if s else ''
            for s in street.unique()
        }
        canon_street = street.map(canon_map)
        number = (
            frame[number_column].astype('string').str.strip().str.upper().fillna('')
        )
        admin4 = (
            frame[admin4_column].astype('string').fillna('')
            if admin4_column in frame.columns
            else pd.Series('', index=frame.index)
        )
        has_base = canon_street.ne('') & number.ne('')
        return (
            admin4 + '|' + canon_street + '|' + number,
            has_base,
            canon_street,
            admin4,
        )

    def _register_key(index_by_key: dict, key, row_id) -> None:
        bucket = index_by_key.setdefault(key, [])
        if row_id not in bucket:
            bucket.append(row_id)

    # Reference key index: each row's own combined key, plus -- for a row
    # whose own number is itself a range -- each half's key too, pointing
    # back to the same row. A key claimed by more than one distinct row
    # (e.g. a real plain parcel at '22 Main St' *and* a range parcel
    # '20-22 Main St' both registering '.../22') is kept as an explicit
    # multi-row bucket rather than silently picking one, so it surfaces as
    # an ambiguous match below like any other multi-parcel case.
    ref_key, ref_has_base, ref_canon_street, ref_admin4 = _base_key(ref)
    ref_key_to_rows: dict[str, list] = {}
    for row_id, key in ref_key[ref_has_base].items():
        _register_key(ref_key_to_rows, key, row_id)

    ref_range_split = (
        ref[number_column]
        .astype('string')
        .map(lambda v: split_number_range(v) if pd.notna(v) else None)
    )
    for row_id in ref.index[ref_range_split.notna() & ref_has_base]:
        num1, num2 = ref_range_split.loc[row_id]
        prefix = f'{ref_admin4.loc[row_id]}|{ref_canon_street.loc[row_id]}|'
        _register_key(ref_key_to_rows, prefix + num1, row_id)
        _register_key(ref_key_to_rows, prefix + num2, row_id)

    pairs = [(c, o) for c, o in _columns_as_pairs(columns) if c in ref.columns]
    output_names = [f'{o}{suffix}' if suffix else o for _, o in pairs]

    _, _, spine_canon_street, spine_admin4 = _base_key(spine)

    existing = [n for n in output_names if n in spine.columns]
    already_matched = (
        spine[existing].notna().any(axis=1)
        if existing
        else pd.Series(False, index=spine.index)
    )
    plain_unmatched = ~is_range & spine[number_column].notna() & ~already_matched
    attempt_rows = is_range | plain_unmatched
    if not attempt_rows.any():
        return state

    matched_index: dict = {}
    ambiguous_rows = []
    n_range_attempted = 0
    for idx in spine.index[attempt_rows]:
        if is_range.loc[idx]:
            n_range_attempted += 1
            num1, num2 = split.loc[idx]
            candidates = [num1, num2]
        else:
            candidates = [str(spine.loc[idx, number_column]).strip().upper()]
        prefix = f'{spine_admin4.loc[idx]}|{spine_canon_street.loc[idx]}|'

        distinct_rows: list = []
        matched_keys: list = []
        for key in dict.fromkeys(prefix + c for c in candidates):
            for row_id in ref_key_to_rows.get(key, []):
                if row_id not in distinct_rows:
                    distinct_rows.append(row_id)
                    matched_keys.append(key)

        if len(distinct_rows) == 1:
            matched_index[idx] = distinct_rows[0]
        elif len(distinct_rows) > 1:
            ambiguous_rows.append(
                {
                    'street': spine_canon_street.loc[idx],
                    'number': spine.loc[idx, number_column],
                    'matched_keys': matched_keys,
                }
            )

    if ambiguous_rows:
        warnings.warn(
            f'link_address_ranges: {len(ambiguous_rows)} row(s) have an '
            f'address number matching more than one distinct parcel in '
            f'{recipe_id!r} (via a range on either side); a multi-family '
            'range should normally resolve to a single parcel, so these '
            'rows are left unmatched until the ambiguity is resolved. '
            f'Sample: {ambiguous_rows[:5]}',
            stacklevel=2,
        )

    match_series = pd.Series(matched_index, dtype='object')
    for col, out_name in pairs:
        name = f'{out_name}{suffix}' if suffix else out_name
        new_vals = match_series.map(ref[col]).reindex(spine.index)
        _write_prioritized(spine, name, new_vals)

    state.spine = spine
    if state.verbose:
        print(
            f'  Link address ranges: {len(matched_index):,d}/'
            f'{int(attempt_rows.sum()):,d} rows matched {recipe_id} '
            f'({n_range_attempted:,d} range-shaped) ({len(pairs)} columns)'
        )
    return state
