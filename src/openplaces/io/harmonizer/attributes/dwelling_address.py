"""A dwelling's address onto the spine
(attribute_dwelling_address).
"""

from __future__ import annotations

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('attribute_dwelling_address')
def attribute_dwelling_address(
    state: HarmonizeState,
    footprint_recipe_id: str,
    on: str = 'parcel_id',
    priority_column: str = 'priority_on_parcel',
    overlap_column: str = 'area_intersection_m2_parcel',
    columns: dict[str, str] | None = None,
) -> HarmonizeState:
    """Relay each parcel's primary footprint's dwelling-point address evidence.

    Dwelling points (e.g. dwelling-overture-2025) are only ever spatially
    linked to footprints, not parcels, so a parcel has no direct access to
    that evidence on its own -- but every parcel's *primary* footprint (per
    :func:`classify_footprint_priority`, the one(s) carrying dwelling/
    building-point evidence) does. This reads *footprint_recipe_id*'s
    harmonized spine, keeps each parcel's primary footprint(s) (largest
    *overlap_column* wins when a parcel has more than one, e.g. a multi-unit
    property with several dwelling-linked buildings), and copies the
    requested *columns* onto the matching parcel row -- so a parcel spine's
    own ``reconcile_addresses`` can declare a source built from this relayed
    evidence. This alone misses a dwelling point whose footprint was never
    detected at all; pair it with a direct parcel<->dwelling spatial link
    (``link_to_reference``/``reconcile_attributes``, same as the footprint
    spine's own dwelling link) as a lower-priority fallback source for that
    case.

    A no-op if ``state.spine`` is ``None``, *footprint_recipe_id* has no
    saved output yet, *on* is not shared between the two spines, or
    *priority_column* is absent from the footprint entity (there would be no
    way to tell which footprint should represent the parcel) -- the same
    "missing evidence is tolerated" convention used throughout this codebase
    (``reconcile_addresses``/``reconcile_attributes``/``reconcile_values``).

    Parameters
    ----------
    footprint_recipe_id : str
        Footprint entity recipe to read (the harmonized spine).
    on : str, optional
        Shared parcel id (default ``'parcel_id'``): a footprint-entity
        column matched against either the parcel spine's current index name
        or its original name recorded in
        ``state.metadata['spine_index_name']`` -- same resolution
        :func:`summarize_footprint_morphology` uses, since a parcel spine's
        own true id lives on the index at this point in the pipeline.
    priority_column : str, optional
        Footprint-entity column marking each footprint's structural role on
        its parcel (default ``'priority_on_parcel'``, written by
        :func:`classify_footprint_priority`). Only ``'primary'`` rows are
        used.
    overlap_column : str, optional
        Footprint-entity column holding each footprint's overlap area (m2)
        with its dominant parcel (default ``'area_intersection_m2_parcel'``).
        Breaks ties among multiple primary footprints on one parcel; ignored
        (first row kept) if absent.
    columns : dict of {footprint column: parcel column}, optional
        Columns to copy (default: the four raw dwelling-overture evidence
        columns ``reconcile_attributes`` writes on the footprint spine,
        mapped to the same names with ``_overture`` swapped for
        ``_footprint`` -- e.g. ``address_street_dwelling_overture`` ->
        ``address_street_dwelling_footprint``, distinct names so a direct
        parcel<->dwelling link's own ``_dwelling_overture`` columns don't
        collide with this relay). Missing source columns are skipped.
    """
    from openplaces.io.readers import get_entities

    if state.spine is None:
        return state
    spine = state.spine

    columns = columns or {
        f'{comp}_dwelling_overture': f'{comp}_dwelling_footprint'
        for comp in ('address_street', 'address_number', 'city', 'postal_code')
    }

    # missing='ignore': footprint_recipe_id genuinely has no output yet for
    # some admin units (not yet harmonized, or itself skipped a reference
    # with zero coverage) -- documented as a tolerated no-op above, not an
    # error; this function already prints its own message below.
    footprints = get_entities(
        footprint_recipe_id, state.admin_id, geom=False, missing='ignore'
    )
    if footprints is None or len(footprints) == 0:
        if state.verbose:
            print('  attribute_dwelling_address: no footprints; skipping.')
        return state

    on_in_spine = on in spine.columns or on in (
        spine.index.name,
        state.metadata.get('spine_index_name'),
    )
    if not (on in footprints.columns and on_in_spine):
        if state.verbose:
            print(f'  attribute_dwelling_address: {on!r} not shared; skipping.')
        return state

    if priority_column not in footprints.columns:
        if state.verbose:
            print(
                f'  attribute_dwelling_address: no {priority_column!r} on '
                f'{footprint_recipe_id}; skipping.'
            )
        return state

    source_cols = [c for c in columns if c in footprints.columns]
    if not source_cols:
        if state.verbose:
            print('  attribute_dwelling_address: no evidence columns found; skipping.')
        return state

    is_primary = footprints[priority_column].astype('string') == 'primary'
    per_fp = footprints[is_primary].copy()
    if per_fp.empty:
        return state
    per_fp['_pid'] = per_fp[on].astype('string')
    per_fp = per_fp.dropna(subset=['_pid'])
    if overlap_column in per_fp.columns:
        per_fp = per_fp.sort_values(overlap_column, ascending=False)
    per_fp = per_fp.drop_duplicates(subset='_pid', keep='first').set_index('_pid')

    key = (
        spine[on].astype('string')
        if on in spine.columns
        else spine.index.to_series().astype('string')
    )

    for src_col in source_cols:
        spine[columns[src_col]] = key.map(per_fp[src_col])

    state.spine = spine
    if state.verbose:
        print(
            f'  attribute_dwelling_address: relayed evidence from '
            f'{len(per_fp):,} primary footprints.'
        )
    return state
