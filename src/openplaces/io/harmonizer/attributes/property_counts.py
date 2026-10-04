"""Preliminary property counts per parcel
(estimate_property_counts).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('estimate_property_counts')
def estimate_property_counts(
    state: HarmonizeState,
    property_count_column: str = 'n_properties_per_parcel',
    source_column: str = 'property_source',
    dwelling_column: str = 'n_dwellings',
    footprint_dwelling_column: str = 'max_dwellings_per_footprint',
) -> HarmonizeState:
    """Fill in ``property_source``/``n_properties_per_parcel`` where still unset.

    A real per-unit property source (a bundled ``additional_layers`` table,
    e.g. MassGIS's L3_ASSESS) or a geometrically inferred shared-land group
    (see the parcel spine's own shared-land step) each write
    *property_count_column*/*source_column* themselves, with
    ``source_column`` set to ``'source'``/``'shared_land_group'``
    respectively -- this step never overwrites either. Everywhere else, it
    estimates a count from evidence that already implies internal
    multiplicity even though no per-unit row exists to count directly:
    *dwelling_column* (the parcel's own recorded dwelling-unit count) or
    *footprint_dwelling_column* (confirmed Overture dwelling points on a
    single footprint on the parcel, from
    :func:`summarize_footprint_morphology` -- run this step after that one).
    A parcel with neither a real source nor multiplicity evidence gets
    ``source_column = 'none'`` and *property_count_column* left at 0/missing.

    This estimate is deliberately not a fabricated 1:1 property row (that
    would misrepresent a state with no per-unit source as having verified
    unit-level data) -- it only ever raises the count, and only when the
    parcel's own evidence already shows multiplicity; ``'estimated'`` marks
    that distinction so downstream consumers (e.g. transaction linking) know
    not to trust it as a real, addressable list of units the way a
    ``'source'``/``'shared_land_group'`` parcel's units are.

    Parameters
    ----------
    property_count_column : str, optional
        Per-parcel property/unit count (default ``'n_properties_per_parcel'``).
    source_column : str, optional
        Provenance of *property_count_column* (default ``'property_source'``).
    dwelling_column : str, optional
        Parcel's own recorded dwelling-unit count (default ``'n_dwellings'``).
    footprint_dwelling_column : str, optional
        Confirmed dwelling count on a single footprint on the parcel (default
        ``'max_dwellings_per_footprint'``).
    """
    if state.spine is None:
        return state
    spine = state.spine

    # Built as fresh Series and assigned back wholesale (never a partial
    # `.loc[mask, col] = ...` mutation of an existing column) -- a column
    # freshly written by an upstream `.map(mapper).astype(...)` call (e.g.
    # link_by_id's count mode) can carry a read-only backing array in some
    # pandas/geopandas dtype combinations, which a partial in-place
    # assignment then fails on.
    source = (
        spine[source_column].astype(object)
        if source_column in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype=object)
    )
    count = (
        pd.to_numeric(spine[property_count_column], errors='coerce')
        if property_count_column in spine.columns
        else pd.Series(np.nan, index=spine.index)
    )

    unset = source.isna()
    has_real_source = unset & (count.fillna(0) > 0)
    source = source.mask(has_real_source, 'source')

    n_dwellings = (
        pd.to_numeric(spine[dwelling_column], errors='coerce').fillna(0)
        if dwelling_column in spine.columns
        else pd.Series(0.0, index=spine.index)
    )
    n_footprint_dwellings = (
        pd.to_numeric(spine[footprint_dwelling_column], errors='coerce').fillna(0)
        if footprint_dwelling_column in spine.columns
        else pd.Series(0.0, index=spine.index)
    )
    estimate = pd.concat([n_dwellings, n_footprint_dwellings], axis=1).max(axis=1)

    needs_estimate = source.isna() & (estimate >= 2)
    count = count.mask(needs_estimate, estimate)
    source = source.mask(needs_estimate, 'estimated')
    source = source.mask(source.isna(), 'none')

    spine[property_count_column] = count
    spine[source_column] = source.astype('category')
    state.spine = spine

    if state.verbose:
        counts = spine[source_column].value_counts()
        print(f'  estimate_property_counts: {counts.to_dict()}')
    return state
