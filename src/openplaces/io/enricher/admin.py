"""
Enrichment step that attaches admin-unit-level covariates to entities.

A municipality-level table (a census population count, a count of
displaced persons, an informality index) says something about every
entity inside the unit. ``link_admin_attribute`` joins one such column
onto the spine through the admin id each entity already carries, and can
express it as a density over the unit's area or as a share of another
joined column, so a recipe never rasterizes a municipality table to get
a per-entity value (the 2023 FEE notebook did exactly that).
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.schema import AdminId
from openplaces.geo.polygon import get_areas
from openplaces.io.enricher import EnrichState, _register
from openplaces.io.readers import get_admin, get_dataset


def _admin_level_from_column(admin_id_column: str, admin_level: int | None) -> int:
    if admin_level is not None:
        return admin_level
    if admin_id_column.startswith('admin') and admin_id_column.endswith('_id'):
        digits = admin_id_column[len('admin') : -len('_id')]
        if digits.isdigit():
            return int(digits)
    raise ValueError(
        'link_admin_attribute: pass admin_level, it cannot be inferred from '
        f'admin_id_column={admin_id_column!r}.'
    )


@_register('link_admin_attribute')
def link_admin_attribute(
    state: EnrichState,
    output: str,
    recipe_id: str,
    attribute: str,
    admin_id_column: str = 'admin3_id',
    admin_level: int | None = None,
    per_km2: bool = False,
    share_of: str | None = None,
) -> EnrichState:
    """Copy an admin-unit attribute from a dataset recipe onto every entity.

    Reads the dataset with :func:`~openplaces.io.readers.get_dataset` at
    the admin unit being processed, keyed by its index (the admin id), and
    joins *attribute* onto the spine through *admin_id_column*. Two
    optional normalisations turn a count into a covariate: *per_km2*
    divides by the unit's own polygon area, and *share_of* divides by
    another column of the same dataset (a displaced-persons count over a
    population count, say). Exactly one of the two may be set.

    A spine row whose admin id is missing from the dataset gets a missing
    value; nothing is imputed here.

    Parameters
    ----------
    output : str
        Evidence column to write. Must exist in the attribute registry.
    recipe_id : str
        Dataset recipe holding one row per admin unit, indexed by admin
        id, at the level of *admin_id_column*.
    attribute : str
        Column of that dataset to copy.
    admin_id_column : str, optional
        Spine column holding the admin id to join on, default
        ``'admin3_id'``.
    admin_level : int, optional
        Level of the units in the dataset. Inferred from
        *admin_id_column* when it is named ``admin{N}_id``.
    per_km2 : bool, optional
        Divide by the admin polygon's area in square kilometres, read
        from the admin spine at *admin_level*.
    share_of : str, optional
        Divide by this other column of the dataset instead.
    """
    if per_km2 and share_of is not None:
        raise ValueError('link_admin_attribute: per_km2 and share_of are exclusive.')
    spine = state.spine
    if admin_id_column not in spine.columns:
        raise ValueError(
            f'link_admin_attribute: spine has no {admin_id_column!r} column; '
            'run link_geographic_ids in the harmonize recipe first.'
        )
    level = _admin_level_from_column(admin_id_column, admin_level)

    table = get_dataset(recipe_id, state.admin_id)
    if attribute not in table.columns:
        raise ValueError(
            f'link_admin_attribute: {recipe_id} has no column {attribute!r}.'
        )
    values = pd.to_numeric(table[attribute], errors='coerce')
    if share_of is not None:
        if share_of not in table.columns:
            raise ValueError(
                f'link_admin_attribute: {recipe_id} has no column {share_of!r}.'
            )
        denominator = pd.to_numeric(table[share_of], errors='coerce')
        values = values / denominator.where(denominator > 0)
    elif per_km2:
        units = get_admin(state.admin_id, level, geom=True)
        area_km2 = pd.Series(
            get_areas(units, unit='km2'), index=units.index, dtype='float64'
        )
        area_km2 = area_km2.reindex(values.index)
        values = values / area_km2.where(area_km2 > 0)

    keys = spine[admin_id_column].map(lambda x: str(AdminId(x)) if pd.notna(x) else x)
    values.index = values.index.map(str)
    state.evidence[output] = keys.map(values).reindex(state.evidence.index)
    if state.verbose:
        n_null = int(state.evidence[output].isna().sum())
        print(
            f'  link_admin_attribute: {output} from {recipe_id}.{attribute}; '
            f'{n_null:,} entities without a value'
        )
    return state
