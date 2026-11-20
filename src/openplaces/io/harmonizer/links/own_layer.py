"""A lot keeps the values of the parcel layer that draws it.

Where two parcel layers cover a unit (a county's own layer under a
statewide one), `resolve_spine` takes each lot's outline from one of
them and stamps its label in `geometry_source`. The id join of
`link_by_id` then copies every layer's values onto every lot it
matches. Under the coverage rule of `_write_prioritized` alone, the
layer joined later replaced the earlier one's values wherever it
covered half the spine, so a lot drawn by the county could carry the
statewide layer's improvement and the county's total (measured
2026-11-18 on US-NC-CAR: 63,023 lots).

The rule here (maintainer, 2026-11-18): on a lot drawn by a parcel
layer, that layer's values stand and another parcel layer only fills
an empty cell, for every column the join writes. The two layers stay
ordered first non-null; neither is weighted and nothing is voted.

The rule reads the reference's entity type, never a place. It covers
a reference whose rows are parcels: a parcel recipe's own table. A
property layer bundled in a parcel recipe (MassGIS's assessor table,
the stacked-units layer) and a roll (a property recipe) describe
properties, whose values a county states for a lot that a coarser
layer may draw, and they keep the coverage rule (US-TX-HAR: the HCAD
roll over the TxGIO layer).

A county whose own layer is older than the statewide one declares it
in its own ingest recipe, ``values_outdated: true``. On the lots that
layer draws, another parcel layer's stated values then replace its
own, and its own join only fills.

The value columns go together (maintainer, 2026-11-19). Filled cell
by cell, one lot could carry the county's land and total beside the
statewide layer's improvement, three figures of two assessments that
do not add up (measured 2026-11-18 on US-NC-HYD: 2,051 such lots).
A step names the group (``value_group``: the value columns and their
tax year), and on a lot drawn by a parcel layer the group comes from
one layer: the layer that overwrites there takes the whole group when
it states any of its values, and a layer that only fills takes the
whole group only where no other source states any of them. Other
columns keep the cell-by-cell fill. The group is still ordered first
non-null between two layers joined by id, applied to the group as one
cell: nothing is weighted, voted or taken from another row.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

from openplaces.io.harmonizer import _PROVENANCE_SUFFIX
from openplaces.io.harmonizer.links.combine import RowRule
from openplaces.recipe import source_id_from_recipe_id

#: The ingest recipe key a parcel layer declares its values stale by.
VALUES_OUTDATED_KEY = 'values_outdated'

#: Keys a `value_group:` block may carry.
_VALUE_GROUP_KEYS = frozenset({'values', 'year'})


def parse_value_group(spec: dict | None) -> dict | None:
    """Validate a step's `value_group:` block.

    Parameters
    ----------
    spec : dict or None
        ``values`` (list of output column names, required) and
        ``year`` (the column holding their tax year, optional).

    Returns
    -------
    dict or None
        ``{'values': tuple, 'year': str or None}``, or None when *spec*
        is empty.

    Raises
    ------
    ValueError
        On an unknown key or an empty ``values`` list.
    """
    if not spec:
        return None
    unknown = set(spec) - _VALUE_GROUP_KEYS
    if unknown:
        raise ValueError(
            f'value_group: unknown key(s) {sorted(unknown)}; expected '
            f'{sorted(_VALUE_GROUP_KEYS)}.'
        )
    values = tuple(spec.get('values') or ())
    if not values:
        raise ValueError('value_group: `values` must name at least one column.')
    return {'values': values, 'year': spec.get('year')}


def values_outdated(recipe: dict) -> bool:
    """Whether a parcel ingest recipe declares its values outdated.

    Parameters
    ----------
    recipe : dict
        The loaded ingest recipe.

    Returns
    -------
    bool
        The recipe's ``values_outdated`` (False when absent).

    Raises
    ------
    ValueError
        When the key is set to anything but true or false, or set
        true on a recipe whose rows are not parcels.
    """
    flag = recipe.get(VALUES_OUTDATED_KEY, False)
    name = recipe.get('recipe_id', 'recipe')
    if not isinstance(flag, bool):
        raise ValueError(
            f'{name}: {VALUES_OUTDATED_KEY} must be true or false, got {flag!r}.'
        )
    entity = recipe.get('entity')
    entity_type = getattr(entity, 'entity_type', None)
    if flag and entity_type is not None and str(entity_type) != 'parcel':
        raise ValueError(
            f'{name}: {VALUES_OUTDATED_KEY} applies to a parcel layer; this '
            f'recipe describes {entity_type} rows.'
        )
    return flag


def own_layer_rule(
    spine: pd.DataFrame,
    match: dict,
    matches: list[dict],
    spine_source_ids: set[str],
    entity_type: str,
    value_group: dict | None = None,
    fill_only: bool = False,
) -> RowRule | None:
    """The rows where a parcel layer's join overwrites or only fills.

    Parameters
    ----------
    spine : DataFrame
        The parcel spine, carrying `geometry_source`.
    match : dict
        One discovered reference (`_discover_link_sources`).
    matches : list of dict
        Every reference discovered beside it, for the declarations.
    spine_source_ids : set of str
        The recipes `resolve_spine` drew outlines from.
    entity_type : str
        The entity type the discovery looked for.
    value_group : dict, optional
        A block from :func:`parse_value_group`: the columns a lot takes
        from one layer or not at all.
    fill_only : bool
        The pass only fills (a lossy key). Without *value_group* such a
        pass needs no rule; with one, every lot a parcel layer draws is
        a fill row, so the group still goes together.

    Returns
    -------
    RowRule or None
        None when the reference's rows are not parcels or the spine
        has no `geometry_source`, or for a *fill_only* pass without a
        group; the coverage rule then decides. Otherwise the reference
        overwrites on the lots it draws and on lots drawn by a layer
        declaring its values outdated, and only fills on lots drawn by
        any other parcel layer and on its own lots when it declares
        itself outdated.
    """
    if fill_only and not value_group:
        return None
    if (match['layer'] or entity_type) != 'parcel':
        return None
    if 'geometry_source' not in spine.columns:
        return None
    label = source_id_from_recipe_id(match['recipe_id'])
    layers = {source_id_from_recipe_id(r) for r in spine_source_ids}
    outdated = {
        source_id_from_recipe_id(m['recipe_id'])
        for m in matches
        if m['layer'] is None and m.get(VALUES_OUTDATED_KEY)
    }
    drawn_by = spine['geometry_source'].astype('string')
    own = drawn_by.eq(label).fillna(False).astype(bool)
    by_layer = drawn_by.isin(layers).fillna(False).astype(bool)
    stale = drawn_by.isin(outdated).fillna(False).astype(bool)
    if label in outdated:
        overwrite = stale & ~own
        fill = (by_layer & ~stale) | own
    else:
        overwrite = own | stale
        fill = by_layer & ~own & ~stale
    if fill_only:
        # A lossy key only fills, on every lot a parcel layer draws.
        fill = overwrite | fill
        overwrite = pd.Series(False, index=spine.index)
    group = value_group or {}
    return RowRule(
        overwrite=overwrite,
        fill=fill,
        values=tuple(group.get('values') or ()),
        year=group.get('year'),
    )


def _settled(
    spine: pd.DataFrame,
    name: str,
    label: str,
    own: pd.Series,
    held: pd.DataFrame | None,
) -> tuple[pd.Series, pd.Series]:
    """A column's values as the lot keeps them, and whose they are.

    A cell a property's copy holds is read as what it returns to once
    `withhold_property_values` takes the copy back (`note_write`). The
    `_source` names who wrote a cell; where it names no one, a value on
    a lot the reference draws counts as its own, elsewhere as
    another's.

    Returns
    -------
    values : pandas.Series
        The column, held cells read as what they return to.
    others : pandas.Series of bool
        Rows where a source other than *label* states the value.
    """
    values = spine[name]
    side = f'{name}{_PROVENANCE_SUFFIX}'
    labels = (
        spine[side].astype(object)
        if side in spine.columns
        else pd.Series(pd.NA, index=spine.index, dtype=object)
    )
    if held is not None:
        held = held.reindex(spine.index)
        is_held = held['held'].fillna(False).astype(bool)
        values = values.astype(object).where(~is_held, held['value'])
        labels = labels.where(~is_held, held['label'])
    labels = labels.astype('string')
    other = labels.ne(label).fillna(False).astype(bool) | (labels.isna() & ~own)
    return values, values.notna() & other


def value_group_writes(
    spine: pd.DataFrame,
    rule: RowRule,
    incoming: dict[str, pd.Series],
    label: str,
    *,
    suffix: str | None = None,
    protected: set[str] | frozenset = frozenset(),
    held: dict | None = None,
) -> dict[str, tuple[pd.Series, RowRule]]:
    """The writes of a value group, each with its per-column rule.

    On a row the reference overwrites, it takes the whole group when it
    states any of the group's values: a value it does not state leaves
    the cell empty rather than beside another layer's. On a row it only
    fills, it takes the whole group where the lot states none of the
    values, none of it where another source states one, and fills the
    gaps where the lot holds only its own values. The year column
    follows the values; it never decides by itself.

    Parameters
    ----------
    spine : DataFrame
        The spine before the reference writes.
    rule : RowRule
        The reference's rule, naming the group.
    incoming : dict of {str: pandas.Series}
        Per output column, the reference's values aligned to the spine.
    label : str
        The reference's source id, as `_source` and `geometry_source`
        name it.
    suffix : str, optional
        The suffix the written columns carry.
    protected : set of str
        Columns the reference does not write on its own lots, because
        `resolve_spine` already put its own value there; that value is
        its statement.
    held : dict, optional
        The property-write record (`PROPERTY_WRITES_KEY`), by column.

    Returns
    -------
    dict of {str: (pandas.Series, RowRule)}
        Per output column of the group present on the spine or in
        *incoming*: the values to write and the rule to write them
        with. Empty when the rule names no group.
    """
    if not rule.values:
        return {}

    def spine_name(column: str) -> str:
        return f'{column}{suffix}' if suffix else column

    drawn_by = spine['geometry_source'].astype('string')
    own = drawn_by.eq(label).fillna(False).astype(bool)
    group = [*rule.values, *([rule.year] if rule.year else [])]
    statements = {}
    for column in group:
        name = spine_name(column)
        if column in incoming:
            values = incoming[column]
        elif name in spine.columns:
            # The reference does not carry the column: it states none.
            values = pd.Series(pd.NA, index=spine.index, dtype=object)
        else:
            continue
        if column in protected and name in spine.columns:
            kept, others = _settled(spine, name, label, own, (held or {}).get(name))
            values = values.astype(object).where(~own, kept.where(~others))
        numeric = name in spine.columns and is_numeric_dtype(spine[name])
        if numeric or is_numeric_dtype(incoming.get(column, values)):
            # Kept numeric, so a write never turns a value column into
            # objects.
            values = pd.to_numeric(values, errors='coerce').astype('float64')
        statements[column] = values
    if not statements:
        return {}

    def any_of(frames) -> pd.Series:
        frames = list(frames)
        if not frames:
            return pd.Series(False, index=spine.index)
        return pd.concat(frames, axis=1).any(axis=1)

    states_any = any_of(statements[c].notna() for c in rule.values if c in statements)
    settled = [
        _settled(spine, spine_name(c), label, own, (held or {}).get(spine_name(c)))
        for c in rule.values
        if spine_name(c) in spine.columns
    ]
    others_any = any_of(others for _values, others in settled)
    spine_any = any_of(values.notna() for values, _others in settled)
    keep = rule.fill & others_any
    # A fill takes the group as it is only onto a lot that states none
    # of it; where the lot holds only the reference's own values (an
    # earlier, more precise pass), it fills their gaps cell by cell, so
    # a lossier key never empties what a precise one wrote.
    exact = (rule.overwrite | (rule.fill & ~spine_any)) & states_any
    per_value = rule._replace(keep=keep, exact=exact)
    # A year with no value of the reference's beside it does not replace
    # or join the year of values the lot already holds; on a lot with no
    # value at all it is written as any other column.
    silent = (rule.overwrite | rule.fill) & ~states_any & spine_any
    per_year = rule._replace(keep=keep | silent, exact=exact)
    return {
        column: (values, per_year if column == rule.year else per_value)
        for column, values in statements.items()
    }
