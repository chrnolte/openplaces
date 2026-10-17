"""Rules for an assessment roll's values as `link_by_id` writes them.

A roll states a parcel's land, improvement and total value for one tax
year. Three rules keep those columns, their year and their provenance
consistent when the roll is joined onto a spine that already holds
values from a coarser source (a statewide parcel layer):

- the year column is written only where the roll wrote a value, so a
  value and its year always name the same record;
- where the roll states no current-year value for a parcel, its own
  prior-year columns, if the roll carries them, supply the values and
  the year before;
- where the roll covers the unit, a coarser source's values do not
  remain on a parcel whose key the roll does not carry.

The recipe declares them on the step (`roll_values:`), so the rules
are the national recipe's and a county roll only maps its columns.
"""

from __future__ import annotations

import pandas as pd

from openplaces.core.provenance import NOT_ON_ROLL, TOKEN_SEPARATOR
from openplaces.core.schema import AdminId
from openplaces.io.harmonizer import _PROVENANCE_SUFFIX, HarmonizeState, _record_source
from openplaces.recipe import source_id_from_recipe_id

# Share of spine rows a roll must key before a parcel it omits is read
# as one the district no longer values. Harris is the one county whose
# omitted parcels were traced account by account: each is a retired,
# split or deleted account. Elsewhere the omitted share is larger (the
# other coastal-Texas rolls on disk key 88.4% to 98.7% of their
# geospine, Fort Bend's 0.06%), and nobody has looked at why: a key
# conversion that fits most but not all accounts would look the same.
# So the rule acts only where a roll omits under 1%, which today is
# Harris alone; lowering the bar for another county needs its omitted
# parcels traced first.
# Measured 2026-10-17 on US-TX-HAR: HCAD keys 99.76% of the geospine.
NOT_ON_ROLL_MIN_COVERAGE = 0.99

#: Keys a `roll_values:` block may carry.
_ROLL_VALUES_KEYS = frozenset(
    {'values', 'year', 'prior_year', 'not_on_roll', 'min_coverage'}
)


def parse_roll_values(spec: dict | None) -> dict | None:
    """Validate a step's `roll_values:` block.

    Parameters
    ----------
    spec : dict or None
        The block as written in the recipe. Keys: ``values`` (list of
        output column names, required), ``year`` (output name of the
        year column, optional), ``prior_year`` (mapping from a value
        column, or the year column, to the roll's own column holding
        the same figure for the year before, optional), ``not_on_roll``
        (bool, default False) and ``min_coverage`` (float in (0, 1],
        default :data:`NOT_ON_ROLL_MIN_COVERAGE`).

    Returns
    -------
    dict or None
        The normalized block, or None when *spec* is empty.

    Raises
    ------
    ValueError
        On an unknown key, an empty ``values`` list, a ``prior_year``
        entry for a column that is neither a value nor the year, or a
        ``min_coverage`` outside (0, 1].
    """
    if not spec:
        return None
    unknown = set(spec) - _ROLL_VALUES_KEYS
    if unknown:
        raise ValueError(
            f'roll_values: unknown key(s) {sorted(unknown)}; expected '
            f'{sorted(_ROLL_VALUES_KEYS)}.'
        )
    values = list(spec.get('values') or [])
    if not values:
        raise ValueError('roll_values: `values` must name at least one column.')
    year = spec.get('year')
    prior = dict(spec.get('prior_year') or {})
    stray = set(prior) - set(values) - {year}
    if stray:
        raise ValueError(
            f'roll_values: prior_year names {sorted(stray)}, which are '
            'neither in `values` nor the `year` column.'
        )
    min_coverage = float(spec.get('min_coverage', NOT_ON_ROLL_MIN_COVERAGE))
    if not 0 < min_coverage <= 1:
        raise ValueError(
            f'roll_values: min_coverage must lie in (0, 1], got {min_coverage}.'
        )
    return {
        'values': values,
        'year': year,
        'prior_year': prior,
        'not_on_roll': bool(spec.get('not_on_roll', False)),
        'min_coverage': min_coverage,
    }


def apply_roll_values(
    spec: dict,
    current: dict[str, pd.Series],
    prior: dict[str, pd.Series],
) -> pd.Series | None:
    """Fill from the roll's prior year, and confine the year column.

    Parameters
    ----------
    spec : dict
        A block returned by :func:`parse_roll_values`.
    current : dict of {str: pandas.Series}
        Per spine row, the roll's aggregated value for each output
        column it joins (value columns and the year column among them),
        aligned to the spine. Changed in place.
    prior : dict of {str: pandas.Series}
        The same for the roll's prior-year columns, keyed by the output
        column each one stands in for.

    Returns
    -------
    pandas.Series or None
        Per spine row, whether the roll wrote a value there (from the
        current or the prior year), or None when the roll joins none of
        the value columns, so the rules do not apply to it (a detail
        table, or a roll without values).
    """
    values = [v for v in spec['values'] if v in current]
    if not values:
        return None
    index = current[values[0]].index
    stated = pd.concat([current[v].notna() for v in values], axis=1).any(axis=1)

    # A parcel whose current-year values are all empty, and whose roll
    # record carries the year before, takes that year's figures. Only
    # where *every* value is empty: mixing one year's land with the
    # next year's total would make a total that is not land plus
    # improvement of any year. The prior figures are the same record's
    # own columns, aggregated per key by the same rule as the current
    # ones; nothing is borrowed from another record.
    filled = pd.Series(False, index=index)
    priors = [v for v in values if v in prior]
    if priors:
        has_prior = pd.concat([prior[v].notna() for v in priors], axis=1).any(axis=1)
        filled = ~stated & has_prior
        if filled.any():
            for v in priors:
                current[v] = current[v].where(~filled, prior[v])

    written = stated | filled
    year = spec['year']
    if year and year in current:
        years = pd.to_numeric(current[year], errors='coerce')
        if filled.any():
            # A roll states the year it was certified for; its prior
            # columns are the year before by definition, unless the
            # roll names that year in a column of its own.
            before = (
                pd.to_numeric(prior[year], errors='coerce')
                if year in prior
                else years - 1
            )
            years = years.where(~filled, before)
        # The year describes the values: where the roll wrote none, the
        # year on the spine still belongs to the values already there.
        current[year] = years.where(written)
    return written


def _recipe_scope(recipe_id: str) -> int:
    """Number of components in a recipe id's admin scope (US-TX-HAR: 3).

    The same specificity auto-discovery orders sources by. A recipe id
    without an admin prefix scores 0.
    """
    head = str(recipe_id).split('_', 1)[0]
    try:
        AdminId(head)
    except ValueError:
        return 0
    return head.count('-') + 1


def _source_scopes(state: HarmonizeState) -> dict[str, int]:
    """Map each source token a spine value may carry to its scope.

    Built from the spine's geometry sources and every parcel or
    property ingest recipe covering the unit, which are the recipes
    whose tokens the geospine's value sidecars record. A token several
    recipes share takes the most specific scope, so a doubt resolves
    toward keeping a value.
    """
    from openplaces.io.harmonizer.links.discovery import (
        _find_admin_scoped_recipe_ids,
    )

    recipe_ids = set(state.metadata.get('spine_source_recipe_ids', set()))
    for entity_type in ('parcel', 'property'):
        recipe_ids.update(_find_admin_scoped_recipe_ids(state, entity_type))
    scopes: dict[str, int] = {}
    for recipe_id in recipe_ids:
        scope = _recipe_scope(recipe_id)
        for token in {source_id_from_recipe_id(recipe_id), recipe_id}:
            scopes[token] = max(scopes.get(token, 0), scope)
    return scopes


def _coarser_than(labels: pd.Series, scopes: dict[str, int], scope: int) -> pd.Series:
    """True where every source a label names is scoped coarser than *scope*.

    A label naming a source this unit does not know, or none at all,
    is not shown to be coarser, so its value is kept.
    """

    def coarser(label) -> bool:
        if label is None or pd.isna(label):
            return False
        parts = [p for p in str(label).split(TOKEN_SEPARATOR) if p]
        if not parts:
            return False
        return all(p in scopes and scopes[p] < scope for p in parts)

    text = labels.astype(object)
    return text.map(coarser).astype(bool)


def blank_values_not_on_roll(
    state: HarmonizeState,
    spine: pd.DataFrame,
    skey: pd.Series,
    rkey: pd.Series,
    *,
    recipe_id: str,
    names: list[str],
    min_coverage: float,
) -> int:
    """Empty a coarser source's values on parcels the roll does not carry.

    A county roll lists every account the appraisal district values.
    Where it covers the unit, a parcel whose key it does not carry is
    one the district no longer values under that account (retired,
    split or merged), and a statewide layer's figure for it is the
    previous year's value of a record that no longer exists. Keeping
    it would mix a superseded value into a roll-year dataset, so the
    cell is emptied and its `_source` says why (`not_on_roll`).

    Coverage decides whether the roll can speak for the unit: the share
    of spine rows whose key the roll carries must reach *min_coverage*.
    Below it, missing keys more likely mean a key conversion that does
    not fit, or a roll that is not the district's full list, and
    nothing is emptied. Only cells whose `_source` names sources all
    scoped coarser than the roll are emptied; a county parcel layer's
    value, the roll's own, or a value of unknown origin stays. A spine
    row without a key stays too: the roll cannot be said to omit it.

    Parameters
    ----------
    state : HarmonizeState
        Run state (admin unit and the spine's geometry sources).
    spine : pandas.DataFrame
        Spine, changed in place.
    skey, rkey : pandas.Series
        Join keys on the spine and on the roll, the latter after the
        roll's unit rows moved onto their stacked lots, so a lot whose
        units the roll carries counts as carried.
    recipe_id : str
        The roll.
    names : list of str
        Spine value columns the rule applies to.
    min_coverage : float
        Share of spine rows the roll must key.

    Returns
    -------
    int
        Number of spine rows with at least one cell emptied.
    """
    if not len(spine):
        return 0
    carried = skey.isin(set(rkey.dropna()))
    coverage = float(carried.mean())
    if coverage < min_coverage:
        if state.verbose:
            print(
                f'  roll_values: {recipe_id} keys {coverage:.1%} of the spine, '
                f'under {min_coverage:.0%}; no value emptied.'
            )
        return 0
    absent = skey.notna() & ~carried
    if not absent.any():
        return 0
    scopes = _source_scopes(state)
    scope = _recipe_scope(recipe_id)
    emptied = pd.Series(False, index=spine.index)
    for name in names:
        side = f'{name}{_PROVENANCE_SUFFIX}'
        if name not in spine.columns or side not in spine.columns:
            continue
        mask = absent & spine[name].notna() & _coarser_than(spine[side], scopes, scope)
        if not mask.any():
            continue
        spine[name] = spine[name].mask(mask)
        _record_source(spine, name, mask, NOT_ON_ROLL)
        emptied |= mask
    n_rows = int(emptied.sum())
    if state.verbose and n_rows:
        print(
            f'  roll_values: {n_rows:,d} spine rows not on {recipe_id} '
            f'(coverage {coverage:.1%}) lost a coarser source value.'
        )
    return n_rows
