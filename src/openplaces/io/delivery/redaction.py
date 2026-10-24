"""
Withhold a restricted source's data from a delivery, value by value.

A source whose terms restrict redistribution can still feed a curated
entity: Edgecombe County's parcels give its footprints their structure
values, their addresses and, for footprints drawn from parcels, their
outlines. Refusing the whole delivery for that let one county hold every
other county's data hostage. A delivery instead withholds exactly the
values such a source supplied and ships the rest.

Which values those are is read from the data where it records it, and
assumed conservatively where it does not:

1. A row whose `geometry_source` names the source is withheld whole: its
   outline is the source's, and a row cannot ship without one.
2. A cell whose `{column}_source` sidecar names the source is withheld,
   wherever the row is.
3. Inside the source's admin scope, a column named after an attribute
   the source's recipe maps (`address`, `address_number`, or with the
   source's own entity suffix, `land_value_parcel`) is withheld unless
   its sidecar names a different, specific source for that row
   (`dwelling_overture`, `parcel.nconemap`).
4. Inside the scope, a cell whose sidecar names only the source's entity
   layer (`parcel`, `parcel+usaddress`) is withheld too: that layer is the
   restricted source's there, and the token cannot say otherwise.

Rules 3 and 4 over-withhold where another source filled the same layer
(an occupancy class voted from statewide use codes on a county's parcel
layer). That costs values, never a leak.

They apply only to a source read as a layer (`layer_source`, set by
`bundle_terms.restricted_inputs`). A source reached only through steps
that name their columns (a permit table linked for one evidence column)
gets those columns withheld by name and nothing else: applying the
layer rules to a national permit source, which maps `year_built` and
`county_fips` like any roll, emptied both across every county of a
bundle on 2026-09-16.

Values openplaces derived from a source's values carry its restriction
too (maintainer's decision, 2026-10-23), within the admin unit:

5. In a unit where the source supplied values, a derived cell is
   withheld: one marked `imputed` (a rate or share learned from the
   unit's own recorded values), or one whose sidecar names no source,
   only the layer it was carried through (`parcel`: an improvement
   value apportioned to a footprint, a roll's year copied onto it).
   An estimate looked up from a place reference instead (a city from
   its ZIP code, `zipcodes+imputed`, a backend in
   `geo.address.POSTAL_CITY_BACKENDS`) is tied to a place, not to the
   unit's values, and stays (maintainer, 2026-10-24).
6. A `{column}_per_area` ratio goes wherever its `{column}` goes:
   multiplied back by the area the bundle ships, it is the value.
7. A column a recipe declares computed from others without a sidecar
   (`share: derived_from:`, the footprint's `occupancy_type_parcel`
   from the roll's use class) goes wherever one of its inputs goes,
   row by row.

Rule 5 is for restricted sources (decision 33, entries with
`reasons`); a share-alike source's reach is measured by rules 1 to 4,
6 and 7.

Which cells are derived is read from the sidecar, and which unit's
inputs they were derived from is read from the bundle itself, not
from the tokens' wording: an estimate's token names its peer group
(`parcel.city__is_residential+imputed`), never the values it was
learned from, and the dependency walk can say only that a source
*could* reach a column, which for a statewide deed registry reached
through the curated parcels is every value column of every town. So
rule 5 holds where both agree: the source is in scope, and its own
token is on cells of that unit (for a layer source, being the layer
in scope is enough). A unit whose values come from an open roll keeps
its estimates.

A token counts as naming a source only when a recipe declares that
source id (`known_sources`), alone or after an entity
(`dwelling_overture`). Tier and method names (`city__is_residential`,
`total_minus_improvement`, `rule`) used to pass for "another specific
source" and exonerated an estimate inside a restricted county.
"""

import re

import pandas as pd

from openplaces.core.attribute_registry import load_registry
from openplaces.core.schema import ENTITY_TYPES, admin_scope_covers
from openplaces.geo.address import POSTAL_CITY_BACKENDS

SOURCE_SUFFIX = '_source'
GEOMETRY_SOURCE = 'geometry_source'

# Written into a withheld cell's `{column}_source`, so an empty value
# says why it is empty rather than reading as missing upstream.
WITHHELD = 'withheld'

# Mirrors `io.curator.provenance`'s marker, which sits above this
# module in the layer hierarchy.
IMPUTED = 'imputed'

# Place references an estimate can be looked up from (rule 5's
# exemption): a city from its ZIP code depends on where a building
# is, not on any roll's values.
_PLACE_REFERENCES = frozenset(POSTAL_CITY_BACKENDS)

# The ratio derive_metrics writes beside a value column.
PER_AREA = '_per_area'

_TOKEN_SEPARATORS = re.compile(r'[+./]')

_ENTITY_NAMES = frozenset(str(entity_type) for entity_type in ENTITY_TYPES)

# Sidecar tokens that name a layer or a method, never a source: a cell
# carrying only these could have come from any source feeding the layer.
# `keywords` and `override` name the kind of table the curate step
# `translate_descriptions` read (io.curator.translation, which sits
# above this module and so cannot be imported here).
_UNSPECIFIC = _ENTITY_NAMES | {
    'reconciled',
    'usaddress',
    IMPUTED,
    'assessor',
    'keywords',
    'override',
}


def _tokens(value) -> frozenset:
    """Split a provenance token such as 'parcel.nconemap+imputed'."""
    if value is None or value is pd.NA:
        return frozenset()
    return frozenset(t for t in _TOKEN_SEPARATORS.split(str(value)) if t)


def _names_source(token: str, known: frozenset) -> bool:
    """True when one token names a source rather than a layer or method.

    A token names a source when it is one of *known* (the source ids
    recipes declare), or one after an entity name (`dwelling_overture`,
    `building_nsi`). Anything else names a layer, a method or a peer
    group (`city__is_residential`), which cannot say whose values a
    cell holds.
    """
    if token in _UNSPECIFIC:
        return False
    if token in known:
        return True
    entity, _, rest = token.partition('_')
    return entity in _ENTITY_NAMES and rest in known


def _known_sources(sources) -> frozenset:
    """The catalog's source ids, plus those of *sources* themselves."""
    from openplaces.io.delivery.terms import known_source_ids

    return known_source_ids() | {str(s['source_id']) for s in sources}


def _test_tokens(series: pd.Series, test) -> pd.Series:
    """Evaluate *test* on each row's tokens, once per distinct value."""
    values = series.astype('string')
    lookup = {value: bool(test(_tokens(value))) for value in values.dropna().unique()}
    return values.map(lookup).fillna(False).astype(bool)


def _names_attribute(
    column: str,
    attribute: str,
    entity_type: str | None,
    other_sources: frozenset = frozenset(),
) -> bool:
    """True when *column* carries *attribute* as the source's layer holds it.

    `land_value_parcel_whole` carries a parcel source's `land_value`;
    `address_street_dwelling_overture` names a different entity, and
    `n_dwellings_overture` another source (*other_sources*), so each
    carries another source's value, not this one's.
    """
    if column == attribute:
        return True
    if not column.startswith(f'{attribute}_'):
        return False
    rest = column[len(attribute) + 1 :].split('_')
    return not any(
        (part in _ENTITY_NAMES and part != entity_type) or part in other_sources
        for part in rest
    )


def _sidecars(frame) -> dict[str, str]:
    """`{column: column_source}` for each provenance sidecar in *frame*.

    A sidecar is an unregistered `{column}_source`, as the registry
    reads one (`core.attribute_registry.get_agg_func`); a registered
    name ending in `_source` is an attribute. `transaction_id_source`,
    a source's own record id, read as a sidecar was skipped by every
    rule, and a restricted source's record ids would have shipped.
    `geometry_source` is registered and stays a sidecar: rule 1 reads
    it. A sidecar may outlive its column (a pass that withheld the
    value), so the column's presence is not asked.
    """
    registered = load_registry().index
    return {
        column[: -len(SOURCE_SUFFIX)]: column
        for column in frame.columns
        if column.endswith(SOURCE_SUFFIX)
        and (column == GEOMETRY_SOURCE or column not in registered)
    }


def _reach_sidecars(source, sidecars) -> list[str]:
    """Sidecars of the columns a source reached by name can land in."""
    return [
        sidecars[column]
        for column in source.get('attributes') or ()
        if column in sidecars
    ]


def units_supplied(frame, sources, admin_id_column) -> dict[str, set]:
    """Units in which each source's own token is on a cell it can reach.

    The evidence rule 5 asks for before it treats a unit's derived
    values as a source's. Read on the cells a source reached by name
    can land in (`attributes`), so a statewide registry that reaches
    the value columns through the curated parcels, but whose own token
    is on none of them, supplies nothing there.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows with their unit in *admin_id_column* and their
        `{column}_source` sidecars. Only the sidecars of the columns in
        a source's `attributes` are read.
    sources : list of dict
        Entries from `bundle_terms.restricted_inputs`.
    admin_id_column : str
        Column holding each row's admin unit.

    Returns
    -------
    dict of str to set
        Source id to the units it supplied values in.
    """
    supplied: dict[str, set] = {}
    if frame.empty or admin_id_column not in frame.columns:
        return supplied
    sidecars = _sidecars(frame)
    units = frame[admin_id_column].astype('string')
    for source in sources:
        source_id = source['source_id']
        hit = pd.Series(False, index=frame.index)
        for sidecar in _reach_sidecars(source, sidecars):
            hit |= _test_tokens(frame[sidecar], lambda tokens: source_id in tokens)
        if hit.any():
            supplied.setdefault(source_id, set()).update(units[hit].dropna().unique())
    return supplied


def _derived_by(source, known):
    """Token test for cells derived from *source*'s values (rule 5).

    A layer source is the layer inside its scope, so rule 4 already
    takes the cells that name only its layer, and this adds the
    estimates. A source reached through another layer (a county roll
    inside the curated parcels a footprint reads) arrives under that
    layer's name, so a cell naming only that layer, or estimated
    through it, is derived from it. A source the recipe reads directly
    (`via_entities` empty) arrives under its own name and derives
    nothing that rule 2 does not already see.

    Only a restricted source (one with `reasons`) has derived cells
    withheld; and an estimate looked up from a place reference
    (`zipcodes+imputed`) is never one of them.
    """
    if not source.get('reasons'):
        return None

    def estimated(tokens):
        return IMPUTED in tokens and not tokens & _PLACE_REFERENCES

    if source.get('layer_source', True):
        return estimated
    layers = frozenset(str(entity) for entity in source.get('via_entities') or ())
    if not layers:
        return None

    def derived(tokens):
        if not tokens & layers:
            return False
        if IMPUTED in tokens:
            return estimated(tokens)
        return not any(_names_source(token, known) for token in tokens)

    return derived


def _source_cells(
    frame, source, sidecars, in_scope, known, supplied, units, derived_from=None
) -> tuple[pd.Series, dict]:
    """Rows and cells one restricted source supplied (rules 1 to 7)."""
    source_id = source['source_id']
    entity_type = source.get('entity_type')
    skip = {'geometry', *sidecars.values()}

    def names_source(tokens):
        return source_id in tokens

    def layer_only(tokens):
        return (
            bool(tokens)
            and entity_type in tokens
            and not any(_names_source(token, known) for token in tokens)
        )

    def other_source(tokens):
        # An estimate inside the source's scope was learned from the
        # unit's values, whatever else its token names.
        if IMPUTED in tokens:
            return False
        return any(
            _names_source(token, known) and token != source_id for token in tokens
        )

    rows = pd.Series(False, index=frame.index)
    if 'geometry_source' in frame.columns:
        rows = _test_tokens(frame['geometry_source'], names_source)

    cells: dict[str, pd.Series] = {}

    def mark(column, mask):
        mask = mask & frame[column].notna()
        if mask.any():
            cells[column] = cells[column] | mask if column in cells else mask

    # Rules 3 and 4 assume the source is its layer inside its scope. A
    # source reached only through steps naming their columns (a permit
    # table linked for one evidence column) is not: its `attributes`
    # already list the only columns it can land, and a layer-only token
    # there says nothing about it.
    as_layer = bool(source.get('layer_source', True))

    for column, sidecar in sidecars.items():
        if column in skip or column not in frame.columns:
            continue
        mark(column, _test_tokens(frame[sidecar], names_source))
        if as_layer and in_scope.any() and entity_type:
            mark(column, in_scope & _test_tokens(frame[sidecar], layer_only))

    if in_scope.any():
        for attribute in source.get('attributes') or ():
            for column in frame.columns:
                if column in skip or not _names_attribute(
                    column, attribute, entity_type, known - {source_id}
                ):
                    continue
                sidecar = sidecars.get(column) or sidecars.get(attribute)
                if not as_layer:
                    # Only the columns it was linked for, and only where
                    # nothing says another source filled them; without a
                    # sidecar the column is its own evidence column.
                    if column != attribute:
                        continue
                    if sidecar:
                        mark(
                            column,
                            in_scope & _test_tokens(frame[sidecar], names_source),
                        )
                    else:
                        mark(column, in_scope)
                    continue
                exonerated = (
                    _test_tokens(frame[sidecar], other_source)
                    if sidecar
                    else pd.Series(False, index=frame.index)
                )
                mark(column, in_scope & ~exonerated)

    # Rule 5: what openplaces derived from the source's values, in the
    # units where it supplied them. A layer source supplied its scope;
    # a source reached by name only where its own token says so.
    derived = _derived_by(source, known)
    if derived is not None and in_scope.any():
        if as_layer:
            supplying = in_scope
        else:
            supplied_units = supplied.get(source_id)
            supplying = (
                in_scope & units.isin(supplied_units).fillna(False).astype(bool)
                if supplied_units
                else None
            )
        if supplying is not None and supplying.any():
            for column, sidecar in sidecars.items():
                if column in skip or column not in frame.columns:
                    continue
                mark(column, supplying & _test_tokens(frame[sidecar], derived))
            if not as_layer:
                # A value its sidecar says nothing about, in a column the
                # source reaches: an apportioned share no step labeled.
                for column in source.get('attributes') or ():
                    sidecar = sidecars.get(column)
                    if sidecar and column in frame.columns:
                        mark(column, supplying & frame[sidecar].isna())

    def withheld_before(column):
        # Emptied by the bundle's other read pass, which leaves
        # 'withheld' in a sidecar this frame repeats.
        sidecar = sidecars.get(column)
        if not sidecar:
            return pd.Series(False, index=frame.index)
        return in_scope & frame[sidecar].astype('string').eq(WITHHELD).fillna(False)

    # Rule 7: a column the recipe declares computed from others, with
    # no sidecar of its own, goes on the rows where an input goes.
    for column, inputs in (derived_from or {}).items():
        if column in skip or column not in frame.columns:
            continue
        for name in inputs:
            if name in cells:
                mark(column, cells[name])
            mark(column, withheld_before(name))

    # Rule 6: a ratio goes with its value, emptied here or before.
    for column in list(frame.columns):
        if not column.endswith(PER_AREA) or column in skip:
            continue
        base = column[: -len(PER_AREA)]
        if base in cells:
            mark(column, cells[base])
        mark(column, withheld_before(base))
    return rows, cells


def find_restricted(
    frame,
    sources,
    admin_id_column,
    known_sources=None,
    supplied=None,
    derived_from=None,
) -> dict:
    """Locate every row and cell the restricted *sources* supplied.

    Parameters
    ----------
    frame : pandas.DataFrame
        Pooled delivery rows, with each row's process unit in
        *admin_id_column* and any `{column}_source` sidecars.
    sources : list of dict
        Entries from `bundle_terms.restricted_inputs`: `source_id`,
        `entity_type`, `admin_id` (the recipe's scope), `attributes`
        (what its recipe maps), `layer_source` and `via_entities`.
    admin_id_column : str
        Column holding each row's admin unit, tested against a source's
        scope.
    known_sources : set of str, optional
        Source ids a recipe declares, which tell a token naming a source
        from one naming a method. Defaults to the recipe catalog's, plus
        those of *sources*.
    supplied : dict of str to set, optional
        Units each source supplied values in (`units_supplied`), for
        rule 5. Defaults to what *frame* itself shows; a bundle read in
        two passes passes what both show, since one pass may hold the
        cells that name the source and the other the cells derived
        from them.
    derived_from : dict of str to list of str, optional
        Columns computed from others without a sidecar
        (`share: derived_from:`), for rule 7.

    Returns
    -------
    dict
        `rows` (bool Series, rows to leave out), `cells` (column to bool
        Series, values to empty) and `counts` (source id to a
        `{'rows': n, 'cells': n}` tally, each cell counted once per
        source however many of its recipes found it).
    """
    rows = pd.Series(False, index=frame.index)
    cells: dict[str, pd.Series] = {}
    counts: dict[str, dict] = {}
    if not sources or frame.empty:
        return {'rows': rows, 'cells': cells, 'counts': counts}

    if known_sources is None:
        known = _known_sources(sources)
    else:
        known = frozenset(known_sources) | {str(s['source_id']) for s in sources}
    if supplied is None:
        supplied = units_supplied(frame, sources, admin_id_column)
    sidecars = _sidecars(frame)
    units = (
        frame[admin_id_column].astype('string')
        if admin_id_column in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype='string')
    )
    distinct = units.dropna().unique()

    by_source: dict[str, tuple[pd.Series, dict]] = {}
    for source in sources:
        scope = source.get('admin_id') or None
        lookup = {unit: admin_scope_covers(scope, unit) for unit in distinct}
        in_scope = units.map(lookup).fillna(False).astype(bool)
        source_rows, source_cells = _source_cells(
            frame, source, sidecars, in_scope, known, supplied, units, derived_from
        )
        # One source id often has several recipes (a roll and its
        # values table), which find the same cells.
        prior_rows, prior_cells = by_source.get(
            source['source_id'], (pd.Series(False, index=frame.index), {})
        )
        for column, mask in source_cells.items():
            prior_cells[column] = (
                prior_cells[column] | mask if column in prior_cells else mask
            )
        by_source[source['source_id']] = (prior_rows | source_rows, prior_cells)
        rows |= source_rows
        for column, mask in source_cells.items():
            cells[column] = cells[column] | mask if column in cells else mask

    for source_id, (source_rows, source_cells) in by_source.items():
        n_cells = sum(int((m & ~source_rows).sum()) for m in source_cells.values())
        if source_rows.any() or n_cells:
            counts[source_id] = {'rows': int(source_rows.sum()), 'cells': n_cells}
    return {'rows': rows, 'cells': cells, 'counts': counts}


def withhold(
    frame,
    sources,
    admin_id_column,
    known_sources=None,
    supplied=None,
    derived_from=None,
):
    """Return *frame* without the values restricted *sources* supplied.

    Rows found by `find_restricted` are left out; cells are emptied and
    their `{column}_source`, where there is one, set to 'withheld'.
    *known_sources*, *supplied* and *derived_from* are passed to
    `find_restricted`.

    Returns
    -------
    tuple of (DataFrame, dict)
        The redacted frame (same type as *frame*) and the per-source
        counts from `find_restricted`.
    """
    found = find_restricted(
        frame, sources, admin_id_column, known_sources, supplied, derived_from
    )
    if not found['counts']:
        return frame, {}
    frame = frame[~found['rows']].copy()
    for column, mask in found['cells'].items():
        mask = mask.reindex(frame.index, fill_value=False)
        if not mask.any():
            continue
        frame[column] = frame[column].mask(mask)
        sidecar = f'{column}{SOURCE_SUFFIX}'
        if sidecar in frame.columns:
            tokens = frame[sidecar].astype('string')
            tokens[mask] = WITHHELD
            frame[sidecar] = tokens
    return frame, found['counts']


def merge_counts(*tallies) -> dict:
    """Add per-source `{'rows', 'cells'}` tallies from several passes."""
    merged: dict[str, dict] = {}
    for tally in tallies:
        for source_id, count in (tally or {}).items():
            into = merged.setdefault(source_id, {'rows': 0, 'cells': 0})
            into['rows'] = max(into['rows'], count['rows'])
            into['cells'] += count['cells']
    return merged
