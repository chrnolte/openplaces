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
"""

import re

import pandas as pd

from openplaces.core.schema import ENTITY_TYPES, admin_scope_covers

SOURCE_SUFFIX = '_source'

# Written into a withheld cell's `{column}_source`, so an empty value
# says why it is empty rather than reading as missing upstream.
WITHHELD = 'withheld'

_TOKEN_SEPARATORS = re.compile(r'[+./]')

_ENTITY_NAMES = frozenset(str(entity_type) for entity_type in ENTITY_TYPES)

# Sidecar tokens that name a layer or a method, never a source: a cell
# carrying only these could have come from any source feeding the layer.
_UNSPECIFIC = _ENTITY_NAMES | {'reconciled', 'usaddress', 'imputed', 'assessor'}


def _tokens(value) -> frozenset:
    """Split a provenance token such as 'parcel.nconemap+imputed'."""
    if value is None or value is pd.NA:
        return frozenset()
    return frozenset(t for t in _TOKEN_SEPARATORS.split(str(value)) if t)


def _test_tokens(series: pd.Series, test) -> pd.Series:
    """Evaluate *test* on each row's tokens, once per distinct value."""
    values = series.astype('string')
    lookup = {value: bool(test(_tokens(value))) for value in values.dropna().unique()}
    return values.map(lookup).fillna(False).astype(bool)


def _names_attribute(column: str, attribute: str, entity_type: str | None) -> bool:
    """True when *column* carries *attribute* as the source's layer holds it.

    `land_value_parcel_whole` carries a parcel source's `land_value`;
    `address_street_dwelling_overture` names a different entity, so it
    carries another source's address, not this one's.
    """
    if column == attribute:
        return True
    if not column.startswith(f'{attribute}_'):
        return False
    rest = column[len(attribute) + 1 :].split('_')
    return not any(part in _ENTITY_NAMES and part != entity_type for part in rest)


def _source_cells(frame, source, sidecars, in_scope) -> tuple[pd.Series, dict]:
    """Rows and cells one restricted source supplied (rules 1 to 4)."""
    source_id = source['source_id']
    entity_type = source.get('entity_type')
    skip = {'geometry', *sidecars.values()}

    def names_source(tokens):
        return source_id in tokens

    def layer_only(tokens):
        return bool(tokens) and tokens <= _UNSPECIFIC and entity_type in tokens

    def other_source(tokens):
        return any(t not in _UNSPECIFIC and t != source_id for t in tokens)

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
                    column, attribute, entity_type
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
    return rows, cells


def find_restricted(frame, sources, admin_id_column) -> dict:
    """Locate every row and cell the restricted *sources* supplied.

    Parameters
    ----------
    frame : pandas.DataFrame
        Pooled delivery rows, with each row's process unit in
        *admin_id_column* and any `{column}_source` sidecars.
    sources : list of dict
        Entries from `bundle_terms.restricted_inputs`: `source_id`,
        `entity_type`, `admin_id` (the recipe's scope) and `attributes`
        (what its recipe maps).
    admin_id_column : str
        Column holding each row's admin unit, tested against a source's
        scope.

    Returns
    -------
    dict
        `rows` (bool Series, rows to leave out), `cells` (column to bool
        Series, values to empty) and `counts` (source id to a
        `{'rows': n, 'cells': n}` tally).
    """
    rows = pd.Series(False, index=frame.index)
    cells: dict[str, pd.Series] = {}
    counts: dict[str, dict] = {}
    if not sources or frame.empty:
        return {'rows': rows, 'cells': cells, 'counts': counts}

    sidecars = {
        column[: -len(SOURCE_SUFFIX)]: column
        for column in frame.columns
        if column.endswith(SOURCE_SUFFIX)
    }
    units = (
        frame[admin_id_column].astype('string')
        if admin_id_column in frame.columns
        else pd.Series(pd.NA, index=frame.index, dtype='string')
    )
    distinct = units.dropna().unique()

    for source in sources:
        scope = source.get('admin_id') or None
        lookup = {unit: admin_scope_covers(scope, unit) for unit in distinct}
        in_scope = units.map(lookup).fillna(False).astype(bool)
        source_rows, source_cells = _source_cells(frame, source, sidecars, in_scope)
        n_cells = sum(int((m & ~source_rows).sum()) for m in source_cells.values())
        if source_rows.any() or n_cells:
            tally = counts.setdefault(source['source_id'], {'rows': 0, 'cells': 0})
            tally['rows'] += int(source_rows.sum())
            tally['cells'] += n_cells
        rows |= source_rows
        for column, mask in source_cells.items():
            cells[column] = cells[column] | mask if column in cells else mask
    return {'rows': rows, 'cells': cells, 'counts': counts}


def withhold(frame, sources, admin_id_column):
    """Return *frame* without the values restricted *sources* supplied.

    Rows found by `find_restricted` are left out; cells are emptied and
    their `{column}_source`, where there is one, set to 'withheld'.

    Returns
    -------
    tuple of (DataFrame, dict)
        The redacted frame (same type as *frame*) and the per-source
        counts from `find_restricted`.
    """
    found = find_restricted(frame, sources, admin_id_column)
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
