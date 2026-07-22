"""Shared address reconciliation, reused by both pipeline stages.

Single implementation of "pick a canonical street address from any number of
evidence sources" (parsing, component normalization, fuzzy cross-source
agreement, admin-completion, canonical-case formatting), so the harmonize- and
curate-stage steps can never drift apart -- the same pattern already used for
value apportionment (see :mod:`openplaces.io.harmonizer.apportion`).

Running this at harmonize time (rather than only at curate time, as before)
means a genuinely parsed, canonical street name is available before any
curate-stage work that needs to group entities by street (e.g. a land-value
imputer comparing nearby parcels) -- previously that work had to derive its
own cheap, error-prone street key by hand.
"""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register

_SOURCE_SUFFIX = '_source'


def _record_source(curated: pd.DataFrame, column: str, mask, token: str) -> None:
    """Set the ``{column}_source`` sidecar to *token* for the *mask* rows.

    Inlined copy of :func:`openplaces.io.curator.provenance.record_source`'s
    (trivial) body: this module may not import from ``io.curator`` (higher
    layer than ``io.harmonizer`` -- see ``CLAUDE.md``'s layer hierarchy).
    """
    side = f'{column}{_SOURCE_SUFFIX}'
    if side not in curated.columns:
        curated[side] = pd.Series(pd.NA, index=curated.index, dtype=object)
    elif isinstance(curated[side].dtype, pd.CategoricalDtype):
        curated[side] = curated[side].astype(object)
    curated.loc[mask, side] = token


def _summarize_conflicts(
    present: list[tuple[str, pd.Series]],
    index: pd.Index,
) -> pd.Series:
    """Summarize disagreeing evidence values per row as a compact string.

    Deliberate duplicate of ``openplaces.io.curator.reconcilers``'s private
    helper of the same name (also used by ``resolve_occupancy``/
    ``reconcile_land_use`` there, so it can't just move) -- same reasoning as
    :func:`_record_source` above: no cross-layer import available.
    """
    conflict = pd.Series(pd.NA, index=index, dtype=object)
    if len(present) < 2:
        return conflict

    differ = pd.Series(False, index=index)
    for (_, class_a), (_, class_b) in combinations(present, 2):
        both = class_a.notna() & class_b.notna()
        differ = differ | (both & class_a.ne(class_b))
    if not differ.any():
        return conflict

    labels = [label for label, _ in present]
    stacked = pd.concat(
        {label: values.astype(object) for label, values in present}, axis=1
    )

    def _row_summary(row) -> str:
        groups: dict[str, list[str]] = {}
        for label in labels:
            value = row[label]
            if pd.notna(value):
                groups.setdefault(str(value), []).append(label)
        return ' | '.join(f'{"/".join(who)}: {value}' for value, who in groups.items())

    conflict.loc[differ] = stacked.loc[differ].apply(_row_summary, axis=1)
    return conflict


def reconcile_addresses_df(
    curated: pd.DataFrame,
    admin_id,
    verbose: bool,
    sources: dict[str, dict[str, str]],
    output_col: str = 'address',
    similarity_threshold: float = 80,
    conflict_column: str = 'address_conflict',
    complete_from_admin: dict[str, int] | None = None,
    complete_city_from_postal: bool = False,
    street_output_col: str | None = 'address_street',
) -> pd.DataFrame:
    """Reconcile street addresses from any number of source inputs.

    State-agnostic core shared by both the harmonize-stage and curate-stage
    ``reconcile_addresses`` steps (see their thin wrappers in
    :mod:`openplaces.io.harmonizer.addresses` and
    :mod:`openplaces.io.curator.reconcilers`).

    Each key of *sources* is a provenance token recorded in the output's
    source sidecar; its value maps component roles to evidence columns.
    Roles: address_full (a one-line address string, parsed via
    openplaces.geo.address.parse_address), address_number, address_street,
    unit_number, city, state, postal_code. A source may mix address_full with
    explicit component roles: address_full seeds every component via
    parse_address, and any explicitly-declared role overrides just that
    component (e.g. a source can parse a full string for street/number and
    still declare its own, separately-sourced ``city``).

    Declaration order is priority. The base address comes from the
    highest-priority source with a usable address on each row (non-empty
    street; sources that declare address_number must also have the number).
    A lower-priority source agrees with the base when house numbers match
    and streets match per openplaces.geo.address.match_streets -- notation
    equivalences from address_equivalences.csv are applied, then rapidfuzz
    similarity against *similarity_threshold* (0-100); agreeing sources fill
    the base's missing components (every component outside MATCH_COMPONENTS)
    and mark the row 'reconciled'. Disagreeing sources are excluded from
    selection but summarized in *conflict_column* (null when the sources
    agree or only one is present; see :func:`_summarize_conflicts`). Missing
    columns are skipped, like in reconcile_values.

    Parameters
    ----------
    curated : pandas.DataFrame
        Entity table to reconcile against (``state.curated`` or
        ``state.spine``, depending on caller).
    admin_id : AdminId or None
        Current admin unit (for ``complete_from_admin`` and locale-aware
        parsing/formatting).
    verbose : bool
        Print a per-run summary.
    sources : dict of {token: {role: column}}
        Ordered mapping of provenance token to role-column spec.

        Example::

            sources:
              parcel:
                address_full: address_parcel
              dwelling_overture:
                address_street: address_street_dwelling_overture
                address_number: address_number_dwelling_overture
    output_col : str
        Canonical output column (default 'address').
    similarity_threshold : float
        Minimum street similarity (0-100) for two sources to agree.
    conflict_column : str
        Output column for the grouped disagreement summary (default
        'address_conflict').
    complete_from_admin : dict of {component: admin_level}, optional
        Fill components that no source provided from the run's admin id,
        e.g. ``{state: 2}`` completes a missing state with the admin unit's
        level-2 code (validated against ISO 3166-2 for the unit's country).
        Only rows that already carry another non-street component are
        completed, so street-only addresses stay untouched.
    complete_city_from_postal : bool, optional
        Fill a still-missing city from the row's resolved postal_code via
        openplaces.geo.address.lookup_postal_city (USPS-preferred city
        name; US only, degrades to no-op elsewhere). Applied after
        complete_from_admin, to rows that have a postal_code but no city
        from any source. Default False.
    street_output_col : str, optional
        Also persist the resolved, normalized (not yet case-formatted)
        street component under this name (default ``'address_street'``) --
        a clean grouping key for consumers that want "same street" without
        parsing *output_col*'s formatted string themselves. Set to ``None``
        to skip.

    Returns
    -------
    pandas.DataFrame
        *curated*, mutated in place and returned for convenience.
    """
    from openplaces.core.schema import AdminId
    from openplaces.geo.address import (
        ADDRESS_COMPONENTS,
        MATCH_COMPONENTS,
        get_admin2_codes,
        harmonize_address_case,
        lookup_postal_city,
        match_streets,
        normalize_address_components,
        parse_address,
    )

    components = list(ADDRESS_COMPONENTS)
    fillable = [c for c in components if c not in MATCH_COMPONENTS]
    admin_str = str(admin_id) if admin_id else ''
    admin_levels = AdminId(admin_str).levels if admin_str else ()
    admin1_id = admin_levels[0] if admin_levels else None

    def normalize_component(value: str, component: str) -> str:
        kwargs = {'address_street': '', 'admin1_id': admin1_id}
        kwargs[component] = value
        return normalize_address_components(**kwargs)[component]

    # Build one normalized component frame per source (over unique values)
    frames: dict[str, pd.DataFrame] = {}
    needs_number: dict[str, bool] = {}
    for name, spec in sources.items():
        unknown = set(spec) - {
            'address_full',
            'address_number',
            'address_street',
            'unit_number',
            'city',
            'state',
            'postal_code',
        }
        if unknown:
            raise ValueError(
                f'reconcile_addresses: unknown role(s) {sorted(unknown)} for '
                f'source {name!r}.'
            )
        present = {role: col for role, col in spec.items() if col in curated.columns}
        if not present:
            continue
        frame = pd.DataFrame('', index=curated.index, columns=components)
        full_col = present.get('address_full')
        if full_col is not None:
            keys = curated[full_col].map(
                lambda v: str(v).strip() if pd.notna(v) else ''
            )
            empty = dict.fromkeys(components, '')
            lookup = {'': empty}
            for value in keys.unique():
                if not value:
                    continue
                parsed = parse_address(value, admin1_id=admin1_id).components
                lookup[value] = normalize_address_components(
                    address_street=parsed['address_street'] or '',
                    address_number=parsed['address_number'],
                    unit_number=parsed['unit_number'],
                    postal_code=parsed['postal_code'],
                    city=parsed['city'],
                    state=parsed['state'],
                    admin1_id=admin1_id,
                )
            for comp in components:
                frame[comp] = keys.map(lambda k: lookup[k][comp])
        for role, col in present.items():
            if role == 'address_full':
                continue
            comp = role
            values = curated[col].map(lambda v: str(v) if pd.notna(v) else '')
            normed = {v: normalize_component(v, comp) for v in values.unique()}
            frame[comp] = values.map(normed)
        frames[name] = frame
        needs_number[name] = 'address_number' in present

    if not frames:
        if verbose:
            print('  reconcile_addresses: no address source columns found.')
        return curated

    # Base = highest-priority source with a usable address on each row
    usable: dict[str, pd.Series] = {}
    for name, frame in frames.items():
        mask = frame['address_street'].str.len() > 0
        if needs_number[name]:
            mask &= frame['address_number'].str.len() > 0
        usable[name] = mask

    base = pd.Series('', index=curated.index, dtype=object)
    for name in frames:
        take = base.eq('') & usable[name]
        base.loc[take] = name

    merged = pd.DataFrame('', index=curated.index, columns=components)
    for name, frame in frames.items():
        take = base.eq(name)
        merged.loc[take] = frame.loc[take]

    # Agreeing lower-priority sources fill the base's missing components;
    # sources that fail the agreement check are tracked for conflict flagging
    corroborated = pd.Series(False, index=curated.index)
    disagreeing = pd.Series(False, index=curated.index)
    for name, frame in frames.items():
        others = usable[name] & base.ne('') & base.ne(name)
        candidate = others & (
            merged['address_number'].ne('')
            & frame['address_number'].ne('')
            & merged['address_number'].eq(frame['address_number'])
        )
        agree = pd.Series(False, index=curated.index)
        if candidate.any():
            pairs = pd.DataFrame(
                {
                    'a': merged.loc[candidate, 'address_street'],
                    'b': frame.loc[candidate, 'address_street'],
                }
            )
            uniq = pairs.drop_duplicates()
            matched = {
                (a, b): match_streets(a, b, similarity_threshold, admin1_id)
                for a, b in zip(uniq['a'], uniq['b'])
            }
            hits = pd.Series(
                [matched[(a, b)] for a, b in zip(pairs['a'], pairs['b'])],
                index=pairs.index,
            )
            agree_index = hits.index[hits]
            agree.loc[agree_index] = True
            corroborated |= agree
            for comp in fillable:
                fill = merged.loc[agree_index, comp].eq('') & frame.loc[
                    agree_index, comp
                ].ne('')
                fill_index = fill.index[fill]
                merged.loc[fill_index, comp] = frame.loc[fill_index, comp]
        disagreeing |= others & ~agree

    # Recipe-configured completion from the run's admin unit (e.g. state: 2
    # fills a missing state with the level-2 code, since most spines carry no
    # state evidence). Level-2 codes are validated against ISO 3166-2 for the
    # unit's country; only rows that already carry another non-street
    # component are completed, so street-only addresses stay untouched.
    for comp, level in (complete_from_admin or {}).items():
        level = int(level)
        if comp not in fillable or len(admin_levels) < level:
            continue
        code = admin_levels[level - 1]
        if level == 2 and admin1_id and code not in get_admin2_codes(admin1_id):
            continue
        others = [c for c in fillable if c != comp]
        fill = merged[comp].eq('') & merged[others].ne('').any(axis=1)
        merged.loc[fill, comp] = code

    # USPS-preferred city name for a still-missing city, from the row's own
    # resolved postal_code (e.g. Overture address points often carry a ZIP
    # but no city in this region). lookup_postal_city is US-only and cached.
    if complete_city_from_postal:
        missing_city = merged['city'].eq('') & merged['postal_code'].ne('')
        if missing_city.any():
            zip5 = merged.loc[missing_city, 'postal_code'].str.extract(
                r'(\d{5})', expand=False
            )
            lookups = {
                z: lookup_postal_city(z, admin1_id) for z in zip5.dropna().unique()
            }
            city_fill = zip5.map(lambda z: lookups[z].city if lookups.get(z) else None)
            merged.loc[missing_city, 'city'] = city_fill.fillna('').to_numpy()

    # Summarize the disagreeing evidence per row (exact-equal values are
    # never summarized, and rows the fuzzy check accepted are masked out)
    compare = [
        (
            name,
            frame[MATCH_COMPONENTS[0]]
            .str.cat(frame[list(MATCH_COMPONENTS[1:])], sep=' ')
            .str.strip()
            .where(usable[name]),
        )
        for name, frame in frames.items()
    ]
    conflict = _summarize_conflicts(compare, curated.index).where(disagreeing)
    curated[conflict_column] = conflict

    # Format over unique component tuples, then assign and record provenance
    has = base.ne('')
    subset = merged.loc[has]
    keys = list(zip(*(subset[comp] for comp in components)))
    formats = {
        key: harmonize_address_case(
            **{comp: value or None for comp, value in zip(components, key)},
            admin1_id=admin1_id,
        )
        for key in set(keys)
    }
    formatted = pd.Series([formats[k] for k in keys], index=subset.index)

    if output_col not in curated.columns:
        curated[output_col] = pd.Series(pd.NA, index=curated.index, dtype=object)
    curated.loc[formatted.index, output_col] = formatted

    if street_output_col:
        if street_output_col not in curated.columns:
            curated[street_output_col] = pd.Series(
                pd.NA, index=curated.index, dtype=object
            )
        curated.loc[has, street_output_col] = merged.loc[has, 'address_street']

    tokens = base.copy()
    tokens[corroborated] = 'reconciled'
    for token in sorted(set(tokens[has])):
        _record_source(curated, output_col, has & tokens.eq(token), token)

    if verbose:
        counts = tokens[has].value_counts()
        summary = ', '.join(f'{k}={v:,d}' for k, v in counts.items()) or 'none'
        n_conflicts = int(conflict.notna().sum())
        print(
            f'  reconcile_addresses: {output_col} populated -> {summary}, '
            f'conflicts={n_conflicts:,d}'
        )
        if n_conflicts:
            print('    sample conflicts:')
            for sample in conflict.dropna().head(5):
                print(f'      {sample}')
            print(
                '    tip: pairs that denote the same address in different '
                'notations\n    (e.g. HIGHWAY ~ HWY) can be added to '
                'src/openplaces/geo/address_equivalences.csv (kind=match).'
            )

    return curated


@_register('reconcile_addresses')
def reconcile_addresses(state: HarmonizeState, **kwargs) -> HarmonizeState:
    """Harmonize-stage wrapper: reconcile addresses on ``state.spine``.

    See :func:`reconcile_addresses_df` for parameters and behavior. A no-op
    when ``state.spine`` is ``None`` (nothing built yet for this entity).
    """
    if state.spine is None:
        return state
    state.spine = reconcile_addresses_df(
        state.spine, state.admin_id, state.verbose, **kwargs
    )
    return state
