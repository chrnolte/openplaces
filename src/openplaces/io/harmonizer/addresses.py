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

from openplaces.core.attribute_registry import (
    PROVENANCE_SOURCE_SUFFIX as _SOURCE_SUFFIX,
)
from openplaces.io.harmonizer import HarmonizeState, _register


def _ensure_object_column(curated: pd.DataFrame, column: str) -> None:
    """Ensure *column* exists on *curated* and can hold string values.

    Creates it as an all-missing object column if absent. An existing
    column can carry a non-object dtype (e.g. float64, when a prior join
    left it entirely missing on this admin unit -- pandas infers a numeric
    dtype for an all-NaN column with no explicit dtype of its own) which
    would otherwise raise a ``TypeError`` the first time a string is
    assigned into it.
    """
    if column not in curated.columns:
        curated[column] = pd.Series(pd.NA, index=curated.index, dtype=object)
    elif isinstance(
        curated[column].dtype, pd.CategoricalDtype
    ) or pd.api.types.is_numeric_dtype(curated[column].dtype):
        curated[column] = curated[column].astype(object)


def _record_source(curated: pd.DataFrame, column: str, mask, token: str) -> None:
    """Set the ``{column}_source`` sidecar to *token* for the *mask* rows.

    Inlined copy of :func:`openplaces.io.curator.provenance.record_source`'s
    (trivial) body: this module may not import from ``io.curator`` (higher
    layer than ``io.harmonizer``).
    """
    side = f'{column}{_SOURCE_SUFFIX}'
    _ensure_object_column(curated, side)
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
    number_output_col: str | None = None,
    unit_output_col: str | None = None,
    city_output_col: str | None = None,
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

    The *_output_col* columns below persist each component's case-formatted
    value (via openplaces.geo.address.format_address_components), matching
    *output_col*'s own formatting -- not the internal uppercase, USPS-
    abbreviated representation used for cross-source matching. The
    ``{output_col}_source`` sidecar's token gets a '+usaddress' suffix (e.g.
    'parcel' -> 'parcel+usaddress') on rows whose city could only be derived
    by parsing a source's address_full string (no source contributing that
    row's city declared an explicit ``city`` role) -- a meaningfully less
    trustworthy derivation than reading a structured field.

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
    number_output_col, unit_output_col, city_output_col : str, optional
        Also persist the resolved, normalized ``address_number``,
        ``unit_number``, and ``city`` components under these names (default
        ``None``, i.e. not persisted). Together with *street_output_col*,
        this lets a later step (e.g. ``impute_postal_city``) fill a still-
        missing component and cheaply re-render *output_col* from the saved
        components, without re-parsing the formatted string.

    Returns
    -------
    pandas.DataFrame
        *curated*, mutated in place and returned for convenience.
    """
    from openplaces.core.schema import AdminId
    from openplaces.geo.address import (
        ADDRESS_COMPONENTS,
        MATCH_COMPONENTS,
        format_address_components,
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
    # A source's city is usaddress-derived when it only supplies address_full
    # (parsed via parse_address) and declares no explicit city role -- an
    # explicit role always overrides the parsed value (see class docstring),
    # so a source with both never actually depends on the parse for city.
    city_via_address_full: dict[str, bool] = {}
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
        city_via_address_full[name] = (
            'address_full' in present and 'city' not in present
        )
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
    # Tracks whether the row's final `city` came from a source that had to
    # parse it out of address_full (see city_via_address_full above) rather
    # than an explicit city field, for the address_source annotation below.
    city_from_usaddress = pd.Series(False, index=curated.index)
    for name, frame in frames.items():
        take = base.eq(name)
        merged.loc[take] = frame.loc[take]
        if city_via_address_full.get(name):
            city_from_usaddress.loc[take & frame['city'].ne('')] = True

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
                if comp == 'city' and city_via_address_full.get(name):
                    city_from_usaddress.loc[fill_index] = True
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

    # Format over unique component tuples, then assign and record provenance.
    # format_address_components is called separately from harmonize_address_case
    # (which repeats the same normalization internally) rather than reusing its
    # private _assemble step directly -- this module doesn't reach into another
    # module's underscore-prefixed internals, same reasoning as _record_source
    # and _summarize_conflicts above. Both calls are over the same deduplicated
    # unique tuples, so the repeated work is cheap.
    has = base.ne('')
    subset = merged.loc[has]
    keys = list(zip(*(subset[comp] for comp in components)))
    unique_keys = set(keys)
    formats = {
        key: harmonize_address_case(
            **{comp: value or None for comp, value in zip(components, key)},
            admin1_id=admin1_id,
        )
        for key in unique_keys
    }
    formatted_components = {
        key: format_address_components(
            **{comp: value or None for comp, value in zip(components, key)},
            admin1_id=admin1_id,
        )
        for key in unique_keys
    }
    formatted = pd.Series([formats[k] for k in keys], index=subset.index)

    _ensure_object_column(curated, output_col)
    curated.loc[formatted.index, output_col] = formatted

    for component, out_col in (
        ('address_number', number_output_col),
        ('address_street', street_output_col),
        ('unit_number', unit_output_col),
        ('city', city_output_col),
    ):
        if not out_col:
            continue
        _ensure_object_column(curated, out_col)
        values = pd.Series(
            [formatted_components[k][component] for k in keys], index=subset.index
        )
        curated.loc[has, out_col] = values.mask(values.eq(''))

    # A row's token is annotated '+usaddress' when its final city could only
    # be derived by parsing address_full (see city_via_address_full above),
    # rather than read from an explicit city field on any contributing source.
    tokens = base.copy()
    tokens[corroborated] = 'reconciled'
    for token in sorted(set(tokens[has])):
        token_mask = has & tokens.eq(token)
        plain_mask = token_mask & ~city_from_usaddress
        usaddress_mask = token_mask & city_from_usaddress
        if plain_mask.any():
            _record_source(curated, output_col, plain_mask, token)
        if usaddress_mask.any():
            _record_source(curated, output_col, usaddress_mask, f'{token}+usaddress')

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


@_register('reconcile_postal_code')
def reconcile_postal_code(
    state: HarmonizeState,
    sources: list[str],
    output_column: str = 'postal_code',
    conflict_column: str = 'postal_code_conflict',
) -> HarmonizeState:
    """Coalesce ZIP-code evidence from multiple columns by priority.

    Some states have essentially no address-parsed ZIP coverage (e.g. no
    Overture address data at all), so a single-source column like
    ``postal_code_dwelling_overture`` alone leaves ``postal_zip5`` empty
    there even though a spatially-derived ``zcta5_id`` (see
    :func:`~openplaces.io.harmonizer.spine.link_geographic_ids`) could fill
    the gap. This coalesces any number of ZIP-like evidence columns, in
    priority order, into one column for :func:`impute_postal_city` to read.

    Extracts a 5-digit ZIP from each column in *sources* (via the same
    ``\\d{5}`` pattern :func:`impute_postal_city` uses) and writes the
    first (highest-priority) non-null value per row into *output_column*.
    Where two sources both have a value and they disagree, the
    higher-priority value still wins, but the disagreement is recorded in
    *conflict_column* (same shape as ``address_conflict``) and reported in
    aggregate to the cache (see
    :func:`openplaces.io.harmonizer.diagnostics.save_postal_code_conflicts`)
    -- always attempted, a no-op unless ``state.save_statistics`` is set.

    A no-op if none of *sources* is present on the spine.

    Parameters
    ----------
    sources : list of str
        ZIP-evidence columns, highest priority first (e.g.
        ``['postal_code_dwelling_overture', 'zcta5_id']``).
    output_column : str, optional
        Coalesced output column (default ``'postal_code'`` -- also
        :func:`impute_postal_city`'s own default ``column``).
    conflict_column : str, optional
        Output column for the grouped disagreement summary (default
        ``'postal_code_conflict'``).
    """
    if state.spine is None:
        return state
    spine = state.spine
    available = [c for c in sources if c in spine.columns]
    if not available:
        if state.verbose:
            print('  reconcile_postal_code: no source columns found.')
        return state

    zips = pd.DataFrame(
        {
            c: spine[c].astype('string').str.extract(r'(\d{5})', expand=False)
            for c in available
        },
        index=spine.index,
    )

    result = pd.Series(pd.NA, index=spine.index, dtype=object)
    winner = pd.Series(pd.NA, index=spine.index, dtype=object)
    for c in available:
        take = result.isna() & zips[c].notna()
        result.loc[take] = zips.loc[take, c]
        winner.loc[take] = c

    spine[output_column] = result
    spine[conflict_column] = _summarize_conflicts(
        [(c, zips[c]) for c in available], spine.index
    )
    for token in winner.dropna().unique():
        _record_source(spine, output_column, winner.eq(token), token)

    from openplaces.io.harmonizer.diagnostics import save_postal_code_conflicts

    save_postal_code_conflicts(state, zips, available)

    state.spine = spine
    if state.verbose:
        n = int(result.notna().sum())
        n_conflicts = int(spine[conflict_column].notna().sum())
        print(
            f'  reconcile_postal_code: {output_column} resolved {n:,} of '
            f'{len(spine):,} rows (conflicts={n_conflicts:,}).'
        )
    return state


@_register('impute_postal_city')
def impute_postal_city(
    state: HarmonizeState,
    column: str = 'postal_code',
    city_column: str = 'city',
    address_column: str = 'address',
) -> HarmonizeState:
    """Derive the USPS-preferred city for a ZIP code, and complete ``address``.

    A ZIP code has exactly one USPS-preferred city -- unlike most curate-stage
    work, there is no real dispute about how to derive it, so it belongs here
    rather than in curate. Writes
    ``postal_zip5``/``postal_city``/``postal_city_acceptable``/
    ``postal_city_unacceptable`` evidence columns from *column* via
    :func:`openplaces.geo.address.lookup_postal_city` (US-only; other
    countries resolve to missing). Where *city_column* is still missing, fills
    it from ``postal_city`` and re-renders *address_column* from the
    components :func:`reconcile_addresses_df` already persisted
    (``address_number``/``address_street``/``address_unit``/*city_column*)
    plus the run's admin-derived state code -- no re-parsing of the rendered
    string, and bounded to just the rows a city was actually filled for. A
    no-op if *column* is absent, or if *city_column*/*address_column* were
    never persisted by ``reconcile_addresses``.

    Parameters
    ----------
    column : str, optional
        ZIP-code evidence column to read (default ``'postal_code'``); point
        this at a recipe's raw, ungated ZIP evidence column (e.g.
        ``'postal_code_dwelling_overture'``) for the broadest coverage --
        deliberately not gated by address-source agreement the way
        ``reconcile_addresses``'s own ``complete_city_from_postal`` is.
    city_column, address_column : str, optional
        Canonical city / formatted-address columns to backfill (defaults
        ``'city'``, ``'address'``).
    """
    if state.spine is None or column not in state.spine.columns:
        return state

    from openplaces.core.schema import AdminId
    from openplaces.geo.address import harmonize_address_case, lookup_postal_city

    spine = state.spine
    admin_str = str(state.admin_id) if state.admin_id else ''
    admin_levels = AdminId(admin_str).levels if admin_str else ()
    admin1_id = admin_levels[0] if admin_levels else None

    zip5 = spine[column].astype('string').str.extract(r'(\d{5})', expand=False)
    spine['postal_zip5'] = zip5

    # ZIP lookups are cached over unique ZIPs only (typically far fewer than
    # rows), then broadcast back with .map -- same cost as the pre-existing
    # curate-stage version this replaces, just relocated.
    lookups = {z: lookup_postal_city(z, admin1_id) for z in zip5.dropna().unique()}
    spine['postal_city'] = zip5.map(
        lambda z: lookups[z].city if lookups.get(z) else pd.NA
    )
    spine['postal_city_acceptable'] = zip5.map(
        lambda z: '; '.join(lookups[z].acceptable_cities) if lookups.get(z) else pd.NA
    )
    spine['postal_city_unacceptable'] = zip5.map(
        lambda z: '; '.join(lookups[z].unacceptable_cities) if lookups.get(z) else pd.NA
    )
    resolved = spine['postal_city'].notna()
    if resolved.any():
        _record_source(spine, 'postal_city', resolved, 'zipcodes')

    if city_column in spine.columns:
        missing_city = spine[city_column].isna() | spine[city_column].eq('')
        fillable = missing_city & resolved
        if fillable.any():
            spine.loc[fillable, city_column] = spine.loc[fillable, 'postal_city']
            _record_source(spine, city_column, fillable, 'zipcodes')

            if address_column in spine.columns:
                # Re-render only the rows just backfilled (a strict subset of
                # `resolved`), caching over unique component tuples -- the
                # same micro-optimization reconcile_addresses_df's own
                # full-dataset formatting pass above already relies on.
                empty = pd.Series('', index=spine.index, dtype=object)
                number = spine.get('address_number', empty).fillna('')
                street = spine.get('address_street', empty).fillna('')
                unit = spine.get('address_unit', empty).fillna('')
                admin_state = admin_levels[1] if len(admin_levels) > 1 else ''

                idx = spine.index[fillable]
                keys = list(
                    zip(
                        number.loc[idx],
                        street.loc[idx],
                        unit.loc[idx],
                        spine.loc[idx, city_column].fillna(''),
                        zip5.loc[idx].fillna(''),
                    )
                )
                renders = {
                    key: harmonize_address_case(
                        address_street=key[1],
                        address_number=key[0] or None,
                        unit_number=key[2] or None,
                        city=key[3] or None,
                        state=admin_state or None,
                        postal_code=key[4] or None,
                        admin1_id=admin1_id,
                    )
                    for key in set(keys)
                }
                spine.loc[idx, address_column] = [renders[k] for k in keys]

    state.spine = spine
    if state.verbose:
        n = int(spine['postal_city'].notna().sum())
        print(f'  impute_postal_city: resolved {n:,} of {len(spine):,} rows.')
    return state


@_register('derive_address_id_local')
def derive_address_id_local(
    state: HarmonizeState,
    street_column: str = 'address_street',
    number_column: str = 'address_number',
    city_column: str = 'city',
    admin4_column: str = 'admin4_id',
    output_column: str = 'address_id_local',
    output_column_city: str = 'address_id_local_city',
) -> HarmonizeState:
    """Derive a town-scoped address matching key for id-free entity linking.

    Companion to ``parcel_id_local``, for entities that have no shared parcel
    id to join on (e.g. linking transactions to parcels where the transaction
    source carries no parcel identifier). Building this from
    *street_column*/*number_column*/*admin4_column* rather than a raw address
    string keeps the key deterministic across sources that were reconciled
    through :func:`reconcile_addresses_df` on both sides -- the same
    normalized components produce the same key regardless of formatting
    differences in the original source data.

    *admin4_column* (the containing town), not *city_column*, is what scopes
    the key: a town can contain several villages that each use their own name
    as the address's ``city`` (e.g. Newton, MA contains Newton Centre,
    Waban, ...), so two genuinely different streets of the same name in two
    different villages of one town are not distinguishable by city text alone
    -- but they are already disjoint by *admin4_column*, which is a coded id,
    not free text. ``output_column`` therefore omits city entirely.
    ``output_column_city`` additionally includes *city_column* and is only
    populated where *city_column* is non-empty, for use as a more specific,
    first-choice join key when both sides of a link happen to carry a city
    (see ``link_by_id``'s column-priority gap-fill, which makes running the
    city-inclusive join before the plain one a safe two-pass fallback).

    A no-op when *street_column* or *number_column* is absent from the spine.
    """
    if state.spine is None:
        return state
    spine = state.spine
    if street_column not in spine.columns or number_column not in spine.columns:
        return state

    from openplaces.core.schema import AdminId
    from openplaces.geo.address import canonicalize_for_match

    admin_str = str(state.admin_id) if state.admin_id else ''
    admin_levels = AdminId(admin_str).levels if admin_str else ()
    admin1_id = admin_levels[0] if admin_levels else None

    street = spine[street_column].astype('string').fillna('')
    canon_map = {
        s: canonicalize_for_match(s, admin1_id) if s else '' for s in street.unique()
    }
    canon_street = street.map(canon_map)
    number = spine[number_column].astype('string').str.strip().str.upper().fillna('')
    admin4 = (
        spine[admin4_column].astype('string').fillna('')
        if admin4_column in spine.columns
        else pd.Series('', index=spine.index)
    )

    has_base = canon_street.ne('') & number.ne('')
    base_key = admin4 + '|' + canon_street + '|' + number
    spine[output_column] = base_key.where(has_base, pd.NA)

    if city_column in spine.columns:
        city = spine[city_column].astype('string').str.strip().str.upper().fillna('')
        has_city = has_base & city.ne('')
        spine[output_column_city] = (base_key + '|' + city).where(has_city, pd.NA)
    else:
        spine[output_column_city] = pd.Series(pd.NA, index=spine.index, dtype='string')

    state.spine = spine
    if state.verbose:
        n = int(spine[output_column].notna().sum())
        n_city = int(spine[output_column_city].notna().sum())
        print(
            f'  derive_address_id_local: {n:,} of {len(spine):,} rows keyed '
            f'({n_city:,} with city).'
        )
    return state
