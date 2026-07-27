"""Address parsing, normalization, and harmonization utilities.

Parsing of one-line US address strings is delegated to pluggable backends
via `parse_address`: the CRF-based usaddress library (default for US
addresses, with a regex fallback for strings it cannot tag unambiguously)
or a pass-through 'none' backend that preserves the raw string. Heavyweight
backends (libpostal, deepparse) are documented placeholders only, so the
base installation stays lightweight and cross-platform.

Component keys follow the conventions of dwelling-overture-2025 and the
attribute registry: address_number, address_street, unit_number, city,
state, postal_code. Normalization follows USPS Publication 28
abbreviations; display formatting produces title-cased street lines with
uppercase state codes (e.g. '1200 Seagrass Ln, Coastal City, NC 28500').

All abbreviation and equivalence tables live in address_equivalences.csv
next to this module (case-insensitive, country-scoped via the admin1_id
column; kind=match rows apply only when comparing streets, so notation
variants like 'HIGHWAY 94' and 'HWY 94' match without changing display
output). Per-country display syntax (segment order, postal-code rewrites)
is declared in address_formats.csv — render-only, never used for parsing.
"""

from __future__ import annotations

import csv
import importlib.metadata
import re
import unicodedata
import warnings
from dataclasses import dataclass, field
from functools import cache, lru_cache
from pathlib import Path

import usaddress
from rapidfuzz import fuzz


def clean_text(text: str) -> str:
    """Normalize casing, unicode accents, punctuation, and whitespace."""
    if not isinstance(text, str):
        return ''

    # Unicode normalization (NFKD decomposes characters, e.g. é -> e + accent),
    # then strip the non-spacing (accent) marks
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))

    text = text.upper().strip()

    # Punctuation removal (keep alphanumeric, space, and # for units), then
    # collapse repeated whitespace
    text = re.sub(r'[^\w\s#]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


_EQUIVALENCES_PATH = Path(__file__).parent / 'address_equivalences.csv'
_FORMATS_PATH = Path(__file__).parent / 'address_formats.csv'

# Effective country when a caller provides none: the single place the
# US-first default lives (lift into cfg if deployments ever need to vary it)
DEFAULT_ADMIN1_ID = 'US'


@cache
def _equivalence_tables(admin1_id: str) -> dict[str, dict[str, str]]:
    """Load one country's phrase equivalence tables, one dict per kind.

    Rows with an empty admin1_id column belong to `DEFAULT_ADMIN1_ID`. There
    is no cross-country fallback: USPS abbreviations must not leak into
    other countries' matching. Every phrase and replacement passes through
    `clean_text`, so CSV entries are case- and accent-insensitive.
    """
    tables: dict[str, dict[str, str]] = {}
    with open(_EQUIVALENCES_PATH, encoding='utf8', newline='') as handle:
        for row in csv.DictReader(handle):
            scope = (row.get('admin1_id') or '').strip() or DEFAULT_ADMIN1_ID
            if scope != admin1_id:
                continue
            phrase = clean_text(row['phrase'])
            replacement = clean_text(row['replacement'])
            tables.setdefault(row['kind'].strip(), {})[phrase] = replacement
    return tables


def _tables(kind: str, admin1_id: str | None) -> dict[str, str]:
    """One kind's equivalence table for a country (default country if None)."""
    return _equivalence_tables(admin1_id or DEFAULT_ADMIN1_ID).get(kind, {})


# Default-country USPS Publication 28 tables, bound at import for direct
# consumers; country-aware code paths use _tables(kind, admin1_id) instead
STREET_SUFFIXES = _tables('street_suffix', None)
DIRECTIONALS = _tables('directional', None)
SECONDARY_UNITS = _tables('unit', None)


@cache
def _match_maps(
    admin1_id: str,
) -> tuple[tuple[tuple[str, str], ...], dict[str, str]]:
    """Country's match-time maps: (multi-word phrases longest-first, token map).

    kind=match rows apply only at comparison time (see canonicalize_for_match):
    multi-word phrases are substituted first, and an empty replacement drops
    the token entirely (noise like literal 'NULL').
    """
    tables = _equivalence_tables(admin1_id)
    match = tables.get('match', {})
    phrases = tuple(
        sorted(
            ((phrase, repl) for phrase, repl in match.items() if ' ' in phrase),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )
    token_map = {
        **tables.get('street_suffix', {}),
        **tables.get('directional', {}),
        **{p: r for p, r in match.items() if ' ' not in p},
    }
    return phrases, token_map


@cache
def get_address_format(admin1_id: str | None = None) -> dict[str, str]:
    """The country's display-syntax row from address_formats.csv.

    Render-only declaration (never compiled into a parse regex): the format
    lists comma-separated segments of {component} fields, with '( ... )?'
    optional groups that drop unless every field inside is non-empty.
    Falls back to the 'default' row for countries without one.
    """
    with open(_FORMATS_PATH, encoding='utf8', newline='') as handle:
        rows = {row['admin1_id'].strip(): row for row in csv.DictReader(handle)}
    return rows.get(admin1_id or DEFAULT_ADMIN1_ID, rows['default'])


@cache
def get_admin2_codes(admin1_id: str) -> frozenset[str]:
    """Within-country level-2 codes (e.g. 'NC', 'BY') from ISO 3166-2.

    Backed by the admin-iso recipe table shipped with the package (the same
    data io.admin.get_admin2_iso reads), so state/region validation needs no
    hand-maintained lists and works for any country.
    """
    from openplaces.recipe import get_recipe

    table = get_recipe(
        None,
        'admin-iso-20210301',
        filename='admin2-iso3166-2',
        keep_default_na=False,
    )
    codes = table.loc[table['country_code'] == admin1_id, 'code']
    prefix = f'{admin1_id}-'
    return frozenset(
        code[len(prefix) :]
        for code in codes
        if isinstance(code, str) and code.startswith(prefix)
    )


# Postal-lookup backend capabilities, mirrors PARSER_BACKENDS below: backend
# -> admin1 ids it supports. zipcodes is a US-only ZIP database; other
# countries degrade to None rather than raising.
POSTAL_CITY_BACKENDS = {'zipcodes': frozenset({'US'})}


@dataclass(frozen=True)
class PostalCityMatch:
    """USPS-preferred and alternate city names for a 5-digit ZIP code.

    Attributes
    ----------
    city : str
        USPS-preferred city name.
    acceptable_cities : tuple of str
        USPS-acceptable alternate city names.
    unacceptable_cities : tuple of str
        City names USPS does not accept for this ZIP code.
    state : str
        Two-letter state code.
    county : str
        County name.
    source : str
        Provenance label for the backing lookup table.
    """

    city: str
    acceptable_cities: tuple[str, ...] = ()
    unacceptable_cities: tuple[str, ...] = ()
    state: str = ''
    county: str = ''
    source: str = ''


@cache
def lookup_postal_city(
    zip5: str, admin1_id: str | None = None
) -> PostalCityMatch | None:
    """USPS-preferred/acceptable city names for a 5-digit ZIP code.

    Backed by the `zipcodes` package, gated to admin1_id == 'US' via
    POSTAL_CITY_BACKENDS the same way PARSER_BACKENDS gates usaddress:
    unsupported countries degrade to None rather than raising.

    Parameters
    ----------
    zip5 : str
        Five-digit ZIP code.
    admin1_id : str, optional
        Country code of the run; defaults to DEFAULT_ADMIN1_ID.

    Returns
    -------
    PostalCityMatch or None
        None for unsupported countries, or unknown/invalid ZIP codes.
    """
    admin1_id = admin1_id or DEFAULT_ADMIN1_ID
    if admin1_id not in POSTAL_CITY_BACKENDS['zipcodes']:
        return None

    import zipcodes  # lazy import: keeps `postal` an optional extra

    matches = zipcodes.matching(zip5)
    if not matches:
        return None
    m = matches[0]
    return PostalCityMatch(
        city=m['city'],
        acceptable_cities=tuple(m['acceptable_cities']),
        unacceptable_cities=tuple(m['unacceptable_cities']),
        state=m['state'],
        county=m['county'],
        source='zipcodes',
    )


# usaddress label -> component key used throughout this module
_TAG_MAPPING = {
    'AddressNumberPrefix': 'address_number',
    'AddressNumber': 'address_number',
    'AddressNumberSuffix': 'address_number',
    'StreetNamePreModifier': 'address_street',
    'StreetNamePreDirectional': 'address_street',
    'StreetNamePreType': 'address_street',
    'StreetName': 'address_street',
    'StreetNamePostType': 'address_street',
    'StreetNamePostDirectional': 'address_street',
    'StreetNamePostModifier': 'address_street',
    'USPSBoxType': 'address_street',
    'USPSBoxID': 'address_street',
    'USPSBoxGroupType': 'address_street',
    'USPSBoxGroupID': 'address_street',
    'OccupancyType': 'unit_number',
    'OccupancyIdentifier': 'unit_number',
    'SubaddressType': 'unit_number',
    'SubaddressIdentifier': 'unit_number',
    'PlaceName': 'city',
    'StateName': 'state',
    'ZipCode': 'postal_code',
}

# Component keys, in the order used for cache tuples and merge frames
ADDRESS_COMPONENTS = (
    'address_number',
    'address_street',
    'unit_number',
    'city',
    'state',
    'postal_code',
)

# Components the street-agreement check consumes (reconcile_addresses); the
# remaining ADDRESS_COMPONENTS are fillable from agreeing sources
MATCH_COMPONENTS = ('address_number', 'address_street')

# Parser backend capabilities: backend -> admin1 ids it supports (None = any
# country). 'auto' resolves to the first non-'none' backend supporting the
# effective country, else degrades to 'none'. Placeholders are documented but
# not bundled (heavy native builds / model downloads break the cross-platform,
# lightweight-install policy) and gracefully fall back to 'auto' resolution.
PARSER_BACKENDS = {
    'auto': None,
    'usaddress': frozenset({'US'}),
    'none': None,
}
_PLACEHOLDER_BACKENDS = ('libpostal', 'deepparse')

try:
    _USADDRESS_VERSION = importlib.metadata.version('usaddress')
except importlib.metadata.PackageNotFoundError:
    _USADDRESS_VERSION = None

_UNIT_PREFIXES = r'APT|STE|UNIT|SUITE|APARTMENT|DEPT|RM|ROOM|FL|FLOOR|BLDG|BUILDING|#'


@dataclass(frozen=True)
class ParsedAddress:
    """Structured result of `parse_address`.

    Attributes
    ----------
    components : dict
        Parsed values keyed by :data:`ADDRESS_COMPONENTS` (None if absent).
    address_raw : str
        The original input string, always preserved.
    address_normalized : str
        Uppercase USPS Publication 28 canonical string.
    address_formatted : str
        Title-cased display string (see `harmonize_address_case`).
    metadata : dict
        Parser provenance: {'parser', 'version', 'status'} where status is
        'success' or 'fallback'.
    """

    components: dict[str, str | None] = field(default_factory=dict)
    address_raw: str = ''
    address_normalized: str = ''
    address_formatted: str = ''
    metadata: dict[str, str | None] = field(default_factory=dict)


def _normalize_postal_code(postal_code: str, admin1_id: str | None = None) -> str:
    """Normalize a postal code per the country's declared rewrite rule.

    The rewrite (e.g. US nine digits -> ZIP+4) comes from address_formats.csv;
    codes that do not match the pattern pass through unchanged.
    """
    postal_code = clean_text(postal_code).replace(' ', '')
    spec = get_address_format(admin1_id)
    pattern = (spec.get('postal_rewrite_pattern') or '').strip()
    if pattern:
        postal_code = re.sub(pattern, spec['postal_rewrite_replacement'], postal_code)
    return postal_code


def normalize_address_components(
    address_street: str,
    address_number: str | None = None,
    unit_number: str | None = None,
    postal_code: str | None = None,
    city: str | None = None,
    state: str | None = None,
    admin1_id: str | None = None,
) -> dict[str, str]:
    """Standardize address components to USPS Publication 28 abbreviations.

    Returns a dict keyed by :data:`ADDRESS_COMPONENTS` with cleaned, uppercase
    values; missing components come back as empty strings. State codes are
    validated against the ISO 3166-2 level-2 codes of *admin1_id* (default
    `DEFAULT_ADMIN1_ID`); invalid codes are dropped.
    """
    suffixes = _tables('street_suffix', admin1_id)
    directionals = _tables('directional', admin1_id)
    units = _tables('unit', admin1_id)

    street_clean = clean_text(address_street)
    number_clean = clean_text(address_number or '')
    unit_clean = clean_text(unit_number or '')
    city_clean = clean_text(city or '')
    state_clean = clean_text(state or '')
    postal_clean = _normalize_postal_code(postal_code or '', admin1_id)

    if state_clean and state_clean not in get_admin2_codes(
        admin1_id or DEFAULT_ADMIN1_ID
    ):
        state_clean = ''

    # Abbreviate directionals anywhere, but the street type only in suffix
    # position (last token, or before a trailing directional): 'MILL CREEK
    # ROAD' -> 'MILL CREEK RD', never 'ML CRK RD'
    tokens = [directionals.get(t, t) for t in street_clean.split()]
    pos = len(tokens) - 1
    if pos > 0 and tokens[pos] in directionals.values():
        pos -= 1
    if pos > 0 and tokens[pos] in suffixes:
        tokens[pos] = suffixes[tokens[pos]]
    street_norm = ' '.join(tokens)

    # Normalize the unit designator prefix
    unit_norm = ''
    if unit_clean:
        unit_tokens = unit_clean.split()
        if unit_tokens and unit_tokens[0] in units:
            unit_tokens[0] = units[unit_tokens[0]]
        unit_norm = ' '.join(unit_tokens)

    return {
        'address_number': number_clean,
        'address_street': street_norm,
        'unit_number': unit_norm,
        'city': city_clean,
        'state': state_clean,
        'postal_code': postal_clean,
    }


_FORMAT_GROUP = re.compile(r'\(([^()]*)\)\?')
_FORMAT_FIELD = re.compile(r'\{(\w+)\}')


def _assemble(components: dict[str, str], admin1_id: str | None = None) -> str:
    """Render components in the country's declared segment layout.

    The single definition of the output layout, shared by the normalized and
    the title-cased formatted strings, driven by address_formats.csv:
    optional '( ... )?' groups drop unless every field inside is non-empty,
    empty comma segments drop, whitespace collapses.
    """
    template = get_address_format(admin1_id)['format']

    def render_group(match: re.Match) -> str:
        inner = match.group(1)
        fields = _FORMAT_FIELD.findall(inner)
        if fields and all(components.get(field) for field in fields):
            return inner
        return ''

    text = _FORMAT_GROUP.sub(render_group, template)
    text = _FORMAT_FIELD.sub(lambda m: components.get(m.group(1)) or '', text)
    segments = (' '.join(segment.split()) for segment in text.split(','))
    return ', '.join(segment for segment in segments if segment)


def _parse_fallback(addr_str: str, admin1_id: str) -> dict[str, str | None]:
    """Regex fallback used when usaddress cannot tag a string unambiguously."""
    addr_str = clean_text(addr_str)
    parsed: dict[str, str | None] = dict.fromkeys(ADDRESS_COMPONENTS)
    if not addr_str:
        return parsed

    # Peel postal code, then a valid state code, off the right end
    zip_match = re.search(r'\b(\d{5}(?:-\d{4})?|\d{9})$', addr_str)
    if zip_match:
        parsed['postal_code'] = zip_match.group(1)
        addr_str = addr_str[: zip_match.start()].strip()
    tokens = addr_str.split()
    if tokens and tokens[-1] in get_admin2_codes(admin1_id) and len(tokens) > 1:
        parsed['state'] = tokens[-1]
        addr_str = ' '.join(tokens[:-1])

    # Peel a trailing unit designator
    unit_match = re.search(rf'\b({_UNIT_PREFIXES})\b\s*(.*)$', addr_str)
    if unit_match:
        parsed['unit_number'] = unit_match.group(0).strip()
        addr_str = addr_str[: unit_match.start()].strip()

    words = addr_str.split()
    if not words:
        return parsed
    if any(c.isdigit() for c in words[0]):
        parsed['address_number'] = words[0]
        parsed['address_street'] = ' '.join(words[1:]) or None
    else:
        parsed['address_street'] = addr_str
    return parsed


@lru_cache(maxsize=200_000)
def _parse_cached(addr_str: str, admin1_id: str) -> tuple[tuple[str | None, ...], str]:
    """Tag one unique address string; returns (component tuple, parser used)."""
    try:
        tagged, _ = usaddress.tag(addr_str, tag_mapping=_TAG_MAPPING)
    except usaddress.RepeatedLabelError:
        parsed = _parse_fallback(addr_str, admin1_id)
        return tuple(parsed[c] for c in ADDRESS_COMPONENTS), 'regex'
    parsed = {c: clean_text(tagged[c]) or None for c in tagged}
    return tuple(parsed.get(c) for c in ADDRESS_COMPONENTS), 'usaddress'


def _resolve_backend(backend: str, admin1_id: str | None) -> tuple[str, str]:
    """Resolve the requested backend; returns (resolved backend, status)."""
    status = 'success'
    if backend in _PLACEHOLDER_BACKENDS:
        warnings.warn(
            f'Address parser backend {backend!r} is a documented placeholder '
            'and is not bundled with openplaces (heavy native builds or model '
            'downloads would break the lightweight cross-platform install); '
            'falling back to automatic backend resolution.',
            stacklevel=3,
        )
        backend, status = 'auto', 'fallback'
    elif backend not in PARSER_BACKENDS:
        raise ValueError(
            f'Unknown address parser backend {backend!r}; supported: '
            f'{tuple(PARSER_BACKENDS) + _PLACEHOLDER_BACKENDS}'
        )
    if backend == 'auto':
        # First real backend whose capability set covers the country; when
        # none does, preserve the raw string via the 'none' pass-through
        effective = admin1_id or DEFAULT_ADMIN1_ID
        backend = next(
            (
                name
                for name, supported in PARSER_BACKENDS.items()
                if name not in ('auto', 'none')
                and (supported is None or effective in supported)
            ),
            'none',
        )
        if backend == 'none':
            status = 'fallback'
    return backend, status


def parse_address(
    addr_str: str,
    backend: str = 'auto',
    admin1_id: str | None = None,
) -> ParsedAddress:
    """Parse a one-line address into a structured `ParsedAddress`.

    Parameters
    ----------
    addr_str : str
        The raw address string; always preserved in address_raw.
    backend : str
        'auto' (default) resolves to 'usaddress' for US or unspecified
        addresses and degrades to 'none' otherwise; 'usaddress' forces the
        cached CRF parser; 'none' skips parsing and preserves the raw string.
        'libpostal' and 'deepparse' are unbundled placeholders that warn and
        resolve like 'auto'.
    admin1_id : str, optional
        Level-1 admin (country) id providing routing context for 'auto' and
        state-code validation; None is treated as `DEFAULT_ADMIN1_ID`.
    """
    raw = addr_str if isinstance(addr_str, str) else ''
    resolved, status = _resolve_backend(backend, admin1_id)
    metadata = {'parser': resolved, 'version': None, 'status': status}

    if resolved == 'none' or not raw.strip():
        cleaned = clean_text(raw)
        formatted = ' '.join(w.capitalize() for w in cleaned.split())
        return ParsedAddress(
            components=dict.fromkeys(ADDRESS_COMPONENTS),
            address_raw=raw,
            address_normalized=cleaned,
            address_formatted=formatted,
            metadata=metadata,
        )

    effective_admin1 = admin1_id or DEFAULT_ADMIN1_ID
    values, parser = _parse_cached(raw.strip(), effective_admin1)
    components = dict(zip(ADDRESS_COMPONENTS, values))
    if parser == 'regex':
        metadata.update(parser='regex', status='fallback')
    else:
        metadata.update(version=_USADDRESS_VERSION)

    norm = normalize_address_components(
        address_street=components['address_street'] or '',
        address_number=components['address_number'],
        unit_number=components['unit_number'],
        postal_code=components['postal_code'],
        city=components['city'],
        state=components['state'],
        admin1_id=effective_admin1,
    )
    return ParsedAddress(
        components=components,
        address_raw=raw,
        address_normalized=_assemble(norm, effective_admin1),
        address_formatted=harmonize_address_case(
            **{k: v or None for k, v in norm.items()},
            admin1_id=effective_admin1,
        ),
        metadata=metadata,
    )


def parse_address_string(addr_str: str) -> dict[str, str | None]:
    """Parse a one-line US address into components via `parse_address`.

    Thin convenience wrapper returning only the components dict, keyed by
    :data:`ADDRESS_COMPONENTS`. Results are cached per unique input string.
    """
    return parse_address(addr_str).components


def get_address_similarity(addr1: str, addr2: str) -> float:
    """Fuzzy similarity between two normalized address strings (0-100)."""
    return fuzz.ratio(addr1, addr2)


def canonicalize_for_match(street: str, admin1_id: str | None = None) -> str:
    """Collapse a street string to its comparison form.

    Applies the country's multi-word match phrases first (longest first),
    then maps every token through the union of its suffix, directional, and
    match tables — position-independent, unlike display normalization, so
    'HIGHWAY 94' and 'HWY 94' compare equal. Tokens with an empty
    replacement are dropped.
    """
    phrases, token_map = _match_maps(admin1_id or DEFAULT_ADMIN1_ID)
    text = clean_text(street)
    for phrase, replacement in phrases:
        text = re.sub(rf'\b{re.escape(phrase)}\b', replacement, text)
    tokens = (token_map.get(t, t) for t in text.split())
    return ' '.join(t for t in tokens if t)


def match_streets(
    street1: str,
    street2: str,
    threshold: float = 80,
    admin1_id: str | None = None,
) -> bool:
    """Decide whether two street strings refer to the same street.

    Both sides are canonicalized via `canonicalize_for_match` (with the
    country's tables) and compared with `get_address_similarity`. A dangling
    trailing directional on one side only ('HWY 94 S' vs 'HWY 94') is
    tolerated: the longer side is retried with it trimmed.
    """
    a = canonicalize_for_match(street1, admin1_id)
    b = canonicalize_for_match(street2, admin1_id)
    if not a or not b:
        return False
    if get_address_similarity(a, b) >= threshold:
        return True
    directionals = set(_tables('directional', admin1_id).values())
    for x, y in ((a, b), (b, a)):
        tokens = x.split()
        if len(tokens) > 1 and tokens[-1] in directionals:
            trimmed = ' '.join(tokens[:-1])
            if get_address_similarity(trimmed, y) >= threshold:
                return True
    return False


def format_address_components(
    address_street: str,
    address_number: str | None = None,
    unit_number: str | None = None,
    city: str | None = None,
    state: str | None = None,
    postal_code: str | None = None,
    admin1_id: str | None = None,
) -> dict[str, str]:
    """Case-format address components without assembling them into a line.

    Applies title casing while preserving uppercase directionals (N, S, NE),
    state codes (NC, MA), Roman numerals, and unit alphanumerics (4B), and
    standardizing suffixes (St, Ave) and ordinals (1st, 2nd). Units keep their
    designator ('Apt 4B'); bare identifiers render as '#4B'. Returns a dict
    keyed by :data:`ADDRESS_COMPONENTS`. Shared by :func:`harmonize_address_case`
    (which assembles the result into a single line) and any caller that wants
    to persist formatted components individually rather than a rendered
    string, e.g. :func:`openplaces.io.harmonizer.addresses.reconcile_addresses_df`.
    """
    norm = normalize_address_components(
        address_street,
        address_number,
        unit_number,
        postal_code,
        city,
        state,
        admin1_id=admin1_id,
    )
    directionals = set(_tables('directional', admin1_id).values())
    designators = {v.capitalize() for v in _tables('unit', admin1_id).values()}

    street_words = []
    for word in norm['address_street'].split():
        if word in directionals:
            street_words.append(word)
        elif re.match(r'^\d+(ST|ND|RD|TH)$', word):
            street_words.append(word.lower())
        elif re.match(r'^(II|III|IV|VI{0,3}|IX|XI{0,3})$', word):
            street_words.append(word)
        else:
            street_words.append(word.capitalize())
    street_harmonized = ' '.join(street_words)

    unit_harmonized = ''
    if norm['unit_number']:
        unit_words = []
        for word in norm['unit_number'].split():
            if word == '#':
                unit_words.append(word)
            elif re.match(r'^\d+[A-Z]?$', word):
                unit_words.append(word)
            else:
                unit_words.append(word.capitalize())
        unit_harmonized = ' '.join(unit_words)
        if not unit_harmonized.startswith('#') and unit_harmonized.split()[0] not in (
            designators
        ):
            unit_harmonized = f'#{unit_harmonized}'

    city_harmonized = ' '.join(w.capitalize() for w in norm['city'].split())

    # State code stays uppercase
    return {
        'address_number': norm['address_number'],
        'address_street': street_harmonized,
        'unit_number': unit_harmonized,
        'city': city_harmonized,
        'state': norm['state'],
        'postal_code': norm['postal_code'],
    }


def harmonize_address_case(
    address_street: str,
    address_number: str | None = None,
    unit_number: str | None = None,
    city: str | None = None,
    state: str | None = None,
    postal_code: str | None = None,
    admin1_id: str | None = None,
) -> str:
    """Format address components into a single harmonized address line.

    Thin wrapper around :func:`format_address_components`: applies the same
    case formatting, then assembles the result per the country's segment
    layout (``address_formats.csv``), e.g. the default '1200 Seagrass Ln,
    Coastal City, NC 28500' and DE's 'Hauptstrasse 12, 80331 Munich'.
    """
    return _assemble(
        format_address_components(
            address_street,
            address_number,
            unit_number,
            city,
            state,
            postal_code,
            admin1_id=admin1_id,
        ),
        admin1_id,
    )
