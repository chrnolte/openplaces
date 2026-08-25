"""Read one named part out of a hierarchical national code.

Many national coding schemes are concatenated hierarchies: a US Census
GEOID is state(2) + county(3) + subdivision(5), INSEE is departement(2)
+ commune(3), DANE is departamento(2) + municipio(3). The parent's code
is a prefix of the child's, and that is a guarantee the source
publishes rather than an accident to be hidden.

What must not be inferred from such a code is which **openplaces admin
level** a given part corresponds to. That mapping is ours, it varies by
geography, and it changes when a government reorganizes. The US county
segment is the worked example: it names an openplaces level-3 unit in
44 states and names nothing in the openplaces hierarchy in the six New
England states, whose level 3 is towns. Connecticut shows the same
thing over time, having replaced its eight counties with nine planning
regions in 2022 while every town kept its own code.

So callers ask for a *segment*, named in the source's own vocabulary,
and never for a level. A segment survives a reorganization that removes
a level, and resolves one-to-many into finer units where a level is
subdivided. The segmentation itself is declared as data, in
``admin-spine-2026_code-segments.csv``, so that a scheme this module
has never seen needs a row rather than a code change.

Declaring the widths also makes damage detectable. A code whose length
matches no declared total for its scheme is reported rather than
silently sliced, which is how a national code that lost a leading zero
to a numeric round-trip surfaces instead of truncating every lookup
built on it.
"""

from __future__ import annotations

from functools import cache
from typing import NamedTuple

import pandas as pd

from openplaces.io.admin_codes.registry import spine_path

DEFAULT_SCHEME = 'us-census-geoid'


def _segments_path():
    """Return the path of the committed code-segment declaration."""
    return spine_path(3).parent / 'admin-spine-2026_code-segments.csv'


class Segment(NamedTuple):
    """One named part of a hierarchical code.

    Parameters
    ----------
    scheme : str
        Identifier of the coding scheme, e.g. ``'us-census-geoid'``.
    admin1_id : str
        Country whose units the scheme codes. A scheme is only ever
        applied to that country's units, so a five-character code
        elsewhere in the world is never mistaken for a US county FIPS.
    segment : str
        The source's own name for this part, e.g. ``'county'``.
    start : int
        Zero-based character offset where the part begins.
    length : int
        Number of characters the part occupies.
    total_length : int
        Full width of a code that terminates at this part, used to
        validate a value before slicing it.
    """

    scheme: str
    admin1_id: str
    segment: str
    start: int
    length: int
    total_length: int


@cache
def load_code_segments() -> dict[tuple[str, str], Segment]:
    """Return every declared segment, keyed by (scheme, segment).

    Reads the declaration CSV once and caches it.

    Returns
    -------
    dict of (str, str) to Segment
        All declared segments.
    """
    table = pd.read_csv(_segments_path(), dtype=str, keep_default_na=False)
    segments = {}
    for row in table.itertuples(index=False):
        scheme = str(row.scheme).strip()
        segment = str(row.segment).strip()
        if not scheme or not segment:
            continue
        segments[(scheme, segment)] = Segment(
            scheme=scheme,
            admin1_id=str(row.admin1_id).strip(),
            segment=segment,
            start=int(row.start),
            length=int(row.length),
            total_length=int(row.total_length),
        )
    return segments


@cache
def _scheme_widths(scheme: str) -> frozenset[int]:
    """Return the code widths a scheme declares as complete."""
    return frozenset(
        seg.total_length for (s, _), seg in load_code_segments().items() if s == scheme
    )


def get_segment(segment: str, scheme: str = DEFAULT_SCHEME) -> Segment:
    """Return one declared segment.

    Parameters
    ----------
    segment : str
        The source's own name for the part, e.g. ``'county'``.
    scheme : str, optional
        Coding scheme, default ``'us-census-geoid'``.

    Returns
    -------
    Segment
        The declaration.

    Raises
    ------
    KeyError
        When the scheme or segment is not declared, listing what is.
    """
    segments = load_code_segments()
    try:
        return segments[(scheme, segment)]
    except KeyError:
        known = sorted(s for (sch, s) in segments if sch == scheme)
        if not known:
            schemes = sorted({sch for (sch, _) in segments})
            raise KeyError(
                f'No coding scheme {scheme!r} is declared in '
                f'{_segments_path().name}. Declared schemes: '
                f'{", ".join(schemes)}.'
            ) from None
        raise KeyError(
            f'Scheme {scheme!r} declares no segment {segment!r}. '
            f'Declared segments: {", ".join(known)}.'
        ) from None


def slice_segment(
    codes,
    segment: str,
    scheme: str = DEFAULT_SCHEME,
    strict: bool = True,
) -> pd.Series:
    """Take one named part out of each code.

    Parameters
    ----------
    codes : pandas.Series or sequence
        National codes, one per unit. Values are compared as strings;
        a missing or blank value yields a missing result.
    segment : str
        The source's own name for the part wanted, e.g. ``'county'``.
    scheme : str, optional
        Coding scheme, default ``'us-census-geoid'``.
    strict : bool, optional
        When True (default), raise if any non-blank code has a width the
        scheme does not declare. Pass False to leave those unsliced as
        missing values, which is what a mixed-provenance column needs.

    Returns
    -------
    pandas.Series
        The segment's characters per input, dtype ``string``, indexed
        like *codes*. Missing where the input was blank, or where its
        width was undeclared and *strict* is False.

    Raises
    ------
    ValueError
        When *strict* and a code's width is not declared for the scheme.
        A width that matches nothing usually means the column mixes two
        coding schemes, or that a numeric round-trip dropped a leading
        zero.
    """
    declaration = get_segment(segment, scheme)
    widths = _scheme_widths(scheme)

    values = codes if isinstance(codes, pd.Series) else pd.Series(list(codes))
    values = values.astype('string').str.strip()
    present = values.notna() & (values != '')

    lengths = values.str.len()
    undeclared = present & ~lengths.isin(widths)
    if undeclared.any():
        sample = sorted(set(values[undeclared].dropna()))[:5]
        message = (
            f'{int(undeclared.sum()):,d} code(s) have a width the '
            f'{scheme!r} scheme does not declare '
            f'(declared: {sorted(widths)}). Examples: '
            f'{", ".join(sample)}.'
        )
        if strict:
            raise ValueError(
                message + ' A width that matches nothing usually means the '
                'column mixes two coding schemes, or that a leading zero '
                'was lost. Pass strict=False to skip these rows instead.'
            )

    end = declaration.start + declaration.length
    # A code that stops before this segment ends does not contain it.
    # Slicing it anyway would return a short string that looks like a
    # valid answer, which is the failure this module exists to prevent.
    usable = present & (lengths >= end) & lengths.isin(widths)
    out = pd.Series(pd.NA, index=values.index, dtype='string')
    out[usable] = values[usable].str[declaration.start : end]
    return out


def admin_code_segment(
    admin_ids=None,
    segment: str = 'county',
    level: int = 3,
    scheme: str = DEFAULT_SCHEME,
    strict: bool = True,
) -> pd.Series:
    """Return one segment of each admin unit's own national code.

    Reads the spine rather than any entity table, so the answer does not
    depend on which datasets happen to be built. Nothing is stored: the
    segment is taken from the code the spine already carries, which is
    why a unit whose national code equals the segment (a US county,
    whose ``admin3_id_admin1`` *is* its county FIPS) needs no row
    anywhere and returns the same value as one where the segment is a
    prefix (a New England town).

    Parameters
    ----------
    admin_ids : sequence of str, optional
        Restrict to these units. Defaults to every unit at *level* that
        carries a national code.
    segment : str, optional
        The source's own name for the part wanted, default
        ``'county'``.
    level : int, optional
        Spine level to read, default 3.
    scheme : str, optional
        Coding scheme, default ``'us-census-geoid'``.
    strict : bool, optional
        Forwarded to :func:`slice_segment`. Defaults to True. The spine
        is global and holds several schemes at once, so a caller that
        has not narrowed *admin_ids* to one country will usually want
        False.

    Returns
    -------
    pandas.Series
        Segment value per admin id, indexed by ``admin{level}_id``.
    """
    column = f'admin{level}_id'
    code_column = f'{column}_admin1'
    spine = pd.read_csv(spine_path(level), dtype=str, keep_default_na=False)
    if code_column not in spine.columns:
        raise KeyError(f'Spine level {level} has no {code_column!r}.')

    spine = spine.set_index(column)
    # A scheme codes one country's units. Without this, a five-character
    # national code anywhere else in the world would be read as a US
    # county FIPS and returned as though it meant something.
    country = get_segment(segment, scheme).admin1_id
    if country:
        spine = spine[spine.index.str.startswith(f'{country}-')]
    if admin_ids is not None:
        wanted = [a for a in dict.fromkeys(str(x) for x in admin_ids)]
        spine = spine.reindex(wanted)

    values = slice_segment(
        spine[code_column], segment=segment, scheme=scheme, strict=strict
    )
    values.index = spine.index
    values.name = segment
    return values
