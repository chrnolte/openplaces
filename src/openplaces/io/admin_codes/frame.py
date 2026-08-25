"""Assign admin ids to a table of units, one sibling group at a time.

The recipe-facing entry point. :func:`~openplaces.io.admin_codes.derive.derive_codes`
works on one group of names; a recipe hands over a whole country's table at
once, so this module splits it by parent, derives each group independently,
and reassembles the full identifiers.

Three things only show up at table level and are handled here rather than in
the code generator: units that share a name with a sibling, units that have
no name at all, and the country prefix that selects the language vocabulary.
"""

import itertools
import string

import pandas as pd

from openplaces.core.constants import STRING_SEPARATOR_WITHIN_IDS
from openplaces.io.admin_codes.anchors import load_group_code_lengths
from openplaces.io.admin_codes.candidates import is_valid_code
from openplaces.io.admin_codes.derive import derive_codes
from openplaces.io.admin_codes.registry import (
    external_codes,
    level_of,
    load_registry,
)


def _placeholder_codes(count, width, taken):
    """Return codes for units carrying no name.

    Ordered so the X block is consumed first, which keeps a nameless unit
    visually distinct from a derived code and matches the convention the
    predecessor implementation used.
    """
    letters = 'X' + ''.join(c for c in string.ascii_uppercase if c != 'X')
    tail = string.ascii_uppercase + string.digits
    out = []
    for first in letters:
        for rest in itertools.product(tail, repeat=width - 1):
            if len(out) >= count:
                return out
            code = first + ''.join(rest)
            if code not in taken and is_valid_code(code):
                out.append(code)
                taken.add(code)
    raise ValueError(
        f'Ran out of {width}-character codes: needed {count}, {len(out)} available.'
    )


def _population_weights(raw):
    """Turn a population column into weights for the assignment.

    Weights are the raw population, deliberately not its logarithm.

    An earlier version took log10, reasoning that population spans four
    orders of magnitude between siblings and that a raw weight lets one
    large unit displace several small ones to win a code it barely
    prefers. That reasoning holds if the quantity being protected is
    *units*. It does not if the quantity is *people*. An identifier is
    read and retyped by a person, so what a tie-break should minimize is
    the number of people facing an unreadable code, and that is linear in
    population, not logarithmic.

    The compression was also large enough to make the weighting inert.
    Cook County has 110 times the population of Coles County, which log10
    turns into a 1.44x weight difference -- less than the 0.65-per-rank
    penalty the assignment charges for moving a unit off its first
    choice. Measured under log10, Cook still lost `CO` to Coles, King
    lost `KI` to Kitsap, and Somerville lost `SOM` to Somerset. Under raw
    population all three go the other way.

    Note this does not make the largest unit win every contest. The
    assignment maximizes a weighted sum over the whole sibling group, so
    a unit with a cheap uncontested fallback can still be moved off its
    first choice when displacing its rival would cascade further.

    A unit with no population, or none recorded, must not lose every
    contest because of a coverage gap. It takes the group's median
    weight, which is what "neither advantaged nor disadvantaged" means;
    a literal 1.0 would sit far below every real weight and amount to
    always losing.

    Parameters
    ----------
    raw : pandas.Series
        Population per name, possibly with missing or zero entries.

    Returns
    -------
    dict of str to float or None
        Weight per name, suitable for `derive_codes(weights=)`. None when
        no unit in the group has a recorded population, which leaves the
        assignment to preference order alone.
    """
    positive = raw.where(raw > 0)
    neutral = positive.median()
    if pd.isna(neutral):
        # Nothing in this group has a population; let preference order
        # decide, as it would with no weight column at all.
        return None
    return positive.fillna(neutral).astype(float).to_dict()


def assign_admin_ids(
    df,
    new_admin_id_col='admin4_id',
    parent_admin_id_col='admin3_id',
    name_col='name',
    id_separator=STRING_SEPARATOR_WITHIN_IDS,
    weight_col=None,
    lengths=None,
    pin_to_spine=True,
    verbose=False,
):
    """Build a unique admin id for every row, grouped by parent unit.

    Wired from a recipe via ``create_index.function`` with the column names
    as ``args``. Replaces ``openplaces.io.admin.generate_admin_ids``, whose
    priority waterfall handed out whichever code happened to be free first
    and so depended on row order.

    Parameters
    ----------
    df : pandas.DataFrame or geopandas.GeoDataFrame
        Units to identify. Must carry the parent id and name columns.
    new_admin_id_col : str, optional
        Column to build and set as the index, e.g. 'admin4_id'.
    parent_admin_id_col : str, optional
        Column holding the parent's full admin id, e.g. 'admin3_id'. Its
        first segment selects the language vocabulary and anchor table.
    name_col : str, optional
        Column holding the unit name. Default 'name'.
    id_separator : str, optional
        Separator between the parent id and the new code.
    weight_col : str, optional
        Numeric column, typically population, deciding which of two units
        wanting the same code gets it. Unweighted by default.
    lengths : tuple of int, optional
        Force specific code lengths. Leave unset for the default policy:
        two characters unless the group cannot carry them, then three for
        the whole group.
    pin_to_spine : bool, optional
        Reproduce the codes the committed spine already records, assigning
        only units it does not name, and keep the codes of units that have
        since disappeared reserved. Default True, which is what makes an
        id stable across re-runs; see
        :mod:`~openplaces.io.admin_codes.registry`. Pass False to mint a
        level from scratch, which is a deliberate migration, not a rerun.
    verbose : bool, optional
        Print a per-rule summary of how the codes were derived.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        The input indexed by ``new_admin_id_col``, with a
        ``{new_admin_id_col}_source`` column naming the rule behind each
        code.

    Raises
    ------
    ValueError
        If a required column is absent, a parent id is missing, or the
        assembled ids are not unique.
    """
    for column in (parent_admin_id_col, name_col):
        if column not in df:
            raise ValueError(
                f"Column '{column}' not found; available: {sorted(df.columns)[:20]}"
            )

    admin = df.copy()
    parents = admin[parent_admin_id_col].astype('string')
    blank_parents = parents.isna() | (parents.str.strip() == '')
    if blank_parents.any():
        raise ValueError(
            f'{int(blank_parents.sum())} rows have no {parent_admin_id_col}; '
            'every unit needs a parent before it can be identified.'
        )
    names = admin[name_col].astype('string').fillna('').str.strip()

    codes = pd.Series(pd.NA, index=admin.index, dtype='string')
    rules = pd.Series(pd.NA, index=admin.index, dtype='string')

    pins, by_external, issued = {}, {}, {}
    externals = pd.Series('', index=admin.index, dtype=object)
    level = level_of(new_admin_id_col)
    if pin_to_spine and level is not None:
        pins, by_external, issued = load_registry(level, id_separator)
        externals = external_codes(admin, level)

    for parent, rows in names.groupby(parents, sort=True):
        country = str(parent).split(id_separator)[0]
        named = rows[rows != '']

        # Units the spine already names keep their code; only the rest
        # are assigned, out of what those leave free.
        held = {n: pins[(parent, n)] for n in set(named) if (parent, n) in pins}
        # Rows a name cannot identify are pinned per row, on the
        # source's own code, and so are resolved outside `held`.
        pinned_rows = {}
        for idx, external in externals[rows.index].items():
            name = names[idx]
            if name in held or not external:
                continue
            code = by_external.get((parent, str(external)))
            if code is not None:
                pinned_rows[idx] = code
        # Codes this parent has issued but no current unit claims. Held
        # in reserve rather than reissued, so a retired id resolves to
        # nothing instead of to some other unit.
        claimed = set(held.values()) | set(pinned_rows.values())
        reserved = set(issued.get(parent, ())) - claimed

        rows_by_name = {}
        for idx, name in named.items():
            rows_by_name.setdefault(name, []).append(idx)
        open_names = [
            n
            for n in dict.fromkeys(named)
            if n not in held and any(idx not in pinned_rows for idx in rows_by_name[n])
        ]

        weights = None
        if weight_col is not None:
            # A name shared by two siblings takes the larger weight, so
            # the group as a whole competes on its strongest claim.
            raw = (
                pd.to_numeric(admin.loc[named.index, weight_col], errors='coerce')
                .groupby(named.values)
                .max()
            )
            weights = _population_weights(raw)
        # Pinned codes fix the group's width, so a unit added later does
        # not arrive three characters wide among two-character siblings.
        pinned_widths = {len(code) for code in held.values()}
        group_lengths = lengths
        if group_lengths is None and len(pinned_widths) == 1:
            group_lengths = (pinned_widths.pop(),)
        # A reviewed decision about this specific group of siblings
        # outranks the country convention, which describes a different
        # level and cannot know that one state's counties are crowded
        # where another's are not.
        if lengths is None:
            reviewed = load_group_code_lengths().get(str(parent).upper())
            if reviewed:
                group_lengths = (reviewed,)

        assigned = derive_codes(
            open_names,
            admin1_id=country,
            weights=weights,
            lengths=group_lengths,
            reserved=reserved | claimed,
        )
        assigned = {name: (code, 'pinned') for name, code in held.items()} | assigned
        taken = {code for code, _ in assigned.values()} | reserved | claimed
        # Clamped to the format rule: a pinned legacy code may be longer
        # than three characters, but nothing newly minted may be.
        width = min(3, max((len(c) for c, _ in assigned.values()), default=2))

        consumed = set()
        for idx, name in named.items():
            if idx in pinned_rows:
                code, rule = pinned_rows[idx], 'pinned'
            elif name not in consumed and name in assigned:
                code, rule = assigned[name]
                consumed.add(name)
            else:
                # Siblings sharing a name cannot share a code. The
                # second and later ones re-derive against everything
                # already taken, so they fall to their next-best
                # candidate.
                code, rule = derive_codes(
                    [name],
                    admin1_id=country,
                    lengths=(width,),
                    reserved=taken,
                )[name]
            taken.add(code)
            codes[idx], rules[idx] = code, rule

        nameless = [idx for idx in rows.index if rows[idx] == '']
        fresh = [idx for idx in nameless if idx not in pinned_rows]
        for idx in nameless:
            if idx in pinned_rows:
                codes[idx], rules[idx] = pinned_rows[idx], 'pinned'
                taken.add(pinned_rows[idx])
        for idx, code in zip(fresh, _placeholder_codes(len(fresh), width, taken)):
            codes[idx], rules[idx] = code, 'placeholder'

    admin[new_admin_id_col] = parents + id_separator + codes
    admin[new_admin_id_col + '_source'] = rules

    if admin[new_admin_id_col].duplicated().any():
        duplicates = admin[admin[new_admin_id_col].duplicated(keep=False)]
        raise ValueError(
            f'{int(admin[new_admin_id_col].duplicated().sum())} duplicate ids:\n'
            + duplicates[[new_admin_id_col, name_col, parent_admin_id_col]]
            .sort_values(new_admin_id_col)
            .head(10)
            .to_string()
        )
    invalid = ~admin[new_admin_id_col].str.fullmatch(rf'[A-Z0-9{id_separator}]+')
    if invalid.any():
        raise ValueError(
            f'{int(invalid.sum())} ids contain characters outside [A-Z0-9]:\n'
            + admin.loc[invalid, [new_admin_id_col, name_col]].head(10).to_string()
        )

    if verbose:
        summary = admin[new_admin_id_col + '_source'].value_counts()
        print(f'Assigned {len(admin):,} ids across {parents.nunique():,} parents')
        print(summary.to_string())

    return admin.set_index(new_admin_id_col)
