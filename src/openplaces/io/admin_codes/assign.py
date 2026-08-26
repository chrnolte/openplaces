"""Assign one unique code per unit within a group of siblings.

Two units under the same parent may want the same code. Resolving that is
an assignment problem, not a priority question: choose the injective
mapping from units to codes that maximizes total satisfaction. It is solved
exactly by maximum-weight bipartite matching, so no heuristic waterfall or
tie-breaking rule is needed, and the result does not depend on row order.

Weights let a larger unit's preference outweigh a smaller one's, which is
the behavior wanted when a well-known city and a minor district nearby
share a name. Passing population as the weight is the intended use;
weighting everything equally is the default so the function is usable
before any population data exists.
"""

import itertools
import string
from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from openplaces.io.admin_codes.candidates import Candidate, is_valid_code

# Utility assigned to a pairing that a unit did not propose at all.
# Large and negative so the optimizer only reaches for it when nothing
# else is free, but finite, because linear_sum_assignment rejects
# infinities.
UNWANTED = -1.0e6


def rank_score(rank: int, decay: float = 0.65) -> float:
    """Return the satisfaction of receiving a unit's rank-th choice.

    A stand-in for a real ballot. Once units score codes directly, pass
    those scores instead; the assignment itself is unchanged.

    Parameters
    ----------
    rank : int
        Zero-based position in the unit's preference list.
    decay : float, optional
        Geometric decay per rank step, default 0.65.

    Returns
    -------
    float
        Score in (0, 1], monotonically decreasing in rank.
    """
    return float(decay**rank)


def _fallback_codes(count: int, taken: set[str], width: int = 3) -> list[str]:
    """Return sequential codes guaranteeing the assignment is feasible.

    Emitted at the width the group is being solved at, so a group that
    exhausts its candidates does not end up carrying two code widths at
    once. Digits lead so a fallback is visually distinct from a code
    derived from the name.
    """
    out = []
    for letter in string.ascii_uppercase:
        for rest in itertools.product(
            string.digits + string.ascii_uppercase, repeat=width - 1
        ):
            if len(out) >= count:
                return out
            code = letter + ''.join(rest)
            if code not in taken and is_valid_code(code):
                out.append(code)
    return out


def assign_codes(
    candidates_by_unit: Mapping[str, Sequence[Candidate]],
    weights: Mapping[str, float] | None = None,
    length_penalty: float = 0.15,
    score_fn: Callable[[int], float] = rank_score,
    reserved: set[str] | None = None,
    fallback_width: int | None = None,
) -> dict[str, tuple[str, str]]:
    """Choose a unique code for every unit in one sibling group.

    Parameters
    ----------
    candidates_by_unit : mapping of str to sequence of Candidate
        Preference-ordered candidates per unit key.
    weights : mapping of str to float, optional
        Relative weight per unit, typically population. Defaults to 1.0
        for every unit, which makes the result depend only on preference
        order.
    length_penalty : float, optional
        Satisfaction subtracted for each character beyond two. Set low
        where two-character codes are scarce and a third character buys
        real recognizability; set high to keep codes short. Default 0.15.
    score_fn : callable, optional
        Maps a zero-based preference rank to a satisfaction score.
    reserved : set of str, optional
        Codes already used elsewhere and not available to this group.
    fallback_width : int, optional
        Width of the sequential codes handed to units that offered no
        candidate. Defaults to the widest candidate in the group, which
        is right when the group is solved at one width and wrong when a
        name offered nothing at all: a five-digit territory name in a
        two-character group has no candidate to read a width from and
        arrived three characters wide. The caller solving at a fixed
        width passes it here.

    Returns
    -------
    dict of str to tuple of (str, str)
        Unit key mapped to its assigned code and the rule that produced
        it. Units assigned a fallback carry the rule 'sequential'.

    Notes
    -----
    The objective is the total weighted satisfaction

        sum over units of weight(unit) * (score(rank) - penalty(length))

    maximized over injective unit-to-code mappings. Both the utilitarian
    form above and a Nash-welfare form (summing weighted log scores) are
    additive, so switching between them requires only transforming the
    scores before they reach this function.
    """
    # Sort both axes canonically. The optimum value never depends on
    # input order, but the optimal assignment is not unique when units
    # tie, and linear_sum_assignment then returns whichever solution the
    # input ordering happens to favor. Building the matrix from sorted
    # keys makes the chosen solution reproducible for the same set of
    # units.
    units = sorted(candidates_by_unit)
    if not units:
        return {}

    weights = weights or {}
    reserved = set(reserved or ())

    codes = sorted(
        {
            candidate.code
            for unit in units
            for candidate in candidates_by_unit[unit]
            if candidate.code not in reserved
        }
    )
    seen = set(codes)

    # Guarantee at least one column per unit so a solution always
    # exists.
    if len(codes) < len(units):
        width = fallback_width or max(
            (
                candidate.length
                for unit in units
                for candidate in candidates_by_unit[unit]
            ),
            default=3,
        )
        codes += _fallback_codes(len(units) - len(codes), seen | reserved, width=width)

    index_of = {code: i for i, code in enumerate(codes)}
    utility = np.full((len(units), len(codes)), UNWANTED, dtype=float)
    rules: dict[tuple[int, int], str] = {}

    for row, unit in enumerate(units):
        weight = float(weights.get(unit, 1.0))
        for rank, candidate in enumerate(candidates_by_unit[unit]):
            column = index_of.get(candidate.code)
            if column is None:
                continue
            penalty = length_penalty * max(0, candidate.length - 2)
            utility[row, column] = weight * (score_fn(rank) - penalty)
            rules[(row, column)] = candidate.rule

    rows, columns = linear_sum_assignment(-utility)
    return {
        units[r]: (codes[c], rules.get((r, c), 'sequential'))
        for r, c in zip(rows, columns)
    }
