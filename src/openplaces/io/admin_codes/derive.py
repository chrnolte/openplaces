"""Derive a unique code for every unit in one sibling group.

The entry point most callers want. It combines the three pieces that have to
work together and are easy to get wrong separately:

1. the country's own published code, where one exists, since codes like AK
   and AZ are conventions no rule recovers from the name;
2. candidates generated from the name's grammar, for everything else;
3. an optimal assignment across the group, so two siblings wanting the same
   code are resolved by weight rather than by row order.
"""

from collections.abc import Iterable, Mapping

from openplaces.io.admin_codes.anchors import (
    get_anchor_codes,
    get_code_length_convention,
    get_override_codes,
    normalize_name,
)
from openplaces.io.admin_codes.assign import assign_codes
from openplaces.io.admin_codes.candidates import Candidate, generate_candidates

# Rules that produce a code carrying no signal from the name:
# an arbitrary letter pair, or a sequential fallback.
OPAQUE_RULES = frozenset({'any2', 'any3', 'sequential', 'letter_number'})

# Share of a group allowed to end up with an unrecognizable code
# before the whole group widens to three characters.
#
# Calibrated against measurement, not chosen for roundness. Existing
# published two-letter subdivision schemes run 41.9% opaque at the
# median (85 countries, ISO 3166-2 and national), so this is already
# stricter than world practice. At 25% with the minimum group size
# below, 881 of 17,014 parents widen; the opacity test, not the size
# test, accounts for most of that.
DEFAULT_MAX_OPAQUE = 0.25

# Above this many siblings, two characters are abandoned outright.
#
# The two-character space is 26*36 = 936, but the candidate pool a
# group actually proposes saturates near 600, because names cluster:
# a hundred municipalities beginning San or Santa propose largely the
# same codes. Measured opacity by group size runs 21% at 200 siblings,
# 26% at 250, 31% at 300 and 50% at 500. 250 is where candidates per
# unit falls to about 2.0 and crowding starts to bite. It affects 81
# of 19,742 parents, which is what a marked exception should mean.
MAX_SIBLINGS_TWO_CHAR = 250

# Below this many siblings the opacity test does not apply.
#
# A share is meaningless on a tiny group: one awkward code out of two
# reads as 50% opaque, and a single-unit group as 100%. Without this
# floor, 658 parents of fewer than ten units widen on arithmetic
# rather than crowding -- 43% of all widenings, every one of them
# spurious. A group this small can always be carried by two
# characters.
MIN_GROUP_FOR_OPACITY = 10


def derive_codes(
    names: Iterable[str],
    admin1_id: str | None = None,
    lengths: tuple[int, ...] | None = None,
    weights: Mapping[str, float] | None = None,
    use_anchors: bool = True,
    length_penalty: float = 0.15,
    reserved: set[str] | None = None,
    max_opaque: float = DEFAULT_MAX_OPAQUE,
) -> dict[str, tuple[str, str]]:
    """Assign a unique code to each name in one sibling group.

    Parameters
    ----------
    names : iterable of str
        Unit names sharing a parent. Duplicates are collapsed.
    admin1_id : str, optional
        Country identifier, used to pick the language vocabulary and the
        anchor table.
    lengths : tuple of int, optional
        Code lengths to allow. Leave unset for the default policy: two
        characters unless the group cannot carry them, then three for the
        whole group. Passing a value overrides that and is honored exactly.
    weights : mapping of str to float, optional
        Relative weight per name, typically population. Decides which unit
        wins a contested code.
    use_anchors : bool, optional
        Whether to prefer the country's published code where one matches
        the name. Default True.
    length_penalty : float, optional
        Satisfaction cost per character beyond two. Ignored when only one
        length is allowed.
    reserved : set of str, optional
        Codes unavailable to this group, e.g. those held by units that sit
        outside the navigational path.

    Returns
    -------
    dict of str to tuple of (str, str)
        Name mapped to its assigned code and the rule that produced it.
        Anchored codes carry the rule 'anchor'.

    Examples
    --------
    >>> derive_codes(['Alaska', 'Alabama'], admin1_id='US')['Alaska']
    ('AK', 'anchor')
    """
    names = list(dict.fromkeys(names))
    anchors = get_anchor_codes(admin1_id) if (use_anchors and admin1_id) else {}

    def solve(widths: tuple[int, ...]) -> dict[str, tuple[str, str]]:
        candidates_by_unit: dict[str, list[Candidate]] = {}
        for name in names:
            generated = generate_candidates(name, admin1_id=admin1_id, lengths=widths)
            anchor = anchors.get(normalize_name(name))
            if anchor and len(anchor) in widths:
                # The published code goes first and is
                # de-duplicated out of the generated list, so
                # anchoring never costs a candidate.
                generated = [Candidate(anchor, 'anchor', len(anchor))] + [
                    c for c in generated if c.code != anchor
                ]
            candidates_by_unit[name] = generated
        return assign_codes(
            candidates_by_unit,
            weights=weights,
            length_penalty=length_penalty,
            reserved=reserved,
            fallback_width=min(widths),
        )

    # An explicit request is honored exactly as given.
    if lengths is not None:
        return solve(lengths)

    # A reviewed override records a decision a person made about these
    # specific units. A length convention is inferred from a published
    # scheme that may enumerate an entirely different level -- ISO
    # 3166-2:GB lists 252 districts, not the four home nations. So where
    # overrides cover this whole group and agree on a width, that width
    # wins over the convention. Requiring *every* unit to be covered
    # keeps this from splitting a group across two widths.
    overrides = get_override_codes(admin1_id) if admin1_id else {}
    if overrides:
        keys = [normalize_name(name) for name in names]
        if all(key in overrides for key in keys):
            widths = {len(overrides[key]) for key in keys}
            if len(widths) == 1:
                return solve((widths.pop(),))

    # A country that publishes three-character codes otherwise keeps
    # them: adopting the published convention is what makes a code
    # recognizable.
    convention = get_code_length_convention(admin1_id) if admin1_id else None
    if convention == 3:
        return solve((3,))

    # A group this large cannot be carried by two characters at all,
    # so there is no point solving for them first.
    if len(names) > MAX_SIBLINGS_TWO_CHAR:
        return solve((3,))

    narrow = solve((2,))
    if not narrow:
        return narrow
    # Opacity is judged on the weighted assignment, because that is the
    # one that will actually ship: weights change which unit wins a
    # contested code and therefore which codes end up carrying no
    # signal.
    opaque = sum(1 for _, rule in narrow.values() if rule in OPAQUE_RULES)
    if len(narrow) < MIN_GROUP_FOR_OPACITY:
        return narrow
    if opaque / len(narrow) <= max_opaque:
        return narrow

    # Widening is all-or-nothing. A group carrying some two- and some
    # three-character codes reads as a mistake and sorts badly, so the
    # whole group moves together.
    return solve((3,))
