"""Generate ranked short codes for an administrative-unit name.

The output is a preference-ordered list of candidate codes rather than a
single answer. Choosing among them is a separate concern, handled in
:mod:`openplaces.io.admin_codes.assign`, because the right code for a unit
depends on what its siblings want.

This replaces the fixed priority waterfall in
:func:`openplaces.io.admin.generate_admin_ids`, which assigned the first
code that happened to be free and was therefore sensitive to row order. A
candidate list plus an assignment step is both order-independent and, given
scores, optimal.

Design note on independent derivation
-------------------------------------
The rules here are derived from the grammar of the names themselves, not
from any existing code table. That matters because the obvious prior art,
HASC, is published without a license. HASC's own documented construction
rule is "the first letter in the subdivision name, followed by a letter
that occurs later in that name" -- an unprotectable heuristic that this
module reaches independently, and only as a late fallback. No HASC table is
read, shipped, or consulted anywhere in this codebase.
"""

import re
from itertools import combinations
from typing import NamedTuple

from openplaces.io.admin_codes.coverage import HONORIFICS
from openplaces.io.admin_codes.languages import (
    LanguagePack,
    fold_diacritics,
    get_language_pack,
)

# A code is a letter followed by one or two alphanumerics. The leading
# letter keeps codes sortable and unambiguous against numeric source
# identifiers, and keeps every path segment a valid identifier on all
# platforms.
CODE_PATTERN = re.compile(r'[A-Z][A-Z0-9]{1,2}')

VOWELS = frozenset('AEIOU')


class Candidate(NamedTuple):
    """One proposed code for a unit.

    Parameters
    ----------
    code : str
        The candidate code, matching CODE_PATTERN.
    rule : str
        Name of the rule that produced it, retained as provenance so a
        generated identifier can be explained after the fact.
    length : int
        Number of characters, used by the assigner's length penalty.
    """

    code: str
    rule: str
    length: int


def is_valid_code(code: str) -> bool:
    """Return True if code is a well-formed admin id segment.

    Parameters
    ----------
    code : str
        Candidate code.

    Returns
    -------
    bool
        True when code is a letter followed by one or two alphanumerics.

    Examples
    --------
    >>> is_valid_code('HSA')
    True
    >>> is_valid_code('3DR')
    False
    >>> is_valid_code('BU3SAM')
    False
    """
    return bool(code) and CODE_PATTERN.fullmatch(code) is not None


def tokenize(name: str, pack: LanguagePack) -> tuple[list[str], list[str]]:
    """Split a name into all tokens and significant tokens.

    Parameters
    ----------
    name : str
        Administrative unit name, in any script.
    pack : LanguagePack
        Vocabulary deciding which tokens are articles, prepositions or
        conjunctions.

    Returns
    -------
    tuple of (list of str, list of str)
        All tokens with articles removed, and the significant tokens with
        prepositions and conjunctions also removed. Both are folded to
        upper-case ASCII.

    Examples
    --------
    >>> pack = get_language_pack('fr')
    >>> tokenize("Bouches-du-Rhone", pack)
    (['BOUCHES', 'DU', 'RHONE'], ['BOUCHES', 'RHONE'])
    """
    raw = [t for t in re.split(r'[^A-Za-z0-9]+', fold_diacritics(name)) if t]
    # Administrative type words say what a unit is, not which one it is, so
    # they must not reach a code. Sources disagree about carrying them --
    # Wikidata gives "Haines Borough" and "Dillingham Census Area" where
    # the spine gives "Haines" and "Dillingham" -- and leaving them in
    # produced HB and DC instead of HA and DI, which is how 56% of US
    # county codes ended up unrecognizable.
    # HONORIFICS as well as the pack's own articles. The two lists were
    # applied inconsistently: `coverage.intuitive_codes` dropped them and
    # so judged a group two-character-codeable, while this generator kept
    # them and produced codes built on the honorific. Dominica is the
    # measured case -- ten parishes named "Saint X" whose two-character
    # solution came out SA SD SG SI SJ SL SM SP SN ST, every code on
    # Saint's S and three of them opaque, which tripped the opacity gate
    # and widened the whole group to three characters against its own
    # recorded policy of two.
    tokens = [
        t
        for t in raw
        if not pack.is_article(t)
        and not pack.is_type_word(t)
        and t.upper() not in HONORIFICS
    ] or raw
    significant = [
        t for t in tokens if not pack.is_preposition(t) and not pack.is_conjunction(t)
    ]
    # A name made entirely of particles still has to yield something.
    return tokens, significant or tokens


def _consonant_skeleton(word: str) -> str:
    """Return the word's distinct consonants, first occurrence order."""
    return ''.join(dict.fromkeys(c for c in word if c not in VOWELS))


def generate_candidates(
    name: str,
    pack: LanguagePack | None = None,
    admin1_id: str | None = None,
    lengths: tuple[int, ...] = (2, 3),
) -> list[Candidate]:
    """Return preference-ordered candidate codes for one name.

    Rules are applied in decreasing order of how recognizable their output
    is to a speaker of the language, not in order of convenience. The
    caller is expected to take the first candidate that is still free, or
    to score the whole list.

    Parameters
    ----------
    name : str
        Administrative unit name. Prefer the local-language spelling; the
        anglicized form yields worse codes.
    pack : LanguagePack, optional
        Vocabulary to use. Resolved from admin1_id when omitted.
    admin1_id : str, optional
        Country identifier, used to look up the language when pack is not
        given.
    lengths : tuple of int, optional
        Code lengths to generate, default both 2 and 3.

    Returns
    -------
    list of Candidate
        Unique candidates, best first, all satisfying CODE_PATTERN.

    Examples
    --------
    >>> [c.code for c in generate_candidates('Haute-Savoie', admin1_id='FR')][:1]
    ['HSA']
    >>> [c.code for c in generate_candidates('Pas-de-Calais', admin1_id='FR')][:1]
    ['PDC']
    """
    if pack is None:
        pack = get_language_pack(admin1_id=admin1_id)

    tokens, significant = tokenize(name, pack)
    if not tokens:
        return []

    out: list[Candidate] = []
    seen: set[str] = set()

    def add(code: str | None, rule: str) -> None:
        if not code:
            return
        code = code.upper()
        if len(code) in lengths and is_valid_code(code) and code not in seen:
            seen.add(code)
            out.append(Candidate(code, rule, len(code)))

    has_preposition = any(pack.is_preposition(t) for t in tokens)
    has_conjunction = any(pack.is_conjunction(t) for t in tokens)

    # A qualifier only counts when it leads the name AND attaches
    # directly to the head. Two cases would otherwise be mishandled:
    # buried in a compound, Alpes-de-Haute-Provence must give AHP not
    # HAL; separated by a preposition, Hauts-de-Seine must give HDS not
    # HSE, because speakers treat that as a prepositional compound.
    leads = tokens and pack.is_qualifier(tokens[0])
    attaches = len(tokens) > 1 and not pack.is_preposition(tokens[1])
    leading_qualifier = tokens[0] if leads and attaches else None
    head = [t for t in significant if t != leading_qualifier]

    if leading_qualifier and head:
        # Keep the qualified and unqualified units visibly paired:
        # Savoie gives SAV, Haute-Savoie gives HSA.
        add(leading_qualifier[0] + head[0][:2], 'qualifier')
        add(leading_qualifier[0] + head[0][0], 'qualifier')
        add(leading_qualifier[0] + head[0][0] + head[0][2:3], 'qualifier.spread')
        add(leading_qualifier[:2] + head[0][0], 'qualifier.alt')

    elif len(significant) == 1:
        word = significant[0]
        add(word[:3], 'name')
        add(word[:2], 'name')
        # Guarded: stripping type words can reduce a name to a single
        # character, and this rule reads word[1] unconditionally. The
        # guard was missing before type-word stripping ever produced such
        # a word, so the crash was latent rather than new.
        if len(word) >= 2:
            add(word[0] + word[1] + word[-1], 'name.ends')
        add(_consonant_skeleton(word)[:3], 'skeleton')
        add(_consonant_skeleton(word)[:2], 'skeleton')

    else:
        first, second = significant[0], significant[1]
        if has_conjunction:
            # Two co-equal names: speakers drop the conjunction.
            add(first[:2] + second[0], 'conjunction')
            add(first[0] + second[:2], 'conjunction.alt')
            add(first[0] + second[0], 'conjunction')
        if has_preposition:
            # One compound name: speakers keep the preposition's letter,
            # which is what produces BDR, PDC, VDM and PDD.
            add(''.join(t[0] for t in tokens), 'preposition')
            add(''.join(t[0] for t in significant)[:3], 'preposition.significant')
            add(first[:2] + significant[-1][0], 'preposition.alt')
        if len(significant) >= 3:
            add(''.join(t[0] for t in significant)[:3], 'initials')
        add(first[:2] + second[0], 'compound')
        add(first[0] + second[:2], 'compound.alt')
        add(first[0] + second[0], 'initials')

    # Late fallbacks. Deliberately last: these carry little signal, and
    # a high share of them at any level is a symptom worth reporting.
    word = significant[0]
    for size in sorted(lengths):
        for combo in combinations(word, size):
            add(''.join(combo), f'any{size}')
    digits = ''.join(c for c in ''.join(tokens) if c.isdigit())
    if digits:
        add(word[0] + digits[:2], 'letter_number')

    return out
