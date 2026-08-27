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

from openplaces.io.admin_codes.coverage import (
    ARTICLE_PREFIXES,
    SAINT_PREFIXES,
)
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


def tokenize(
    name: str, pack: LanguagePack, drop_saints: bool = False
) -> tuple[list[str], list[str]]:
    """Split a name into all tokens and significant tokens.

    Parameters
    ----------
    name : str
        Administrative unit name, in any script.
    pack : LanguagePack
        Vocabulary deciding which tokens are articles, prepositions or
        conjunctions.
    drop_saints : bool, optional
        Also drop a saint particle (San, Santa, Saint, St). Default
        False: naming a place for a saint is deliberate and speakers
        keep it, so San Francisco is SF. Set True to generate the
        fallback reading a mostly-Saint sibling group needs.

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
    # A leading number set off by punctuation is an ordinal the source
    # prefixed to a real name ("2, Pollocksville" is township 2 of Jones
    # County, named Pollocksville; 357 of the 359 such US subdivisions
    # take this shape), so it is dropped. The punctuation is the tell:
    # a number joined by a space is part of the name, as in Ecuador's
    # "27 De Abril" or "105 Mile Post 2". A name that is nothing but a
    # number keeps it, and `generate_candidates` turns that into a
    # number code rather than a sequential fallback.
    unprefixed = re.sub(r'^\s*\d+\s*[,.;:/]\s*', '', name)
    if unprefixed.strip():
        name = unprefixed
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
    drop = ARTICLE_PREFIXES | (SAINT_PREFIXES if drop_saints else frozenset())
    # A type word is only a legal-form suffix in the trailing position,
    # which is where a source appends it: "Agawam Town", "Archer City".
    # Anywhere else it is part of the name and removing it destroys the
    # name -- District of Columbia became "of Columbia", Town Creek
    # became "Creek", Huntsville Ward 1 became "Huntsville 1". Strip a
    # trailing run only, and never all of it: a name that is nothing but
    # type words keeps them.
    kept = list(raw)
    while len(kept) > 1 and pack.is_type_word(kept[-1]):
        kept.pop()
    tokens = [
        t for t in kept if not pack.is_article(t) and t.upper() not in drop
    ] or kept
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

    # A name that is only a number (a numbered township or district:
    # "1", "12", "3A") carries its number into the code, behind the
    # letter the format requires and zero-padded to the group's width,
    # so US-NC-PAM's township "1" is A01 and sorts with its siblings.
    # Without this the name yields no candidate at all and the unit
    # falls to the sequential pool, whose A00, A01, ... look the same
    # but bear no relation to the number the source gave.
    body = ''.join(tokens).upper()
    if re.fullmatch(r'\d+[A-Z]?', body):
        for width in sorted(lengths):
            if len(body) <= width - 1:
                add('A' + body.zfill(width - 1), 'number')
        return out

    # A name the source numbered ("4, Cumberland") offers its number
    # behind the name's first letters as the next choice after the plain
    # name, so siblings that share the name after the ordinal is dropped
    # (Allegany County, Maryland has several election districts called
    # Cumberland) come out CU4, CU6, C22: still found under the letter a
    # reader searches by, and told apart by the number the source gave.
    ordinal = re.match(r'^\s*(\d+)\s*[,.;:/]\s*\S', name)

    def add_numbered() -> None:
        if not ordinal:
            return
        number = ordinal.group(1)
        for width in sorted(lengths):
            if len(number) < width:
                add(significant[0][: width - len(number)] + number, 'name_number')

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
        add_numbered()

    elif len(significant) == 1:
        word = significant[0]
        add(word[:3], 'name')
        add(word[:2], 'name')
        add_numbered()
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
        add_numbered()

    # Late fallbacks. Deliberately last: these carry little signal, and
    # a high share of them at any level is a symptom worth reporting.
    word = significant[0]
    for size in sorted(lengths):
        for combo in combinations(word, size):
            add(''.join(combo), f'any{size}')
    digits = ''.join(c for c in ''.join(tokens) if c.isdigit())
    if digits:
        add(word[0] + digits[:2], 'letter_number')

    # The saint-dropped reading, ranked below everything above. A saint
    # particle is kept by preference, but a sibling group that is mostly
    # "Saint X" cannot tell its members apart on the particle they share:
    # Dominica's ten parishes came out SA SD SG SI SJ SL SM SP SN ST,
    # every code on Saint's S. Offering both readings lets the assigner
    # keep the particle where it distinguishes and drop it where it does
    # not, instead of the tokenizer deciding for a group it cannot see.
    if any(t.upper() in SAINT_PREFIXES for t in tokens):
        bare = ' '.join(t for t in tokens if t.upper() not in SAINT_PREFIXES)
        if bare:
            for candidate in generate_candidates(bare, pack, admin1_id, lengths):
                add(candidate.code, f'saintless.{candidate.rule}')

    # A reader's strongest cue is the letter the name starts with, so a
    # code carrying it outranks one that does not, whatever rule made
    # it. Carteret County is the case: its own name yields CA, CR, CT
    # and CE, all held by heavier C-named neighbours, and it fell
    # through to AR, a code with no C in it. Preferring first-letter
    # codes does not conjure a free one, but it makes the assigner trade
    # within the group rather than exile one member to an opaque code.
    #
    # A stable partition, so every rule keeps its relative order inside
    # each half and the preference only ever breaks ties between them.
    lead = significant[0][0] if significant and significant[0] else ''
    if lead:
        out = [c for c in out if lead in c.code] + [
            c for c in out if lead not in c.code
        ]

    return out
