"""Can a group of sibling units be coded in two letters intuitively?

A code is only worth having if a reader can reconstruct it from the name
without being taught. Four mappings meet that bar:

    first two    Somerville -> SO
    acronym      North East -> NE   (initials of the significant words)
    syllables    Somerville -> SM   (onsets of the first two syllables)
    ends         Somerville -> SE   (first and last letter)

Whether a whole sibling group can be covered by these alone is a maximum
bipartite matching question, not a counting one: each unit has a small set
of acceptable codes and they must be distinct. Solving it exactly gives a
deficiency, the number of units that would otherwise need a code nobody
would guess.

Measured across the 3,650 first-level subdivisions in the admin spine, 228
of 239 countries are fully coverable in two letters and only 15 units
(0.4%) are not.

Group size does not predict this, which is the reason the policy is a
measured test rather than a count cutoff. The smallest country that cannot
be covered has 8 subdivisions; the largest that can has 81. The two groups
overlap across the whole range, so no threshold separates them. What does
predict failure is a shared honorific or article across siblings (Saint
Andrew, Saint George, Saint John; Al Butnan, Al Jabal al Akhdar), which
stripping resolves, and genuine crowding in a romanization tradition where
many names begin alike.
"""

import re

import numpy as np
from scipy.optimize import linear_sum_assignment

from openplaces.core.constants import ADMIN_NAME_PREFIXES
from openplaces.io.admin_codes.languages import (
    LanguagePack,
    fold_diacritics,
    get_language_pack,
)

VOWELS = frozenset('AEIOUY')

# Honorifics carry no identifying signal and are dropped when people
# abbreviate, so Saint George is George. Kept separate from the language
# packs' articles because these are cross-linguistic name particles.
HONORIFICS = frozenset(
    {w.upper() for w in ADMIN_NAME_PREFIXES} | {'SAINT', 'ST', 'STE', 'SANTO', 'SAO'}
)

# A group is treated as two-letter-codeable only when every unit can be
# given a distinct intuitive code. Relaxing this admits units whose code
# no reader could reconstruct, which is the cost the whole scheme exists
# to avoid. Exposed as a parameter so the tradeoff can be measured.
DEFAULT_MIN_COVERAGE = 1.0


def syllable_onsets(word: str) -> list[str]:
    """Return the first letter of each syllable, approximated.

    Applies the maximal onset principle: the consonant that opens a
    syllable is the last one before its vowel, so a cluster stays with the
    following vowel rather than being split. Somerville divides as
    So-mer-ville and yields S, M, V. Splitting at the first consonant
    after a vowel instead would yield S, M, R, which is not how the name
    is said or clipped.

    Parameters
    ----------
    word : str
        Folded upper-case word.

    Returns
    -------
    list of str
        Onset letters, first syllable first. The word's first letter is
        always the first onset, whether it is a vowel or a consonant.

    Examples
    --------
    >>> syllable_onsets('SOMERVILLE')
    ['S', 'M', 'V', 'L']
    >>> syllable_onsets('OSLO')
    ['O', 'L']
    """
    if not word:
        return []

    def is_vowel(i: int) -> bool:
        char = word[i]
        if char != 'Y':
            return char in VOWELS
        # Y glides to consonant before a vowel and is one otherwise, which
        # is what recovers BY for Boyaca rather than BC.
        following = word[i + 1] if i + 1 < len(word) else ''
        return following not in VOWELS or following == 'Y'

    onsets = [word[0]]
    for i in range(1, len(word)):
        if not is_vowel(i):
            continue
        # The onset is the consonant immediately before this vowel, unless
        # the vowel is part of a run or opens the word.
        if i > 1 and not is_vowel(i - 1):
            onsets.append(word[i - 1])
    return onsets


def intuitive_codes(
    name: str, pack: LanguagePack | None = None, admin1_id: str | None = None
) -> set[str]:
    """Return the two-letter codes a reader could reconstruct from a name.

    Parameters
    ----------
    name : str
        Administrative unit name.
    pack : LanguagePack, optional
        Vocabulary to use. Resolved from admin1_id when omitted.
    admin1_id : str, optional
        Country identifier, used to look up the language.

    Returns
    -------
    set of str
        Two-letter codes from the four intuitive mappings.

    Examples
    --------
    >>> sorted(intuitive_codes('Somerville', admin1_id='US'))
    ['SE', 'SM', 'SO']
    """
    if pack is None:
        pack = get_language_pack(admin1_id=admin1_id)
    raw = [w for w in re.split(r'[^A-Z0-9]+', fold_diacritics(name)) if w]
    words = [w for w in raw if not pack.is_article(w) and w not in HONORIFICS] or raw
    significant = [
        w for w in words if not pack.is_preposition(w) and not pack.is_conjunction(w)
    ] or words

    # A name with no alphanumeric content yields no tokens at all, which
    # an alias row can easily be. Guarded rather than assumed, because the
    # same missing guard in `candidates` crashed a full build once.
    if not significant:
        return set()

    codes = set()
    head = significant[0]
    if len(head) >= 2:
        codes.add(head[:2])
        codes.add(head[0] + head[-1])
    if len(significant) >= 2:
        codes.add(significant[0][0] + significant[1][0])
    onsets = syllable_onsets(head)
    if len(onsets) >= 2:
        codes.add(onsets[0] + onsets[1])
    return {c for c in codes if re.fullmatch(r'[A-Z][A-Z]', c)}


def intuitive_coverage(
    names, pack: LanguagePack | None = None, admin1_id: str | None = None
) -> tuple[int, int]:
    """Return how many of a sibling group can get a distinct intuitive code.

    Solved as a maximum bipartite matching, so the result is exact and does
    not depend on the order names are supplied.

    Parameters
    ----------
    names : iterable of str
        Names of units sharing a parent.
    pack : LanguagePack, optional
        Vocabulary to use.
    admin1_id : str, optional
        Country identifier, used to look up the language.

    Returns
    -------
    tuple of (int, int)
        Number of units matched, and the total number of units.

    Examples
    --------
    >>> intuitive_coverage(['Alabama', 'Alaska'], admin1_id='US')
    (2, 2)
    """
    names = [n for n in names if str(n).strip()]
    if not names:
        return 0, 0
    if pack is None:
        pack = get_language_pack(admin1_id=admin1_id)

    candidate_sets = [intuitive_codes(n, pack) for n in names]
    codes = sorted({c for s in candidate_sets for c in s})
    if not codes:
        return 0, len(names)

    position = {code: i for i, code in enumerate(codes)}
    cost = np.ones((len(names), len(codes)))
    for row, candidates in enumerate(candidate_sets):
        for code in candidates:
            cost[row, position[code]] = 0.0
    rows, columns = linear_sum_assignment(cost)
    matched = int(sum(cost[r, c] == 0 for r, c in zip(rows, columns)))
    return matched, len(names)


def recommend_code_length(
    names,
    admin1_id: str | None = None,
    iso_length: int | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> tuple[int, str]:
    """Recommend a code length for one sibling group, with the reason.

    A country's own published convention wins outright where one exists,
    since adopting it is what makes a code recognizable. Otherwise the
    group is measured: two letters if every unit can be covered
    intuitively, three if not.

    Parameters
    ----------
    names : iterable of str
        Names of units sharing a parent.
    admin1_id : str, optional
        Country identifier, used to look up the language.
    iso_length : int, optional
        Length of the country's published alphabetic codes, if any.
    min_coverage : float, optional
        Share of units that must be coverable intuitively for two letters
        to be recommended. Default 1.0, meaning all of them.

    Returns
    -------
    tuple of (int, str)
        Recommended length, and a short reason naming what decided it.

    Examples
    --------
    >>> recommend_code_length(['Alabama', 'Alaska'], 'US', iso_length=2)
    (2, 'published convention')
    """
    if iso_length in (2, 3):
        return iso_length, 'published convention'
    matched, total = intuitive_coverage(names, admin1_id=admin1_id)
    if total == 0:
        return 2, 'no names'
    share = matched / total
    if share >= min_coverage:
        return 2, f'intuitive coverage {share:.0%}'
    return 3, f'intuitive coverage only {share:.0%}'
