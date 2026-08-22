"""Language vocabularies used when abbreviating administrative-unit names.

The vocabularies themselves live in CSV files under the recipe tree, not as
Python literals, for two reasons. Contributors who know a language but not
Python can extend them, and per the repository's conventions geography- and
language-specific data belongs in data files rather than in source.

Three token categories drive abbreviation, and the distinction between them
is grammatical rather than orthographic:

article
    Dropped entirely. Carries no identifying information.
preposition
    Kept in an initialism. In a name like Bouches-du-Rhone the preposition
    binds one compound name, and speakers keep its letter: BDR, not BOR.
conjunction
    Dropped from an initialism. In a name like Eure-et-Loir the conjunction
    joins two co-equal names, and speakers omit it: EUL, not EEL.
qualifier
    A relative or directional modifier that leads a name (Haute-Savoie,
    Nieder-Bayern, Upper Austria). Treated specially so the qualified and
    unqualified units stay visibly paired: Savoie gives SAV and
    Haute-Savoie gives HSA.
type_word
    A word saying what kind of unit this is rather than which one: County,
    Borough, ilce, rayon, -shi. Dropped before a code is derived, because a
    code exists to identify the unit and these identify the class. This has
    been the single most recurrent defect in code generation -- it appeared
    once per language, each time looking country-specific -- which is why
    the vocabulary lives here as data rather than in source.

    The 'und' type words apply to every language, since Wikidata serves
    English labels for units in every country. Language-specific entries
    apply only to their own language: short ones like 'ku', 'fu' or 'qu'
    would eat real names if applied globally.

The preposition/conjunction split is what makes generated codes look native
rather than mechanical, and it reproduces abbreviations that speakers of a
language already use. See the France validation in the accompanying tests.
"""

import unicodedata
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import pandas as pd

_RECIPE_DIR = (
    Path(__file__).parents[2] / 'recipes' / '_all' / 'admin' / 'openplaces' / '2026'
)
_VOCABULARY_CSV = _RECIPE_DIR / 'admin-openplaces-2026_language-vocabulary.csv'
_COUNTRY_LANGUAGE_CSV = _RECIPE_DIR / 'admin-openplaces-2026_country-language.csv'

# Language code for "undetermined", per ISO 639-2. Used when a country
# has no entry in the country-language table, so callers never have to
# special-case a missing language.
UNDETERMINED = 'und'


@dataclass(frozen=True)
class LanguagePack:
    """Token vocabularies for one language.

    Parameters
    ----------
    language : str
        ISO 639-1 code, or 'und' when undetermined.
    articles : frozenset of str
        Tokens dropped entirely before abbreviating.
    prepositions : frozenset of str
        Tokens dropped from the significant-word list but retained in an
        initialism.
    conjunctions : frozenset of str
        Tokens dropped from both the significant-word list and initialisms.
    qualifiers : frozenset of str
        Relative or directional modifiers handled as a leading qualifier.
    type_words : frozenset of str
        Words naming the kind of unit, dropped before a code is derived.
        Includes the universal ('und') set as well as this language's own.
    """

    language: str
    articles: frozenset[str] = field(default_factory=frozenset)
    prepositions: frozenset[str] = field(default_factory=frozenset)
    conjunctions: frozenset[str] = field(default_factory=frozenset)
    qualifiers: frozenset[str] = field(default_factory=frozenset)
    type_words: frozenset[str] = field(default_factory=frozenset)

    def is_article(self, token: str) -> bool:
        """Return True if token is an article in this language."""
        return token.lower() in self.articles

    def is_preposition(self, token: str) -> bool:
        """Return True if token is a preposition in this language."""
        return token.lower() in self.prepositions

    def is_conjunction(self, token: str) -> bool:
        """Return True if token is a conjunction in this language."""
        return token.lower() in self.conjunctions

    def is_qualifier(self, token: str) -> bool:
        """Return True if token is a leading qualifier in this language."""
        return token.lower() in self.qualifiers

    def is_type_word(self, token: str) -> bool:
        """Return True if token names a kind of unit rather than a unit."""
        return token.lower() in self.type_words


def fold_diacritics(text: str) -> str:
    """Strip diacritics and upper-case, leaving ASCII letters and digits.

    Codes are ASCII so they stay safe in file paths on every platform, but
    scoring and matching happen against the original spelling. Names are
    therefore folded only at the point a code is emitted.

    Parameters
    ----------
    text : str
        Text to fold.

    Returns
    -------
    str
        Upper-cased text with combining marks removed.

    Examples
    --------
    >>> fold_diacritics('Haute-Savoie')
    'HAUTE-SAVOIE'
    >>> fold_diacritics('Ardeche')
    'ARDECHE'
    """
    decomposed = unicodedata.normalize('NFD', str(text))
    return ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn').upper()


@cache
def load_language_packs() -> dict[str, LanguagePack]:
    """Return every language pack, keyed by language code.

    Reads the vocabulary CSV once and caches the result. The 'und' pack is
    always present; a language listed in the CSV inherits nothing from it,
    so each pack states its own vocabulary in full.

    Returns
    -------
    dict of str to LanguagePack
        Packs keyed by ISO 639-1 code (plus 'und').
    """
    table = pd.read_csv(_VOCABULARY_CSV, keep_default_na=False)
    by_category: dict[str, dict[str, set[str]]] = {}
    for row in table.itertuples(index=False):
        language = str(row.language).strip().lower()
        category = str(row.category).strip().lower()
        token = str(row.token).strip().lower()
        if not language or not token:
            continue
        by_category.setdefault(language, {}).setdefault(category, set()).add(token)

    universal_types = frozenset(by_category.get(UNDETERMINED, {}).get('type_word', ()))
    packs = {}
    for language, categories in by_category.items():
        packs[language] = LanguagePack(
            language=language,
            articles=frozenset(categories.get('article', ())),
            prepositions=frozenset(categories.get('preposition', ())),
            conjunctions=frozenset(categories.get('conjunction', ())),
            qualifiers=frozenset(categories.get('qualifier', ())),
            # Every pack carries the universal set plus its own.
            type_words=universal_types | frozenset(categories.get('type_word', ())),
        )
    packs.setdefault(UNDETERMINED, LanguagePack(language=UNDETERMINED))
    return packs


@cache
def load_country_languages() -> dict[str, str]:
    """Return the naming language for each country, keyed by admin1_id.

    A country absent from the table is not an error; callers fall back to
    the undetermined pack, which still abbreviates names correctly for
    languages whose particles happen not to appear.

    Returns
    -------
    dict of str to str
        Mapping from admin1_id (e.g. 'FR') to language code (e.g. 'fr').
    """
    table = pd.read_csv(_COUNTRY_LANGUAGE_CSV, keep_default_na=False, dtype=str)
    return {
        str(row.admin1_id).strip().upper(): str(row.language).strip().lower()
        for row in table.itertuples(index=False)
        if str(row.admin1_id).strip() and str(row.language).strip()
    }


def get_language_pack(language: str | None = None, admin1_id: str | None = None):
    """Return the language pack for a language code or a country.

    Parameters
    ----------
    language : str, optional
        ISO 639-1 code. Takes precedence over admin1_id when both are given.
    admin1_id : str, optional
        Country identifier, used to look up the naming language.

    Returns
    -------
    LanguagePack
        The requested pack, or the undetermined pack if it is not known.

    Examples
    --------
    >>> get_language_pack(admin1_id='FR').language
    'fr'
    >>> get_language_pack(admin1_id='ZZ').language
    'und'
    """
    packs = load_language_packs()
    if language:
        return packs.get(language.strip().lower(), packs[UNDETERMINED])
    if admin1_id:
        code = load_country_languages().get(admin1_id.strip().upper())
        if code:
            return packs.get(code, packs[UNDETERMINED])
    return packs[UNDETERMINED]
