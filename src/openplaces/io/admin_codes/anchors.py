"""Prior codes a country already uses for its own subdivisions.

Some codes are conventions rather than abbreviations. No rule derived from
the name produces AK for Alaska, AZ for Arizona or CT for Connecticut, and a
generator left to itself reproduces only 30 of the 51 US state codes.
Anchoring on the country's own published code fixes all 51.

The reference table is built from the ISO 3166-2 list committed to this
repository plus a Wikidata harvest, both permissively licensed. It also
records that code length is a national convention rather than a universal
one: of 151 countries with alphabetic codes, 25 use one character, 96 use
two and 30 use three. Colombia's ANT for Antioquia is not an exception to
accommodate, it is one of thirty such countries.

A caution for anyone using this as ground truth. ISO 3166-2 codes were
assigned by an international body working from romanized names, so they
track those forms by construction. That makes them good evidence of
international legibility and weak evidence of what residents themselves
would choose. Systems chosen locally are the better guide to the latter.
"""

import re
from functools import cache
from pathlib import Path

import pandas as pd

from openplaces.io.admin_codes.languages import fold_diacritics

_RECIPE_DIR = (
    Path(__file__).parents[2] / 'recipes' / '_all' / 'admin' / 'openplaces' / '2026'
)
PRIOR_CODES_CSV = _RECIPE_DIR / 'admin-openplaces-2026_prior-codes.csv'
CONVENTIONS_CSV = _RECIPE_DIR / 'admin-openplaces-2026_code-conventions.csv'
LENGTH_POLICY_CSV = _RECIPE_DIR / 'admin-openplaces-2026_code-length-policy.csv'
OVERRIDES_CSV = _RECIPE_DIR / 'admin-openplaces-2026_code-overrides.csv'

# Administrative type words carried inside official subdivision names.
# They are stripped before matching a name against the reference table,
# because one source writes "Stockholm County" where another writes
# "Stockholm".
TYPE_WORDS = frozenset(
    {
        'PROVINCE',
        'PROVINCIA',
        'PROVINSI',
        'PROVINCIE',
        'REGION',
        'REGIONE',
        'DISTRICT',
        'DISTRITO',
        'RAYONU',
        'RAYON',
        'OBLAST',
        'OBLYSY',
        'KRAY',
        'RESPUBLIKA',
        'REPUBLIC',
        'COUNTY',
        'PARISH',
        'BOROUGH',
        'DEPARTMENT',
        'DEPARTEMENT',
        'DEPARTAMENTO',
        'MUNICIPALITY',
        'MUNICIPIO',
        'STATE',
        'GOVERNORATE',
        'PREFECTURE',
        'CITY',
        'ISLAND',
        'ISLANDS',
        'TERRITORY',
        'LAN',
        'FYLKE',
        'AMT',
        'KOMMUNE',
        'CENSUS',
        'AREA',
        # Turkish, where Wikidata labels a district as "Kas ilcesi" or
        # "Demre (ilce)" while the spine carries the bare name.
        'ILCE',
        'ILCESI',
    }
)


def normalize_name(name: str) -> str:
    """Return a name reduced to its distinctive words, for matching only.

    Folds diacritics, upper-cases, and drops administrative type words so
    that "Stockholm County" and "Stockholm" compare equal. Never use the
    result to build a code; it discards information a code may need.

    Parameters
    ----------
    name : str
        Administrative unit name.

    Returns
    -------
    str
        Space-joined distinctive words, or the folded name if every word
        turned out to be a type word.

    Examples
    --------
    >>> normalize_name('Stockholm County')
    'STOCKHOLM'
    >>> normalize_name('Province of Ontario')
    'OF ONTARIO'
    """
    # Apostrophes and soft signs mark transliteration, not word breaks.
    # Splitting on them turned Dnipropetrovs'k into two tokens while
    # Wikidata holds Dnipropetrovsk, so the two never compared equal.
    folded = re.sub(r"['’ʼ´`]", '', fold_diacritics(name))
    words = [w for w in re.split(r'[^A-Z0-9]+', folded) if w]
    kept = [w for w in words if w not in TYPE_WORDS]
    return ' '.join(kept or words)


@cache
def load_prior_codes() -> pd.DataFrame:
    """Return the world prior-code reference table.

    Returns
    -------
    pandas.DataFrame
        One row per published subdivision code, with country_code,
        subdivision_code, name, name_native, parent_code, code_kind and
        code_length.
    """
    return pd.read_csv(PRIOR_CODES_CSV, dtype=str, keep_default_na=False)


@cache
def load_code_conventions() -> pd.DataFrame:
    """Return the per-country summary of code-length conventions.

    Returns
    -------
    pandas.DataFrame
        One row per country, with n_alpha_codes, modal_length, min_length,
        max_length and share_derivable.
    """
    return pd.read_csv(CONVENTIONS_CSV, keep_default_na=False)


@cache
def load_length_policy() -> pd.DataFrame:
    """Return the reviewed per-country code-length policy.

    Returns
    -------
    pandas.DataFrame
        One row per admin1, with n_units, the ISO code length, measured
        two-character coverage, the recommended_length that governs, and
        the reason it was chosen.

    Notes
    -----
    This is the decision; :func:`load_code_conventions` is only one of the
    inputs to it. Keeping the two apart matters, because a country's ISO
    code length is measured over whatever level ISO happens to enumerate,
    which is not always the level being minted -- ISO 3166-2:GB lists 252
    three-character district codes, so reading a convention off it would
    force three characters onto the four home nations, which two
    characters cover completely.
    """
    return pd.read_csv(LENGTH_POLICY_CSV, keep_default_na=False)


@cache
def load_code_overrides() -> pd.DataFrame:
    """Return codes assigned by hand rather than derived from the name.

    Returns
    -------
    pandas.DataFrame
        Columns admin_id, code, name, and a note recording who decided it
        and when.

    Notes
    -----
    The escape hatch for cases the generator cannot reach on principle,
    not merely cases where its answer is unpopular. The clearest is a
    language whose customary short form is not recoverable from the
    romanised name at all: 전라북도 romanises with "Jeolla-" but contracts
    to 전북, Jeonbuk, so no rule applied to "Jeollabuk-do" yields JB.
    """
    if not OVERRIDES_CSV.exists():
        return pd.DataFrame(columns=['admin_id', 'code', 'name', 'note'])
    return pd.read_csv(OVERRIDES_CSV, keep_default_na=False)


@cache
def get_override_codes(admin1_id: str) -> dict[str, str]:
    """Return only a country's reviewed code overrides, keyed by name.

    :func:`get_anchor_codes` merges these into the published codes, which
    is what callers assigning a code want. This returns them alone, for
    the one caller that needs to tell a reviewed human decision apart
    from an inference off a published scheme.

    Parameters
    ----------
    admin1_id : str
        Country identifier, e.g. 'CO'.

    Returns
    -------
    dict of str to str
        Normalized subdivision name mapped to its overridden code.
    """
    overrides = load_code_overrides()
    country = admin1_id.strip().upper()
    out: dict[str, str] = {}
    for row in overrides.itertuples(index=False):
        if str(row.admin_id).split('-')[0].upper() != country:
            continue
        key = normalize_name(row.name)
        if key:
            out[key] = str(row.code).upper()
    return out


@cache
def get_anchor_codes(admin1_id: str) -> dict[str, str]:
    """Return a country's published subdivision codes, keyed by name.

    Only alphabetic codes are returned; numeric ones carry no name signal
    and are not usable as identifier segments under the letter-first rule.

    Parameters
    ----------
    admin1_id : str
        Country identifier, e.g. 'US'.

    Returns
    -------
    dict of str to str
        Normalized subdivision name mapped to its published code.

    Examples
    --------
    >>> get_anchor_codes('US')['ALASKA']
    'AK'
    """
    table = load_prior_codes()
    rows = table[
        (table['country_code'].str.upper() == admin1_id.strip().upper())
        & (table['code_kind'] == 'alpha')
    ]
    anchors: dict[str, str] = {}
    for row in rows.itertuples(index=False):
        code = str(row.subdivision_code).upper()
        for candidate in (row.name, row.name_native):
            key = normalize_name(candidate)
            if key and key not in anchors:
                anchors[key] = code

    # A reviewed override wins over the published code. This is the seam
    # for short forms no rule can reach: Korean provinces contract by
    # syllable, so Chungcheongbuk-do is Chungbuk and takes CB, which no
    # amount of work on the romanised string would ever produce.
    anchors.update(get_override_codes(admin1_id))
    return anchors


@cache
def get_code_length_convention(admin1_id: str) -> int | None:
    """Return the code length a country uses for its own subdivisions.

    Parameters
    ----------
    admin1_id : str
        Country identifier, e.g. 'CO'.

    Returns
    -------
    int or None
        Modal alphabetic code length, or None when the country publishes
        no alphabetic codes.

    Examples
    --------
    >>> get_code_length_convention('US')
    2
    >>> get_code_length_convention('CO')
    3
    """
    wanted = admin1_id.strip().upper()

    # The reviewed policy governs where it has an entry; the derived
    # convention is the fallback for a country it does not cover.
    policy = load_length_policy()
    row = policy[policy['admin1_id'].astype(str).str.upper() == wanted]
    if not row.empty:
        try:
            return int(float(row.iloc[0]['recommended_length']))
        except (TypeError, ValueError):
            pass

    table = load_code_conventions()
    row = table[table['country_code'].astype(str).str.upper() == wanted]
    if row.empty:
        return None
    value = row.iloc[0]['modal_length']
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
