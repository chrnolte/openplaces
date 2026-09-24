"""
Clean geographic names so they compare across sources: fold to
ASCII, strip the generic words and prefixes a source adds, and
keep what identifies the unit.
"""

import re
import unicodedata

import pandas as pd

from openplaces.core.constants import (
    ADMIN_GENERIC_WORDS,
    ADMIN_NA_TOKENS,
    ADMIN_NAME_PREFIXES,
)

_ASCII_FOLD_MAP = str.maketrans(
    {
        'Đ': 'D',
        'đ': 'd',
        'Ð': 'D',
        'ð': 'd',
        'Ø': 'O',
        'ø': 'o',
        'Ł': 'L',
        'ł': 'l',
        'Æ': 'AE',
        'æ': 'ae',
        'Œ': 'OE',
        'œ': 'oe',
        'ß': 'ss',
        'Þ': 'TH',
        'þ': 'th',
    }
)


def fold_to_ascii(text):
    """Transliterate text to its closest ASCII representation.

    Strips diacritics via NFKD normalization, maps non-decomposing letters
    (e.g. Đ, Ø, ß) to ASCII equivalents, converts non-ASCII decimal digits
    (e.g. Arabic-Indic) to ASCII digits, and drops anything else non-ASCII.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        ASCII-only version of the input.
    """
    decomposed = unicodedata.normalize('NFKD', str(text).translate(_ASCII_FOLD_MAP))
    chars = []
    for c in decomposed:
        if c.isascii():
            chars.append(c)
        elif unicodedata.combining(c):
            continue
        elif c.isdigit():
            chars.append(str(unicodedata.digit(c)))
    return ''.join(chars)


def clean_geographic_name(
    name,
    *,
    na_tokens=ADMIN_NA_TOKENS,
    prefixes=ADMIN_NAME_PREFIXES,
    generic_words=ADMIN_GENERIC_WORDS,
):
    """Clean an administrative-unit name into structured components.

    Language-agnostic: the vocabulary lists default to the broadened
    English/Spanish sets in `openplaces.core.constants` but can be overridden
    (e.g. per recipe) to support other languages without editing this function.

    Parameters
    ----------
    name : str
        Raw administrative-unit name.
    na_tokens : Iterable[str], optional
        Lower-cased tokens treated as "no name".
    prefixes : Iterable[str], optional
        Leading articles/honorifics stripped from the name.
    generic_words : Iterable[str], optional
        Generic administrative words detected alongside a number.

    Returns
    -------
    tuple
        ``(clean_text, digits, letter_suffix, generic_word)``.
    """
    # Handle None/NA/null cases
    if pd.isna(name) or str(name).strip().lower() in set(na_tokens):
        return '', '', '', ''

    text = str(name).strip()

    # Initialize variables at the start
    extracted_num = ''
    letter_suffix = ''
    detected_generic = ''

    # 1. FIRST: Special handling for "n.a. (1234)" pattern - treat as pure numeric
    na_num_pattern = re.search(r'^n\.?a\.?\s*\((\d+)\)$', text, re.I)
    if na_num_pattern:
        return '', na_num_pattern.group(1), '', ''

    # 2. Special handling: If text outside parens is NA/None,
    # keep only parenthetical content
    na_with_parens = re.match(r'^(none|na|n\.?a\.?)\s*\(([^)]+)\)\s*$', text, re.I)
    if na_with_parens:
        text = na_with_parens.group(2).strip()
    else:
        # 3. Handle other parentheses
        paren_num_match = re.search(r'\((\d+)\)', text)
        if paren_num_match:
            extracted_num = paren_num_match.group(1)
            text = re.sub(r'\s*\(\d+\)', '', text)
        else:
            text = re.sub(r'[()]', ' ', text)

    # 4. Clean up extra whitespace
    text = ' '.join(text.split())

    # 5. Handle remaining "NA" or "None" prefix
    if re.match(r'^(none|na|n\.?a\.?)$', text, re.I):
        text = ''
    else:
        text = re.sub(r'^(none|na|n\.?a\.?)\s+', '', text, flags=re.I).strip()

    # 6. Remove "No." prefix
    text = re.sub(r'\bNo\.?\s+', '', text, flags=re.I).strip()

    # 7. Remove leading articles/honorifics (e.g. The, San, El, La)
    prefix_re = r'\b(' + '|'.join(re.escape(p) for p in prefixes) + r')\b'
    text = re.sub(prefix_re, '', text, flags=re.I).strip()

    # 8. Handle "Division No. X" pattern
    div_pattern = re.search(r'Division\s+No\.?\s+(\d+)', text, re.I)
    if div_pattern and not extracted_num:  # Only set if not already set
        extracted_num = div_pattern.group(1)
        text = re.sub(r'Division\s+No\.?\s+\d+', '', text, flags=re.I).strip()

    # 10. Remove ordinal suffixes
    text = re.sub(r'(\d+)(st|nd|rd|th)\b', r'\1', text, flags=re.I)

    # 11. Handle letter suffixes (e.g., "5o", "3sam")
    num_letter_pattern = re.search(r'(\d+)\s*([a-z]+)$', text, re.I)
    if num_letter_pattern and len(num_letter_pattern.group(2)) <= 3:
        if not extracted_num:  # Only set if not already set
            extracted_num = num_letter_pattern.group(1)
        letter_suffix = num_letter_pattern.group(2).lower()
        text = re.sub(r'\d+\s*[a-z]+$', '', text, flags=re.I).strip()

    # 12. DETECT generic words
    for word in generic_words:
        if re.search(rf'\b{word}\b', text, re.I):
            match = re.search(rf'\b({word})\b', text, re.I)
            if match:
                detected_generic = match.group(1).lower()
                break

    # 13. Special handling for "Subd. X"
    subd_pattern = re.search(r'subd\.?\s+([A-Z0-9]+)', text, re.I)
    if subd_pattern:
        subd_code = subd_pattern.group(1).upper()
        if subd_code.isalpha():
            text = subd_code
            detected_generic = ''
        elif subd_code.isdigit():
            if not extracted_num:  # Only set if not already set
                extracted_num = subd_code
            text = ''
            detected_generic = 'subd'

    # 14. Remove non-alphanumeric
    text = re.sub(r'[\-_\.,]', ' ', text)

    # 15. Extract digits if not already extracted
    if not extracted_num:
        digit_matches = re.findall(r'\d+', text)
        if digit_matches:
            extracted_num = digit_matches[0]
            if len(extracted_num) > 5:
                extracted_num = ''

    # 16. Extract clean text (remove all digits)
    clean_text = re.sub(r'\d+', '', text)
    clean_text = ''.join(re.findall(r'[A-Z\s]', clean_text.upper())).strip()

    # 17. FINAL CHECK: Remove any remaining parentheses
    clean_text = re.sub(r'[()]', '', clean_text)
    letter_suffix = re.sub(r'[()]', '', letter_suffix)

    return clean_text, extracted_num, letter_suffix, detected_generic
