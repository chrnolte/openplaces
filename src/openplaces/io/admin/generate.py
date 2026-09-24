"""
Mint admin ids for one level from names and codes: the waterfall
that io.admin_codes.assign_admin_ids has since replaced for the
spine (see that function), kept for the callers that still use it.
"""

import re
from itertools import combinations

from openplaces.core.constants import (
    STRING_SEPARATOR_WITHIN_IDS,
)
from openplaces.io.admin.names import clean_geographic_name, fold_to_ascii


def generate_admin_ids(
    df,
    new_admin_id_col='admin4_id',
    parent_admin_id_col='admin3_id',
    name_col='name',
    id_separator=STRING_SEPARATOR_WITHIN_IDS,
    name_cleaning=None,
    verbose=False,
):
    """
    Generate unique two-letter admin unit codes within parent units.

    Generate unique admin ID codes for administrative units

    Level-agnostic design: works for any parent-child relationship:
    admin2->admin3 (state->county), admin3->admin4 (county->town)

    Strategy
    --------
    Each name is first cleaned into structured components: a text portion,
    digit portion, letter suffix, and detected generic word. IDs are then
    assigned through a waterfall of prioritized strategies. Each row moves
    to the next strategy only if it remains unassigned:

    0. Pure numeric — If the name reduces to only digits with no text
       (e.g., "N.A. (12)") use the number directly.

    1. Generic word + number — If a recognized generic word (ward, zone,
       barangay, district, etc.) is detected alongside a number, prefix
       the number with the generic word's initial(s). A letter suffix is
       appended if present (e.g., "Ward 3B" → "W3B").

    2. Name + number for duplicates — If the same base name appears
       multiple times under the same parent and a digit is present,
       disambiguate by combining the name's initial(s) with the number
       (and any letter suffix).

    3. Initials from multi-word names — For names with two or more words,
       take the first letter of the first two words
       (e.g., "North East" → "NE").

    4. First two letters — Take the first two characters of the cleaned
       name, assigned only where unique within the parent.

    5. Any two letters — Try all pairwise letter combinations from the
       cleaned name until a unique code is found.

    6. Letter + number combinations — Combine any letter from the name
       with any digit from the name; fall back to "X" + digit if no
       letters exist.

    7. Swapping — If a desired two-letter code is taken by another row,
       check whether that row can be reassigned to an alternative code,
       freeing up the preferred code for the current row.

    8. Three-letter codes — Try the first three letters, then all
       three-letter combinations from the name.

    9. Sequential fallback — Assign codes like X01, X02, … (with a
        letter disambiguator if needed) to any rows that all prior
        strategies failed to place.

    After assignment, all IDs are verified to be non-null and globally
    unique; an exception is raised if either condition is violated.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with administrative unit data
    new_admin_id_col : str
        Name for the new administrative ID column (default 'admin4_id')
    parent_admin_id_col : str
        Column name containing parent admin ID (e.g., 'admin3_id')
    name_col : str
        Column name containing subdivision name
    id_separator : str
        Separator to use in IDs (default ``STRING_SEPARATOR_WITHIN_IDS``)
    name_cleaning : dict, optional
        Overrides forwarded to :func:`clean_geographic_name` (e.g.
        ``{'prefixes': [...], 'generic_words': [...]}``) to support a specific
        language. Defaults to the broadened English/Spanish vocabulary.
    verbose : bool
        If True, prints statistics and other outputs

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by new_admin_id_col with diagnostics column

    Raises
    ------
    ValueError
        If unable to generate unique IDs for all rows

    See Also
    --------
    openplaces.io.admin_codes.assign_admin_ids : the successor, wired into
        every recipe that used to call this function.

    Notes
    -----
    Superseded by
    :func:`openplaces.io.admin_codes.assign_admin_ids`. The waterfall below
    assigns whichever code is free when a unit's turn comes, so its output
    depends on row order and a later sibling can be left with an
    unrecognizable code that an earlier one did not need. The successor
    solves the whole sibling group as an assignment problem instead, which
    is order-independent, and keeps one code width per parent.
    Measured 2026-08-22 on US: codes carrying no signal fall from 48.8%
    to 11.5% of counties under the successor. Kept for reference and for
    reproducing identifiers generated before the switch; do not wire it
    into new recipes.
    """

    admin = df.copy()

    id_source_col = new_admin_id_col + '_source'
    admin[new_admin_id_col] = None
    admin[id_source_col] = None

    # Apply cleaning logic
    cleaned_data = admin[name_col].apply(
        lambda n: clean_geographic_name(n, **(name_cleaning or {}))
    )
    admin['_name_clean'] = cleaned_data.apply(lambda x: x[0].replace(' ', ''))
    admin['_name_words'] = cleaned_data.apply(lambda x: x[0].split())
    admin['_digits'] = cleaned_data.apply(lambda x: x[1])
    admin['_letter_suffix'] = cleaned_data.apply(lambda x: x[2])
    admin['_generic_word'] = cleaned_data.apply(lambda x: x[3])

    admin = admin.sort_values([parent_admin_id_col, name_col]).copy()
    used_ids = set()

    # Priority 0: Pure Numeric Extraction - prefix with X
    if verbose:
        print('Priority 0: Pure Numeric Extraction...')
    mask = (admin['_digits'] != '') & (admin['_name_clean'] == '')
    for idx, row in admin[mask].iterrows():
        candidate = f'{row[parent_admin_id_col]}{id_separator}X{row["_digits"]}'
        if candidate not in used_ids:
            admin.at[idx, new_admin_id_col] = candidate
            admin.at[idx, id_source_col] = 'numeric_only'
            used_ids.add(candidate)

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 1: Generic word + number — prefix number with generic word initial(s),
    # appending any letter suffix (e.g., "Ward 3B" → "W3B")
    if verbose:
        print('Strategy 1: Generic word + number...')
    mask = (
        admin[new_admin_id_col].isna()
        & (admin['_digits'] != '')
        & (admin['_generic_word'] != '')
    )
    for idx, row in admin[mask].iterrows():
        generic_word = row['_generic_word']
        nums = row['_digits']
        letter_suffix = (
            str(row['_letter_suffix']).upper() if row['_letter_suffix'] else ''
        )
        letter_suffix = re.sub(r'[()]', '', letter_suffix)

        prefix = generic_word[0].upper()
        candidate = (
            f'{row[parent_admin_id_col]}{id_separator}{prefix}{nums}{letter_suffix}'
        )

        if candidate in used_ids:
            if len(generic_word) >= 2:
                prefix = generic_word[:2].upper()
                candidate = (
                    f'{row[parent_admin_id_col]}{id_separator}{prefix}{nums}'
                    f'{letter_suffix}'
                )

        if candidate not in used_ids:
            admin.at[idx, new_admin_id_col] = candidate
            admin.at[idx, id_source_col] = 'generic_word_num'
            used_ids.add(candidate)

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 2: Name + number for duplicates — same base name appears
    # more than once under the same parent; combine name initial(s) with
    # the digit (and any letter suffix)
    if verbose:
        print('Strategy 2: Name + number for duplicates...')
    mask = (
        admin[new_admin_id_col].isna()
        & (admin['_digits'] != '')
        & (admin['_generic_word'] == '')
    )

    admin['_needs_number'] = False
    for idx, row in admin[mask].iterrows():
        base_name = row['_name_clean']
        parent = row[parent_admin_id_col]

        same_parent_mask = (admin[parent_admin_id_col] == parent) & (
            admin['_name_clean'] == base_name
        )
        count = same_parent_mask.sum()

        if count > 1:
            admin.at[idx, '_needs_number'] = True

    mask = admin[new_admin_id_col].isna() & admin['_needs_number']
    for idx, row in admin[mask].iterrows():
        words = row['_name_words']
        nums = row['_digits']
        letter_suffix = (
            str(row['_letter_suffix']).upper() if row['_letter_suffix'] else ''
        )
        letter_suffix = re.sub(r'[()]', '', letter_suffix)

        if words:
            prefix = words[0][0].upper()
            candidate = (
                f'{row[parent_admin_id_col]}{id_separator}{prefix}{nums}{letter_suffix}'
            )

            if candidate in used_ids and len(words[0]) >= 2:
                prefix = words[0][:2].upper()
                candidate = (
                    f'{row[parent_admin_id_col]}{id_separator}{prefix}{nums}'
                    f'{letter_suffix}'
                )

            if candidate not in used_ids:
                admin.at[idx, new_admin_id_col] = candidate
                admin.at[idx, id_source_col] = 'name_num_duplicate'
                used_ids.add(candidate)

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 3: Initials from multi-word names
    if verbose:
        print('Strategy 3: Initials from multi-word names...')
    mask = admin[new_admin_id_col].isna()
    if mask.any():
        unassigned = admin.loc[mask]  # work on the subset
        has_multiple_words = unassigned['_name_words'].apply(lambda x: len(x) > 1)

        for idx in unassigned[has_multiple_words].index:
            words = admin.at[idx, '_name_words']
            if len(words) >= 2:
                code = words[0][0] + words[1][0]
                candidate = admin.at[idx, parent_admin_id_col] + id_separator + code
                if candidate not in used_ids:
                    admin.at[idx, new_admin_id_col] = candidate
                    admin.at[idx, id_source_col] = 'initials'
                    used_ids.add(candidate)

    # Strategy 4: First two letters — unique within parent only
    if verbose:
        print('Strategy 4: First two letters...')
    mask = admin[new_admin_id_col].isna() & (admin['_name_clean'].str.len() >= 2)
    if mask.any():
        codes = (
            admin.loc[mask, '_name_clean'].str[:2].str.replace(r'[()]', '', regex=True)
        )
        candidates = admin.loc[mask, parent_admin_id_col] + id_separator + codes
        is_unique = ~candidates.duplicated(keep=False) & ~candidates.isin(used_ids)
        admin.loc[mask & is_unique, new_admin_id_col] = candidates[is_unique]
        admin.loc[mask & is_unique, id_source_col] = 'first2'
        used_ids.update(candidates[is_unique])

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')
        still_unassigned = admin[admin[new_admin_id_col].isna()]
        if len(still_unassigned) > 0:
            print(
                '  Still unassigned:',
                still_unassigned[[name_col, '_name_clean', '_name_words']].head(),
            )

    # Strategy 5: any two letters, trying every pairwise combination
    # from the cleaned name
    if verbose:
        print('Strategy 5: Any two letters...')
    mask = admin[new_admin_id_col].isna()
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        indices = unassigned.index.tolist()
        names_clean = unassigned['_name_clean'].tolist()
        parent_ids = unassigned[parent_admin_id_col].tolist()

        for idx, name_clean, parent_id in zip(indices, names_clean, parent_ids):
            name_clean = re.sub(r'[()]', '', name_clean)
            if len(name_clean) < 2:
                continue

            for c1, c2 in combinations(name_clean, 2):
                code = c1 + c2
                new_id = parent_id + id_separator + code
                if new_id not in used_ids:
                    admin.loc[idx, new_admin_id_col] = new_id
                    admin.loc[idx, id_source_col] = 'any2'
                    used_ids.add(new_id)
                    break

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 6: Letter + number combinations — any letter paired with any digit;
    # falls back to "X" + digit when no letters exist
    if verbose:
        print('Strategy 6: Letter + number combinations...')
    mask = admin[new_admin_id_col].isna()
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        indices = unassigned.index.tolist()
        parent_ids = unassigned[parent_admin_id_col].tolist()
        names_upper = (
            unassigned[name_col]
            .fillna('')
            .map(fold_to_ascii)
            .str.upper()
            .str.replace(r'[()]', '', regex=True)
            .tolist()
        )

        for idx, parent_id, name_upper in zip(indices, parent_ids, names_upper):
            letters = [c for c in name_upper if c.isalpha()]
            numbers = [c for c in name_upper if c.isdigit()]

            found = False
            if letters and numbers:
                for letter in letters:
                    for number in numbers:
                        code = letter + number
                        new_id = parent_id + id_separator + code
                        if new_id not in used_ids:
                            admin.loc[idx, new_admin_id_col] = new_id
                            admin.loc[idx, id_source_col] = 'letter_num'
                            used_ids.add(new_id)
                            found = True
                            break
                    if found:
                        break

            if not found and not letters and numbers:
                for number in numbers:
                    code = 'X' + number
                    new_id = parent_id + id_separator + code
                    if new_id not in used_ids:
                        admin.loc[idx, new_admin_id_col] = new_id
                        admin.loc[idx, id_source_col] = 'x_num'
                        used_ids.add(new_id)
                        break

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 7: Swapping — if a desired code is held by another row, attempt to
    # reassign that row to an alternative code, freeing the preferred code
    if verbose:
        print('Strategy 7: Swapping...')
    mask = admin[new_admin_id_col].isna()
    swaps_made = 0
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        indices = unassigned.index.tolist()
        names_clean = (
            unassigned['_name_clean'].str.replace(r'[()]', '', regex=True).tolist()
        )
        parent_ids = unassigned[parent_admin_id_col].tolist()

        for idx, name_clean, parent_id in zip(indices, names_clean, parent_ids):
            if len(name_clean) < 2:
                continue

            found = False
            for c1, c2 in combinations(name_clean, 2):
                code = c1 + c2
                new_id = parent_id + id_separator + code

                if new_id in used_ids:
                    existing_mask = admin[new_admin_id_col] == new_id
                    if not existing_mask.any():
                        continue
                    existing_idx = existing_mask.idxmax()
                    existing_name_clean = re.sub(
                        r'[()]', '', admin.at[existing_idx, '_name_clean']
                    )
                    existing_parent_id = admin.at[existing_idx, parent_admin_id_col]

                    if len(existing_name_clean) < 2:
                        continue

                    if existing_parent_id == parent_id:
                        swap_found = False
                        for d1, d2 in combinations(existing_name_clean, 2):
                            alt_code = d1 + d2
                            alt_new_id = existing_parent_id + id_separator + alt_code
                            if alt_new_id not in used_ids and alt_code != code:
                                used_ids.remove(new_id)
                                admin.loc[existing_idx, new_admin_id_col] = alt_new_id
                                admin.loc[existing_idx, id_source_col] = 'swapped'
                                used_ids.add(alt_new_id)

                                admin.loc[idx, new_admin_id_col] = new_id
                                admin.loc[idx, id_source_col] = 'any2_after_swap'
                                used_ids.add(new_id)

                                swap_found = True
                                swaps_made += 1
                                break

                        if swap_found:
                            found = True
                            break

            if found:
                break

    if verbose:
        print(f'  Swaps made: {swaps_made}')
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 8: Three-letter codes — first three letters,
    # then all three-letter combinations
    if verbose:
        print('Strategy 8: Three-letter codes...')
    mask = admin[new_admin_id_col].isna()

    if mask.any():
        has_three = admin.loc[mask, '_name_clean'].str.len() >= 3
        if has_three.any():
            codes = (
                admin.loc[mask & has_three, '_name_clean']
                .str[:3]
                .str.replace(r'[()]', '', regex=True)
            )
            candidates = (
                admin.loc[mask & has_three, parent_admin_id_col] + id_separator + codes
            )
            is_unique = ~candidates.duplicated(keep=False) & ~candidates.isin(used_ids)
            admin.loc[mask & has_three & is_unique, new_admin_id_col] = candidates[
                is_unique
            ]
            admin.loc[mask & has_three & is_unique, id_source_col] = 'first3'
            used_ids.update(candidates[is_unique])

    mask = admin[new_admin_id_col].isna()
    unassigned = admin[mask].copy()

    if len(unassigned) > 0:
        indices = unassigned.index.tolist()
        names_clean = (
            unassigned['_name_clean'].str.replace(r'[()]', '', regex=True).tolist()
        )
        parent_ids = unassigned[parent_admin_id_col].tolist()

        for idx, name_clean, parent_id in zip(indices, names_clean, parent_ids):
            if len(name_clean) < 3:
                continue

            for c1, c2, c3 in combinations(name_clean, 3):
                code = c1 + c2 + c3
                new_id = parent_id + id_separator + code
                if new_id not in used_ids:
                    admin.loc[idx, new_admin_id_col] = new_id
                    admin.loc[idx, id_source_col] = 'any3'
                    used_ids.add(new_id)
                    break

    if verbose:
        print(f'  Assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Strategy 9: sequential fallback (X01, X02, ...) with a letter
    # disambiguator if needed
    if verbose:
        print('Strategy 9: Sequential fallback (X01, X02, ...)...')
    mask = admin[new_admin_id_col].isna()
    if mask.any():
        remaining = admin[mask].groupby(parent_admin_id_col)
        for parent_id, group in remaining:
            for i, idx in enumerate(group.index, start=1):
                code = f'X{i:02d}'
                new_id = parent_id + id_separator + code
                counter = 1
                while new_id in used_ids:
                    code = f'X{i:02d}{chr(64 + counter)}'
                    new_id = parent_id + id_separator + code
                    counter += 1

                admin.loc[idx, new_admin_id_col] = new_id
                admin.loc[idx, id_source_col] = 'sequential'
                used_ids.add(new_id)

    if verbose:
        print(f'  Final assigned: {admin[new_admin_id_col].notna().sum()}/{len(admin)}')

    # Verify uniqueness
    if admin[new_admin_id_col].isna().any():
        n_missing = admin[new_admin_id_col].isna().sum()
        raise ValueError(f'Failed to assign IDs to {n_missing} rows')

    if admin[new_admin_id_col].duplicated().any():
        n_dupes = admin[new_admin_id_col].duplicated().sum()
        dupes = admin[admin[new_admin_id_col].duplicated(keep=False)][
            [new_admin_id_col, name_col, parent_admin_id_col]
        ]
        raise ValueError(f'Found {n_dupes} duplicate IDs:\n{dupes}')

    # Verify charset: AdminId only accepts uppercase ASCII letters and digits
    is_invalid = ~admin[new_admin_id_col].str.match(
        rf'^[A-Z0-9{re.escape(id_separator)}]+$'
    )
    if is_invalid.any():
        offenders = admin.loc[
            is_invalid, [new_admin_id_col, name_col, parent_admin_id_col]
        ]
        raise ValueError(
            f'Generated {is_invalid.sum()} IDs with invalid characters:\n{offenders}'
        )

    if verbose:
        print('\n✓ All IDs assigned and verified unique!')
        print('\nID Generation Summary:')
        print(admin[id_source_col].value_counts().to_string())

    # Final cleanup - remove any parentheses that might have slipped through
    admin[new_admin_id_col] = admin[new_admin_id_col].str.replace(
        r'[()]', '', regex=True
    )

    # Cleanup temp columns
    admin = admin.drop(
        columns=[
            '_name_clean',
            '_name_words',
            '_digits',
            '_letter_suffix',
            '_generic_word',
            '_needs_number',
        ],
        errors='ignore',
    )
    admin = admin.set_index(new_admin_id_col)

    return admin


#: Spine columns a redistribution-restricted source may not fill: its
#: alternative and native-script spellings. Its own codes are matched
#: by pattern in `_publishable_spine_columns`.
