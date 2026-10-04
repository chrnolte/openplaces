"""The combined land-use label the parcel classifier votes
on (derive_use_classes).
"""

from __future__ import annotations

import pandas as pd

from openplaces.io.harmonizer import HarmonizeState, _register


@_register('derive_use_classes')
def derive_use_classes(
    state: HarmonizeState,
    combined_column: str = 'use_group_combined',
    columns: list[str] | None = None,
    labeled_column: str | None = 'use_group_combined_labeled',
) -> HarmonizeState:
    """Build the combined use_group_combined label from use_group / use_subgroup.

    Source-specific raw use codes are mapped to the openplaces use_group and
    use_subgroup vocabulary upstream, via a ``*-remap.csv`` crosswalk that
    ``link_by_id``'s auto-discovery applies automatically when joining a
    source that ships one (see
    :func:`~openplaces.io.harmonizer.links._apply_remap_csvs`); this step only
    combines the two into the label the parcel land-use classifier groups and
    votes on, so it holds no code vocabulary of its own.

    Falls back to whichever of ``use_group`` / ``use_subgroup`` reached the
    spine when only one did (e.g. a source, like Florida's DOR use code, with
    no subgroup taxonomy to crosswalk) rather than skipping the whole column
    -- per-row as well as per-column, so a row with one field blank (empty or
    whitespace-only, not just ``NaN``) falls back to the other alone instead
    of producing a degenerate ``' | '`` label.

    Parameters
    ----------
    combined_column : str, optional
        Output combined-label column (default ``use_group_combined``).
    labeled_column : str or None, optional
        Second output carrying the same label built from the crosswalked
        columns alone -- the first name in each group -- and left missing
        where only a raw code was available. Pass None to skip it.

        This is what a keyword ruleset should read. Such rulesets match
        English text, so a raw source code reaching one is inert at best
        and a false match at worst; the coalesced column keeps the codes
        because a code is a sound grouping key for cohort statistics even
        when it is meaningless to a text rule.

    columns : list, optional
        The label's parts, joined in order. Each entry is either a column
        name, or a list of alternative column names coalesced per row --
        the first of them that has a value wins, and the rest are ignored
        for that row. Defaults to
        ``[['use_group', 'use_group_code'], ['use_subgroup', 'use_subgroup_code']]``.

        The alternative groups exist because land use arrives in whichever
        column the source happens to populate. A source that ships a raw
        code has it crosswalked into the ``use_group`` / ``use_subgroup``
        vocabulary upstream, so the code contributes nothing extra and is
        skipped; but a source whose code has *no* crosswalk (or whose
        crosswalk does not cover that code) would otherwise contribute
        nothing at all, even though the code is perfectly good grouping
        and voting evidence. Galveston County, TX is the measured case:
        no state category code and no crosswalk for its local codes, but
        98% coverage of ``use_subgroup_code``. Coalescing rather than
        appending is what keeps this from fragmenting cohorts -- where the
        crosswalk did fire, the code adds no new label values.

        Listing ``building_style`` alongside them lets counties whose
        land-use text carries no occupancy signal contribute one from
        their structure description -- Sampson County, NC is the
        motivating case: its land-use column is a *land segment* type
        (homesite / cropland / woodland) that matches no land-use keyword,
        while its style column names the structure and identifies
        thousands of manufactured homes the classifier would otherwise
        never see. Note that the combined label is also the grouping key
        for cohort statistics such as ``footprint_area_log_zscore``, so
        adding a high-cardinality column fragments those cohorts; add one
        only where it earns its place.
    """
    if state.spine is None:
        return state
    spine = state.spine
    groups = [
        [entry] if isinstance(entry, str) else list(entry)
        for entry in (
            columns
            or [
                ['use_group', 'use_group_code'],
                ['use_subgroup', 'use_subgroup_code'],
            ]
        )
    ]
    present = [[c for c in group if c in spine.columns] for group in groups]
    present = [group for group in present if group]
    if not present:
        if state.verbose:
            flat = [c for group in groups for c in group]
            print(f'  derive_use_classes: none of {flat} on spine; skipping.')
        return state

    def _clean(column: str) -> pd.Series:
        cleaned = spine[column].astype('string').str.strip()
        return cleaned.mask(cleaned.eq(''))

    filled: dict[str, int] = {}

    def _coalesce(group: list[str]) -> tuple[pd.Series, pd.Series]:
        """First column of *group* with a value, per row.

        Returns the coalesced part and the part restricted to the group's
        first column -- the crosswalked vocabulary, before any fallback.
        """
        primary = _clean(group[0])
        part = primary
        for column in group[1:]:
            alternative = _clean(column)
            gap = part.isna() & alternative.notna()
            if gap.any():
                filled[column] = int(gap.sum())
            part = part.mask(gap, alternative)
        return part, primary

    # Join whichever parts a row actually has, so a row missing one field
    # falls back to the others alone rather than producing a degenerate
    # ' | ' label -- per row, not just per column.
    def _join(label, part):
        if label is None:
            return part
        both = label.notna() & part.notna()
        only_part = label.isna() & part.notna()
        label = label.mask(both, label.fillna('') + ' | ' + part.fillna(''))
        return label.mask(only_part, part)

    label = None
    labeled = None
    for group in present:
        part, primary = _coalesce(group)
        label = _join(label, part)
        labeled = _join(labeled, primary)

    spine[combined_column] = pd.Categorical(label)
    if labeled_column:
        # The same label with the raw-code fallback withheld. Keyword
        # rulesets match English text ('SINGLE WIDE', 'DUPLEX'), so a raw
        # code reaching them is at best inert and at worst a false match:
        # measured across 45 Eastern NC counties, 17 of 1,941 distinct
        # codes hit a rule, and `03-MFR-CONST(04-COND/TWN/DUP-SFR)` -- a
        # *multi-family* code -- matched the Single-Family rule on its
        # trailing '-SFR', on 9,221 parcels. None fire today only because
        # every county carrying such codes also has a populated crosswalk.
        # Cohort statistics keep the coalesced column: there a code is a
        # perfectly good grouping key, which is what it is good for.
        spine[labeled_column] = pd.Categorical(labeled)
    state.spine = spine
    if state.verbose:
        mapped = int(label.notna().sum())
        print(f'  derive_use_classes: combined {mapped:,d}/{len(spine):,d} parcels')
        for column, count in filled.items():
            print(
                f'    fell back to {column} for {count:,d} row(s) with no '
                f'crosswalked value'
            )
    return state
