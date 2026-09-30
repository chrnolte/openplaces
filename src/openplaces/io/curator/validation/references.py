"""Reference classes from a keyword ruleset and the confidence
tier of a reference row.
"""

from __future__ import annotations

import pandas as pd


def class_from_ruleset(
    recipe,
    terms: pd.Series,
    ruleset: str,
    *,
    reviewed_only: bool = False,
) -> pd.Series | None:
    """Reconstruct a class column from raw evidence through a recipe ruleset.

    Applies the same ordered ruleset a curate vote used, so a
    reconstructed column scores identically to one the recipe's
    formatting stage dropped from published output.

    Parameters
    ----------
    recipe : str or dict
        Curate recipe (id or dict) whose sidecar rulesets to read.
    terms : pandas.Series
        Raw label text the class column is derived from.
    ruleset : str
        Filename of the ruleset CSV beside the curate recipe.
    reviewed_only : bool, optional
        Keep only matches whose winning rule is marked reviewed,
        mirroring the recipe's own flag. Nulling unreviewed matches
        after the fact, rather than dropping those rules up front, is
        deliberate: pre-filtering would let a term fall through to a
        later reviewed rule and assert a class the vote never saw.

    Returns
    -------
    pandas.Series or None
        The reconstructed class column, or None when the ruleset cannot
        be located, leaving the caller to omit that source.
    """
    from types import SimpleNamespace

    from openplaces.io.curator.occupancy import load_ruleset, match_ruleset
    from openplaces.recipe import get_recipe_by_id

    if isinstance(recipe, str):
        recipe = get_recipe_by_id(recipe)
    try:
        state = SimpleNamespace(recipe=recipe)
        rules = load_ruleset(state, ruleset)
    except (FileNotFoundError, KeyError):
        return None
    proposal, reviewed = match_ruleset(terms.astype(object), rules)
    if reviewed_only:
        proposal = proposal.where(reviewed)
    return proposal


def reference_confidence_tier(
    frame: pd.DataFrame,
    *,
    count_column: str = 'n_permits_with_occupancy_type',
    mode_pct_column: str = 'occupancy_type_mode_pct',
    mode_column: str = 'occupancy_type_mode',
    matched_via_column: str = 'matched_via',
    id_match_value: str = 'parcel_id_local',
) -> pd.Series:
    """Confidence tier for a reference-label table, high to low.

    Two things separate a strong claim from a weak one: how the
    reference reached the entity (an id join beats an address or point
    match) and whether its records agree (a unanimous mode over at
    least two label-bearing records beats a single uncorroborated one).
    A point match scores in the `addr` tiers, with an address match:
    both locate the parcel rather than naming it, which is why
    `id_match_value` names only the id route.

    Parameters
    ----------
    frame : pandas.DataFrame
        Entity-keyed reference labels carrying the four columns named by
        the keyword arguments (the permit pair-table schema by default).

    Returns
    -------
    pandas.Series
        One of `1_id_strong`, `2_id_weak`, `3_addr_strong`,
        `4_addr_weak`, or `none` where no record names a label.
    """
    n_labels = pd.to_numeric(frame[count_column], errors='coerce')
    unanimous = frame[mode_pct_column].ge(0.999) & n_labels.ge(2)
    by_id = frame[matched_via_column].eq(id_match_value)
    spoke = frame[mode_column].notna()
    tier = pd.Series('none', index=frame.index)
    tier[spoke & ~by_id & ~unanimous] = '4_addr_weak'
    tier[spoke & ~by_id & unanimous] = '3_addr_strong'
    tier[spoke & by_id & ~unanimous] = '2_id_weak'
    tier[spoke & by_id & unanimous] = '1_id_strong'
    return tier
