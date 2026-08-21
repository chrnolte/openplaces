"""
Usage-profile gate for sources whose terms condition access on the user.

Some sources place conditions not on redistribution but on *who may
download at all*: non-commercial use only, a licensed or restricted
computing environment, a declared interest in the source's jurisdiction.
A recipe records such a condition on its `Source` as a
`usage_requirement` -- set deliberately, only when a person has actually
determined one from the source's terms, never inferred from the
free-text `license` field. This module checks that requirement against
the user's self-declared usage profile (`config.get_usage_profile`)
before an automatic download.

The control flow mirrors `io.consent.require_terms_consent`: in-process
memo, then a persisted per-source standing decision, then an interactive
prompt when attended, then an error when unattended and unresolved. One
asymmetry of consent's does not carry over: there is no forbidden
recipe-side lever here, because a recipe recording a `usage_requirement`
is stating a fact about the source, not consenting to anything on the
user's behalf -- the only discretionary call is the user's own, made at
the prompt or via `--set-usage-profile`.
"""

from __future__ import annotations

import sys

from openplaces.config import (
    get_usage_override,
    get_usage_profile,
    set_usage_override,
)

__all__ = [
    'UsageProfileMismatchError',
    'forget_usage_answers',
    'require_usage_compatible',
]

# Prompt answers. 'a' records a standing decision in the user's config;
# 'y' covers only the current process. Anything else, including empty
# input, is a refusal -- the safe direction for a question about
# someone else's terms.
_YES = ('y', 'yes')
_ALWAYS = ('a', 'always')

# Answers given in this process, by source. A recipe partitioned by
# admin unit reaches this gate once per unit, and asking a person the
# same question fifty times teaches them to stop reading it. Standing
# decisions live in the user config instead (`config.set_usage_override`).
_ANSWERED: dict[str, bool] = {}

_PROMPT = """\
Proceed with this download?
  [y] yes, for this run
  [a] yes, and remember for this source (recorded in your openplaces config)
  [N] no, skip this download
"""


class UsageProfileMismatchError(RuntimeError):
    """A source's usage requirement was unmet and no one could be asked.

    Raised instead of returning False when a run cannot ask, so an
    unattended job stops with an explanation rather than silently
    downloading nothing or silently proceeding.
    """


def forget_usage_answers(source: str | None = None) -> None:
    """Drop this process's remembered answers so the next gate asks again.

    Affects only the in-process memo, never a standing decision recorded
    in the user config -- use `config.set_usage_override` to change one
    of those.

    Parameters
    ----------
    source : str, optional
        Forget only this source. None forgets every source.
    """
    if source is None:
        _ANSWERED.clear()
    else:
        _ANSWERED.pop(source, None)


def require_usage_compatible(
    source,
    recipe_id: str | None = None,
    admin_id: str | None = None,
    verbose: bool = False,
) -> bool:
    """Decide whether a source with a usage requirement may auto-download.

    Consulted in order: the requirement itself (absent or empty means
    yes), the declared usage profile, this process's earlier answer, a
    standing decision in the user's config, then the person at the
    keyboard.

    Parameters
    ----------
    source : openplaces.core.schema.Source
        The source whose download is about to run. Its
        `usage_requirement` drives the check; its `source_id` keys the
        remembered answers.
    recipe_id : str, optional
        Fallback key for remembered answers when the source has no
        `source_id`, and named in messages so the reader knows which
        recipe reached the gate.
    admin_id : str, optional
        The recipe's own admin unit, for an `admin_interest` condition.
    verbose : bool, default False
        Report decisions that were made without prompting.

    Returns
    -------
    bool
        True when the download may proceed.

    Raises
    ------
    UsageProfileMismatchError
        When the requirement is unmet, no standing decision exists, and
        no one can be asked because stdin is not a terminal (a cluster
        job, CI, a Snakemake rule). The message names the unmet
        conditions and the `--set-usage-profile` command that declares
        a profile.
    """
    requirement = getattr(source, 'usage_requirement', None)
    if requirement is None or requirement.is_empty():
        return True

    key = getattr(source, 'source_id', None) or recipe_id or str(source)
    label = recipe_id or key

    unmet = requirement.unmet(get_usage_profile(), admin_id=admin_id)
    if not unmet:
        return True

    if key in _ANSWERED:
        return _ANSWERED[key]

    standing = get_usage_override(key)
    if standing is not None:
        if verbose:
            settled = 'proceed' if standing else 'skip'
            print(
                f'  {label}: usage mismatch previously resolved as '
                f'{settled} by this user.'
            )
        return standing

    reasons = '\n'.join(f'  - {reason}' for reason in unmet)
    if sys.stdin is None or not sys.stdin.isatty():
        raise UsageProfileMismatchError(
            f'{label}: this source conditions access on its users, and the '
            'declared usage profile does not meet its conditions:\n'
            f'{reasons}\n'
            'Declare your context with '
            '`python -m openplaces.config --set-usage-profile ...`, or run '
            'this download once interactively and answer [a] to record a '
            'standing decision; unattended runs will then honor it.'
        )

    print('\n' + '-' * 70)
    print(f'{label}: this source conditions access on its users.')
    if getattr(source, 'terms_url', None):
        print(f'Read the terms at: {source.terms_url}')
    print('\nYour declared usage profile does not meet its conditions:')
    print(reasons)
    print(
        '\nWhether the conditions apply to you is your judgment as the '
        "operator\nof this download, not this software's."
    )
    print()
    answer = input(_PROMPT + '> ').strip().lower()

    accepted = answer in _YES or answer in _ALWAYS
    _ANSWERED[key] = accepted
    if answer in _ALWAYS:
        set_usage_override(key, True, reason='; '.join(unmet))
        print(f'Recorded: proceeding with {key}. Change it in your config.')
    elif not accepted:
        print('Not confirmed; skipping this download.')
    return accepted
