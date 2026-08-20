"""
Human consent for third-party terms-of-use gates.

Some sources publish behind a click-through agreement rather than a static
URL, so the scraper that reaches them has to pass a gate reading "I Agree".
Clicking it forms a contract, and openplaces is not a party to it -- whoever
runs the download is.

One rule follows, and `require_terms_consent` enforces it: **a person
accepts, or nothing is accepted.** There is no accepting default anywhere,
and no recipe can grant consent, because a recipe is committed and public --
setting a flag in one would bind everyone who ever runs it to terms they
never read. A standing "always accept this source" is available, but only
as a choice a person makes at the prompt, recorded in their own config.

A scraper facing a gate calls `require_terms_consent` before clicking, and
skips the click when it returns False.
"""

from __future__ import annotations

import sys

from openplaces.config import get_terms_consent, set_terms_consent

__all__ = [
    'ConsentNotDelegableError',
    'TermsNotAcceptedError',
    'forget_terms_consent',
    'require_terms_consent',
]

# Prompt answers. 'a' records a standing decision in the user's config;
# 'y' covers only the current process. Anything else, including empty
# input, is a refusal -- the safe direction for a question about a
# contract.
_YES = ('y', 'yes')
_ALWAYS = ('a', 'always')

# Answers given in this process, by source. A recipe partitioned by month
# asks its scraper once per month, and asking a person the same question
# twelve times teaches them to stop reading it. Standing decisions live in
# the user config instead (`config.set_terms_consent`).
_ANSWERED: dict[str, bool] = {}

_PROMPT = """\
Accept the terms of use?
  [y] yes, for this run
  [a] yes, and remember for this source (recorded in your openplaces config)
  [N] no, skip this download
"""


class TermsNotAcceptedError(RuntimeError):
    """A source's terms gate was reached without a human accepting it.

    Raised instead of returning False when a run cannot ask, so an
    unattended job stops with an explanation rather than silently
    downloading nothing or silently agreeing.
    """


class ConsentNotDelegableError(ValueError):
    """A recipe tried to accept a source's terms on its users' behalf.

    A recipe-configuration error, not a runtime condition: consent to
    someone else's contract cannot be delegated to a committed, public
    file, so `accept_terms: true` is refused outright rather than quietly
    downgraded to a prompt. Declining (`false`) is still a recipe's to
    make.
    """


def forget_terms_consent(source: str | None = None) -> None:
    """Drop this process's remembered answers so the next gate asks again.

    Affects only the in-process memo, never a standing decision recorded
    in the user config -- use `config.set_terms_consent` to change one of
    those.

    Parameters
    ----------
    source : str, optional
        Forget only this source. None forgets every source.
    """
    if source is None:
        _ANSWERED.clear()
    else:
        _ANSWERED.pop(source, None)


def require_terms_consent(
    source: str,
    terms_url: str | None = None,
    accept_terms: bool | None = None,
    verbose: bool = False,
) -> bool:
    """Decide whether a scraper may accept a source's terms of use.

    Consulted in order: a recipe's refusal, this process's earlier answer,
    a standing decision in the user's config, then the person at the
    keyboard. Only the last two can produce acceptance.

    Parameters
    ----------
    source : str
        Human-readable name of the source whose gate was reached, shown in
        the prompt and used as the key a standing decision is recorded
        under (e.g. a recipe id or portal name).
    terms_url : str, optional
        Where the terms can be read. Printed so the person answering can
        read what they are agreeing to before answering.
    accept_terms : bool or None, default None
        A recipe may pass False to declare that this gate must never be
        accepted, which is honored -- refusing has no legal consequence.
        True raises: a committed recipe cannot consent for the people who
        run it.
    verbose : bool, default False
        Report decisions that were made without prompting.

    Returns
    -------
    bool
        True when the scraper may accept the gate.

    Raises
    ------
    ConsentNotDelegableError
        When *accept_terms* is True. Refused outright rather than
        downgraded to a prompt, so a recipe that reads as though consent
        were handled cannot ship.
    TermsNotAcceptedError
        When nothing has been accepted and no one can be asked, because
        stdin is not a terminal (a cluster job, CI, a Snakemake rule). The
        message says to run the download interactively once, which is the
        only way consent is ever created.

    Notes
    -----
    The asymmetry between True and False is deliberate. Declining is a
    decision a recipe author can reasonably make on a user's behalf,
    because its worst outcome is a skipped download. Accepting binds the
    person running it to someone else's contract, so it has to come from
    that person.
    """
    if accept_terms is False:
        if verbose:
            print(f'  {source}: terms of use declined by recipe configuration.')
        return False

    if accept_terms is True:
        raise ConsentNotDelegableError(
            f'{source}: `accept_terms: true` is not allowed. A recipe cannot '
            'accept terms of use on behalf of whoever runs it -- it is '
            'committed and public, so it would bind every future user to '
            'terms they never read.\n'
            'Remove the key from the recipe. Each user records their own '
            'decision by answering [a] at the prompt, which unattended runs '
            'then honor. `accept_terms: false` remains available to declare '
            'that a gate must never be passed.'
        )

    if source in _ANSWERED:
        return _ANSWERED[source]

    standing = get_terms_consent(source)
    if standing is not None:
        if verbose:
            settled = 'accepted' if standing else 'declined'
            print(f'  {source}: terms of use previously {settled} by this user.')
        return standing

    if sys.stdin is None or not sys.stdin.isatty():
        raise TermsNotAcceptedError(
            f'{source} is published behind a terms-of-use gate, and this run '
            'cannot ask anyone to accept it.\n'
            + (f'Read the terms at: {terms_url}\n' if terms_url else '')
            + 'Run this download once interactively and answer [a] to record '
            'your decision; unattended runs will then honor it.\n'
            'There is no recipe setting that accepts on your behalf.'
        )

    print('\n' + '-' * 70)
    print(f'{source} is published behind a terms-of-use gate.')
    if terms_url:
        print(f'Read the terms at: {terms_url}')
    print(
        '\nAccepting is your decision as the operator of this download, not\n'
        "this software's. openplaces will not agree on your behalf."
    )
    print()
    answer = input(_PROMPT + '> ').strip().lower()

    accepted = answer in _YES or answer in _ALWAYS
    _ANSWERED[source] = accepted
    if answer in _ALWAYS:
        set_terms_consent(source, True)
        print(f'Recorded: {source} terms accepted. Change it in your config.')
    elif not accepted:
        print('Not accepted; skipping this download.')
    return accepted
