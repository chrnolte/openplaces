"""What openplaces says about itself to other people's servers: the
application names, the agent environment variables, the identity
notice, the User-Agent and the identity prompt.
"""

from openplaces.core.constants import (
    VERSION,
)

APPNAME = 'openplaces'
APPAUTHOR = 'placeslab'

# Project page named in the User-Agent, so a data provider reading its
# logs can find out what openplaces is without contacting anyone.
PROJECT_URL = 'https://openplaces.io'

# Environment variables meaning "an AI coding agent is driving this
# run", mapped to the name reported in the User-Agent. Checked in
# order; the first match wins. OPENPLACES_AGENT is the explicit escape
# hatch for an agent this list does not know.
AGENT_ENV_VARS = {
    'OPENPLACES_AGENT': None,  # value is the agent name itself
    'CLAUDECODE': 'claude-code',
    'CLAUDE_CODE': 'claude-code',
    'CURSOR_AGENT': 'cursor',
    'AIDER_MODEL': 'aider',
    'CODEX_SANDBOX': 'codex',
    'GITHUB_ACTIONS': 'github-actions',
}
IDENTITY_NOTICE = f"""\
How should openplaces identify itself?

openplaces downloads from public servers run by other people -- county GIS
portals, state agencies, national statistical offices. Every request it
makes carries a User-Agent naming the project, so an operator seeing
unexpected load knows what the traffic is and has someone to ask.

You are not registering anything and nothing is sent to this project. Pick
any nickname and place you are willing to have appear in a server log; a
work handle and an institution or city is the usual choice. Leave it blank
to stay unidentified.

  openplaces/{VERSION} (+{PROJECT_URL}; ada@some-university)\
"""


def build_user_agent(
    nickname: str | None,
    place: str | None,
    agent: str | None = None,
) -> str:
    """Assemble the User-Agent string for a given identity.

    Shaped like a conventional crawler identity -- product token, then a
    parenthesized comment holding the project URL and a contact handle --
    because that is the form a server operator's log tooling already knows
    how to read.

    Parameters
    ----------
    nickname : str or None
        Self-chosen handle. Never verified and never an email address.
    place : str or None
        Institution, city, or organization the nickname belongs to.
    agent : str or None
        Name of the AI coding agent driving the run, appended so a
        provider can tell autonomous traffic from a person at a keyboard.

    Returns
    -------
    str
        e.g. ``openplaces/0.1.0 (+https://openplaces.io; ada@some-university)``
    """
    nickname = (nickname or '').strip()
    place = (place or '').strip()
    if nickname and place:
        who = f'{nickname}@{place}'
    else:
        # An installation that set neither reports 'unidentified',
        # honest and still better than impersonating a browser.
        who = nickname or place or 'unidentified'

    parts = [f'+{PROJECT_URL}', who]
    if agent:
        parts.append(f'agent: {agent}')
    return f'openplaces/{VERSION} ({"; ".join(parts)})'


def prompt_identity() -> tuple[str, str]:
    """Show the identity notice and ask for a nickname and place.

    Returns
    -------
    tuple of str
        (nickname, place), either possibly empty. An empty nickname skips
        the place question: half an identity is not worth a second prompt.
    """
    print('\n' + '-' * 70)
    print(IDENTITY_NOTICE)
    print()

    nickname = input('Nickname (Enter to stay unidentified): ').strip()
    place = input('Place (university, city, or org): ').strip() if nickname else ''
    print(f'\nRequests will be sent as:\n  {build_user_agent(nickname, place)}')
    return nickname, place
