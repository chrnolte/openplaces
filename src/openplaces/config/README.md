<!-- Reference for `openplaces.config` and the outbound-request rules,
moved verbatim from AGENTS.md on 2026-10-17. AGENTS.md keeps the three
rules in short form. AGENTS.md's "Maintaining this file" rules apply
here too. -->

# openplaces.config: reference

## Package layout

A package since 2026-09-30: `identity` (User-Agent, identity prompt),
`user_config` (the user's own file, the prompts that fill it, agent
detection), `settings` (`OpenPlacesConfig`), and the package file, which
keeps `cfg`, `_cfg`, `get_config`, `reload_config` and every setter that
reads or rebinds them (consent, personal columns, identity, usage
profile), because the tests patch those names on `openplaces.config` and
the callers look them up there. `python -m openplaces.config` still runs
the command line (`__main__.py`).

`cfg` (singleton `OpenPlacesConfig`) holds directory paths (`data_root`, `dir_core`,
`dir_external`, `dir_heap`, etc.), CRS, and the installation's `identity`.

## Talking to other people's servers

Two rules govern every outbound request, both of them about openplaces not
speaking for its user.

**Identify yourself.** `cfg.user_agent` builds
`openplaces/{version} (+https://openplaces.io; {nickname}@{place})` from the
per-user `identity` block, appending `; agent: claude-code` when
`detect_agent()` finds an AI agent driving the run. Unset reports
`unidentified`. `io.request_headers()` is the single accessor; use it rather
than passing headers by hand, and **never** send a browser User-Agent: a
test (`tests/core/test_request_identity.py`) fails the build on any
`Mozilla/` string in `src/`. The identity is asked for at first use
(`_interactive_setup`) and during `dev.py setup`, which cannot import the
package it is installing and so shells out to
`python -m openplaces.config --set-identity`. That is also why the first-use
prompt triggers on a missing `directories` key rather than a missing file.

**Never agree to terms on the user's behalf.** A source behind a
click-through gate goes through `io.consent.require_terms_consent`, which
asks the operator and raises `TermsNotAcceptedError` when it cannot ask.
A person accepts or nothing is accepted: **`accept_terms: true` in a recipe
raises `ConsentNotDelegableError`**, because a committed public recipe would
bind everyone who runs it to terms they never read: refused outright, not
downgraded to a prompt, so a recipe that reads as though consent were handled
cannot ship. `accept_terms: false` *is* honored: declining only costs a
download, so a recipe author may do it.
A standing "always accept this source" exists but only as an answer given
at the prompt (`[a]`), stored per user in `consent.terms` in their own
config (`config.get_terms_consent` / `set_terms_consent`). An answer is also
remembered per source for the process, so a year-partitioned recipe asks
once. Apply the same asymmetry to any future decision with legal
consequences: a recipe may refuse for a user, never consent for them.

**Respect who a source says it is for.** A source whose terms condition
access on the user (non-commercial only, a restricted environment, a
jurisdictional interest) records that as `usage_requirement` on its
`Source`, set only when a person determined it from the terms, never
inferred from the free-text `license` field. At download time,
`io.usage_profile.require_usage_compatible` checks it against the user's
self-declared profile (`python -m openplaces.config
--set-usage-profile`): a match proceeds silently, a mismatch prompts the
operator (with a rememberable `[a]`, mirroring the consent prompt), and
an unattended unresolved mismatch raises `UsageProfileMismatchError`. A
resolved decline soft-skips the partition like a missing download URL.
Unlike consent there is no forbidden recipe-side lever, because
recording a requirement states a fact about the source rather than
deciding anything for the user. Restrictions on the redistribution/fee
axis (NHGIS, Shovels, Edgecombe, GADM) stay out of this mechanism
(`redistribution_restricted`, `resale_restricted` and
`io/delivery/terms.py` cover those), and a blanket ban on automated
access is expressed by having no `download_url` at all, not by a
requirement.

`arcgis_rest_scraper` additionally paces itself (`DEFAULT_REQUEST_INTERVAL_S`,
module-level so the whole paging loop is bounded), because backing off only
after a failure means the load that caused it was already applied.
