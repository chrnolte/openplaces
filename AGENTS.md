# AGENTS.md

This file provides guidance to AI coding agents working with code in the `openplaces`
repository.

## Maintaining this file
- Give each entry depth proportional to its surprise-factor (how far actual behavior
  departs from what the name implies), not to how recently or thoroughly it was just
  implemented.
- When adding detail to one entry in a list, check sibling entries -- don't leave
  newly thin-by-comparison entries as an unintended side effect.
- Verify claims against current code before writing them; this file should describe
  what the code does now, not a historical narrative of how it changed.

## Privacy: no personal data, ever
- Never commit real personal information into any tracked file: source code, recipe
  YAML, tests, fixtures, notebooks (including cell outputs), or docs. This means no
  real names, addresses tied to a named individual, phone numbers, personal emails,
  government ID numbers, or similar identifiers of identifiable individuals — for any
  dataset, in any jurisdiction, ever. This applies even to a single illustrative value
  copied from a real record for a docstring, test, or commit message.
- Recipes describe a source's *schema* (column names, field codes) — never its
  *records*. Example values in docstrings, tests, or notebook markdown must be
  fabricated, not sourced from real data.
- Notebooks are committed with outputs stripped (`nbstripout`, wired via
  `.gitattributes`/`dev.py`) specifically because raw pipeline output can carry
  personal fields from upstream sources (e.g. an owner name column) — never disable
  or bypass this for a notebook that touches entity data.
- If you find real personal data already committed, stop, do not build on top of it,
  and flag it to the user immediately. Do not "fix forward" quietly — purging it from
  git history is a decision for a human, not an agent.
- `docs/5_contribute/no-personal-data.rst` is the contributor-facing version of
  this section: what counts as a record, how to fabricate a fixture that still
  tests something, and the failure modes a history rewrite hits.
- **Personal columns in data (not in git) are a per-user choice, and never
  leave in a delivery.** Owner, grantor and grantee attributes
  (`core.attribute_registry.is_personal_attribute`, the one definition) are
  ingested, because sources carry them. A curate recipe's
  `keep_registered_columns` step drops them from curated tables unless the
  person running it set `python -m openplaces.config --keep-personal-columns
  true` in their own config (a research use: telling a sale within a family
  from one at arm's length where the source states no relationship). No
  recipe key can switch that on, for the reason a recipe cannot accept terms.
  `export_delivery` withholds these columns from every file whatever the
  setting and raises `PersonalColumnError` on a recipe that lists one under
  `share`. The same step is an allow-list against *unregistered* source
  headers, which are dropped either way: Wisconsin's RETR is ingested with
  78 raw columns, among them agent, preparer and tax-bill names that no
  deny-list would have named.
- See `DISCLAIMER.md` for the project's broader privacy and liability posture.

## Intellectual property: one notice, in one place
- Copyright is held by the Trustees of Boston University and Christoph Nolte
  (maintainer's decision, 2026-09-20), and the work is licensed under
  Apache-2.0. `NOTICE` carries the copyright lines and is the single source
  of truth; `LICENSE.md` is the verbatim licence text and must stay verbatim;
  `README.md`, `DISCLAIMER.md` section 8, `docs/conf.py` and `CITATION.cff`
  repeat the holders and have to change together with `NOTICE`.
- Never add a "Copyright <name>" header to a new file: files carry no
  per-file notice, except third-party code, which keeps its upstream header
  (see the next section). Never edit copyright, ownership or hosting language
  in any of the files above without asking the user first, even to "fix" or
  "simplify" it: a wrong assertion here is a legal problem, not a style one.
- Contributions come in under the Developer Certificate of Origin (`DCO.md`,
  `CONTRIBUTING.md`): every commit by a contributor other than the
  maintainer carries a `Signed-off-by` line (`git commit -s`). The
  maintainer's own commits carry none: the sign-off certifies a
  contributor's right to submit work to the project, which a copyright
  holder has no one to certify to. Never suggest that the maintainer add
  one, and never list a missing sign-off on the maintainer's commits as an
  open item. An agent never writes a sign-off for a person and never adds
  one in its own name; the sign-off is the human committer's statement, in
  the same way co-authorship is reserved for people.
- The repository is hosted under a personal GitHub account, not an
  institutional one. Don't propose or perform an org transfer, mirror, or
  similar hosting change unprompted.

## Restricted-licence sources: the terms decide what may be committed

- **A connector is not a secret.** That a research group holds a licence
  to a private dataset, and that this project can read it, is ordinary
  and may be said in public. A recipe naming such a source, and a
  pipeline step that links to it, are acceptable in git on the same
  footing as any other recipe, subject to the rule below.
- **What may not be committed is anything the source's terms do not
  permit, or that evidences a use they do not permit.** Before committing
  a recipe for a source whose licence restricts redistribution,
  derivative works, or disclosure, read the terms and check each of the
  following against them:
  - **Schema disclosure.** A recipe reproduces column names, field codes,
    and value vocabularies. Some licences treat data documentation, or
    "any portion" of the data, as confidential or as non-redistributable.
    If the terms can be read that way, keep the recipe (and any sidecar
    crosswalk that reproduces the source's codes) out of git until the
    licensor, or Boston University's review, has cleared it.
  - **Derivative products.** A licence that forbids "derivative works"
    or "data products based on the Data" forbids publishing anything
    derived from it, and a recipe or pipeline step written so that such a
    product would be published is evidence of intent to do so. Restricted
    evidence may inform an internal check; it must never reach a shipped
    output. Pin that with a test, as
    `test_restricted_shovels_columns_are_never_published` does.
  - **Scope of use.** Internal validation against a licensed dataset is
    a different use from training, redistribution, or commercial
    delivery. Describe the actual use accurately in the recipe's notes,
    and do not describe or configure a use the terms do not cover.
- **Record the restriction on the `Source`** (`license`, `terms_url`,
  `redistribution_restricted: true`) so `bundle_terms` reports it and the
  usage-profile gate (below) can act on it. A recipe that reads as though
  the source were open is the failure mode; a recipe that says plainly
  what the terms are is not. A no-resale clause ("not to be resold",
  common on county assessor downloads) is a different fact: it forbids
  selling, not sharing. Record it as `resale_restricted: true`, which
  the notice reports in its own section, and leave
  `redistribution_restricted` to describe sharing.
- **When in doubt, hold the file, do not delete it.** The question is
  institutional, the answer is not an agent's to assume, and the cost of
  waiting is a file that sits uncommitted for a while. Say so in the
  handoff; never commit to "fix it later."
- **Shovels (`US-NC_property-shovels-2026`) is the worked example.** Its
  terms forbid derivative works "based on or trained on the Data". The
  project uses it only to validate an occupancy classification, and none
  of its data or any column derived from it is published (the test above
  pins that). The recipe was committed on 2026-08-15 and removed from
  the tracked tree on 2026-08-22 (`4bf1588`, git-ignored so it stays
  local) as a precaution while Boston University reviews whether
  publishing its schema description is within the terms. Of the held
  files only the occupancy-type value crosswalk reproduces any part of
  the Data (a field's value vocabulary); it was removed from git history
  on 2026-08-27 (see `plans/shovels-terms-and-the-withheld-connector.md`).
  The recipe YAML and the county-name remaps, which carry no Data, stay
  in history and out of the tree. The `link_by_id` steps that name the
  recipe in the spine recipes are unaffected by that hold. Any further
  history change is a decision for a human, exactly like the
  personal-data rule above: flag it, do not quietly rewrite, and do not
  re-add the files until the review has answered.

## Third-party code: attribute it, check its license
- Porting or adapting code from another project (a GitHub repo, a paper's
  reference implementation): name the source (repo/paper URL), its original
  author(s), and its license **in that file's own header** — not only in a
  README or a sibling file in the same directory. A reader of one file
  shouldn't have to find a different file to learn where its code came from.
- Before merging a port, confirm the upstream license is compatible with this
  repo's Apache-2.0 (permissive licenses — MIT, BSD, Apache-2.0 — are fine;
  copyleft licenses — LGPL, GPL — are not automatically compatible and need a
  NOTICE-file carve-out, not silence). A vendored copyleft subtree with no
  license text or notice marking it as different from the rest of the repo
  is a real gap, not a hypothetical one — check for this rather than
  assuming every ported file already got it right.
- This applies to source code specifically; adapting a data *source* into a
  recipe is a separate, parallel concern — check the source's license and
  redistribution terms the same way before writing the recipe.

## Patent risk: new algorithms in the parcel/property-matching and valuation space
- **"Open-source, non-commercial, public-benefit" is not a legal shield
  against patent infringement in the U.S.** — this is a common and
  understandable misconception, but it's wrong, and a specific, on-point
  precedent says so directly: *Madey v. Duke University*, 307 F.3d 1351
  (Fed. Cir. 2002) held that a university's own research use of a patented
  invention did not qualify for the "experimental use" defense, because it
  still furthered the university's legitimate institutional
  objectives — the defense is limited to use "for amusement, to satisfy
  idle curiosity, or for strictly philosophical inquiry," which a
  maintained, publicly-used research tool is not. Don't let the project's
  mission stand in for an actual legal analysis when a specific technique
  looks close to a known patent.
- **Where risk actually concentrates, based on research done so far**:
  the parcel/property-record-linking space has real, active patent
  holders — CoreLogic/Cotality, Black Knight/ICE Mortgage Technology, and
  First American chief among them — whose claims cluster around four
  specific technique *shapes*, not the general goal of "link/match
  property records" (which itself isn't patentable, only a particular
  claimed method for doing it is):
  1. Geometric neighbor/"community" detection (buffer-enlarge, union,
     reduce a group of parcel boundaries) used to impute or validate a
     *missing address number* by interpolating between neighbors
     (CoreLogic **US10248731B1**, to 2037-03-25; continuation
     **US11061985B2**, whose claimed purpose is validating or correcting
     address information; second continuation US20220067117A1, grant
     status unconfirmed). Claim 1 recites five steps and **two of them
     are worth knowing before writing any geometry that groups
     parcels**: the community is parcels *within a threshold distance*
     with contiguous boundaries, and the border is built by *enlarging*
     each boundary, unioning, then *reducing* it back. A proximity
     threshold is therefore a recited element, which is why a distance
     tolerance is a different proposition from strict adjacency. The
     back half (border-intersection points, a reduced neighbor set, and
     bracketing a missing address number) is what the whole claim is
     for, and `openplaces` does none of it.
  2. A trained machine-learning model used to score match probability
     between two property/record representations (as opposed to a fixed,
     deterministic rule set with no learned parameters).
  3. Detecting records *internally inconsistent* with their own source's
     mapping/address data, grouping them, and normalizing the group
     (CoreLogic **US9298740B2**; all three independent claims start from
     a parcel that *failed verification* against mapping or addressing
     data, which nothing here tests).
  4. A **cascading match that falls through to fuzzy matching**, scores the
     resulting link with a strength indicator, *calibrates that scorer from
     the links it just made*, and then detects and removes an incorrect
     earlier link (Black Knight US10606854B2, in force to 2038). This is
     the shape closest to what `openplaces` already does: it normalizes to
     a comparable form and matches in tiers. **The three independent
     claims differ, and the narrowest reading is not the safe one** (claims
     read from the patent text 2026-09-21; an agent's reading, not legal
     advice). Claim 1 (machine) and claim 15 (method) require the strength
     indicator and its calibration from the links just made (claim 1 also
     the removal of an incorrect link). **Claim 17, the computer-readable
     medium claim and so the one a library is measured against, requires
     none of that back half** (it appears only in dependent claim 18). Its
     elements are: an input record with two attributes; a transformation
     into a comparable form; a first non-fuzzy match on the first
     attribute; on failure a cascade to a second non-fuzzy match on a
     second attribute; **on failure of both, a cascade to a fuzzy matching
     comparison, and a link made on the fuzzy match**. What keeps
     `openplaces` clear of claim 17 is therefore the *front* half: **no
     record-linking path falls through from exact key passes to a fuzzy
     comparison.** `link_by_id`, `link_entities_by_id`, the stacked-units
     passes, the transaction lane's parcel and address keys and
     `assign_entity_ids` are exact matches on normalized keys, and a row
     no exact pass reaches stays unlinked. `rapidfuzz` is used in two
     places, neither a fall-through of that kind: `reconcile_addresses`
     compares two address columns *already on one row* to decide whether
     two sources agree (no record is linked by it), and
     `io.curator.validation.link_points_to_entities` matches validation
     points by house number plus fuzzy street name *first* and falls back
     to distance, the reverse order, for scoring only. **Do not add a fuzzy
     tier behind exact key passes in any linking step** (parcel, property,
     transaction, address), do not add link-confidence scoring that learns
     from its own past links, and do not add an unlink-on-reconsideration
     step, without a specific check against that patent and the
     maintainer's decision.
  A new feature that does one of these four things, in this domain,
  deserves a specific check against that sub-area before merging — not a
  general "we checked patents once" assumption. `openplaces`'s existing
  `parcel_id_local`/`geo_id` matching is deterministic string/geometry
  fingerprinting with no learned parameters and no neighbor-comparison or
  address-imputation step, which is why it reads as a different mechanism
  from all four shapes above — that reasoning doesn't automatically carry
  over to a new ML-based imputation or inference feature, which may
  resemble shape 2 much more closely by design. If ML matching is ever
  added, note that every independent claim of the shape-2 patent
  (US11372900B1) needs *two* trained models — one scoring record-pair
  matches, a second identifying a "context" that then selects the cleansing
  rules. A single match-scoring model does not read on it; adding the
  context model and context-selected rules is what would.
- **Process for a new imputation/inference/matching/valuation feature
  touching parcel, property, or transaction data**: (1) identify the
  specific technique, not just the goal, and check whether it resembles
  one of the four shapes above or another known patent in this space;
  (2) if it does, flag it to the user explicitly before merging — this is
  a judgment call for a human, not something an agent should silently wave
  through, the same posture as the IP-ownership section above — **but
  count the differing steps before flagging anything** (maintainer's
  rule, 2026-09-22). Infringement needs *every* element of a claim; one
  element absent is enough to be clear of it. The maintainer set four
  levels, in these words:

  > Three steps different: silent. Two steps: report but don't ask for
  > permissions. Moving from two to one steps: warn. Removing the last
  > step: forbidden

  So three or more differing steps is not a close call and saying so
  only wastes attention; two is reported in passing, not escalated into
  a blocking question; going from two to one is the point to warn,
  because one further change would close the gap; and **taking the count
  to zero is forbidden outright**, not warned about, because every
  element present is infringement. The floor is the part of the rule
  that protects anything, so never treat the warn level as the top of
  the scale.
  Count against the claim text, not against a paraphrase, and say which
  steps you counted. **Count against every independent claim, and let
  the one with the fewest absent steps decide**, because clearing one
  claim clears nothing on its own: US10606854B2's claim 17 drops the
  whole score/calibrate/unlink back half that claims 1 and 15 recite, so
  a change measured only against claim 1 would read as three steps clear
  while sitting one step from claim 17. For a library, the
  computer-readable-medium claim is usually the broadest and is the one
  to count first. Worked example: a strict contiguity gate on a lot
  union differs from US10248731B1 claim 1 on the community-and-threshold
  step, the border-intersection step, the neighbor-set step and the
  address-bracketing step, so it needed no warning at all, and the 2026-09-22
  stop on it was over-cautious by this rule; (3) where
  more than one technically valid approach exists, prefer the one that is
  most clearly mechanistically different from a known patented approach —
  this is not purely defensive: a genuinely distinct method is also a
  stronger, more citable methodological contribution for a research
  project, so the incentive runs the same direction as the science; (4)
  document the technical rationale for a new method in the code itself
  (why this approach, not a more obvious alternative) — ordinary good
  practice that also creates a contemporaneous record of independent
  development.
- **Publishing openly and promptly is itself a protective strategy, not
  just a defensive one** — a clearly dated, technically detailed
  description of a novel method (in code, docs, or a paper) becomes prior
  art that keeps the technique in the public domain and available to
  everyone, rather than leaving room for someone else to patent it later
  and assert it against future users of this or a similar tool. This is
  directly aligned with the project's own public-benefit mission, not a
  tradeoff against it.

## Code style
- Line length:
  - 88 characters maximum for code. Use ``ruff format`` to enforce.
  - 72 characters maximum for comments.
- Use the `|` union syntax for `isinstance` checks, never a tuple:
  `isinstance(x, Foo | Bar)` ✓
  `isinstance(x, (Foo, Bar))` ✗
- Do not use sequences of lines (`─`, `-`, `=`) in comments
- Do not use em-dashes (literal `—` or text-based `---`) in documentation, docstrings, or code comments. Use commas, colons, parentheses, or spaced hyphens (` - `) instead.

## Docstrings
- Use NumPy-style docstrings exclusively throughout the codebase (no Google-style docstrings).
- Avoid double backticks.
- Only add Sphinx cross-references when they are genuinely useful for navigation, e.g:
  - Important public functions or classes in this package
  - Key workflow entry points
  - Custom data structures or configuration objects
- Do not include the .py filepath in the top-level docstring of scripts
- Provide detailed individual docstrings for each class constructor detailing their specific parameters and validation rules in NumPy style (no placeholder or duplicate constructor docstrings).
- Keep docstrings strictly accurate and formatted to the function, resolving copy-paste typos.
- Standardize on American English spelling ('meter', 'center', 'reproject') throughout all comments and docstrings.
- Retain deep technical/algorithmic rationale and historical context in comments and docstrings to help developers understand why decisions were made.
- Keep docstrings clean and focused on user-facing API contracts, moving implementation notes (like psutil or RAM heuristics) to internal inline comments.
- **Quote a statistic that can move; do not avoid it** (maintainer's
  rule, 2026-09-22). A number is the evidence for a method, so a
  docstring states it in one form, on one line, so that it can be found,
  dated and flagged when the data it was measured on is rebuilt:
  `Measured <YYYY-MM-DD> on <scope>: <claim>`, where the scope is an
  admin id, a region id or another single token (`US`, `CO`,
  `cheer-coastal-tx`, `admin-gadm-4.1`). `python -m
  openplaces.flow.measured_claims` lists every claim in the tree and
  reports `stale` where the scope's curated output or delivered bundle
  is newer than the claim's date, `current` where it is not, and
  `no build` where the scope resolves to nothing on this machine. A
  stale number is documentation debt to re-measure, never a reason to
  delete the number.

## Directory structure
Write all plans to ``<repository_root>/plans/``. Never write a plan outside the
repository -- not to a home directory, and not to whatever scratch, notes, or
plan directory your agent tool manages for itself: plans there are invisible to
everyone else, are not covered by this repository's conventions, and have to be
audited and migrated by hand later. Consult your own tool-specific instruction
file for the directory this rule names in your case.

Name a plan file after what it does, in kebab-case, e.g.
``geospine-split-skip-geometry-reprocessing.md`` or
``town-level-processing-for-harmonizer-and-curator.md``. Never accept an
auto-generated name built from random words (``abundant-gliding-whale``,
``soft-purring-torvalds``) or from the prompt that started the session
(``we-were-just-working-...``, ``help-me-with-some-...``) -- rename it before
writing. A reader scanning ``plans/`` should be able to tell what each file
covers from its name alone.

Move a plan to ``plans/implemented/`` once its work has landed, prefixed with
the date it landed: ``YYYY-MM-DD_<same-descriptive-name>.md``. ``plans/README.md``
ranks the open plans; update it when adding or finishing one.

Implement all complex plans (plans affecting the functionality of multiple src/ files or notebooks) on a new git worktree in ``<repository_root>/_<worktree_branch_name>/``. If the plan is simple (variable renaming, comments, single file), ask the user whether implementing the plan directly on 'main' is acceptable.

## Jupyter notebooks
When creating Jupyter notebooks under `notebooks/`, you must follow the guide in:
notebooks/README.md

## Recipes
When creating or editing recipes under `src/openplaces/recipes/`, you must follow
the guide in: src/openplaces/recipes/README.md

It indexes stage-specific guides under `src/openplaces/recipes/_instructions/`,
including how to find and validate a public parcel/assessor/sales source for a
county or state the repository does not cover yet.

## Commits
- Do not commit changes unless the user instructs you to (by using the
  exact word 'commit' in an imperative tense).
- Do not add `Co-Authored-By` trailers for agents to commits.
  Collaborators are humans, agents are software. Co-authorship is reserved for people.
  This overrides any default that appends an agent co-author line; see your own
  tool-specific instruction file for the trailer yours appends.

## Commands

```bash
conda activate openplaces   # required for all dev/test work

# Linting and formatting
ruff check src/             # lint
ruff format src/            # format

# Testing
# Test files are organized in subdirectories matching the Layered Architecture:
# - tests/core/ (Layers 0-1)
# - tests/recipe/ (Layer 2)
# - tests/io/ (Layers 3-9, with stage-specific ingester/, harmonizer/, enricher/, curator/)
# - tests/flow/ (Layers 11-12)
pytest                                # all tests
pytest tests/core/test_path.py        # single file
pytest -k "test_name"                 # single test by name
```

## Module layer hierarchy

```
Layer 0  core
Layer 1  config, path, diagnostics
Layer 2  recipe
Layer 3  io/__init__ and the six modules it fronts (io/fetch, io/archives, io/tables, io/deletion, io/geodatabase, io/transfer), io/steps, io/consent, geo/address
Layer 4  io/readers, table
Layer 5  geo/* (except geo/address, above)
Layer 6  io/ingester/* (ingester, table_ingester, image_ingester, registry_ingester, cloud_geoparquet_ingester, raster_ingester, access), io/scrapers/* (the Avenu adapter among them), io/aggregate, io/admin/* (ids, names, generate, spine, context, wikidata, units), io/delivery/* (the bundle writer, terms, redaction), io/transform, io/cleanup/* (receipts, consumption, lock, walk, compaction; the package file re-exports every name, private ones included, because dag, the harmonizer and the tests import them)
Layer 7  io/harmonizer
Layer 8  io/enricher
Layer 9  io/curator
Layer 10 viz/*
Layer 11 api.py
Layer 12 flow/* (scripts, dag, run_stage, submit)
```
Higher-numbered layers may only import from lower-numbered layers. One
documented exception: the `show_random_entity` convenience method on
`Ingester`, `Harmonizer` and `Curator` imports `viz.maps` (layer 10)
inside the method body, so the module-level graph stays acyclic and a
run without matplotlib never pays for it. Do not add a second one.

Four placements are worth knowing. `table` holds the registry-driven row
helpers (`aggregate_rows`, `add_unique_suffix`, the `join_nonnull_*`
functions) and `summarize_conflicts`, the per-row disagreement summary
both the harmonize and curate stages write into `{col}_conflict`
columns; they sit below `geo/` because `geo/crosswalk` and `geo/ids`
call them, and keeping them in `io/aggregate`/`io/transform` created a
module-level import cycle. `io/aggregate` and `io/transform` re-export
them, so the older import paths still work. `geo/address` is listed
separately because it depends on nothing above `recipe`, which is what
lets `table` use `strip_unit_suffix` without reintroducing that cycle.
`io/steps` holds the step-registry decorator and the lazy loader that
harmonize, enrich and curate each carried a copy of until 2026-09-22; it
imports nothing from openplaces, so it sits below all three, and each
stage keeps its own `_STEP_REGISTRY` dict (tests monkeypatch those by
name). `io/__init__` is a facade: everything it exports is defined in
one of the six modules beside it, and the facade only re-exports, so a
helper that several stages share lives in one of those modules, never in
a stage folder, because the layer-3 facade cannot import from layer 6.
**A module in `io/` must not share a name with a facade export**:
`from openplaces.io import download` yields the *function*, which is
why the module that defines it is `io/fetch`. A test that binds a
module object and patches a name on it patches the module the function
under test lives in, not the facade or a package file; a patch on those
never reaches a function looking names up in its own globals.

## Architecture overview

`openplaces` integrates property and geospatial data (parcels, buildings, transactions, environmental datasets) for reproducible research. All data access and processing is driven by **recipes** (YAML files).

### Core schema (`core/schema.py`)

Four key identifiers form the naming system used throughout the codebase:

- **`AdminId`** — hierarchical geographic identifier, e.g. `AdminId('US', 'NC', 'CUR')`. Level 0 = global, 1 = country, 2 = state/region, 3 = county/district (a town in New England, where the county is skipped: Somerville is `US-MA-SOM`), 4 = municipality/town. String form uses `-` separator: `'US-NC-CUR'`.
- **`Entity`** — a data entity defined by `entity_type` + `source` + `version`, e.g. `Entity('parcel', 'massgis', '2025')`. String form: `'parcel-massgis-2025'`. Valid entity types are in `ENTITY_TYPES` (parcel, building, footprint, property, transaction, admin, ...).
- **`DataSet`** — a non-entity dataset defined by `Theme` + `source` + `version`, e.g. `DataSet('land-elevation', 'usgs', '3dep')`. Themes are hierarchical, separated by `-`, with the top level constrained to `TOP_LEVEL_THEMES` (land, landcover, water, built, people, risk, ...).
- **`Source`** — a data source with download URL(s), portal URL, DOI, etc.

### Entity model and stage roles (`core/schema.ENTITY_DEFINITIONS`)

**One row of each entity type means one thing, and these are easy to
confound.** `ENTITY_DEFINITIONS` holds the authoritative text (a test pins
that every type has one); in short:

| entity | one row is | worked cases |
|---|---|---|
| `parcel` | a unit of land as the cadastre draws it | |
| `footprint` | a building outline polygon | a townhome row is one footprint |
| `building` | one structure | a townhome row is several buildings; a condo building is one |
| `dwelling` | one housing unit | single-family: one; multi-family: one per unit |
| `property` | one unit of ownership, what a sale conveys | a condo unit is one; a house on its lot is one |
| `transaction` | one recorded sale | |

A recipe's rows must be the entity its `entity_type` names. **A tax roll's
rows are properties**, condo rows included: a roll or CAMA table is one
`property` recipe, never split by a condo flag, and a table whose rows are
*details* of those properties (one row per building component) declares
`supplements: <roll id>` so that it adds columns to them, never rows. The
roll may sit at a containing scope (a county appraiser's building table
detailing the state DOR roll; a city table detailing MassGIS's statewide
layer), and `supplements_layer: <entity type>` names an `additional_layers`
entry of the host when the roll has no recipe id of its own
(`recipe.get_supplemented_table` validates both). A supplement joins its
roll on `parcel_id_local` unless `supplements_key: <column>` names a column
both tables carry, for a roll whose `parcel_id_local` is built from an id
the detail table lacks (Travis County TX links parcels on `geo_id` while
its detail tables carry `prop_id`); keyed supplements are skipped by every
other auto-discovered join, the parcel geospine's included.

The stages divide the work, and **harmonize keeps redundancy minimal**:

- **Harmonize** builds each entity's core table (its spine) and the keys
  that relate entities. An attribute lands once, on the entity it
  describes. Do not copy heavy or wide columns onto another entity here:
  a property's bedrooms belong on the property spine, not also on every
  parcel. Harmonize *may* derive a small number of preliminary columns
  or filters, but only as far as an enrich step needs them to choose
  what to run on: a preliminary `occupancy_type` so an imagery step
  runs on residential footprints only, not a reconciled value that
  curate will decide. No enrich step selects rows by attribute today
  (they subset by admin unit only), so this is an open path, not a
  used one; when one is added, the column it selects on is the
  justification for deriving it here, and it stays labeled preliminary.
- **Enrich** adds evidence from external sources to any entity, keyed by
  that entity (imagery predictions, raster statistics, reference
  inventories).
- **Curate** assembles one dataset per entity type (parcels, footprints,
  buildings, dwellings, properties, transactions), aggregating or
  translating from the others where needed: a footprint's bedrooms come
  from summing its properties' in a curate recipe, not from a
  harmonize-time copy.

The parcel geospine's
`link_by_id(auto_discover: true, entity_type: property, mode: aggregate)`
copies an explicit list of parcel-level columns (values, use codes,
`building_style`, `n_dwellings`, address) and nothing else since
2026-09-12; property-level attributes reach curated parcels through
`aggregate_from_entities` (curate, `aggregation.py`). A county whose
geospine predates that still carries the old wider copy until its next
geometry rerun, which is why the curate step fills only what is empty.

**Build order, and why it is fixed.** A source that holds parcels and
properties together is separated at ingest (an `additional_layers` entry
writes its own property table), so by harmonize every property table exists
independently of any parcel spine. Per admin unit the order is: every
ingest; `US_property-spine-2026`; the footprint geospine and spine; the
parcel geospine, which reads the ingest-level property tables for
parcel-level values and the property spine for a count; the parcel spine;
enrichment; parcel curation; footprint curation. `RecipeDAG` derives this
from the recipes and a test pins it
(`test_property_parcel_link_is_written_after_both_of_its_sides`); a script
that calls stages by hand has to follow it, and **a parcel geospine built
before one of its county's rolls was ingested silently lacks that roll's
values until it is rerun**.

**A property's id is the account number its assessor issued**, prefixed with
the admin unit that scopes it: `US-TX-VIC_000123`
(`io/harmonizer/entity_ids.py`, step `assign_entity_ids` in
`US_property-spine-2026`). The number comes from the column a source's
recipe names in `entity_id`, else whichever of `property_id_assessor`,
`property_id_admin2`, `parcel_id_assessor` repeats on the fewest rows: the
*account*, never the lot (New Hanover County NC's account number repeats on
no row, its PIN on 19,519). Nothing raises: a source whose best column still
repeats on most rows issues no account number and is named by content, with
a warning. A source holding several `tax_year` values keeps only its latest
roll before ids are minted (Florida's DOR roll is 24 yearly rolls per
county, kept at ingest as a panel; Lake County's 3,952,270 rows are 210,057
properties), the unit's latest year and not the latest row per account,
which would revive every account retired since. Case is folded and runs of separators become one hyphen, but their
positions are kept, because in a map-block-lot number they carry meaning
(dropping them made 426 Somerville MA accounts collide). Two sources
publishing the same accounts therefore mint the same ids, and **rows sharing
an id merge into one**, the first-loaded (most specific) source winning each
cell and `source` becoming `a+b`: Boston's city table and MassGIS's layer are
364,997 rows and 185,132 properties. The few differing rows under one number
get `_2`, `_3` ordered by content, and a source with no issued number is
named `{admin}_{source}:{content hash}`. A spatial entity's id comes from its
geometry; this is the equivalent for an entity that has none.

**Relationships between entities are link tables** (`geo/link.py`,
`io/harmonizer/entity_links.py`), one row per pair, stored beside the finer
entity's output (`get_entity_link_path`, finer by `ENTITY_LINK_ORDER`). The
spatial joins write theirs from an overlay; `link_entities_by_id` writes the
property-to-parcel table from an exact match on `parcel_id_local`, with
columns `property_id`, `parcel_id`, `link_method`, `link_source` and `share`
(a test pins that a link table holds nothing else, so it can never carry a
person). It runs in `US_parcel-spine-2026` because that is the first recipe
in which both sides exist. `link_method` is a fixed label naming the rule
that found the pair, never a score, and no link is removed once written
(patent shape 4). A key on more rows than `link_by_id`'s placeholder cutoff
keeps its pairs under `parcel_id_local_shared_key`, because a link table,
unlike a sum, need not decide whether it is a placeholder or a large stack:
a reader that sums values leaves that method out. The footer
(`openplaces:entity_link`) records both input files, so a reader can tell a
link that predates a rebuilt spine. The step sits **after** the parcel
spine's checkpointed step: a restored checkpoint skips every step before
it and validates against the geospine only, so anything placed earlier
that reads another recipe's output goes stale unnoticed (the transaction
`link_by_id` sat near the top of that recipe with this weakness until
2026-09-23; it now follows the link table, and
`tests/recipe/test_parcel_spine_step_order.py` pins both). Curate's
`aggregate_from_entities` reads the table; see the stacked-units paragraph
under Stage 1 for the passes the ingest-time split adds.

### Recipes (`recipe.py`, `src/openplaces/recipes/`)

Recipes are YAML files that define how to ingest, harmonize, enrich, or curate
a dataset. They are stored in a path-encoded directory structure:

```
src/openplaces/recipes/{admin_id_path}/{entity_type_or_theme_path}/{source}/{version}/{recipe_id}.yaml
```

A recipe ID encodes its parts: `{admin_id}_{entity_type_or_theme}-{source}-{version}[_{filename}]`, e.g. `US-MA_parcel-massgis-2025`.

**Recipe roots.** The bundled tree is one root; an installed package can
contribute another through the `openplaces.recipes` entry-point group
(`path.recipe_roots()`, bundled first, bundled wins on a name collision).
Recipe lookup, the recipe index, source discovery and the suffix vocabulary
read every root, so a recipe in a package is found exactly like a bundled
one. Because auto-discovery then picks the most specific recipe *whatever
roots are installed*, every attribute table `to_parquet` writes records the
roots it was built from in its footer (`openplaces:recipe_roots`: bundled
version or checkout commit, plus each package's distribution and version),
and a delivery's terms notice lists them under "Recipes". Without that
record, "the best source available at build time" is not reproducible.

Key recipe fields:
- `admin_id` — geographic scope
- `entity` or `dataset` — what is being ingested
- `description` — short human-readable summary; kept to 25 words or less, with any
  implementation detail in `notes` instead
- `stage` — `'ingest'`, `'harmonize'`, `'enrich'`, `'curate'`
- `columns` — mapping from openplaces attribute names → source column names
- `download_by` — controls download partitioning: `admin_level`, or `partition:
  year|year_month|table|tile_id|latlon_tile`. `year_month` computes a YYYY-MM date range,
  `table` takes an explicit table-name list; unrecognized partition types raise
  `NotImplementedError`.
- `process_by` — controls chunking granularity during processing (`admin_level`, `admin_id_column`).
  For sources that ship one already-split file per admin unit inside a single shared
  download (rather than one file with an in-data admin column to filter rows by), use
  `file_pattern` instead of `admin_id_column`: a filename pattern relative to the recipe's
  heap directory, with `{partition_id}` and the crosswalk's raw-code column name (e.g.
  `{admin3_id_admin2}`) as placeholders, resolved per admin unit
  (`TableIngester._resolve_file_pattern_path`).
- `save_to` — output location (`data_dir` from `STANDARD_DIRS`, `admin_level`, optional
  `retention` class). When `save_to.admin_level` is coarser than `process_by.admin_level`,
  the Ingester aggregates intermediate files to that level.
- `join_partitions_by` — for `download_by: {partition: table}` recipes, left-joins the
  per-table outputs into one entity file per admin unit after ingest (`join_key_name`,
  optional `keep_original`). See `io.aggregate.join_partitions_by_index`.
- `additional_layers` — list of secondary entities extracted from the same source file (e.g., a property table alongside a parcel table)
- `entity_recipe` — predecessor entity recipe used by enrichment or curation
- `pipeline` — ordered named steps for harmonization, enrichment, or curation

Key recipe functions (`recipe.py`):
- `get_recipe(admin_id, entity, ...)` / `get_recipe_by_id(recipe_id)` — load a recipe dict
- `find_entity_recipe_id(...)` — select by stage rank
  `ingest < harmonize < enrich < curate`; pass `stage=` when a caller needs
  a specific predecessor rather than the latest pipeline product
- `get_output_path(recipe, admin_id, partition_id, geo, layer)` — resolve the parquet path where output is written
- `get_save_admin_level(recipe)` / `get_process_admin_level(recipe)` — determine output/processing granularity
- `get_layers(recipe)` — list secondary layers in `additional_layers`
- `build_table_recipe(primary, layer_spec)` — merge primary recipe with an additional_layers spec

### Data pipeline: ingest → harmonize → enrich → curate

**Stage 1 - Ingest** (`io/ingester/`, `io/ingester/table_ingester.py`):

`Ingester` orchestrates download, unzip, and processing of one recipe:
1. Resolves admin IDs to save/process/download (three potentially different levels)
2. For each download partition (admin unit × partition ID): downloads + unzips source,
   calls `TableIngester` to process each chunk
3. After tile partitions: merges per-tile partials into per-admin files
4. In aggregate mode: calls `aggregate_to_admin_level()` to merge process-level chunks
   into save-level files

`TableIngester` handles reading, transforming, and saving one table from an
already-resolved source file. It applies column mappings, type casts, spatial filtering,
and the attribute registry type checks.

**Every parcel table is split into lots and properties at ingest**
(`io/stacked_units.py`). Sixteen parcel sources stack several ownership
records on one lot polygon (Florida's statewide layer alone carries 1.13
million condo-unit rows on shared outlines), and a stacked row is a
property, not a parcel. After preprocessing, rows are grouped by `lot_key`
(default `geo_id`, the geometry hash; a recipe may name a source lot id):
exact duplicates collapse, a group with one row is a lot and passes
through, and a group with several distinct rows becomes one parcel row
keeping only the values its members agree on (a varying value is left
missing, never summed, and a column the registry aggregates by `sum` is
left missing even where they agree, because one unit's floor area is not
the lot's and curate's `aggregate_from_entities` fills only empty cells)
plus one property row per member. Rows that repeat a record on another
polygon are parts of one lot, not units: they merge into one row and
yield no property rows. Under a source `lot_key` that row takes the union
of the lot's polygons with `geo_id` recomputed (Wilson County NC draws
1,060 lot ids as 2 to 7 polygons); under `geo_id` nothing is unioned,
since two polygons in one group are a collision of the quantized hash
and a union would invent an outline. A unit id repeated within a lot
drops nothing and is counted in the ingest log. The property rows go to an
implicit `additional_layers` entry of the same recipe
(`property-<source>-<version>`, `layer_key: parcel_id_local`) that the
recipe loader adds to every parcel ingest recipe without a declared
property layer. Discovery, the property spine, readers and cleanup see
that layer like a declared one; the ingester's layer loop skips it
(`STACKED_UNITS_LAYER_KEY`) because the parcel table's own processing
wrote it. `stacked_units: false` opts a recipe out.

**A unit keeps its own keys and names its lot in `lot_id_local`.** Where
units have their own account numbers, the lot's parcel row cannot carry any
one of them and takes the lot key, which no tax roll knows (Galveston County
TX: 5,186 roll rows in 119 of 184 stacks). Nothing is rewritten to bridge
that. The pair (unit key, lot key) the split records is read three times:
- `assign_entity_ids` labels the split's rows `<source>:units`, loads them
  last, and merges a unit with its roll row where both carry the account
  number (Galveston: 5,205 merged, 440 units the roll does not know). Units
  whose number repeats inside the layer are named by content, the rest keep
  it, so a layer that numbers some stacks and not others still converges.
- `link_entities_by_id` adds two exact passes to the key pass:
  `stacked_units` (a unit links to the lot it names) and
  `stacked_units_crosswalk` (a roll row keyed on a unit links to that
  unit's lot, to every lot where the split saw the key on several). A
  property on several lots gets a `share` by lot area, `share_basis` saying
  so; it is wrong for improvements, which stand on one lot, and stays until
  a property-to-footprint link can replace it.
- the parcel geospine's property `link_by_id` moves a roll row keyed on a
  unit to its lot before joining (`_move_units_to_lots`), and reads the
  split layer itself by `lot_id_local`. An account on several lots becomes
  one row per lot: its land is divided by lot area, every other additive
  value goes whole to the largest lot and stays empty on the rest, because
  a building stands on one lot (`total_value` so overstates the largest
  lot by the others' land).
`aggregate_from_entities` sums over the link table when a valid one exists
and falls back to the key column otherwise: on a lot any roll describes,
rows whose `link_source` is only `:units` are left out, so a roll and the
units split off the parcel layer are never summed together; shares scale
summed columns; `*_shared_key` links are skipped. Measured on Galveston,
rebuilt end to end 2026-09-21: curated parcel `total_value` $151.4bn to
$79.2bn against a roll of $75.9bn, roll rows on a parcel unchanged at
173,697. A lot with a single record yields no property row, so where no
roll exists the property spine holds stacked units only (all of Wisconsin:
Dane County's layer is 218,093 rows, 188,518 lots and 31,285 units, value
conserved to the dollar). The module docstring
carries the patent rationale (US9298740B2, all-elements rule: claims 1
and 15 recite the same six steps, claim 11 adds a wilderness test, and all
three start from a parcel that *failed verification* against mapping or
addressing data, which the split never tests; read from the patent text
2026-09-21, an agent's reading, not legal advice). Data on disk keeps the old row shape until a county is
re-ingested.

Public entrypoint:

`ingest(recipe, admin_ids, partition_ids, reprocess, redownload, verbose)`.

**Stage 2 — Harmonize** (`io/harmonizer/`):

`Harmonizer` runs a composable step pipeline, executing steps declared in the recipe's
`pipeline` list. Each step is a function registered via `@_register('step_name')` in one
of the sub-modules. Registrations carry a `phase` tag (`'geometry'` for steps that
mutate spine rows/geometry or run spatial joins; `'attributes'`, the default, for
steps that only read or annotate) — the tag drives the link-sidecar fingerprint and
defines the geometry/attribute recipe split below.

**The geospine split.** Each expensive spine is two recipes. A *geospine* recipe
(`US_footprint-geospine-2026`, `US_parcel-geospine-2026`) runs the geometry phase —
spine resolution, spatial overlays and point joins, group detections, morphology —
and persists its spine plus one n:m link sidecar per `link_to_reference` join
(`save_link` is default-on; a step opts *out* with `save_link: false`). The
*attribute* recipe keeps the established id (`US_footprint-spine-2026`,
`US_parcel-spine-2026`), declares the geospine as its `entity_recipe`, starts with
`load_geospine`, and writes an attribute-only table (`save_to: geometry: false`);
its geometry lives with the geospine and is resolved by `get_entities` through the
`entity_recipe` chain — by declaration, never by probing for a `_geo` sidecar, so a
stale pre-split sidecar is ignored. Enricher and curator load their entity spine
through `get_entities` for the same reason. Rerunning an attribute recipe involves
no spatial computation (`--reprocess attributes` in the driver). Each sidecar's
footer fingerprint (format 2) covers the step config, the configs of every *prior
geometry-phase* pipeline step, and size/mtime of the ingest inputs, plus a
per-source content sha256 stamped at write time: an input whose mtime moved but
whose bytes did not (a sync-tool touch; Dropbox re-hydration bumped every cache
mtime on 2026-08-24) revalidates through the hash instead of forcing a geometry
rerun (`_fingerprints_match`). Anything else fails closed — a stale sidecar raises with instructions
to rerun the geospine, never silently recomputing geometry. Curate readers resolve
the sidecar's owner through `geo/link.get_link_owner_recipe_id` (the geospine when
split, the recipe itself when not). Geospine recipes declare
`save_to: retention: keep`: their outputs and link tables are the normalized
geometry store, exempt even from aggressive cleanup (an explicit retention wins
over the aggressive core-bucket demotion). Bucket policy: `core` holds the
normalized store and intermediate evidence; terminal curate outputs ship
self-contained (attributes + geometry) in `share`.

The step sub-modules:
- `spine.py` — build/merge the primary entity GeoDataFrame (`resolve_spine`),
  and assign each row's containing polygon id from a space-partitioning
  reference layer such as an admin level or a Census statistical geography
  (`link_geographic_ids`; unlike `link_to_reference` below, every configured
  reference is assumed to tile space without overlaps, so the relationship
  is always exactly one containing polygon, not a many-to-many crosswalk).
  An `inherit_from` option rolls up an already-linked recipe's own output
  (e.g. a parcel spine inheriting from its footprint spine, which runs
  first) wherever every row in the group agrees, so the direct spatial join
  only runs on the residual.
- `links.py` — join to reference datasets spatially or via crosswalks
  (`link_to_reference`)
- `load.py` — restore a geospine recipe's spine, crosswalks, overlays, and
  prepared references from its persisted output and link sidecars
  (`load_geospine`); which links to restore is read from the geospine
  recipe's own pipeline, so the two YAMLs cannot drift
- `attributes.py` — attribute source columns to the spine as suffixed evidence
  columns (`reconcile_attributes`), assign each footprint's parcel priority
  (`classify_footprint_priority`), and build the combined land-use label the
  parcel classifier votes on (`derive_use_classes`). Its `columns` list is
  ordered, and each entry is either a column or a list of alternatives
  **coalesced per row** — the default is `[use_group, use_group_code]` then
  `[use_subgroup, use_subgroup_code]`, so a source whose raw code has no
  crosswalk still contributes that code as grouping and voting evidence,
  while a source whose crosswalk did fire contributes only the vocabulary
  and adds no new label values (so cohort statistics keyed on
  `use_group_combined` are not fragmented). This matters more than it
  looks: 20 recipes map a `use_*_code` but only 5 ship a crosswalk for it.
  The parcel spine appends `building_style` last so a county whose
  land-use text carries no occupancy signal can contribute one from its
  structure description.
  Value selection, gap-filling, and occupancy inference now run in the
  curation stage, not here; the harmonized spine (`US_footprint-spine-2026`)
  is an evidence-only table.
- `addresses.py` — reconcile a canonical street address from any number of
  source inputs (`reconcile_addresses`); coalesce ZIP-code evidence from
  multiple columns by priority (`reconcile_postal_code` — e.g. an
  address-parsed ZIP, then the spatially-derived `zcta5_id`, so a state with
  no address-parsed coverage at all still gets a ZIP-like value); and derive
  the USPS-preferred city for a ZIP code (`impute_postal_city`).
  - These are a deliberate exception to the "gap-filling belongs in curate" rule: the
    dividing line is whether there is dispute about how to derive its output. A ZIP code
    has exactly one USPS-preferred city — nothing parameter-sensitive to defer — so it's
    resolved once, in harmonize, where it's available early for downstream linking.
- `diagnostics.py` — conflict-inspection reports (agreement rates, a
  crosstab of disagreeing value-pairs, a bounded sample of conflicting rows)
  written to the cache when `save_statistics` is set, mirroring
  `io.curator.diagnostics`'s established convention. Never raises, never
  changes the harmonized spine.
- `filter.py` — subset rows (`filter_entities`)
- `discover.py` — discover available data sources for an admin unit
- `admin_names.py` — `assign_admin_id_from_name`: the admin unit a record
  only *names*, as an id, matched exactly and uniquely against the units of
  its county or not at all. It exists because an address key is scoped by
  `admin4_id`: a source with an empty scope matches no parcel by address,
  silently (0 of Vilas County WI's keyed returns, 0.759 with the scope
  ignored). The source's spelling is a recipe-side regex; a no-op where the
  level does not exist (New England).
- `link_methods.py` — `record_link_method`: after each `link_by_id` pass of
  the transaction spine, writes `parcel_link_method`, the name of the first
  rule that reached the row (`parcel_id_local`, then an exact address key).
  A fixed label, written once, never a score and never revised (patent
  shape 4). It re-reads the pass's reference instead of instrumenting
  `link_by_id`, so a step added between a pass and its label would make the
  label describe the wrong pass.
- `last_sales.py` — `append_last_sales`: turn the last-sale fields an
  assessment roll carries into transaction rows
  (`sale_record_kind = assessor_last_sale`; rows already on the spine read
  `deed`), so a state with no recorder's feed still has prices. It is a
  reshape, not a match: the row keeps the roll row's own `parcel_id_local`.
  Three things are not what the name suggests. **"Has a last sale" is never
  "non-null"**: Florida's parcel layer fills every last-sale column on every
  row and 0 is the placeholder, in the year as well as the price, so a row
  needs a price and a real date or year. **A roll that records the month
  only arrives as first-of-month dates** (all 73,253 of Pitt County NC's);
  the step keeps year and month and writes no day, so a deed and its
  last-sale echo compare at month precision. **A roll ingested for several
  years repeats each last sale once per year** (Florida's DOR roll holds 24);
  the same ids, date and price are one row, while different last sales of
  one property across years are kept, which is history a single roll year
  lacks. Property tables are read before parcel tables, and a later source
  adds a sale only for a `parcel_id_local` no earlier one supplied. Only an
  allow-list of columns is read, so owner fields cannot enter. A last-sale
  field is the most recent sale only: these rows support cross-sectional
  work, not a repeat-sales index.

All steps share a `HarmonizeState` dataclass (spine, references, crosswalks, overlays,
metadata). Each step receives state and returns the updated state.

Public entrypoint: `harmonize(recipe, admin_ids, reprocess, verbose)`.

When working on `src/openplaces/io/harmonizer/`, read
`src/openplaces/io/harmonizer/README.md` for a full pipeline reference before making
changes.


**Stage 3 — Enrich** (`io/enricher/`):

`Enricher` reads a harmonized entity recipe and writes entity-keyed evidence
tables. Enrichment adds observations or model outputs without selecting a
canonical value, reconciling disagreements, or filling unrelated gaps.

- `attributes.py` — registered evidence-producing steps (`classify_roof_shape`,
  `classify_occupancy`, `detect_n_stories`); image-based steps fetch the
  recipe's `image_recipe` imagery **in memory, per run**
  (`io.ingester.image_ingester.fetch_images_in_memory`) and keep only the
  predictions. There is no image ingest stage and no image cache: Google's
  Static API policy prohibits pre-fetching, indexing, storing, or caching
  its content, so an image recipe declares no `save_to` and carries camera
  configuration only. The cost is that every enrichment pass re-fetches, and
  for Street View re-pays; a step whose scraper cannot initialize warns and
  leaves its evidence columns empty rather than aborting the batch.
- `buildings.py` — `enrich_footprints_from_reference_buildings`: attach an
  already-built reference *building* entity's attributes onto footprints,
  each footprint taking the single reference building it overlaps most by
  IoU (not raw intersection area — a large reference building clipping a
  small footprint's corner shares more area with it than the correct small
  building does). Source-agnostic: any building recipe with polygon
  geometry works, so a precomputed inventory can substitute for re-running
  imagery inference. An admin unit the reference does not cover still gets
  the declared columns written as all-null, because curate treats a present
  evidence file missing a declared column as a recipe error.
- `parcels.py` — attach a reference *parcel* dataset's attributes to current
  parcels through a fractional area-weighted crosswalk, driven by a sidecar
  `{recipe_id}_column-notes.csv` beside the reference recipe
- `zonal.py` — `zonal_stats`: per-entity raster statistics over polygon
  geometry, dispatching to one of three backends in `geo/raster.py`
  (`exactextract` for fractional pixel-area weighting, `rasterstats`, or
  `rasterized` for a burn-and-groupby). Requires `spine_geom: true`.
- `vicinity.py` — `vicinity_coverage`: derives a neighborhood-percentage
  raster from a boolean source raster by FFT convolution, windowed to one
  admin unit and padded so neighboring units still count, then samples it
  through `zonal.py`. The derived raster is cached per admin unit and reused.
- `derived.py` — `derive_from_spine`: per-entity metrics from spine geometry
  and existing spine columns alone, no raster involved. Overlaps the
  harmonize step `derive_geometry_attributes` and shares its area
  measurement (`geo.polygon.get_areas`); it exists for spines that do not
  run that step.
- `detectors/` — attribute-specific detectors and shared inference runtimes
  (EfficientDet/EfficientNet ports; torch is conda-only)
- `models.py` — pretrained-model download and cache handling

Raster-consuming steps name their raster by a path relative to the configured
`rasters` directory, resolved by `path.resolve_raster_path()`, so a recipe
stays portable across machines. An absolute path passes through unchanged.

Examples include roof-shape, occupancy, and story-count evidence from imagery,
and the same attributes read off a precomputed inventory
(`US-NC_footprint_building-cheer-v0`). Evidence columns retain
provenance-oriented names such as `roof_shape_brails`, `n_stories_brails`, and
`foundation_type_building_cheer`.

Public entrypoint: `enrich(recipe, admin_ids, entity_recipe_id, reprocess, verbose)`.

**Stage 4 — Curate** (`io/curator/`):

`Curator` creates the canonical entity dataset. It starts from a harmonized
(evidence-only) entity, incorporates enrichment evidence, and applies explicit
recipe steps that select values, fill gaps, infer canonical attributes, format
the output, and remove records. Each step is a registered function operating on
a shared `CurateState` (canonical GeoDataFrame in `state.curated`).

Steps are organized by the nature of the transformation:

- `evidence.py` — incorporate enrichment evidence (`merge_enrichments`)
- `indicators.py` — the shared voting vocabulary and scoring cores
  (`evaluate_indicator` predicates; `score_decisions` enumerated votes;
  `vote_dynamic_values` open-vocabulary votes). Pure functions over a
  DataFrame — no thresholds, class names, or geography of their own.
  Both report provenance, and both report *evidence* rather than the class:
  `vote_dynamic_values` joins the labels that agreed (`nsi/fema/parcel`),
  and `score_decisions` joins the `label` of each indicator that fired for
  the winning decision (`no_improvement_value+block_context`), falling back
  to the decision's declared `source` only where it labeled nothing. Labels
  are opt-in per recipe; unlabeled decisions keep their old behavior, which
  is why a `{col}_source` can still read as a synonym of the value beside
  it until its indicators are labeled.
- `reconcilers.py` — resolve conflicts between competing source columns
  (`reconcile_values` priority selection; `resolve_occupancy` parcel-vs-NSI;
  `resolve_by_vote`, the single voting seam every curate classification
  resolves through)
- `imputers.py` — fill missing canonical values (`impute_n_dwellings`,
  `impute_from_group_statistic`, `impute_occupancy_type`)
- `aggregation.py` — reduce another entity's rows onto the curated rows
  they belong to (`aggregate_from_entities`): a parcel's `year_built` is
  the earliest of its properties', its `living_area_sqft` and
  `gross_floor_area_sqft` their sums, read from
  `US_property-spine-2026` by `parcel_id_local` with the registry's rule
  or a per-column override, filling only what the parcel's own layer
  left empty. It exists because the parcel geospine's property
  `link_by_id` no longer copies property-level attributes (its explicit
  `columns:` list stops at parcel-level values, use codes and address):
  a copy made in the geometry phase was a second, lossy home for them and
  tied every fix to a geometry rerun. Measured on Victoria County, TX,
  2026-09-12: the curate-side reduction reproduces the old copy's
  coverage exactly (63.9% `year_built` and floor area, the PACS segment
  sum now named `gross_floor_area_sqft`), so a county
  whose geospine predates the change loses nothing. An all-missing group
  reduces to missing, not to pandas' zero sum. A column named under
  `recover` (the parcel recipe names `improvement_value`) is the one
  departure from fill-only: a stacked lot keeps only what its units
  agree on, so its improvement is missing or zero while the units
  record it, and the units' positive sum replaces that. Nothing is
  written where no unit records a positive figure: a lot the roll does
  not value stays as it is, because estimating it would be an
  imputation step, which openplaces does not do for structure values
  (maintainer, 2026-09-22). Measured 2026-09-22 on cheer-eastern-nc:
  179 of 503 such lots hold a recorded figure ($44.1M); on
  cheer-coastal-tx 447 of 15,555 ($64.8M).
- `inferers.py` — derive new canonical features (`derive_metrics`,
  `derive_indicators` — named indicator columns holding values, never
  pre-thresholded booleans; every cutoff lives in the vote decisions).
  `derive_group_class_share` adds the context an entity cannot supply about
  itself: the share of its group (any id column it already carries, e.g.
  `census_block_id`) whose evidence reads as a given class, excluding the
  row itself. It is a **groupby, deliberately not a spatial operation** — no
  buffering, no boundary union, no nearest-neighbor search — both because
  geometric neighbor/"community" detection is a patented technique shape in
  this domain (see the patent-risk section) and because aggregating within a
  published administrative unit is older, plainer practice. It exists
  because `flag_manufactured_home_communities` counts per *parcel* and so
  cannot see a subdivided community where every home has its own lot. Feed
  it only evidence a downstream vote has not written, or the class
  reinforces itself. **No shipping recipe votes on it**: measured
  2026-08-21, a block share computed from the assessor's own text
  correlates +0.577 with the keyword rule that reads the same column, and
  the points it uniquely moves are 0.294 precise against a 0.425 base rate.
  It is kept as evidence, and as the mechanism
  `notebooks/05_curate/mmh_separability.py` measures with.
  `derive_group_count` and `derive_group_rank` are the same kind of
  groupby, counting or ranking (largest first, ties by id) the rows of
  a group that satisfy voting indicators. **The footprint recipe votes
  twice on `occupancy_type`**: they read the first vote's classes of a
  parcel's primaries, and a second `resolve_by_vote` (`base_output:
  occupancy_type_pass1`) turns small secondary Manufactured Home
  footprints into Secondary. It is safe only because the second pass
  writes secondaries and reads primaries; it reads which labels carried
  the first pass through the `has_token` predicate, which matches whole
  `+`-separated parts of `occupancy_type_source`.
- `formatters.py` — structural/type-only output shaping (`cast_categoricals`,
  `order_columns`)
- `filters.py` — (stub) remove records that do not belong in the canonical
  dataset
- `transactions.py` — steps for the transaction entity, which **grade and
  flag and never drop**: `derive_document_id` (the deed behind a row, spelled
  per source; a part that is empty or all zeros names no document, since
  Florida's roll writes a single space for book and page and that once folded
  1,333 sales of one county into a single "deed"; scoped by record kind, so a
  deed and the last-sale row citing its book and page are never aggregated as
  one sale of two parcels), `count_parcels_per_document`,
  `aggregate_multi_parcel_sales`, `collapse_double_closings` (compared within
  one record kind) and `flag_sales_matching_other_kind`
  (`sale_matches_deed`: exact equality of parcel, year, month and price, no
  tolerance and no score). `sale_arms_length_confidence` is graded per source
  in the recipe: Florida from the state's qualification codes (labels and the
  raw two-digit codes alike, because last-sale rows read from the parcel
  layer are undecoded), Wisconsin as the lower of two grades, the conveyance
  type and the relationship the parties themselves state on the RETR form
  (`sale_party_relationship`; about 20% of transfers are `Family`), through
  the `minimum` indicator type and `fill_only`.

Alongside the step modules sit support modules that register no steps of their
own: `occupancy.py` (shared, vocabulary-neutral occupancy helpers),
`provenance.py` (`{col}_source` sidecars), `land_value.py` (land-value
estimation, split out because it is expected to grow), `diagnostics.py`
(cache-written conflict reports), and `validation.py` — scoring a curated
classification against hand-labelled points. `validation.py`'s
`link_points_to_entities` links by address first and distance only as a
fallback; because a house and its shed share one address, callers break the
resulting ties with `prefer_column`/`prefer_values` (e.g. rank
`priority_on_parcel == 'primary'` first) rather than letting row order decide.

**Every validation step writes a confusion matrix.** `confusion_matrix` keeps
reference classes as rows and predictions as columns, then `Secondary`,
`Non-residential` (any other asserted class; `(other class)` for a recipe
declaring no residential classes) and `No class`, so no row is dropped and
"said nothing" stays apart from "said something else". NSI and FEMA derived
sources set `keep_unmapped: true`, keeping a raw class their residential-only
class map does not cover instead of scoring it as no prediction;
`accuracy_from_matrix` derives producer's (recall) and consumer's (precision)
accuracy, kappa and the abstention rate, and `score_classification` is
computed from the same matrix. `write_confusion_report` (and
`write_year_agreement_report`, whose "matrix" is one reference row of
exact/within 1/within 5/more than 5 years bins) writes
`{name}_confusion.csv` (long form, pooled plus strata),
`{name}_accuracy.csv` and a JSON sidecar into the delivery's `accuracies/`
folder, refusing any stratum under 10 rows so no cell points at one building.
`ValidationContext.score_sources(linked, out_dir)` writes them for every
survey source; the permit notebooks call the writer directly. Only these
aggregates may leave memory: permit modes per footprint are restricted
row-level data.

`provenance.py` carries one invariant worth knowing before writing any step
that produces a value: **a cell openplaces itself filled must say so**, by
carrying the `imputed` marker in its `{col}_source`. Tokens are therefore
composite, joined by `+` (`parcel+imputed`, and the harmonize stage's
`parcel+usaddress`) — never assume a token equals a bare source name.
`mark_imputed` is the only place the marker is written and `is_imputed` the
only place it is read (it matches whole `+`-separated parts, so a source named
`imputed_rates` is not swept up); `record_sources` writes a per-row token
series, which is what a step carrying an upstream sidecar forward needs rather
than `record_source`'s one-token-per-mask.

**The marker means openplaces filled the cell, not that the number is
modeled.** A value read from a dataset keeps that dataset's name even when the
dataset is a model: `nsi` is a FEMA-modeled structure value and stays plain
`nsi`, because the token already names what produced it. So classification
votes are not marked (`nsi`/`keyword`/`classifier` each name the winning
evidence), and neither is apportioning a reference's value across the entities
on it — the parcel's total is a real assessed figure being divided, not a value
invented where none existed. Only estimation of a genuinely missing value
counts.

The hard part is propagation, not the decision: `impute_land_value` knows which
rows it estimated, and the marker has to survive `apportion_curated_values`
(which crosses parcel → footprint) and `select_value_source_by_admin_unit`
(whose `output` and `parcel_column` are commonly the *same* column, so they
share one sidecar) to reach the delivered `structure_value_source`. All three
dropped or flattened it before 2026-08-21.

These concern-based modules mirror the processor categories used by related
inventory systems, while remaining native to the openplaces recipe and state
architecture. Value selection, gap-filling, and occupancy inference were
migrated here from the harmonize stage; the harmonized spine
(`US_footprint-spine-2026`) is now an evidence-only table. Curation outputs are
full entity recipes, not sidecar evidence tables.

Public entrypoint: `curate(recipe, admin_ids, reprocess, verbose)`.


### Data access (public API, `api.py`)

```python
import openplaces as op

op.get_admin(admin_id, level, geom, recipe)        # load admin unit table/geodataframe
op.get_admin_ids(admin_level, admin_id)            # list admin ID strings
op.get_entities(recipe, admin_id, geom, layer)     # load entity parquet
op.get_dataset(recipe, admin_id, partition_id)     # load dataset parquet (or path for rasters)
op.ingest(recipe, admin_ids, ...)
op.harmonize(recipe, admin_ids, ...)
op.enrich(recipe, admin_ids, ...)
op.curate(recipe, admin_ids, ...)
op.aggregate(recipe, admin_level, ...)
op.export_delivery(recipe, admin_id, ...)     # split a curated region into a shareable bundle
```

### Named regions (`_all/admin/regions/2026/admin-regions-2026.csv`)

A **region** is any named group of admin units the hierarchy cannot express: a
study area, a delivery footprint, a funder's geography. The registry is a flat
1:n CSV (`region_id`, `name`, `region_admin_id`, `admin_id`, one row per member)
read by `op.get_regions()` and `op.get_region_admin_ids(region_id)`, loaded
through `get_recipe_by_id` exactly like the admin spine CSVs beside it.

It exists because the same county list is wanted by delivery, by mapping, and
by ad-hoc analysis, and three copies drift. Anything needing "the CHEER Texas
counties" asks the registry rather than restating 42 ids. A grouping used in
exactly one place need not be registered -- `share: delivery: admin_ids:` still
takes an inline list. `region_admin_id` is the unit a region rolls up to
(`US-TX`); blank means callers derive it, which is what delivery does.

### Delivery bundles (`io/delivery/`, with `terms.py` and `redaction.py` beside the package file)

`export_delivery` pools a curation recipe's per-process-unit files into a
region-wide, shareable set of four files sharing one index: the canonical
attributes as a plain table, the same attributes on centroid points, the
boundary polygons alone, and an `_evidence` supplement holding every remaining
column. Which columns are canonical is declared per recipe in a `share:` block
(`columns`, plus `point_columns` for the point file only) -- not inferred, because
the curated schema holds far more technically-canonical columns than belong in a
compact delivery. Each canonical column's `{col}_source` sidecar is appended
automatically and written into *both* the canonical and the evidence file, so
either reads on its own.

A `share: delivery:` block names the regions: `admin_level` (a bundle's own
level) plus either `admin_ids` (one grouping, listed inline) or `regions` (a
list of ids from the shared region registry, below). The declared member list
wins over walking the admin hierarchy, because a region is rarely all of a
state's children -- the CHEER regions are 45 of North Carolina's 100 counties
and 42 of Texas's 254.

One recipe ships as many regions as it declares: `US_footprint-cheer-2026`
produces a Carolina bundle and a Texas bundle from identical curation logic, so
the membership is data in the registry rather than structure in the YAML.
`delivery_regions(recipe)` is the resolver; `delivery_spec`, `delivery_members`,
`delivery_paths`, `delivery_admin_id` and `export_delivery` all take a `region=`
selector, and **raise rather than guess** when a multi-region recipe is asked
for "the" region -- silently picking one would ship a region's rows under
another's filename. A single-region recipe needs no selector, so
`export_delivery(recipe_id)` still works unchanged.

Sibling regions are told apart by containment, not by level: both CHEER bundles
sit at admin level 2, so `RecipeDAG._delivery_in_scope` asks whether a requested
unit is the bundle's own unit or an ancestor of it (`US`, `US-TX`), not merely
whether it is coarse enough. `--config deliver=true` likewise only forces the
regions the run actually touches.

**It withholds what restricted sources supplied, and ships the rest.** A
source recording `redistribution_restricted: true` or a `usage_requirement`
may feed the recipe; `bundle_terms.restricted_inputs` lists every such input,
walked per pooled unit because county parcel and property recipes reach a
spine only through auto-discovery, which resolves per admin unit (an unscoped
walk never saw Edgecombe's parcels feeding the NC bundle).
`io.delivery.redaction.withhold` then removes that source's values from both read
passes: a row whose `geometry_source` names it is left out, a cell whose
`{col}_source` names it is emptied, and inside the source's county, columns
named after an attribute its recipe maps (or with its entity suffix), or whose
sidecar names only its layer (`parcel`), are emptied unless a sidecar names
another specific source. Emptied cells' sidecars read `withheld`, and the
notice lists each source's counts. `RestrictedInputError` is only the
backstop: it fires if restricted values are still in the frames about to be
written, never because a source merely feeds the recipe, so one county's
terms cannot hold the rest of a bundle hostage. The rules over-withhold where
a layer's sidecar cannot say which source filled it; that costs values, not
a leak. A no-resale clause withholds nothing; the notice reports it.
**Team bundles are the one exception, and never published.** A region listed
under `share: delivery: team_regions:` also ships as `<region>.team`, in a
`team/` subdirectory beside the region's own bundle (whose path does not
move). A team bundle keeps the values of sources a person has cleared with
`team_sharing_permitted: true` (Edgecombe's parcels, decided by the user
2026-09-13), and its notice opens "TEAM-INTERNAL BUNDLE"; every other bundle
withholds them, and uncleared sources are withheld from team bundles too.
Asked for a delivery by admin unit alone, the DAG and `delivery_node` resolve
to the public bundle: a team twin is only ever shipped by naming it.

Three behaviors are worth knowing. It reads each unit twice (canonical+geometry,
then evidence) so the wide evidence columns are never in memory alongside the
polygons. It deduplicates the index: an entity on a county line is curated by
both neighbors and the two copies share an Open Location Code, so the
better-covered copy wins, ties broken on admin id. And it leaves its four
outputs read-only, unlocking them itself on the next run -- deliberately not
Snakemake's `protected()`, which additionally refuses to ever regenerate a file
and would abort the workflow on every reship.

The QGIS side is `qgis/load_joined_parquet.py`, which resolves the whole set from
any one of its files. It picks the join key at load time (`_join_id`, then
`geo_id`, then whichever entity id the two files share), so one algorithm reads
both a core split-layout file and a delivery bundle -- the only difference
between them is that key.

`viz/qgis_map` renders a delivery as a project rather than a layer. Asked for
the admin unit a recipe delivers, `resolve_layers` returns two standalone
output layers instead of the per-unit curated file: the `_point` centroids
carrying the canonical attributes (`RENDER_POINTS`, template fills rewritten as
markers) and the `_geo` polygons as plain outlines (`RENDER_OUTLINE`). Neither
joins -- classifying millions of polygons to color them costs far more than
reading the centroids. `include_inputs=False` drops the ingest-stage layers for
a map of the product rather than of how it was built, and a style variant whose
classifying column is absent from the data is skipped rather than shipped
empty.

### Orchestration (`flow/dag.py`, `workflow/Snakefile`)

`RecipeDAG` derives one job per (stage, recipe, admin unit) from the recipe tree,
plus one terminal **`deliver`** job per delivery region the target recipe
declares (`dag.delivery_nodes`; the older `dag.delivery_node` still returns the
single node when there is exactly one, and raises when there are several).
`deliver` is not a recipe stage -- recipe stages are ranked
(`ingest < harmonize < enrich < curate`) and drive `find_entity_recipe_id`, so a
fifth would ripple into recipe resolution for nothing. It is a node kind this
graph derives, the way `extra_outputs` derives link sidecars from `save_link`.

Scope decides whether each runs: an unscoped run builds and ships every declared
region, a run naming a region (or covering every member of it) ships that one,
and a narrower debug run stops at curate so a one-county rebuild cannot
overwrite a shipped regional file. `--config deliver=true|false` overrides
either way, `true` still limited to the regions the run touches.

A **`validate`** job follows each delivery, one per shipped region whose
recipe lists notebooks for it (`validation: notebooks:`, each naming the
reference it scores; `ground_truth` or a sidecar `references:` key resolves to
a region). It shares the delivery's recipe, unit and region, so `node_key`
adds the stage as a fourth part. `run_stage validate <recipe> <region>`
executes the notebooks with `jupyter nbconvert --execute`, passing arguments
through `OPENPLACES_NOTEBOOK_ARGS` (read in each notebook's test-arguments
cell), writes executed copies to the cache `_logs/validate/` tree, runs
`docs/_ext/generate_validation_tables.py`, and records a
`{recipe}_validation-run.json` manifest as its output. It follows `deliver`'s
scope; `--config validate=false|true` overrides it.

### File layout on disk

Output files are parquet, stored under the configured `cfg.data_root`:
```
{data_dir}/{admin_id_path}/{entity_or_dataset_path}/{admin_id}_{entity}_[suffix].parquet
```

Geometry sidecar: as above, but with a `'_geo'` suffix added to the filename

`path()` in `path.py` generates these paths from:

  `(admin_id, entity, dataset, filename, root)`

`external_path()` / `heap_path()` generate paths for downloaded and unzipped source
files, respectively.

### Attribute registry (`core/attribute_registry.csv`, `core/attribute_registry.py`)

Maps well-known column names to their expected data type, units, default aggregation
function, and a `sort` rank. Used by the harmonizer and ingester to validate dtypes and
drive groupby aggregations without hardcoded column lists, and by the curate
`order_columns` step to order output columns deterministically. Loaded once and cached
via `@cache`.

**Column naming convention.** Evidence columns carry a provenance suffix with exactly
the disambiguating components: point/building-level refs use entity-level + source
(`_building_nsi`, `_dwelling_overture`, because footprint/building/dwelling are easily
conflated); parcel refs use entity-level only (`_parcel`, since parcel layers are
interchangeable). Relational counts use `n_{counted}s_per_{grouping}`
(`n_parcels_per_footprint`). Final output order is computed from the suffix + registry
`sort` rank, so no explicit per-recipe column list is needed.

### Configuration (`config.py`)

`cfg` (singleton `OpenPlacesConfig`) holds directory paths (`data_root`, `dir_core`,
`dir_external`, `dir_heap`, etc.), CRS, and the installation's `identity`.

### Talking to other people's servers

Two rules govern every outbound request, both of them about openplaces not
speaking for its user.

**Identify yourself.** `cfg.user_agent` builds
`openplaces/{version} (+https://openplaces.io; {nickname}@{place})` from the
per-user `identity` block, appending `; agent: claude-code` when
`detect_agent()` finds an AI agent driving the run. Unset reports
`unidentified`. `io.request_headers()` is the single accessor; use it rather
than passing headers by hand, and **never** send a browser User-Agent -- a
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
bind everyone who runs it to terms they never read -- refused outright, not
downgraded to a prompt, so a recipe that reads as though consent were handled
cannot ship. `accept_terms: false` *is* honored -- declining only costs a
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
`Source` -- set only when a person determined it from the terms, never
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
axis (NHGIS, Shovels, Edgecombe, GADM) stay out of this mechanism --
`redistribution_restricted`, `resale_restricted` and
`io/delivery/terms.py` cover those -- and
a blanket ban on automated access is expressed by having no
`download_url` at all, not by a requirement.

`arcgis_rest_scraper` additionally paces itself (`DEFAULT_REQUEST_INTERVAL_S`,
module-level so the whole paging loop is bounded), because backing off only
after a failure means the load that caused it was already applied.
