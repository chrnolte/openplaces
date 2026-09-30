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

## Module READMEs
Before editing files in a folder that has a `README.md`, read it. The
detailed architecture reference lives there, beside the code:
`io/ingester`, `io/harmonizer`, `io/enricher`, `io/curator`,
`io/delivery` and `flow` (all under `src/openplaces/`), as well as
`recipes` and `notebooks`.

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
Layer 6  io/ingester/* (ingester, table_ingester, image_ingester, registry_ingester, cloud_geoparquet_ingester, raster_ingester, access), io/scrapers/* (the Avenu adapter among them), io/aggregate, io/admin/* (ids, names, generate, spine, context, wikidata, units), io/admin_migration, io/delivery/* (the bundle writer, terms, redaction), io/transform, io/sale_records (the frame operations that put recorded sales into the transaction entity's terms and mint their ids; the harmonize and curate steps of the same names both call them), io/cleanup/* (receipts, consumption, lock, walk, compaction; the package file re-exports every name, private ones included, because dag, the harmonizer and the tests import them)
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
ingest; `US_property-spine-2026`; the footprint geospine; the building
geospine (one building per outline) and the building spine, which runs
the evidence, address and permit steps against the footprint geospine's
links re-keyed to building ids; the footprint spine, a projection of the
building spine onto the outlines (`adopt_entity_attributes`, so its
consumers read unchanged values until they read the building spine); the
parcel geospine, which reads the ingest-level property tables for
parcel-level values and the property spine for a count; the parcel spine;
enrichment; parcel curation; footprint curation. `RecipeDAG` derives this
from the recipes and a test pins it
(`test_property_parcel_link_is_written_after_both_of_its_sides`); a script
that calls stages by hand has to follow it, and **a parcel geospine built
before one of its county's rolls was ingested silently lacks that roll's
values until it is rerun**.

**A property's id is the account number its assessor issued**, scoped by
its admin unit (`assign_entity_ids`), and rows sharing an id merge into
one. **Relationships between entities are link tables**, one row per
pair (`link_entities_by_id` for property to parcel); `link_method` is a
fixed label, never a score, and no link is removed once written (patent
shape 4). Both are described in full in
`src/openplaces/io/harmonizer/README.md`.

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
- `patches` / `pipeline_patch` — a patch recipe: a file at a county's or
  state's scope that names a national harmonize or curate recipe and
  lists operations on its pipeline (`set`, `extend`, `insert_before`,
  `insert_after`, `replace`, `remove`, each addressing a step by name,
  `occurrence` when the name repeats). `recipe.apply_recipe_patches`
  applies the patches whose scope covers the unit, coarse to fine, once
  per unit inside the harmonizer and the curator; the output footer
  records them (`openplaces:recipe_patches`). Auto-discovery never picks
  a patch recipe as an entity's recipe and the DAG keeps one recipe id
  per job. A county's rule goes there, never into a branch on a place
  in the national recipe or in `src/` (`recipes/README.md`).

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

Each stage's full reference lives in a README beside its code; read it
before editing that package.

**Stage 1 - Ingest** (`io/ingester/`, reference in
`src/openplaces/io/ingester/README.md`): downloads, unzips and maps one
recipe's source into per-admin parquet (`Ingester`, `TableIngester`).
**Every parcel table is split into lots and properties at ingest**
(`io/stacked_units.py`): a row stacked on a shared lot polygon is a
property, not a parcel, and units keep their own keys and name their
lot in `lot_id_local`.
Public entrypoint:
`ingest(recipe, admin_ids, partition_ids, reprocess, redownload, verbose)`.

**Stage 2 - Harmonize** (`io/harmonizer/`, reference in
`src/openplaces/io/harmonizer/README.md`): runs a recipe's registered
steps over a shared `HarmonizeState`. Each expensive spine is two
recipes: a geospine (geometry phase, persisted link sidecars that fail
closed when stale) and an attribute recipe that loads it and runs no
spatial computation.
Public entrypoint: `harmonize(recipe, admin_ids, reprocess, verbose)`.

**Stage 3 - Enrich** (`io/enricher/`, reference in
`src/openplaces/io/enricher/README.md`): writes entity-keyed evidence
(imagery predictions, raster statistics, reference inventories) without
selecting canonical values. Imagery is fetched in memory per run and
never cached, because Google's terms prohibit storing it.
Public entrypoint:
`enrich(recipe, admin_ids, entity_recipe_id, reprocess, verbose)`.

**Stage 4 - Curate** (`io/curator/`, reference in
`src/openplaces/io/curator/README.md`): builds one canonical dataset
per entity type from harmonized evidence and enrichment, through
registered steps over `CurateState`. One invariant binds every step
that writes a value: **a cell openplaces itself filled must say so**,
through the `imputed` marker in its `{col}_source`
(`provenance.mark_imputed` writes it, `is_imputed` reads it).
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

### Named regions and delivery bundles (`io/delivery/`)

A **region** is a named group of admin units the hierarchy cannot
express, listed in `_all/admin/regions/2026/admin-regions-2026.csv`
(`op.get_regions()`, `op.get_region_admin_ids()`). `export_delivery`
pools a curation recipe's per-unit files into a region-wide bundle
(canonical table, centroid points, polygons, evidence supplement).
**It withholds what restricted sources supplied, and ships the rest**;
team bundles (`<region>.team`) are the one exception and are never
published. Full reference: `src/openplaces/io/delivery/README.md`.

### Orchestration (`flow/dag.py`, `workflow/Snakefile`)

`RecipeDAG` derives one job per (stage, recipe, admin unit), plus a
`deliver` and a `validate` job per shipped region. A narrow debug run
stops at curate, so a one-county rebuild cannot overwrite a shipped
regional file. Full reference: `src/openplaces/flow/README.md`.

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
