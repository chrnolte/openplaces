# AGENTS.md

This file provides guidance to AI coding agents working with code in the `openplaces`
repository.

## Maintaining this file
- This file holds rules and traps an agent would get wrong without being
  told. Architecture reference lives in the module READMEs (see "Module
  READMEs"); put new reference material there, not here.
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
  government ID numbers, or similar identifiers of identifiable individuals, for any
  dataset, in any jurisdiction, ever, even a single illustrative value
  copied from a real record for a docstring, test, or commit message.
- Recipes describe a source's *schema* (column names, field codes), never its
  *records*. Example values must be fabricated.
  `docs/5_contribute/no-personal-data.rst` explains what counts as a record and how
  to fabricate a fixture that still tests something.
- Notebooks are committed with outputs stripped (`nbstripout`, wired via
  `.gitattributes`/`dev.py`) because raw pipeline output can carry personal fields
  from upstream sources. Never disable or bypass this.
- If you find real personal data already committed, stop, do not build on top of it,
  and flag it to the user immediately. Purging it from git history is a decision for
  a human, not an agent.
- **Personal columns in data (not in git) are a per-user choice, and never
  leave in a delivery.** Owner, grantor and grantee attributes
  (`core.attribute_registry.is_personal_attribute`, the one definition) are
  ingested. A curate recipe's `keep_registered_columns` step drops them unless
  the person running it set `python -m openplaces.config --keep-personal-columns
  true` in their own config; no recipe key can switch that on.
  `export_delivery` withholds them from every file whatever the setting and
  raises `PersonalColumnError` on a recipe that lists one under `share`. The
  same step drops *unregistered* source headers either way, as an allow-list.
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
  to a private dataset, and that this project can read it, may be said in
  public. A recipe naming such a source is acceptable in git, subject to
  the rule below.
- **What may not be committed is anything the source's terms do not
  permit, or that evidences a use they do not permit.** Before committing
  a recipe for a source whose licence restricts redistribution,
  derivative works, or disclosure, read the terms and check:
  - **Schema disclosure.** If the terms can be read to treat data
    documentation or "any portion" of the data as confidential, keep the
    recipe (and any sidecar that reproduces the source's codes) out of git
    until the licensor or Boston University's review has cleared it.
  - **Derivative products.** Restricted evidence may inform an internal
    check; it must never reach a shipped output. Pin that with a test, as
    `test_restricted_shovels_columns_are_never_published` does.
  - **Scope of use.** Describe the actual use accurately in the recipe's
    notes, and do not describe or configure a use the terms do not cover.
- **Record the restriction on the `Source`** (`license`, `terms_url`,
  `redistribution_restricted: true`) so `bundle_terms` reports it. A
  no-resale clause ("not to be resold") forbids selling, not sharing:
  record it as `resale_restricted: true` and leave
  `redistribution_restricted` to describe sharing.
- **When in doubt, hold the file, do not delete it**, and say so in the
  handoff. The question is institutional, not an agent's to answer.
- **Shovels (`US-NC_property-shovels-2026`) is held**: its recipe is
  git-ignored while Boston University reviews its terms. Do not re-add
  its files or change git history about it; the state is in
  `plans/shovels-terms-and-the-withheld-connector.md`.

## Third-party code: attribute it, check its license
- Porting or adapting code from another project (a GitHub repo, a paper's
  reference implementation): name the source (repo/paper URL), its original
  author(s), and its license **in that file's own header**, not only in a
  README or a sibling file in the same directory.
- Before merging a port, confirm the upstream license is compatible with this
  repo's Apache-2.0 (MIT, BSD, Apache-2.0 are fine; LGPL and GPL are not
  automatically compatible and need a NOTICE-file carve-out, not silence).
  A vendored copyleft subtree with no license text marking it is a real gap;
  check for it rather than assuming every ported file got it right.
- Adapting a data *source* into a recipe is a parallel concern: check the
  source's license and redistribution terms the same way.

## Design limits
- Some techniques are kept out of this repository by decision of the
  maintainer. Contributors with access keep the list in
  `plans/design-limits/DESIGN_LIMITS.md` (a private repository cloned
  into the git-ignored `plans/` folder).
- Before planning or implementing work that touches record linking or
  matching, identifiers derived from geometry, grouping or merging of
  parcels or lots, reconciling values across sources, imagery or other
  trained models, transaction deduplication, price indices, or
  valuation models, read that file if it exists, and state in the plan
  which of its limits you checked.
- If the file does not exist on your machine, do not add a new
  technique in those areas without asking the maintainer first.
- Document the technical rationale for a new method in the code itself
  (why this approach, not a more obvious alternative).

## Code style
- 88 characters maximum for code (`ruff format` enforces it); 72 for comments.
- `isinstance(x, Foo | Bar)`, never a tuple.
- No sequences of line characters (`─`, `-`, `=`) in comments.
- No em-dashes (literal `—` or text-based `---`) in documentation, docstrings, or
  code comments. Use commas, colons, parentheses, or spaced hyphens (` - `).

## Docstrings
- NumPy style only. Avoid double backticks. American English ('meter', 'center').
- Sphinx cross-references only where they help navigation (important public
  functions or classes, workflow entry points, configuration objects).
- No .py filepath in a script's top-level docstring.
- Each class constructor gets its own docstring with its parameters and
  validation rules (no placeholders or duplicates).
- Docstrings describe the user-facing contract, accurately; keep the deep
  rationale and history of a decision, and put implementation notes (RAM
  heuristics and the like) in inline comments.
- **Quote a statistic that can move; do not avoid it** (maintainer's
  rule, 2026-09-22). A docstring states a measured number in one form, on
  one line: `Measured <YYYY-MM-DD> on <scope>: <claim>`, where the scope is
  an admin id, a region id or another single token (`US`, `CO`,
  `cheer-coastal-tx`, `admin-gadm-4.1`). `python -m
  openplaces.flow.measured_claims` lists every claim and reports `stale`
  where the scope's curated output or delivered bundle is newer than the
  claim's date. A stale number is documentation debt to re-measure, never
  a reason to delete the number.

## Directory structure
Write all plans to ``<repository_root>/plans/``. Never write a plan outside the
repository, not to a home directory and not to whatever scratch, notes, or plan
directory your agent tool manages for itself: plans there are invisible to
everyone else. Consult your own tool-specific instruction file for the directory
this rule names in your case.

Name a plan file after what it does, in kebab-case
(``geospine-split-skip-geometry-reprocessing.md``). Never accept an
auto-generated name built from random words or from the prompt that started the
session; rename it before writing.

Move a plan to ``plans/implemented/`` once its work has landed, prefixed with
the date it landed: ``YYYY-MM-DD_<same-descriptive-name>.md``. ``plans/README.md``
ranks the open plans; update it when adding or finishing one. It is an index:
keep it under 150 lines, one line per row, and put measurements, build logs and
history in the plan they describe, never in the index. Overwrite its state
block rather than appending to it.

Implement all complex plans (plans affecting the functionality of multiple src/ files or notebooks) on a new git worktree in ``<repository_root>/_<worktree_branch_name>/``. If the plan is simple (variable renaming, comments, single file), ask the user whether implementing the plan directly on 'main' is acceptable.

## Module READMEs
Before editing files in a folder that has a `README.md`, read it. The
detailed architecture reference lives there, beside the code:
`recipe`, `config`, `io/ingester`, `io/harmonizer`, `io/enricher`,
`io/curator`, `io/delivery` and `flow` (all under `src/openplaces/`), as
well as `recipes` and `notebooks`.

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
Layer 6  io/ingester/*, io/scrapers/*, io/aggregate, io/admin/*, io/admin_migration, io/delivery/*, io/transform, io/sale_records, io/cleanup/*
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

`table`, `geo/address` and `io/steps` sit low on purpose; their module
docstrings say why. `io/__init__` is a facade that only re-exports from
the six modules beside it, so a helper several stages share lives in one
of those modules, never in a stage folder. **A module in `io/` must not
share a name with a facade export**: `from openplaces.io import download`
yields the *function*, which is why the module that defines it is
`io/fetch`. A test patches the module the function under test lives in,
not the facade or a package file; a patch on those never reaches a
function looking names up in its own globals.

## Architecture overview

`openplaces` integrates property and geospatial data (parcels, buildings,
transactions, environmental datasets) for reproducible research. All data access
and processing is driven by **recipes** (YAML files, `src/openplaces/recipes/`;
loader reference in `src/openplaces/recipe/README.md`).

Identifiers (`core/schema.py`): `AdminId('US', 'NC', 'CUR')` -> `'US-NC-CUR'`
(level 1 country, 2 state, 3 county, 4 municipality; in New England level 3 is
the town and the county is skipped: `US-MA-SOM`); `Entity('parcel', 'massgis',
'2025')` -> `'parcel-massgis-2025'`; `DataSet('land-elevation', 'usgs', '3dep')`
for non-entity data; `Source` for a download. A recipe id is
`{admin_id}_{entity_or_theme}-{source}-{version}[_{filename}]`, e.g.
`US-MA_parcel-massgis-2025`.

### Entity model and stage roles (`core/schema.ENTITY_DEFINITIONS`)

**One row of each entity type means one thing, and these are easy to
confound.** `ENTITY_DEFINITIONS` holds the authoritative text; in short:

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
*details* of those properties declares `supplements: <roll id>` so that it
adds columns, never rows (`src/openplaces/recipe/README.md`).

The stages divide the work, and **harmonize keeps redundancy minimal**:

- **Harmonize** builds each entity's core table (its spine) and the keys
  that relate entities. An attribute lands once, on the entity it
  describes: a property's bedrooms belong on the property spine, not also
  on every parcel. Harmonize may derive a preliminary column only where an
  enrich step needs it to choose what to run on, and labels it
  preliminary.
- **Enrich** adds evidence from external sources to any entity, keyed by
  that entity.
- **Curate** assembles one dataset per entity type, aggregating or
  translating from the others where needed: a footprint's bedrooms come
  from summing its properties' in a curate recipe, not from a
  harmonize-time copy.

**The build order is fixed** (full order in
`src/openplaces/io/harmonizer/README.md`; `RecipeDAG` derives it and a test
pins it). A script that calls stages by hand has to follow it, and **a parcel
geospine built before one of its county's rolls was ingested silently lacks
that roll's values until it is rerun**.

**A property's id is the account number its assessor issued**, scoped by
its admin unit, and rows sharing an id merge into one. **Relationships
between entities are link tables**, one row per pair; `link_method` is a
fixed label, never a score, and no link is removed once written. **A
transaction carries what its document states and nothing it conveyed**:
a deed of several parcels is one spine row and several link rows. Both
are described in full in the harmonizer README.

**A county's rule goes into a patch recipe** at the county's scope, never
into a branch on a place in the national recipe or in `src/`
(`src/openplaces/recipes/README.md`).

### Pipeline: ingest, harmonize, enrich, curate

Each stage's reference is the README beside its code (`io/ingester`,
`io/harmonizer`, `io/enricher`, `io/curator`). Three invariants hold
everywhere:

- **Every parcel table is split into lots and properties at ingest**: a
  row stacked on a shared lot polygon is a property, not a parcel.
- **Imagery is fetched in memory per run and never cached**, because
  Google's terms prohibit storing it.
- **A cell openplaces itself filled must say so** in curate, through the
  `imputed` marker in its `{col}_source` (`provenance.mark_imputed`).

`export_delivery` pools a curated region into a bundle and **withholds what
restricted sources supplied**; a team bundle (`<region>.team`) is
published only on a password-protected team value page, never openly
(`io/delivery/README.md`). A narrow debug run of the DAG stops at
curate, so a one-county rebuild cannot overwrite a shipped regional file
(`flow/README.md`).

### Column naming (`core/attribute_registry.csv`)

The registry gives each well-known column its dtype, units, aggregation and
`sort` rank. Evidence columns carry a provenance suffix with exactly the
disambiguating components: point/building-level refs use entity-level + source
(`_building_nsi`, `_dwelling_overture`, because footprint/building/dwelling are
easily conflated); parcel refs use entity-level only (`_parcel`). Relational
counts use `n_{counted}s_per_{grouping}` (`n_parcels_per_footprint`). Output
column order comes from the suffix and the registry, never a per-recipe list.

### Talking to other people's servers

Full reference: `src/openplaces/config/README.md`.

- **Identify yourself.** Use `io.request_headers()` for every request;
  **never** send a browser User-Agent (a test fails the build on any
  `Mozilla/` string in `src/`).
- **Never agree to terms on the user's behalf.** Click-through sources go
  through `io.consent.require_terms_consent`; `accept_terms: true` in a
  recipe raises `ConsentNotDelegableError`. A recipe may refuse for a user,
  never consent for them; apply that asymmetry to any future decision with
  legal consequences.
- **Respect who a source says it is for.** A `usage_requirement` on a
  `Source` is set only when a person determined it from the terms, never
  inferred from the `license` text; `io.usage_profile` checks it against
  the user's declared profile at download time.
