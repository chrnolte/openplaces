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
- See `DISCLAIMER.md` for the project's broader privacy and liability posture.

## Intellectual property: ownership is under review
- Ownership of this repository's IP — Boston University vs. individual
  contributors, and how any personal-time contributions are separated from
  grant-funded or in-scope faculty work — is currently under institutional
  review (see `DISCLAIMER.md`). Until that's resolved: never add a
  "Copyright <name>" header to a new file, and never edit
  copyright/ownership/hosting language in `LICENSE.md`, `README.md`, or
  `DISCLAIMER.md` without asking the user first, even to "fix" or "simplify"
  it — a wrong assertion here is a legal problem, not a style one.
- This includes the repository's hosting arrangement (currently a personal
  GitHub account, not an institutional one) — don't propose or perform an org
  transfer, mirror, or similar hosting change unprompted.

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
  First American chief among them — whose claims cluster around three
  specific technique *shapes*, not the general goal of "link/match
  property records" (which itself isn't patentable, only a particular
  claimed method for doing it is):
  1. Geometric neighbor/"community" detection (buffer-enlarge, union,
     reduce a group of parcel boundaries) used to impute or validate a
     *missing address number* by interpolating between neighbors.
  2. A trained machine-learning model used to score match probability
     between two property/record representations (as opposed to a fixed,
     deterministic rule set with no learned parameters).
  3. Detecting records *internally inconsistent* with their own source's
     mapping/address data, grouping them, and normalizing the group.
  A new feature that does one of these three things, in this domain,
  deserves a specific check against that sub-area before merging — not a
  general "we checked patents once" assumption. `openplaces`'s existing
  `parcel_id_local`/`geo_id` matching is deterministic string/geometry
  fingerprinting with no learned parameters and no neighbor-comparison or
  address-imputation step, which is why it reads as a different mechanism
  from all three shapes above — that reasoning doesn't automatically carry
  over to a new ML-based imputation or inference feature, which may
  resemble shape 2 much more closely by design.
- **Process for a new imputation/inference/matching/valuation feature
  touching parcel, property, or transaction data**: (1) identify the
  specific technique, not just the goal, and check whether it resembles
  one of the three shapes above or another known patent in this space;
  (2) if it does, flag it to the user explicitly before merging — this is
  a judgment call for a human, not something an agent should silently wave
  through, the same posture as the IP-ownership section above; (3) where
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
Layer 3  io/__init__, geo/address
Layer 4  io/readers, table
Layer 5  geo/* (except geo/address, above)
Layer 6  io/ingester/* (ingester, table_ingester, image_ingester, registry_ingester, cloud_geoparquet_ingester, raster_ingester), io/scrapers/*, io/aggregate, io/admin, io/delivery, io/transform, io/cleanup
Layer 7  io/harmonizer
Layer 8  io/enricher
Layer 9  io/curator
Layer 10 viz/*
Layer 11 api.py
Layer 12 flow/* (scripts, dag, run_stage, submit)
```
Higher-numbered layers may only import from lower-numbered layers.

Two placements are worth knowing. `table` holds the registry-driven row
helpers (`aggregate_rows`, `add_unique_suffix`, the `join_nonnull_*`
functions); they sit below `geo/` because `geo/crosswalk` and `geo/ids`
call them, and keeping them in `io/aggregate`/`io/transform` created a
module-level import cycle. `io/aggregate` and `io/transform` re-export
them, so the older import paths still work. `geo/address` is listed
separately because it depends on nothing above `recipe`, which is what
lets `table` use `strip_unit_suffix` without reintroducing that cycle.

## Architecture overview

`openplaces` integrates property and geospatial data (parcels, buildings, transactions, environmental datasets) for reproducible research. All data access and processing is driven by **recipes** (YAML files).

### Core schema (`core/schema.py`)

Four key identifiers form the naming system used throughout the codebase:

- **`AdminId`** — hierarchical geographic identifier, e.g. `AdminId('US', 'MA', 'MI')`. Level 0 = global, 1 = country, 2 = state/region, 3 = county/district, 4 = municipality/town. String form uses `-` separator: `'US-MA-MI'`.
- **`Entity`** — a data entity defined by `entity_type` + `source` + `version`, e.g. `Entity('parcel', 'massgis', '2025')`. String form: `'parcel-massgis-2025'`. Valid entity types are in `ENTITY_TYPES` (parcel, building, footprint, property, transaction, admin, ...).
- **`DataSet`** — a non-entity dataset defined by `Theme` + `source` + `version`, e.g. `DataSet('land-elevation', 'usgs', '3dep')`. Themes are hierarchical, separated by `-`, with the top level constrained to `TOP_LEVEL_THEMES` (land, landcover, water, built, people, risk, ...).
- **`Source`** — a data source with download URL(s), portal URL, DOI, etc.

### Recipes (`recipe.py`, `src/openplaces/recipes/`)

Recipes are YAML files that define how to ingest, harmonize, enrich, or curate
a dataset. They are stored in a path-encoded directory structure:

```
src/openplaces/recipes/{admin_id_path}/{entity_type_or_theme_path}/{source}/{version}/{recipe_id}.yaml
```

A recipe ID encodes its parts: `{admin_id}_{entity_type_or_theme}-{source}-{version}[_{filename}]`, e.g. `US-MA_parcel-massgis-2025`.

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

Public entrypoint:

`ingest(recipe, admin_ids, partition_ids, reprocess, redownload, verbose)`.

**Stage 2 — Harmonize** (`io/harmonizer/`):

`Harmonizer` runs a composable step pipeline, executing steps declared in the recipe's
`pipeline` list. Each step is a function registered via `@_register('step_name')` in one
of the sub-modules:
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
- `attributes.py` — attribute source columns to the spine as suffixed evidence
  columns (`reconcile_attributes`), assign each footprint's parcel priority
  (`classify_footprint_priority`), and build the combined land-use label the
  parcel classifier votes on (`derive_use_classes`, which joins an ordered
  `columns` list, default `use_group` + `use_subgroup`; the parcel spine
  appends `building_style` last so a county whose land-use text carries no
  occupancy signal can contribute one from its structure description).
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
  `classify_occupancy`, `detect_n_stories`); image-based steps build their
  input from the metadata of the recipe's `image_recipe` (per-building imagery
  ingested at admin level 4; admin units without imagery are skipped). Missing
  imagery is fetched automatically on first ingest; `redownload` only re-fetches
  images that already exist on disk (cached images are otherwise reused).
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
- `reconcilers.py` — resolve conflicts between competing source columns
  (`reconcile_values` priority selection; `resolve_occupancy` parcel-vs-NSI;
  `resolve_by_vote`, the single voting seam every curate classification
  resolves through)
- `imputers.py` — fill missing canonical values (`impute_n_dwellings`,
  `impute_from_group_statistic`, `impute_occupancy_type`)
- `inferers.py` — derive new canonical features (`derive_metrics`,
  `derive_indicators` — named indicator columns holding values, never
  pre-thresholded booleans; every cutoff lives in the vote decisions)
- `formatters.py` — structural/type-only output shaping (`cast_categoricals`,
  `order_columns`)
- `filters.py` — (stub) remove records that do not belong in the canonical
  dataset

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

### Delivery bundles (`io/delivery.py`)

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

A `share: delivery:` block names the region: `admin_level` (the bundle's own
level) and `admin_ids` (the units pooled into it). The declared member list wins
over walking the admin hierarchy, because a region is rarely all of a state's
children -- the CHEER region is 45 of North Carolina's 100 counties. With it
declared, `export_delivery(recipe_id)` needs no other argument, and
`delivery_paths(recipe)` is the single resolver both the writer and the
orchestrator use, so their idea of the output files cannot drift.

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
plus one terminal **`deliver`** job when the target recipe declares a delivery
region. `deliver` is not a recipe stage -- recipe stages are ranked
(`ingest < harmonize < enrich < curate`) and drive `find_entity_recipe_id`, so a
fifth would ripple into recipe resolution for nothing. It is a node kind this
graph derives, the way `extra_outputs` derives link sidecars from `save_link`.

Scope decides whether it runs: an unscoped run builds and ships the declared
region, a run naming the region (or covering every member) ships it, and a
narrower debug run stops at curate so a one-county rebuild cannot overwrite a
shipped regional file. `--config deliver=true|false` overrides either way.

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
`dir_external`, `dir_heap`, etc.) and CRS.
