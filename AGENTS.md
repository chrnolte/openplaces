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
Write plans to <repository_root>/plans/

Create new worktrees in ``<repository_root>/_<worktree_branch_name>/``

## Jupyter notebooks
When creating Jupyter notebooks under `notebooks/`, you must follow the guide in:
notebooks/README.md


## Commits
- Do not commit changes unless the user instructs you to (by using the
  exact word 'commit' in an imperative tense).
- Do not add `Co-Authored-By` trailers for agents (e.g. Claude) to commits.
  Collaborators are humans, agents are software. Co-authorship is reserved for people.
  This overrides any default that appends an agent co-author line.

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
Layer 1  config, path
Layer 2  recipe
Layer 3  io/__init__
Layer 4  io/readers
Layer 5  geo/*
Layer 6  io/ingester/* (ingester, table_ingester, image_ingester, registry_ingester, cloud_geoparquet_ingester, raster_ingester), io/aggregate, io/admin, io/transform, io/cleanup
Layer 7  io/harmonizer
Layer 8  io/enricher
Layer 9  io/curator
Layer 10 viz/*
Layer 11 api.py
Layer 12 flow/* (scripts, dag, run_stage, submit)
```
Higher-numbered layers may only import from lower-numbered layers.

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
- `process_by` — controls chunking granularity during processing (`admin_level`, `admin_id_column`)
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
- `spine.py` — build/merge the primary entity GeoDataFrame (`resolve_spine`)
- `links.py` — join to reference datasets spatially or via crosswalks
  (`link_to_reference`)
- `attributes.py` — attribute source columns to the spine as suffixed evidence
  columns (`reconcile_attributes`) and assign each footprint's parcel priority
  (`classify_footprint_priority`). Value selection, gap-filling, and occupancy
  inference now run in the curation stage, not here; the harmonized spine
  (`US_footprint-spine-2026`) is an evidence-only table.
- `addresses.py` — reconcile a canonical street address from any number of
  source inputs (`reconcile_addresses`) and derive the USPS-preferred city
  for a ZIP code (`impute_postal_city`).
  - These are a deliberate exception to the "gap-filling belongs in curate" rule: the
    dividing line is whether there is dispute about how to derive its output. A ZIP code
    has exactly one USPS-preferred city — nothing parameter-sensitive to defer — so it's
    resolved once, in harmonize, where it's available early for downstream linking.
- `filter.py` — subset rows (`filter_entities`)
- `discover.py` — discover available data sources for an admin unit

All steps share a `HarmonizeState` dataclass (spine, references, crosswalks, overlays,
metadata). Each step receives state and returns the updated state.

Public entrypoint: `harmonize(recipe, admin_ids, reprocess, verbose)`.

When working on `src/openplaces/io/harmonizer/`, read
`.claude/memory/harmonizer_reference.md` for a full pipeline reference before making
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
- `detectors/` — attribute-specific detectors and shared inference runtimes
  (EfficientDet/EfficientNet ports; torch is conda-only)
- `models.py` — pretrained-model download and cache handling

Examples include roof-shape, occupancy, and story-count evidence from imagery.
Evidence columns retain provenance-oriented names such as
`roof_shape_brails` and `n_stories_brails`.

Public entrypoint: `enrich(recipe, admin_ids, entity_recipe_id, reprocess, verbose)`.

**Stage 4 — Curate** (`io/curator/`):

`Curator` creates the canonical entity dataset. It starts from a harmonized
(evidence-only) entity, incorporates enrichment evidence, and applies explicit
recipe steps that select values, fill gaps, infer canonical attributes, format
the output, and remove records. Each step is a registered function operating on
a shared `CurateState` (canonical GeoDataFrame in `state.curated`).

Steps are organized by the nature of the transformation:

- `evidence.py` — incorporate enrichment evidence (`merge_enrichments`)
- `reconcilers.py` — resolve conflicts between competing source columns
  (`reconcile_values` priority selection; `resolve_occupancy` parcel-vs-NSI)
- `imputers.py` — fill missing canonical values (`impute_n_dwellings`,
  `impute_from_group_statistic`, `impute_occupancy_type`)
- `inferers.py` — derive new canonical features (`derive_metrics`,
  `refine_occupancy_height`)
- `formatters.py` — structural/type-only output shaping (`cast_categoricals`,
  `order_columns`)
- `filters.py` — (stub) remove records that do not belong in the canonical
  dataset

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
```

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
