# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code style
- Line length: 88 characters maximum
- Use the `|` union syntax for `isinstance` checks, never a tuple:
  `isinstance(x, Foo | Bar)` ✓
  `isinstance(x, (Foo, Bar))` ✗
- Do not use sequences of lines (`─`, `-`, `=`) in comments

## Docstrings
- Use NumPy-style docstrings.
- Avoid double backticks.
- Only add Sphinx cross-references when they are genuinely useful for navigation, such as:
  - Important public functions or classes in this package
  - Key workflow entry points
  - Custom data structures or configuration objects
  - Specialized external concepts that readers may not know
- Do not include the .py filepath in the top-level docstring of scripts

## Commands

```bash
conda activate openplaces   # required for all dev/test work

# Linting and formatting
ruff check src/             # lint
ruff format src/            # format

# Testing
pytest                      # all tests
pytest tests/test_path.py   # single file
pytest -k "test_name"       # single test by name
```

## Module layer hierarchy

```
Layer 0  core
Layer 1  config, path
Layer 2  recipe
Layer 3  io/__init__
Layer 4  io/readers
Layer 5  geo/*
Layer 6  io/ingester, io/table_ingester, io/aggregate, io/admin, io/transform, io/harmonize
Layer 7  viz/*
Layer 8  api.py
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

Recipes are YAML files that define how to ingest or harmonize a dataset. They are stored in a path-encoded directory structure:

```
src/openplaces/recipes/{admin_id_path}/{entity_or_theme_path}/{source}/{version}/{recipe_id}.yaml
```

A recipe ID encodes its parts: `{admin_id}_{entity}-{source}-{version}[_{filename}]`, e.g. `US-MA_parcel-massgis-2025`.

Key recipe fields:
- `admin_id` — geographic scope
- `entity` or `dataset` — what is being ingested
- `stage` — `'ingest'` or `'harmonize'`
- `columns` — mapping from openplaces attribute names → source column names
- `download_by` — controls download partitioning (by `admin_level`, `partition: year|table|tile_id|latlon_tile`)
- `process_by` — controls chunking granularity during processing (`admin_level`, `admin_id_column`)
- `save_to` — output location (`data_dir`, `admin_level`). When `save_to.admin_level` is coarser than `process_by.admin_level`, the Ingester runs in aggregate mode.
- `additional_layers` — list of secondary entities extracted from the same source file (e.g. a property table alongside a parcel table)
- `pipeline` — list of named steps for harmonization recipes

Key recipe functions (`recipe.py`):
- `get_recipe(admin_id, entity, ...)` / `get_recipe_by_id(recipe_id)` — load a recipe dict
- `get_output_path(recipe, admin_id, partition_id, geo, layer)` — resolve the parquet path where output is written
- `get_save_admin_level(recipe)` / `get_process_admin_level(recipe)` — determine output/processing granularity
- `get_layers(recipe)` — list secondary layers in `additional_layers`
- `build_table_recipe(primary, layer_spec)` — merge primary recipe with an additional_layers spec

### Data pipeline: ingest → harmonize

**Stage 1 — Ingest** (`io/ingester.py`, `io/table_ingester.py`):

`Ingester` orchestrates download, unzip, and processing of one recipe:
1. Resolves admin IDs to save/process/download (three potentially different levels)
2. For each download partition (admin unit × partition ID): downloads + unzips source, calls `TableIngester` to process each chunk
3. After tile partitions: merges per-tile partials into per-admin files
4. In aggregate mode: calls `aggregate_to_admin_level()` to merge process-level chunks into save-level files

`TableIngester` handles reading, transforming, and saving one table from an already-resolved source file. It applies column mappings, type casts, spatial filtering, and the attribute registry type checks.

Public entrypoint: `ingest(recipe, admin_ids, partition_ids, reprocess, redownload, verbose)`.

**Stage 2 — Harmonize** (`io/harmonizer/`):

`Harmonizer` runs a composable step pipeline, executing steps declared in the recipe's `pipeline` list. Each step is a function registered via `@_register('step_name')` in one of the sub-modules:
- `spine.py` — build/merge the primary entity GeoDataFrame (`resolve_spine`)
- `links.py` — join to reference datasets spatially or via crosswalks (`link_to_reference`)
- `attributes.py` — compute/enrich attributes (`add_attributes`, `classify_footprint_role`, ...)
- `filter.py` — subset rows (`filter_entities`)
- `discover.py` — discover available data sources for an admin unit

All steps share a `HarmonizeState` dataclass (spine, references, crosswalks, overlays, metadata). Each step receives state and returns the updated state.

Public entrypoint: `harmonize(recipe, admin_ids, reprocess, verbose)`.

### Data access (public API, `api.py`)

```python
import openplaces as op

op.get_admin(admin_id, level, geom, recipe)        # load admin unit table/geodataframe
op.get_admin_ids(admin_level, admin_id)            # list admin ID strings
op.get_entities(recipe, admin_id, geom, layer)     # load entity parquet
op.get_dataset(recipe, admin_id, partition_id)     # load dataset parquet (or path for rasters)
op.ingest(recipe, admin_ids, ...)
op.harmonize(recipe, admin_ids, ...)
op.aggregate(recipe, admin_level, ...)
```

### File layout on disk

Output files are parquet, stored under the configured `cfg.data_root`:
```
{data_dir}/{admin_id_path}/{entity_path}/{admin_id}_{entity}_[suffix].parquet
{data_dir}/{admin_id_path}/{entity_path}/{admin_id}_{entity}_geo.parquet  ← geometries sidecar
```

`path()` in `path.py` generates these paths from `(admin_id, entity, dataset, filename, root)`. `external_path()` / `heap_path()` generate paths for downloaded/unzipped source files.

### Attribute registry (`core/attribute_registry.csv`, `core/attribute_registry.py`)

Maps well-known column names to their expected data type, units, and default aggregation function. Used by the harmonizer and ingester to validate dtypes and drive groupby aggregations without hardcoded column lists. Loaded once and cached via `@cache`.

### Configuration (`config.py`)

`cfg` (singleton `OpenPlacesConfig`) holds directory paths (`data_root`, `dir_core`, `dir_external`, `dir_heap`, etc.) and CRS. Loaded from `~/.config/openplaces/<username>.yaml` (user overrides) and `./openplaces.yaml` (project defaults). The `cfg.crs` is the project-wide target CRS for geometries.
