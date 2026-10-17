<!-- Reference for `openplaces.recipe`, moved verbatim from AGENTS.md on
2026-10-17. AGENTS.md's "Maintaining this file" rules apply here too:
depth proportional to surprise, verified against the current code.
How to write a recipe is in `src/openplaces/recipes/README.md`; this
file describes the code that loads and resolves them. -->

# openplaces.recipe: reference

Recipes are YAML files that define how to ingest, harmonize, enrich, or curate
a dataset. They are stored in a path-encoded directory structure:

```
src/openplaces/recipes/{admin_id_path}/{entity_type_or_theme_path}/{source}/{version}/{recipe_id}.yaml
```

A recipe ID encodes its parts: `{admin_id}_{entity_type_or_theme}-{source}-{version}[_{filename}]`, e.g. `US-MA_parcel-massgis-2025`.

## Recipe roots

The bundled tree is one root; an installed package can
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

## Key recipe fields

- `admin_id` - geographic scope
- `entity` or `dataset` - what is being ingested
- `description` - short human-readable summary; kept to 25 words or less, with any
  implementation detail in `notes` instead
- `stage` - `'ingest'`, `'harmonize'`, `'enrich'`, `'curate'`
- `columns` - mapping from openplaces attribute names to source column names
- `download_by` - controls download partitioning: `admin_level`, or `partition:
  year|year_month|table|tile_id|latlon_tile`. `year_month` computes a YYYY-MM date range,
  `table` takes an explicit table-name list; unrecognized partition types raise
  `NotImplementedError`.
- `process_by` - controls chunking granularity during processing (`admin_level`, `admin_id_column`).
  For sources that ship one already-split file per admin unit inside a single shared
  download (rather than one file with an in-data admin column to filter rows by), use
  `file_pattern` instead of `admin_id_column`: a filename pattern relative to the recipe's
  heap directory, with `{partition_id}` and the crosswalk's raw-code column name (e.g.
  `{admin3_id_admin2}`) as placeholders, resolved per admin unit
  (`TableIngester._resolve_file_pattern_path`).
- `save_to` - output location (`data_dir` from `STANDARD_DIRS`, `admin_level`, optional
  `retention` class). When `save_to.admin_level` is coarser than `process_by.admin_level`,
  the Ingester aggregates intermediate files to that level.
- `join_partitions_by` - for `download_by: {partition: table}` recipes, left-joins the
  per-table outputs into one entity file per admin unit after ingest (`join_key_name`,
  optional `keep_original`). See `io.aggregate.join_partitions_by_index`.
- `additional_layers` - list of secondary entities extracted from the same source file (e.g., a property table alongside a parcel table)
- `entity_recipe` - predecessor entity recipe used by enrichment or curation
- `pipeline` - ordered named steps for harmonization, enrichment, or curation
- `patches` / `pipeline_patch` - a patch recipe: a file at a county's or
  state's scope that names a national harmonize or curate recipe and
  lists operations on its pipeline (`set`, `extend`, `insert_before`,
  `insert_after`, `replace`, `remove`, each addressing a step by name,
  `occurrence` when the name repeats). `recipe.apply_recipe_patches`
  applies the patches whose scope covers the unit, coarse to fine, once
  per unit inside the harmonizer and the curator; the output footer
  records them (`openplaces:recipe_patches`). Auto-discovery never picks
  a patch recipe as an entity's recipe and the DAG keeps one recipe id
  per job.

## Supplements: detail tables of a roll

A table whose rows are *details* of a roll's properties (one row per
building component) declares `supplements: <roll id>` so that it adds
columns to them, never rows. The roll may sit at a containing scope (a
county appraiser's building table detailing the state DOR roll; a city
table detailing MassGIS's statewide layer), and
`supplements_layer: <entity type>` names an `additional_layers` entry of
the host when the roll has no recipe id of its own
(`recipe.get_supplemented_table` validates both). A supplement joins its
roll on `parcel_id_local` unless `supplements_key: <column>` names a
column both tables carry, for a roll whose `parcel_id_local` is built
from an id the detail table lacks (Travis County TX links parcels on
`geo_id` while its detail tables carry `prop_id`); keyed supplements are
skipped by every other auto-discovered join, the parcel geospine's
included.

## Key functions

The package has one module per concern since 2026-09-30: `loading`,
`tables`, `naming`, `discovery`, `patches`, `dependencies`, `output`.
Every name is re-exported by the package, and a test that patches
`openplaces.recipe.<name>` reaches a caller that imports it from the
package at call time.

- `get_recipe(admin_id, entity, ...)` / `get_recipe_by_id(recipe_id)` - load a recipe dict
- `find_entity_recipe_id(...)` - select by stage rank
  `ingest < harmonize < enrich < curate`; pass `stage=` when a caller needs
  a specific predecessor rather than the latest pipeline product
- `get_output_path(recipe, admin_id, partition_id, geo, layer)` - resolve the parquet path where output is written
- `get_save_admin_level(recipe)` / `get_process_admin_level(recipe)` - determine output/processing granularity
- `get_layers(recipe)` - list secondary layers in `additional_layers`
- `build_table_recipe(primary, layer_spec)` - merge primary recipe with an additional_layers spec

## File layout on disk

Output files are parquet, stored under the configured `cfg.data_root`:
```
{data_dir}/{admin_id_path}/{entity_or_dataset_path}/{admin_id}_{entity}_[suffix].parquet
```

Geometry sidecar: as above, but with a `'_geo'` suffix added to the filename.

`path()` in `path.py` generates these paths from
`(admin_id, entity, dataset, filename, root)`. `external_path()` /
`heap_path()` generate paths for downloaded and unzipped source files,
respectively.
