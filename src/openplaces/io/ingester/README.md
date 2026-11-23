<!-- Reference for `openplaces.io.ingester`, moved verbatim from AGENTS.md on
2026-09-28. AGENTS.md's "Maintaining this file" rules apply
here too: depth proportional to surprise, verified against the
current code. -->

# openplaces.io.ingester: reference

**Stage 1 - Ingest** (`io/ingester/`, `io/ingester/table_ingester.py`):

`Ingester` orchestrates download, unzip, and processing of one recipe:
1. Resolves admin IDs to save/process/download (three potentially different levels).
   Without `reprocess`, a unit whose output exists is dropped; when none is left,
   the run prints one line naming the recipe and why (all built, or no unit at the
   save level) and returns without running a later step, except the backfill of
   missing `entity_links` files. A requested id outside the spine still raises.
2. For each download partition (admin unit × partition ID): downloads + unzips source,
   calls `TableIngester` to process each chunk. **A download holding no rows
   raises** (`download.EmptyDownloadError`, naming recipe, unit, source host
   and file) before any chunk is read: an empty answer from a service is not
   an observation that the unit has none. When the source has a download
   mechanism the empty file is deleted first, as the REST scraper discards a
   partial download, so a cached empty file is fetched again by the next run;
   a file placed by hand is kept. The check runs in two places, and
   neither parses a file twice:
   - **Before any read**, where a row count is cheap
     (`count_source_rows`: GeoJSON up to 64 KB, parquet, shapefile,
     GeoPackage, geodatabase, FlatGeobuf, header-only CSV, blank text)
     and the partition is one file.
   - **At the primary table's first unfiltered read**
     (`TableIngester._refuse_empty_read`, called from `_read_recipe_data`),
     for everything else: spreadsheets, Access tables, GeoJSON over
     64 KB, XML and JSON tables, archives read in place, one unit's
     `process_by.file_pattern` file or `layer_pattern` layer, and a
     `read_members` or `join_members` set as a whole. The frame is read
     anyway, so its length costs nothing; a read filtered by FID or
     bounding box says nothing about the file and is not checked. One
     blank member among others is legal (it is skipped, and in a
     `join_members` set the strict join names it); a set of members that
     together hold no rows raises. An empty `file_pattern` file or
     `layer_pattern` layer is part of a download other units read, so it
     is kept, and `empty_download_ok` may list the processing unit
     itself.

   Three paths stay unchecked by design. A **tile** with no features is
   an observation (open water, an empty quarter-degree), not a failed
   download. A **raster** has no rows. **Additional layers** are not
   checked: a secondary table of a roll (sales, permits) may be empty
   for a unit while the primary table is not. A read clipped to a
   bounding box (`use_spatial_mask`) is checked only where the format
   allows the cheap count before the read.
   A source that legitimately ships no rows for some units declares
   `empty_download_ok` (true, or a list of admin ids); the ingest then
   proceeds and prints one line.
   **Every table an ingest writes without rows says why**, in its parquet
   footer: `openplaces:empty_reason` and a short `openplaces:empty_detail`
   (`empty_outputs.save_empty_table`; read with `readers.empty_reason`,
   footer only). The reasons:
   - `not_published`: the source does not publish the unit or partition
     (a scraper or `partition_key_lookup` said so, no URL and no file for
     a partition, `file_pattern_missing: skip`).
   - `not_covered`: the source was read and holds rows, none for the
     unit. The detail gives the rows read, the filter (the
     `admin_id_column` and the unit's code, the bounding box, the split
     column) or says preprocessing (a `query`, the overlay) kept none.
   - `source_has_none`: the source itself holds no rows, which
     `empty_download_ok` allowed (or the table is one the empty-download
     check does not cover).
   - `marker`: the base file of a `download_by: partition` recipe, whose
     rows are in the `_all` file (`aggregate_by: single_file`) or the
     per-partition files. It only tells the skip check the unit was
     ingested and is never read as data.
   - `nothing_stacked`: the stacked-units layer of a parcel table whose
     lots each carry one record.
   - `declined`: the user's usage profile (`usage_requirement`) or a
     withheld terms consent (`io.consent`, which counts refusals so a
     scraper's None is told apart from "not published") declined the
     download. The detail names the mechanism and its key, never a
     profile or config value.

   Consumers read every reason as none except `marker`.

   The writers: a processed chunk saved without rows (`table_save`,
   decided from the rows read before and after the unit filter); a split
   that has no rows for a unit, recorded for the after-run fill; the
   after-run fill (`outputs._record_units_the_source_does_not_cover`),
   which also replaces a reasonless empty file and raises
   `UnexplainedEmptyIngestError` for a unit nothing explains; a unit no
   tile holds rows for; the stacked-units layer; and an aggregate or
   join of inputs that all hold no rows, which inherits their reasons
   (none if any input records none). An aggregate-mode `replace_by`
   that drops every row of an existing file stamps the inputs' reasons,
   or raises `UnexplainedEmptyIngestError` when not every input has one.

   **A table with no rows and no reason is not a build**
   (`readers.is_unexplained_empty`): the skip checks
   (`scope._resolve_output_admin_ids`, the aggregate-mode check and
   `partitions.get_ingested_partition_ids`) ingest the unit again, a
   batched ingest's `flow.dag.needs_rerun` reruns it, and
   `scope.unbuilt_outputs` (behind the harmonizer's
   `require_source_ingested`) raises `UnexplainedEmptySourceError`. Each
   costs one parquet footer read per existing output, never a row.
   `classify_empty_ingest_outputs.py` (plan
   `empty-ingest-outputs-say-why`) stamps legacy markers that are
   provable from the files beside them; every other legacy empty file is
   re-ingested. Open: the skip check of a `single_file` recipe still
   reads the base marker, so a new partition (a new year) is not picked
   up without `reprocess`, and a unit with truly no sales never gets an
   `_all` file, so its consumers raise `SourceNotIngestedError`.
3. After tile partitions: merges per-tile partials into per-admin files
4. In aggregate mode: calls `aggregate_to_admin_level()` to merge process-level chunks
   into save-level files

`TableIngester` handles reading, transforming, and saving one table from an
already-resolved source file. It applies column mappings, type casts, spatial filtering,
and the attribute registry type checks.

**Both classes are assembled from mixins, one module per concern**
(since 2026-09-29; until then `__init__.py` held 2,518 lines and
`table_ingester.py` 1,710). `Ingester` keeps its constructor, `ingest`,
the per-partition driver and table processing in `__init__.py`, and
inherits `scope.py` (which units are saved, processed, downloaded),
`partitions.py` (partition, tile and placeholder ids), `download.py`
(URLs, paths, download, unzip, scrapers) and `outputs.py` (entity links,
tile merges, table joins, aggregation); module-level helpers sit in
`_helpers.py`. `TableIngester` keeps `process` in `table_ingester.py` and
inherits `table_scope.py`, `table_read.py`, `table_preprocess.py`,
`table_ids.py` and `table_save.py`. All state stays on the instance.
**A test patches the module the method lives in**
(`openplaces.io.ingester.download.unzip`,
`openplaces.io.ingester.table_read.read_access_table`), because a method
looks names up in its own module's globals; a patch on the package or on
`table_ingester` reaches nothing now, and `monkeypatch` raises rather
than passing silently. `get_admin` is the one name bound in two mixins
(`scope` and `partitions`), so a test that fakes the spine patches both.

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

**A parcel's id is its `geo_id` wherever that hash is unique.**
Preprocessing suffixes every copy of a shared hash (`X-1`, `X-2`), since
a stack's rows share one polygon; after the split, a row whose `geo_id`
no other row carries any more (a collapsed stack, a row shipped twice,
a unioned lot) is indexed by the bare hash
(`table_ids.drop_resolved_suffixes`). The footprint overlay keys
parcels by `geo_id`, so a kept `X-1` named a parcel no table carried
(Harris County TX, 2026-10-17: 31,585 overlay links). A hash still
shared keeps its suffixed ids.

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
records the design: the split groups all rows by the source's own lot
identity and never verifies a row against other mapping or addressing
data, flags it as inconsistent, or normalizes it. Data on disk keeps the
old row shape until a county is re-ingested.

**A source already split by admin unit is read one piece per unit**, not
filtered by row. `process_by.file_pattern` names one file per unit (FL DOR
rolls); `process_by.layer_pattern` names one layer per unit of a single
multi-layer file (MetroGIS's `Parcels{co_name_file}` county layers in one
file geodatabase). Both fill `{partition_id}` and the crosswalk's raw-code
column from the same reverse crosswalk lookup
(`table_scope._fill_unit_pattern`), and may be combined. A
`layer_pattern` that names a layer the file lacks raises with the layer
name and the file's layers; it never falls back to the first layer,
which would save one unit's rows under another unit's id. The recipe's
`layer` stays the default for every other read.

**A row matching no unit of `process_by.admin_id_crosswalk` is reported**
(`table_scope._prepare_table_fid_filter`): it falls in no unit's chunk,
so no file saves it, and the ingest warns with the count and the first
unmatched codes. When no row matches at all, it raises.

Public entrypoint:

`ingest(recipe, admin_ids, partition_ids, reprocess, redownload, verbose)`.
