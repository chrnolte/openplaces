<!-- Reference for `openplaces.io.ingester`, moved verbatim from AGENTS.md on
2026-09-28. AGENTS.md's "Maintaining this file" rules apply
here too: depth proportional to surprise, verified against the
current code. -->

# openplaces.io.ingester: reference

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
records the design: the split groups all rows by the source's own lot
identity and never verifies a row against other mapping or addressing
data, flags it as inconsistent, or normalizes it. Data on disk keeps the
old row shape until a county is re-ingested.

Public entrypoint:

`ingest(recipe, admin_ids, partition_ids, reprocess, redownload, verbose)`.
