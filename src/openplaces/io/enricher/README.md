<!-- Reference for `openplaces.io.enricher`, moved verbatim from AGENTS.md on
2026-09-28. AGENTS.md's "Maintaining this file" rules apply
here too: depth proportional to surprise, verified against the
current code. -->

# openplaces.io.enricher: reference

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
  `{recipe_id}_column-notes.csv` beside the reference recipe. Where the
  reference ships no geometry, a crosswalk its source delivered
  (`delivered_crosswalk_table`) is read instead of built. It names the
  parcels of the bundle it was built against, so it is refused unless
  99 percent of its own ids are in the spine; at least 90 percent of the
  spine's ids must appear in it, and a share between 90 and 99 percent
  warns, since parcels newer than the reference have no reference
  parcel to name them.
- `overlay.py` — `overlay_reference_polygons`: per entity, the share of
  its area inside a reference polygon dataset (a thematic layer whose
  polygons do not overlap, e.g. MassGIS protected open space) and the
  attributes and id of the polygon covering most of it. The reference
  is read at its own save level (a state file for a town's run); a unit
  outside its scope gets the declared columns all-missing. `withhold`
  leaves a label empty on polygons a recipe names (site names on
  privately owned land). The share is a sum of overlaps unless the
  recipe sets `dissolve: true`, which counts land under two overlapping
  records once (PAD-US easements recorded by two sources); summing such
  a layer overstates the share. `where` restricts the reference to
  rows with given codes, and `flag_column` with `min_share` writes a
  boolean at a threshold the recipe states and justifies.
- `zonal.py` — `zonal_stats`: per-entity raster statistics over polygon
  geometry, dispatching to one of three backends in `geo/raster.py`
  (`exactextract` for fractional pixel-area weighting, `rasterstats`, or
  `rasterized` for a burn-and-groupby). Requires `spine_geom: true`.
  With `rasterized`, the spine burned onto a raster window is kept in
  `EnrichState.grid_cache` and reused by every later raster on the same
  window, CRS and pixel grid (each year of an annual layer), so a further
  year costs one read and one pass over the burned pixels. The cache
  belongs to one admin unit's run and is emptied when its pipeline ends;
  a raster on another grid simply burns its own.
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

**A recipe with `series: true` writes a long table, and only what its
requests ask for** (`series.py`). One row per entity and period
(index `{entity id}`, `period`; `period_start` too where a step sets
`period_length_years` above 1), one column per variable. Every enabled
step carries `years`, and its `column` has no `{year}`: the year is the
row. The `requests` block names where requests come from: a recipe's
output (`recipe_id`, whose key gives the DAG its edge; another entity's
rows reach this one through their harmonize link table) or the spine
(`from: spine`), each with a `date` column or `now`, and
`offsets_years`. Each step's `period_rule` (default `before`, the
latest period that ended before the date: a sale in 2016 takes 2015;
`at_or_before`; `now`) maps a request to a period; a date before the
first period gets no row, and two requests one period answers cost one.
A series recipe without `requests` refuses to run; `requests: all`
asks for every entity at every period and prints the row count.

- **The rows are the record of done.** A rerun computes the pairs the
  file lacks and appends them; with none missing it writes nothing.
  `reprocess` recomputes everything. The footer
  (`openplaces:series`) records the request sources with their row
  counts, the periods present and each column's periods and rule, which
  `join_temporal_snapshot` reads. A stored wide file is refused, not
  appended to. A unit whose requests name no pair is saved without rows
  and with the empty reason `nothing_requested`.
- **Each period is sampled on the whole spine**, not on the requested
  entities: the rasterized back end gives a pixel under two overlapping
  polygons to one of them, so a frame without the other gives it to the
  first, and a pair's value would depend on what else was requested.
  Measured 2026-11-24 on CO-AN-CV: the long table of `requests: all` pivoted equals the wide output in all 1,872,584 cells.
  Measured 2026-11-24 on CO-AN-CV: transaction requests plus every parcel at now give 9,619 rows in 8.4 s and 0.43 MB against the wide 9,554 by 190 in 14.7 s and 3.23 MB.
- `merge_enrichments` refuses a series recipe; curate joins it by date.

Raster-consuming steps name their raster by a path relative to the configured
`external` directory, resolved by `path.resolve_raster_path()`, so a recipe
stays portable across machines. An absolute path passes through unchanged.

Examples include roof-shape, occupancy, and story-count evidence from imagery,
and the same attributes read off a precomputed inventory
(`US-NC_footprint_building-cheer-v0`). Evidence columns retain
provenance-oriented names such as `roof_shape_brails`, `n_stories_brails`, and
`foundation_type_building_cheer`.

Public entrypoint: `enrich(recipe, admin_ids, entity_recipe_id, reprocess, verbose)`.
