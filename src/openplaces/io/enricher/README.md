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
