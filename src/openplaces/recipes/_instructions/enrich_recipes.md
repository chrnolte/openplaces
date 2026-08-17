# Writing an enrich recipe

`stage: enrich` reads a harmonized entity and writes an **entity-keyed evidence
table** beside it — observations or model outputs. Enrichment never selects a
canonical value, reconciles a disagreement, or fills an unrelated gap. Those
are curate's job.

Read [`../README.md`](../README.md) for naming and style.

## Shape

```yaml
stage: enrich
admin_id: US
entity:
  entity_type: parcel
dataset: parcel-placeslab-fmv2026        # names the evidence table
entity_recipe: US_parcel-spine-2026      # what it reads
reference_parcel_recipe_id: US_parcel-placeslab-fmv2026   # or image_recipe:
pipeline:
  - step: enrich_parcels_from_reference_crosswalk
    min_overlap_m2: 10
    area_ratio_tolerance: 0.01
    save_crosswalk: true
```

Output lands next to the harmonized entity as
`{entity_output}_{dataset}.parquet`, keyed by the entity id.

## Evidence columns keep their provenance

Name the column for where it came from, not for what you wish it were:
`roof_shape_brails`, `n_stories_brails`, `occupancy_type_nsi`. Curate is what
turns `roof_shape_brails` into a canonical `roof_shape`.

## **Check the reference actually covers your geography**

The trap that has cost the most here. An enrichment whose reference data does
not cover the admin unit still *runs*: it finds nothing, writes an evidence
file containing only the key columns, and reports success in about zero
seconds. Then curate fails for every county, because `merge_enrichments`:

- **skips** an enrichment whose output is *missing* — the correct state for an
  uncovered geography;
- **raises** on one that *exists but lacks a requested column*.

Creating an empty file converts a clean skip into a hard failure.
`US_parcel_parcel-placeslab-fmv2026` covers 8 Massachusetts counties and
PA-Delaware only; running it for North Carolina broke curate for 13 counties.

**Before including an enrichment in a run, confirm its
`reference_parcel_recipe_id` / `image_recipe` has data for those admin units.**
A zero-second stage is not a fast stage.

Recovery, if it happens: delete the empty `*_{dataset}.parquet` evidence files
and re-run curate alone.

## Image-based enrichment

Steps that read imagery build their input from the metadata of the recipe's
`image_recipe`, ingested per building at admin level 4. Admin units without
imagery are skipped rather than erroring. Missing images are fetched on first
ingest; `redownload` only re-fetches images already on disk.

Imagery is billed per request, so the notebook lane gates it behind
`--include_streetview` / `--include_googlesatellite`, and drops the dependent
enrichment recipes entirely when those are off rather than running them to a
no-op. Follow that pattern for any new billed source.

## Re-running

Enrichment evidence is keyed to the harmonized entity's ids. If a spine change
alters those ids — which happens when a new source supplies geometry that a
statewide layer used to — stored evidence no longer joins and must be rebuilt.
If the ids are stable (geometry comes from footprint sources, say, and only
attributes changed), existing evidence survives and re-running is wasted work,
or worse if the reference no longer covers the area.
