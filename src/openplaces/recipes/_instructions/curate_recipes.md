# Writing a curate recipe

`stage: curate` produces the **canonical entity** — the deliverable. It starts
from an evidence-only harmonized spine, merges enrichment evidence, then
selects values, fills gaps, infers features, shapes the output and removes
records. Everything opinionated lives here, which is the point: the spine stays
auditable and every judgement is one explicit, reviewable step.

Read [`../README.md`](../README.md) for naming and style.

## Shape

```yaml
stage: curate
admin_id: US
entity:
  entity_type: parcel
  source: {source_id: openplaces}
  version: 2026
entity_recipe: US_parcel-spine-2026     # the harmonized input
save_to: {data_dir: share}

pipeline:
  - step: merge_enrichments
    recipes:
      - recipe_id: US_parcel_parcel-placeslab-fmv2026
        columns: {...}
  - step: derive_indicators
  - step: resolve_by_vote
  - step: impute_from_group_statistic
  - step: cast_categoricals
  - step: order_columns
```

Steps are grouped by the *nature* of the transformation, and the module a step
lives in tells you what it is allowed to do:

| module | role |
|---|---|
| `evidence.py` | bring enrichment evidence in (`merge_enrichments`) |
| `indicators.py` | the shared voting vocabulary and scoring cores |
| `reconcilers.py` | resolve conflicts between competing sources |
| `imputers.py` | fill missing canonical values |
| `inferers.py` | derive new canonical features |
| `formatters.py` | structural/type-only shaping |
| `filters.py` | remove records that do not belong |

## The rules that matter

**Every classification resolves through one voting seam.** `resolve_by_vote`
is it. Do not add a bespoke if/else classifier beside it — add decisions to the
vote. `indicators.py` stays pure: predicates and scoring over a DataFrame, with
no thresholds, class names or geography of its own.

**Indicators hold values, never pre-thresholded booleans.** `derive_indicators`
emits named columns carrying the measurement; every cutoff lives in the vote's
decisions, where it is visible and tunable. A boolean baked upstream hides the
threshold from review.

**Geography-specific data belongs in YAML or a sidecar CSV, never in `.py`.**
Keyword tables (`*_land-use-keywords.csv`), class maps, id overrides — all sit
beside the recipe. `src/openplaces` functions stay generic.

**Adding a new source value can change classification.** The land-use keyword
table matches on text; a county whose `use_group` was previously empty starts
voting the moment you map it. Check the new values against
`US_parcel-openplaces-2026_land-use-keywords.csv` — `MAN-HOU` does not match
`MANUFACTURED`, and `BUSINESS` does not match `COMMERCIAL`, so both would fall
to the residual class until a pattern is added.

**`order_columns` is computed, not listed.** Output order comes from the
provenance suffix plus the registry's `sort` rank, so no per-recipe column list
is needed — which is another reason a mapped column must exist in the registry.

## Downstream of curate

Curated outputs are consumed by other curated outputs.
`US_footprint-cheer-2026` pulls `use_group_combined`, `land_use_class` and
`manufactured_home_community` from `US_parcel-openplaces-2026` through
`link_curated_entity`. **Re-curating parcels makes curated footprints stale**,
so the parcel lane runs first and the footprint lane follows.

Note that curated deliverables write to `data_dir: share`, not `core`. Resolve
paths with `get_output_path(recipe, admin_id)` rather than assuming — stale
intermediates with the same filename can sit in `core` and quietly answer the
wrong question.

## Verifying

Compare the curated output against the harmonized spine and the ingested
source, per admin unit:

- spine count and curated count for a given attribute should match unless a
  step deliberately changed it;
- where curated exceeds the source roll, the surplus should be explainable
  (gap-fill from a second source);
- where it falls short, the deficit should be explainable (a join residual).

If you cannot explain a difference, it is a bug, not a rounding artifact.
