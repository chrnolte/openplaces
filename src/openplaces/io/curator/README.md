<!-- Reference for `openplaces.io.curator`, moved verbatim from AGENTS.md on
2026-09-28. AGENTS.md's "Maintaining this file" rules apply
here too: depth proportional to surprise, verified against the
current code. -->

# openplaces.io.curator: reference

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
  Both report provenance, and both report *evidence* rather than the class:
  `vote_dynamic_values` joins the labels that agreed (`nsi/fema/parcel`),
  and `score_decisions` joins the `label` of each indicator that fired for
  the winning decision (`no_improvement_value+block_context`), falling back
  to the decision's declared `source` only where it labeled nothing. Labels
  are opt-in per recipe; unlabeled decisions keep their old behavior, which
  is why a `{col}_source` can still read as a synonym of the value beside
  it until its indicators are labeled.
- `reconcilers.py` — resolve conflicts between competing source columns
  (`reconcile_values` priority selection; `resolve_occupancy` parcel-vs-NSI;
  `resolve_by_vote`, the single voting seam every curate classification
  resolves through)
- `imputers.py` — fill missing canonical values (`impute_n_dwellings`,
  `impute_from_group_statistic`, `impute_occupancy_type`)
- `aggregation.py` — reduce another entity's rows onto the curated rows
  they belong to (`aggregate_from_entities`): a parcel's `year_built` is
  the earliest of its properties', its `living_area_sqft` and
  `gross_floor_area_sqft` their sums, read from
  `US_property-spine-2026` by `parcel_id_local` with the registry's rule
  or a per-column override, filling only what the parcel's own layer
  left empty. It exists because the parcel geospine's property
  `link_by_id` no longer copies property-level attributes (its explicit
  `columns:` list stops at parcel-level values, use codes and address):
  a copy made in the geometry phase was a second, lossy home for them and
  tied every fix to a geometry rerun. Measured on Victoria County, TX,
  2026-09-12: the curate-side reduction reproduces the old copy's
  coverage exactly (63.9% `year_built` and floor area, the PACS segment
  sum now named `gross_floor_area_sqft`), so a county
  whose geospine predates the change loses nothing. An all-missing group
  reduces to missing, not to pandas' zero sum. A column named under
  `recover` (the parcel recipe names `improvement_value`) is the one
  departure from fill-only: a stacked lot keeps only what its units
  agree on, so its improvement is missing or zero while the units
  record it, and the units' positive sum replaces that. Nothing is
  written where no unit records a positive figure: a lot the roll does
  not value stays as it is, because estimating it would be an
  imputation step, which openplaces does not do for structure values
  (maintainer, 2026-09-22). Measured 2026-09-22 on cheer-eastern-nc:
  179 of 503 such lots hold a recorded figure ($44.1M); on
  cheer-coastal-tx 447 of 15,555 ($64.8M).
- `inferers.py` — derive new canonical features (`derive_metrics`,
  `derive_indicators` — named indicator columns holding values, never
  pre-thresholded booleans; every cutoff lives in the vote decisions).
  `derive_group_class_share` adds the context an entity cannot supply about
  itself: the share of its group (any id column it already carries, e.g.
  `census_block_id`) whose evidence reads as a given class, excluding the
  row itself. It is a **groupby, deliberately not a spatial operation** — no
  buffering, no boundary union, no nearest-neighbor search — both because
  geometric neighbor/"community" detection is a patented technique shape in
  this domain (see the patent-risk section of AGENTS.md) and because aggregating within a
  published administrative unit is older, plainer practice. It exists
  because `flag_manufactured_home_communities` counts per *parcel* and so
  cannot see a subdivided community where every home has its own lot. Feed
  it only evidence a downstream vote has not written, or the class
  reinforces itself. **No shipping recipe votes on it**: measured
  2026-08-21, a block share computed from the assessor's own text
  correlates +0.577 with the keyword rule that reads the same column, and
  the points it uniquely moves are 0.294 precise against a 0.425 base rate.
  It is kept as evidence, and as the mechanism
  `notebooks/05_curate/mmh_separability.py` measures with.
  `derive_group_count` and `derive_group_rank` are the same kind of
  groupby, counting or ranking (largest first, ties by id) the rows of
  a group that satisfy voting indicators. **The footprint recipe votes
  twice on `occupancy_type`**: they read the first vote's classes of a
  parcel's primaries, and a second `resolve_by_vote` (`base_output:
  occupancy_type_pass1`) turns small secondary Manufactured Home
  footprints into Secondary. It is safe only because the second pass
  writes secondaries and reads primaries; it reads which labels carried
  the first pass through the `has_token` predicate, which matches whole
  `+`-separated parts of `occupancy_type_source`.
- `formatters.py` — structural/type-only output shaping (`cast_categoricals`,
  `order_columns`)
- `filters.py` — (stub) remove records that do not belong in the canonical
  dataset
- `transactions.py` — steps for the transaction entity, which **grade and
  flag and never drop**: `derive_document_id` (the deed behind a row, spelled
  per source; a part that is empty or all zeros names no document, since
  Florida's roll writes a single space for book and page and that once folded
  1,333 sales of one county into a single "deed"; scoped by record kind, so a
  deed and the last-sale row citing its book and page are never aggregated as
  one sale of two parcels), `count_parcels_per_document`,
  `aggregate_multi_parcel_sales`, `collapse_double_closings` (compared within
  one record kind) and `flag_sales_matching_other_kind`
  (`sale_matches_deed`: exact equality of parcel, year, month and price, no
  tolerance and no score). `sale_arms_length_confidence` is graded per source
  in the recipe: Florida from the state's qualification codes (labels and the
  raw two-digit codes alike, because last-sale rows read from the parcel
  layer are undecoded), Wisconsin as the lower of two grades, the conveyance
  type and the relationship the parties themselves state on the RETR form
  (`sale_party_relationship`; about 20% of transfers are `Family`), through
  the `minimum` indicator type and `fill_only`.
  `join_temporal_snapshot` is the one as-of join in the repo (the only
  `pd.merge_asof`), attaching **the property as it was when it sold**: a
  hedonic model wants the sale-date values, and where a county renumbered
  its parcels the panel is the only thing that reaches its older sales at
  all (Pasco and Volusia FL went from 0.03 to 0.998 of pre-2017 sales
  carrying a land value). `require_panel` makes it a no-op where a county
  has no roll or **only one vintage**, which is nearly everywhere: Florida's
  DOR roll keeps 24 yearly vintages and the other 104 assessor sources are
  one each. One vintage is deliberately not a panel, since joining a sale to
  the only year on file reads as a temporal match while adding nothing the
  cross-sectional spine did not have. Its `match_type_column` is written on
  **every** row, not only matched ones (`exact`, `backward_fallback`,
  `forward_fallback`, `not_in_panel`, `single_vintage`, `no_panel`,
  `no_sale_date`): a missing value cannot separate "no panel in this county"
  from "the panel does not know this sale", and the two mean different
  things to anyone modeling with it. It tolerates what varies between
  counties rather than failing the curate, since one nationwide recipe names
  one column list and one restriction: a column the roll lacks is skipped
  (Lake FL has no `land_area_sqft`), a `restrict_to` on an absent column
  selects nothing (`sale_vacant` is Florida's word), and a sale with no year
  is set aside as `no_sale_date` because `merge_asof` refuses null keys.

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

**Every validation step writes a confusion matrix.** `confusion_matrix` keeps
reference classes as rows and predictions as columns, then `Secondary`,
`Non-residential` (any other asserted class; `(other class)` for a recipe
declaring no residential classes) and `No class`, so no row is dropped and
"said nothing" stays apart from "said something else". NSI and FEMA derived
sources set `keep_unmapped: true`, keeping a raw class their residential-only
class map does not cover instead of scoring it as no prediction;
`accuracy_from_matrix` derives producer's (recall) and consumer's (precision)
accuracy, kappa and the abstention rate, and `score_classification` is
computed from the same matrix. `write_confusion_report` (and
`write_year_agreement_report`, whose "matrix" is one reference row of
exact/within 1/within 5/more than 5 years bins) writes
`{name}_confusion.csv` (long form, pooled plus strata),
`{name}_accuracy.csv` and a JSON sidecar into the delivery's `accuracies/`
folder, refusing any stratum under 10 rows so no cell points at one building.
`ValidationContext.score_sources(linked, out_dir)` writes them for every
survey source; the permit notebooks call the writer directly. Only these
aggregates may leave memory: permit modes per footprint are restricted
row-level data.

`provenance.py` carries one invariant worth knowing before writing any step
that produces a value: **a cell openplaces itself filled must say so**, by
carrying the `imputed` marker in its `{col}_source`. Tokens are therefore
composite, joined by `+` (`parcel+imputed`, and the harmonize stage's
`parcel+usaddress`) — never assume a token equals a bare source name.
`mark_imputed` is the only place the marker is written and `is_imputed` the
only place it is read (it matches whole `+`-separated parts, so a source named
`imputed_rates` is not swept up); `record_sources` writes a per-row token
series, which is what a step carrying an upstream sidecar forward needs rather
than `record_source`'s one-token-per-mask.

**The marker means openplaces filled the cell, not that the number is
modeled.** A value read from a dataset keeps that dataset's name even when the
dataset is a model: `nsi` is a FEMA-modeled structure value and stays plain
`nsi`, because the token already names what produced it. So classification
votes are not marked (`nsi`/`keyword`/`classifier` each name the winning
evidence), and neither is apportioning a reference's value across the entities
on it — the parcel's total is a real assessed figure being divided, not a value
invented where none existed. Only estimation of a genuinely missing value
counts.

The hard part is propagation, not the decision: `impute_land_value` knows which
rows it estimated, and the marker has to survive `apportion_curated_values`
(which crosses parcel → footprint) and `select_value_source_by_admin_unit`
(whose `output` and `parcel_column` are commonly the *same* column, so they
share one sidecar) to reach the delivered `structure_value_source`. All three
dropped or flattened it before 2026-08-21.

These concern-based modules mirror the processor categories used by related
inventory systems, while remaining native to the openplaces recipe and state
architecture. Value selection, gap-filling, and occupancy inference were
migrated here from the harmonize stage; the harmonized spine
(`US_footprint-spine-2026`) is now an evidence-only table. Curation outputs are
full entity recipes, not sidecar evidence tables.

Public entrypoint: `curate(recipe, admin_ids, reprocess, verbose)`.

