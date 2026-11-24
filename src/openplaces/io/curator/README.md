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

**Every step declares a phase, and a recipe runs its phases in order.**
`_register('name', phase=...)` takes one of `core.constants.CURATE_PHASES`
(gather, reconcile, standardize, infer, format) and refuses a step without
one, so a curate recipe reads top to bottom as: bring every source's value
onto the entity (gather: reductions over links, apportionments, evidence
tables, admin attributes); choose one where sources disagree and gate
invalid evidence (reconcile); put the source's vocabularies, ids and row
definitions into the schema's by reviewed rules (standardize: translation
tables, placeholder-row filters, one sale per deed); vote, impute and
estimate what remains, marking every cell openplaces filled (infer); shape
the output (format). `phase_order_violations(recipe)` lists steps that run
after a later phase, and `tests/recipe/test_curate_phase_order.py` runs it
over every curate recipe. A step that has to run out of order for a
dependency reason declares `phase_override: <reason>` in the recipe (a
control key like `enabled`, never passed to the step): the building
recipe's `year_built` reconcile stays after the first votes because
the indicators read that column as it stands (the roof, foundation
and construction reconcile stays where it ran beside it), and the
transaction recipe's temporal joins need `sale_year` and the folded deed
rows. A separate assemble stage for the gather phase was measured and
rejected on 2026-09-29: gathering is 7% to 35% of a county's curate time
(Galveston footprints 21 of 99 s, Lake transactions 8 of 111 s).

**A step decides a column only where it rebuilds every row of it.**
A registered step may declare the columns it reads, writes and decides
(`io/steps`, read by `python -m openplaces.flow.lineage`), and no recipe
may declare two deciders of one column (`tests/flow/test_lineage.py`).
`reconcile_values`, `apportion_curated_values`, `divide_account_values`,
`impute_from_group_statistic`, `impute_occupancy_type`,
`flag_manufactured_home_communities`, `decide_dwellings` and a
`resolve_by_vote` with `preserve_base: false` (its target and
`flag_column`) decide. A step that keeps rows it does not claim only
writes: a vote that keeps the base class, a fill (`impute_n_dwellings`,
`reconcile_land_use`), a translation table (`translate_descriptions`),
a gate (`suppress_where`) and a reviewed correction
(`apply_reviewed_corrections`, which declares `*` because its sidecar
names the columns). So the building class is decided once, by
`impute_occupancy_type`, and the votes after it (`resolve_occupancy`,
the main vote, the second pass, the height bands) are writers of it.
Where a state patch puts a translation table in place of the cohort
statistic (`group_parcel` in MA and WI), that column has no declared
decider in the patched recipe. The steps that build a column of their
own on every row decide it too: `collect_link_ids`,
`derive_admin_attribute`, `apportion_common_land`, `derive_group_count`,
`derive_group_rank`, `flag_unconfirmed_height`,
`classify_manufactured_homes` (`p_manufactured_home`),
`derive_duplicate_columns`, a `derive_indicators` spec without
`fill_only`, and `flag_sales_matching_other_kind` and
`collapse_double_closings` where they write a flag column. The format
steps write what they change: `cast_categoricals` every registry
categorical it may cast (a dtype change). A step that removes columns
declares them under `drops`, which is no write: `order_columns` its
`drop` list, and `keep_registered_columns` a rule
(`io.steps.DropRule`) that holds for every column the registry does
not know, plus excluded and personal ones, since no list of names can
state that set. `flow.lineage.droppers` names where a column leaves
the table, and the audit counts such a removal as declared.
`merge_enrichments` declares
the columns it fills but not the enrichment columns it reads, since
those recipes are named inside its list of specs.

**A lot is a manufactured-home park by one rule, on the building
side.** `flag_manufactured_home_communities`, after the class votes,
calls `is_manufactured_home_park` (`estimators/manufactured_homes.py`):
the national building recipe makes a lot a park where it holds 8 or
more manufactured homes, or where 3 or more of its homes with an
`aspect_ratio` of 2.5 or more have `long_axis_bearing`s inside one
15-degree window, modulo 180 (`aligned_homes`; a window across 0 and
180 counts). Both columns come from one `shape_metric` rectangle in
`derive_indicators`, which the recipe drops before output. The
aligned-homes arm raises where its columns are absent rather than
calling every small park a family lot. The delivery's building
imputation reads this flag as its park (`io/delivery/README.md`). The
parcel vote's Manufactured Home Park class still decides the lot side
from the geospine's elongated-outline count, a proxy run before the
buildings have classes; a second parcel pass after the building
curate is to replace it.

**A building's dwelling count is decided after its class.** The
building curate's lanes (`reconcile_values`: the roll's apportioned
share, Overture's address points, NSI; then `impute_n_dwellings`'s
class lookup) write `n_dwellings_preliminary`, which no vote reads, so
the votes see what they saw when the lanes wrote `n_dwellings`.
`decide_dwellings` (`estimators/dwellings.py`), after the last writer
of the class, is the one step that sets `n_dwellings`: the roll's 0
holds on a lot whose own count (`n_dwellings_parcel_whole`) is 0 and
whose class is listed in the recipe's `stated_zero_classes`, credited
to `parcel` (empty in the national recipe, so off; a unit's patch
recipe lists the classes once the rule is measured there, because a
curated lot's 0 is not always the roll's statement); a Secondary
building that is not its lot's largest has 0; a Single-Family building
whose roll share ships as 2 or more has 1, the share kept in
`n_dwellings_parcel`. The two class rules mark their cells imputed.
The largest building of a lot keeps its count whatever its class:
which building is the house there is open
(`plans/occupancy-and-dwellings-one-decider.md`, question 5). The
exception is a lot whose class is one of the recipe's `guard_classes`
(Vacant, Mostly Vacant, Retail, Commercial, Office, Industrial,
Institutional, Recreation): no building there is the house, so its
largest Secondary building stays Secondary and has 0 dwellings, token
`secondary_nonresidential_lot+imputed`. A lot without a class is not
guarded.
Measured 2026-11-22 on US-TX-LAV: simulated on the stored curate, the guard reaches 119 lots, changes 43 buildings and removes 13 dwellings.
Measured 2026-11-22 on US-MA-SOM: simulated on the stored curate of 2026-10-31 with the stated-zero rule off, 2,292 secondary buildings changed to 0 (225 dwellings), 1 house set to 1, 39,113 dwellings to 38,887.

Steps are organized by the nature of the transformation:

- `evidence.py` — incorporate enrichment evidence (`merge_enrichments`).
  An enrichment's output is found under the spine it ran on, which is
  the curated entity's own unless an entry names `entity_recipe`: the
  building curate reads the outline enrichments (roof shape, CHEER v0,
  NCDPS years, Overture and FEMA heights) under
  `US_footprint-spine-2026` and joins them on its `key`, the building's
  `footprint_id`. Without it the output is looked for under the
  building spine and skipped as if the enrichment had not run.
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
  it until its indicators are labeled. A decision's `insufficient_alone`
  names labeled indicators that may support it but not carry it alone
  where a context indicator holds: the building recipe's Manufactured
  Home decision needs a partner for footprint shape and
  `p_manufactured_home` where the roll and NSI both name Single-Family
  or Multi-Family (maintainer, 2026-10-22). It may be a list of guards,
  each with its own labels and context, all applied: the same decision
  also needs a partner for shape and `no_improvement_value` where the
  outline shares a wall of at least 3 m (`shared_wall_m`, a
  `derive_indicators` value measured by strict intersection only;
  maintainer, 2026-10-24).
- `reconcilers.py` — resolve conflicts between competing source columns
  (`reconcile_values` priority selection; `resolve_occupancy` parcel-vs-NSI;
  `resolve_by_vote`, the single voting seam every curate classification
  resolves through)
- `imputers.py` — fill missing canonical values (`impute_n_dwellings`,
  `impute_from_group_statistic`, `impute_occupancy_type`)
- `aggregation.py` — reduce another entity's rows onto the curated rows
  they belong to (`aggregate_from_entities`), on either side of a link
  table: a curated parcel takes its properties' rows through the
  property-to-parcel table, a curated sale its parcels' rows through the
  transaction-to-parcel table (`key: null`, link only, each parcel's sums
  scaled by the sale's `share` of it, a class kept only where the deed's
  parcels agree), and the footprint curate CHEER ships
  (`US_footprint-cheer-2026`) takes every column of the building curate
  through the footprint-to-building link (`columns: all`): sums add, a
  class the buildings disagree on keeps each value joined by " | "
  (`distinct`), a number they disagree on stays missing (`agree`),
  a value's tokens are those of the rows it came from (an `imputed`
  marker on any of them survives), and
  `n_buildings_per_footprint` counts them. One building per outline is
  a copy. Measured 2026-10-26 on US-MA-SOM: the fold equals the
  footprint curate it replaced in all 113 columns. A parcel's `year_built` is
  the earliest of its properties', its `living_area_sqft` and
  `gross_floor_area_sqft` their sums, read from
  `US_property-spine-2026` by `parcel_id_local` with the registry's rule
  or a per-column override, filling only what the parcel's own layer
  left empty. It is the only route for property-level attributes: the
  parcel geospine copies from a roll, and from a property layer bundled
  in a parcel recipe, one explicit list that stops at parcel-level
  values, use codes and address, because a copy made in the geometry
  phase was a second, lossy home for them (a mean of a lot's units'
  stories) and tied every fix to a geometry rerun. On a lot that a roll
  and the units split off the parcel layer both describe, the units'
  rows are left out of a column wherever the roll's rows state it, so
  nothing is summed twice, and give the lot its value where only they
  state it. Measured on Victoria County, TX,
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
  cheer-coastal-tx 447 of 15,555 ($64.8M). Three options narrow a
  reduction: `where` keeps only the reference rows whose code matches
  a pattern, `row_minimum` counts a row stating less (or nothing) as
  the minimum, and `prefer` writes the reduction ahead of the curated
  value wherever it states one. The Massachusetts parcel patch uses
  all three to count a condominium lot's dwellings from its 102 unit
  accounts, each at least one, ahead of the master record's count.
  Each written cell's `{column}_source` names the rows its value came
  from: for `max`, `min`, `first` and `last` the row holding it (rows of
  several sources tied on it are joined by `+`), for every other rule
  each row stating the column. On the property spine a row's token is
  its cell's own `{column}_source` where one is set (a supplement, a
  merged row's other source) and its `source` otherwise, reduced to
  source ids. A written cell no row labels loses the label it held.
- `account_values.py` — `divide_account_values` (gather), **the one
  step that decides a lot's account values** (land, improvement,
  other, total, dwellings) in the parcel curate. Each roll account is
  divided over the lots its links reach, shared-key links included:
  land and other value by lot area (the link `share` renormalized over
  the curated lots), improvement and dwellings by the floor area of the
  building pieces standing on each lot (overlay pairs, each piece
  counted for the lot it stands on, area times stories, pieces under
  10 m2 ignored; a condominium cluster outline only where no other
  drawn piece stands on the account's lots; a parcel-shaped stand-in
  never weighs), by lot area where no lot has any; a stated total in
  proportion to each lot's parts,
  the sum of the parts where none is stated (marked imputed). A value
  the account does not state, which the parcel layer repeats on each
  of its lots, is taken once and divided over the lots where the
  layer states it (`layer_columns`). It replaces every cell a stating
  account reaches, so the parcel geospine's whole copies do not
  survive; `aggregate_from_entities` still fills these columns on a
  unit without a link table. Story counts are `decide_stories`'
  evidence without its roll fill, which reads this curate. Writes a
  conservation report per unit (`account-values-conservation.csv`).
  **A 0 is an observation only where an account states it**: a lot
  whose accounts all leave a column of `unstated_zero_missing` blank
  (the parcel curate names `n_dwellings`) loses the 0 it holds from
  the parcel layer, whose numeric field has no missing value; a lot
  without accounts, and a positive layer value, keep theirs.
  Measured 2026-11-22 on US-FL-LA: step re-run on the stored curate, 70,441 lot zeros became missing, 802 stated zeros stay.
- `common_land.py` — `apportion_common_land` (gather), the plat's
  common land as an area: it reads the parcel spine's `plat key` link
  table and writes `common_land_share_ha` on each lot (the common area
  divided among the plat's lots in proportion to lot area),
  `n_lots_per_plat` and `common_land_lots_area_ha` on each common area,
  and `common_land_plat_id` on both. **It writes no value**: the lots keep
  their recorded land values and the common area none, because the roll
  counts the common land's worth in the lots (maintainer, 2026-10-24); a
  reader divides a lot's land by its area plus its share.
- `inferers.py` — derive new canonical features (`derive_metrics`,
  `derive_indicators` — named indicator columns holding values, never
  pre-thresholded booleans; every cutoff lives in the vote decisions).
  Until 2026-09-29 it also held the group-context steps, the
  manufactured-home classifier, the occupancy and story estimators and
  `derive_admin_attribute` (1,707 lines); those now live in
  `group_context.py`, `estimators/` (one module per attribute:
  `manufactured_homes.py`, `occupancy.py`, `stories.py`, whose
  `decide_stories` is the one step that sets a building's `n_stories`:
  ordered evidence, a lidar height at 3.5 m per story, rounded half up
  once, and a roll gap fill per unit, as the roll states it (a 1.5
  stays 1.5); the package
  file imports them because the shared step loader skips sub-packages)
  and `evidence.py`. `inferers.py` still re-exports every moved step
  name, so `from openplaces.io.curator.inferers import ...` keeps
  working in the tests and the docs.
- `group_context.py` —
  `derive_group_class_share` adds the context an entity cannot supply about
  itself: the share of its group (any id column it already carries, e.g.
  `census_block_id`) whose evidence reads as a given class, excluding the
  row itself. It is a **groupby, deliberately not a spatial operation** — no
  buffering, no boundary union, no nearest-neighbor search — because
  geometric neighbor or "community" detection is kept out of this
  repository by design (see "Design limits" in AGENTS.md) and because
  aggregating within a published administrative unit is older, plainer
  practice. It exists
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
  a group that satisfy voting indicators. **The building recipe votes
  twice on `occupancy_type`**: they read the first vote's classes of a
  parcel's primaries, and a second `resolve_by_vote` (`base_output:
  occupancy_type_pass1`) turns small secondary Manufactured Home
  buildings into Secondary. It is safe only because the second pass
  writes secondaries and reads primaries; it reads which labels carried
  the first pass through the `has_token` predicate, which matches whole
  `+`-separated parts of `occupancy_type_source`.
- `formatters.py` — structural/type-only output shaping (`cast_categoricals`,
  `order_columns`)
- `filters.py` — remove records that do not belong in the canonical
  dataset, by value or by numeric threshold (`exclude_by_value`,
  `keep_by_value`, `filter_numeric_at_least`, `exclude_numeric_above`),
  and `normalize_id_column`. A threshold drops rows whose value is
  missing too, because a missing value is never at least the minimum;
  the parcel recipe's 1 m2 floor on `area_ha` relies on every curated
  parcel carrying an area. Links that name a dropped parcel stay in
  their tables (`apportion_curated_values` warns with a count).
- `transactions.py` — steps for the transaction entity, which **grade and
  flag and never drop**. Since 2026-09-29 the rows arrive from
  `US_transaction-spine-2026` already as the entity: one row per recorded
  sale, indexed by `transaction_id`, with the sale period, the document id
  (the deed behind a row, spelled per source; a part that is empty or all
  zeros names no document, since Florida's roll writes a single space for
  book and page and that once folded 1,333 sales of one county into a single
  "deed"; scoped by record kind, so a deed and the last-sale row citing its
  book and page are never aggregated as one sale of two parcels) and the
  parcel count on every row. `derive_sale_period`, `dedup_transactions`,
  `derive_document_id`, `count_parcels_per_document`,
  `aggregate_multi_parcel_sales` and `assign_transaction_ids` stay
  registered here as wrappers over `io/sale_records.py` for a recipe that
  still lists them; the national recipe does not, and never re-mints an id.
  What a sale conveyed arrives through the spine's link tables, gathered
  by `aggregate_from_entities` at the top of the recipe (the property's
  own values first, since a property is what a sale conveys, then the
  parcel's sums, agreed classes and area filling what no property
  supplied), not through spine columns: since 2026-10-01 a multi-parcel deed's
  row carries no parcel number of its own, so the key-based steps below
  (`collapse_double_closings`, the deed-echo flag, the temporal joins,
  which mark such a row `no_key`) see only single-parcel sales; reaching
  the panel per linked parcel is the next step for the temporal join.
  What runs here: `collapse_double_closings` (compared within
  one record kind) and `flag_sales_matching_other_kind`
  (`sale_matches_deed`: exact equality of parcel, year, month and price, no
  tolerance and no score), and `derive_duplicate_columns`, which
  summarizes the spine's duplicate-sale pair table per row
  (`sale_duplicate_of`, the kept partner; `sale_duplicate_reason`, the
  label by a fixed precedence; `sale_resale_within_window`, a second
  market sale within a month, kept apart so dropping duplicates never
  drops a resale). It reads the pair table and the index only, never a
  gathered value. The two older flags stay unchanged until
  `openplaces-valuation` reads the new columns, then are re-derived from
  the pair table (plans/transaction-sale-deduplication.md, phase 2). `sale_arms_length_confidence` and
  `sale_qualification_group` are graded per state, each in a patch recipe at
  the state's scope inserted right after the national recipe's first
  `derive_indicators` step: Florida from the state's qualification codes
  (labels and the raw two-digit codes alike, because last-sale rows read
  from the parcel layer are undecoded), Wisconsin as the lower of two
  grades, the conveyance type and the relationship the parties themselves
  state on the RETR form (`sale_party_relationship`; about 20% of transfers
  are `Family`), through the `minimum` indicator type and `fill_only`,
  Illinois from the PTAX-203 conditions. A state with no patch leaves both
  columns unwritten.
  `join_temporal_snapshot` is the one as-of join in the repo (the only
  `pd.merge_asof`), attaching **the property as it was when it sold**: a
  hedonic model wants the sale-date values, and where a county renumbered
  its parcels the panel is the only thing that reaches its older sales at
  all (Pasco and Volusia FL went from 0.03 to 0.998 of pre-2017 sales
  carrying a land value). Florida's patch joins on `parcel_id_local` with
  `normalize_key: false`: both sides mint it by one rule per county and era
  (the ranged rows of the id-overrides tables), which is what carries a
  renumbering or a respelling across roll years, and comparing it stripped
  would merge `1|23` with `12|3`. `require_panel` makes it a no-op where a county
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
(cache-written conflict reports), and `validation/` — scoring a curated
classification against hand-labelled points (a package since 2026-09-29:
`linking`, `accuracy`, `reports`, `references`, `context`; every name
re-exported from `openplaces.io.curator.validation`). `validation/linking.py`'s
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
(which crosses parcel → building) to reach the delivered
`structure_value_source`. Both dropped or flattened it before 2026-08-21.

These concern-based modules mirror the processor categories used by related
inventory systems, while remaining native to the openplaces recipe and state
architecture. Value selection, gap-filling, and occupancy inference were
migrated here from the harmonize stage; the harmonized spines are
evidence-only tables (`US_building-spine-2026` holds the structure's
evidence; `US_footprint-spine-2026` is the outline geometry alone, see the
harmonizer README). Curation outputs are
full entity recipes, not sidecar evidence tables.

**Building values are pieces of the parcel's.** `apportion_curated_values`,
in the building curate, splits a parcel's improvement value among the
footprints of its buildings (one outline per building today) by overlap
area,
or by floor area (outline times measured floors, `floor_weight`) where
every outline on the lot has a height or is under `unmeasured_max_m2`;
a secondary structure's share passes to the lot's other footprints. Land
and total value go to the principal footprints, divided by overlap area.
A parcel that is no principal footprint's largest overlap (a lot of a
townhome row drawn as one outline, a unit lot under a condominium
cluster) divides its land and total among the footprints linked to it
that are neither secondary nor dwelling-suppressed. The footprint curate
carries these values onto the outlines (`aggregate_from_entities`). Building
and footprint sums still fall short of parcel sums, by the parcels with no
footprint or only secondary ones; read regional totals from curated parcels.

**A curate never saves an id column empty that its entity filled.**
`require_filled_ids_kept` notes which `*_id` columns of the harmonized
entity hold a value when the curate loads it, and raises
`EmptiedIdColumnError` before saving if one of them is empty on every
curated row. An all-empty column is written with the Arrow null type and
every downstream curate inherits it silently (the census tract and
block-group ids of 70 Massachusetts towns, 2026-11-05). Dropping the
column is not refused; that is a recipe's explicit choice.

**A unit built with none of an entity reads as having none of it.**
Harmonize saves a placeholder for a unit with nothing to build from (no
rows, no column but `_join_id`), and the curate of it saves one too,
running no step (`io.readers.is_placeholder_output`). A later curate
reading it as a reference (`read_curated_reference`, behind
`link_curated_entity`, `apportion_curated_values` and
`keep_curated_ids`) gets a frame without rows whose one column is the
id, so every join matches nothing; `get_entities` returns it without
rows, with an empty geometry column when geometry was asked for, and
`export_delivery` pools the region without it. A reference with rows
but no id column still raises, and so does a missing file: the unit
was then not built, which is a different fault from having no rows.

Public entrypoint: `curate(recipe, admin_ids, reprocess, verbose)`.

