# Acquiring a new parcel / property / sales source

How to find, ingest, and harmonize public parcel, property, tax-assessor and
sales data for a county or state the repo does not cover yet, unsupervised.
Written from the NC (44-county CHEER) and GA (Atlanta metro) acquisition runs;
every trap below was hit for real, not imagined.

Read `../README.md` first for the conventions every recipe follows. This guide
covers the part that comes before writing YAML: finding a source and proving it
is the right one.

Read it end to end before starting. The order of phases matters: most of the
expensive mistakes come from skipping validation and discovering three hours
later that the layer was a different state's county of the same name.

---

## Phase 0 — Frame the job before searching

Establish, and write down:

- **Admin units in scope.** Resolve them to `AdminId` strings up front
  (`op.get_admin_ids(3, admin_id='US-NC')`). Never work from county names alone.
- **Entity types wanted.** `parcel` (geometry + assessment roll), `property`
  (per-unit/building card), `transaction` (recorded sales), `footprint`.
  A county often publishes these as separate layers or tables; each is its own
  recipe.
- **Attributes that justify the work.** Typically `last_sale_price`,
  `last_sale_date`, `land_value`, `improvement_value`, `year_built`,
  `use_group`/`use_group_code`, `zoning_code`, `area_sqft`, `owner_name`.
  A county layer with none of these is rarely worth a recipe: the statewide
  layer usually already covers geometry.
- **The incumbent baseline.** What does the statewide source already give this
  county? If a state layer (NC OneMap, FloridaGIO, TxGIO, MassGIS) already
  supplies 99% of what the county layer has, skip it.

Cheap way to get the baseline: read the existing statewide ingest output and
profile it, rather than re-deriving from the recipe.

---

## Phase 1 — Discovery: four routes, in this order

Run all four before concluding a county has no source. They fail in different
ways and cover each other.

### 1. ArcGIS Online item search

```python
requests.get('https://www.arcgis.com/sharing/rest/search',
             params={'q': f'"{county} County" AND parcels AND type:"Feature Service"',
                     'f': 'json', 'num': 50, 'sortField': 'numViews',
                     'sortOrder': 'desc'})
```

Vary the query — `{county} County {STATE} parcels`, `{county} tax parcel`,
`{county} County GIS parcels`. Require the county name to appear in the item
title, owner, or snippet; otherwise you get every county in the country.

### 2. ArcGIS Hub dataset index

Indexes open-data sites and *on-prem servers registered to them* that item
search misses entirely (this is how Pitt and Wilson County NC were found, both
on county-hosted ArcGIS Servers).

```python
requests.get('https://hub.arcgis.com/api/v3/datasets',
             params={'q': f'{county} County parcels', 'page[size]': 50})
```

Read `attributes.url`, `attributes.name`, `attributes.owner`,
`attributes.recordCount`. **Do not pass `fields[datasets]`** — it silently
strips the attributes you need and every result looks empty.

### 3. The county's own GIS/tax page

Search for `{county} county {state} GIS parcel download` and
`{county} county tax assessor data download`. Look for a zipped shapefile or
file geodatabase, a CSV export, or an ArcGIS Server root at
`gis.{county}.org/arcgis/rest/services` or `maps.{county}gov.com/...`.

### 4. Sibling layers of anything you find

A service that has parcels usually has more: a sales-history table, a building/
improvement table, permits. Enumerate `/{service}?f=json` → `layers` **and**
`tables`. Pitt County's full 285k-row sales history was a sibling table of its
parcel layer and would have been missed by a layer-only scan.

**When to stop.** If all four routes come up empty, record that explicitly
(with the hostnames tried) so nobody re-searches. Henry County GA is the
canonical example: seven dead hostnames, assessor behind qpublic/Schneider,
genuinely no public source.

---

## Phase 2 — Validate before you invest

The dominant false positive is **a same-named county in another state**. Real
examples that passed a naive title/field check: Beaufort SC for Beaufort NC,
Franklin VA for Franklin NC (its schema had `TaxsifterLink` and `DFLAcres`,
both Washington-State assessor artifacts), Cumberland PA, Forsyth NC for
Forsyth GA.

Apply all four gates:

1. **Geography.** Reproject the layer's `extent` to EPSG:4326 and intersect it
   with the real admin polygon. Require >50% of the county polygon covered.
   This alone kills nearly every false positive.
2. **Feature count.** Compare `returnCountOnly` against the county's parcel
   count from the incumbent statewide layer. Accept roughly 0.6–1.5×. Outside
   that you have a subset (an MPO boundary, a municipality, an open-space
   selection) or a superset (multi-county service).
3. **Field population.** A field existing means nothing. Sample ~400–1000 rows
   and measure the share that is non-null, non-empty, and non-zero. Schemas are
   full of declared-but-empty columns (DeKalb GA has `LNDVALUE`, `RESYRBLT`,
   `USECD`, `BLDGAREA` all at 0%).
4. **License / terms of use.** Look for an explicit terms-of-use or license
   page on the source's own site — footer links, an "about this data" page,
   the API documentation. Government open-data portals often have one at a
   predictable path (`/terms`, `/open-data-license`). Record what you find,
   or its absence, in the recipe's `source:` block (see `../README.md`). A
   source that restricts commercial use or redistribution (GADM:
   non-commercial, redistribution requires permission) is not
   disqualifying — it means the recipe says so, the same way a source with a
   degenerate join key gets documented rather than silently used.

---

## Phase 3 — Bulk vs REST, and the equivalence test

**Default to a bulk download.** REST paging costs the county's server hundreds
of sequential requests; a bulk export is one. Preference order:

1. A direct file on the county GIS page (zip/gdb/csv).
2. ArcGIS Hub export:
   `https://hub.arcgis.com/api/v3/datasets/{itemId}_{layerId}/downloads/data?format=geojson&spatialRefId=4326`
   (`format=csv` for tables).
3. `arcgis_rest_scraper` only when neither exists.

### The Hub export is asynchronous

A cold cache answers **404 or HTTP 202** with a job-status JSON for minutes
while the file builds. Nudge it with
`https://hub.arcgis.com/api/download/v1/items/{itemId}/geojson?layers={n}` and
poll until it returns a real `FeatureCollection`. A 404 on first ingest means
"not built yet", not "gone". Wide layers take longest (Carteret NC, ~120
fields, needed several minutes across two polling runs).

The ingester will happily save a 202 status stub as your `.geojson` and then
die with "not recognized as being in a supported file format". If you see that,
delete the stub and re-ingest.

### **A bulk export is not always the same data as the REST layer**

This is the trap that costs the most and looks the most like success. Wilson
County NC's Hub export downloads cleanly, has a plausible feature count, and is
**almost entirely empty**: 42,097 rows with geometry and PIN, every CAMA field
null — 182 rows with a sale price against the server's own 23,189. The layer is
a join/view that the live service resolves per query and Hub materializes
without.

**Always run the equivalence test before choosing bulk:**

```
server:  /query?where=<FIELD> > 0&returnCountOnly=true&f=json
output:  count of rows where that mapped column is > 0
```

Compare *a populated field's exact count*, not just the feature count. If they
disagree materially, use REST and record why in the recipe. Seven of eight NC
bulk exports matched the server exactly; one did not. The same question is open
for DeKalb GA.

---

## Phase 4 — Map the schema

- **Every target column name must exist in `core/attribute_registry.csv`.**
  Check with `load_registry()`. The registry is the canonical vocabulary; a
  recipe never invents a name.
- **Do not mis-home a field just to keep it.** If a source has something
  genuinely useful with no registry equivalent, leave it unmapped and say so
  loudly in the recipe header, then raise the registry question. Real examples:
  Pitt's `SALE_TYPE` (`SLTYLND`/`SLTYPKG` — what the sale conveyed, not a
  document type, so *not* `doc_type`); bedroom/bathroom/room counts, heating,
  basement, garage, renovation year — all common in NC assessor layers, none
  currently in the registry.
- **Check what a field means, not what it is called.** Sampson County NC's
  `USE_DESC` looks like land use and is actually building style
  (`RANCH`, `CONVENTIONAL`, `DOUBLE WIDE MOHO|STORAGE`); its land use lives in
  `SEG_TYPE_D` (`HOMESITE`, `CROPLAND|WOODLAND`). Mapping the first to
  `use_group` would corrupt the land-use classifier. Sample values before
  deciding.
- **Prefer the tax-side field over the GIS-side** when both exist (Nash NC:
  `TAX_PIN` matches 99.6%, `GIS_PIN` only 92.0%).
- **Watch for truncated shapefile field names** (`FINALFULLL`, `PHYSICALAD`) —
  a 10-character cap means the layer round-tripped through a shapefile, and
  names may be ambiguous. Verify against sampled values.

### Dates are the most common silent corruption

- **Esri `f=geojson` serializes date fields as epoch milliseconds**, not ISO
  strings, and nothing casts them at ingest. `last_sale_date` then lands as
  `1345161600000.0`. Existing Gates/Jones/Pender/Chowan NC recipes already
  have this. **Prefer a string-formatted date field where the source offers
  one**; otherwise document the units in the recipe.
- **Integer `YYYYMMDD`** (`20011009`) read as a datetime becomes nanoseconds
  since 1970. Map a clean year field instead, or handle it explicitly.
- **Sentinel dates are everywhere.** Pitt had three families: `1901-01-02`
  (29,394 rows), `1900-01-01` (1,890), and `01/1900`/`01/1901` in the parcel
  roll's own month-year field. Identify them by how sharply they spike against
  neighbouring dates (1–3 rows each), then null them with `null_if_equal`
  *before* `to_datetime`. Keep the rows — the price is usually real, only the
  date is a placeholder.
- **`year_built` uses 0 for "never recorded"** in many counties; the registry's
  `null_placeholder` handles that generically once mapped.

### Numeric ids

Set `csv_dtype: {FIELD: str}` for id columns that are all digits, or pandas
infers an integer and destroys leading zeros (and violates the registry's
string type).

---

## Phase 5 — The join key: `parcel_id_local`

This is where whole datasets disappear silently. `last_sale_price` and the
value fields are **not** in `US_parcel-spine-2026`'s `keep_columns` — they reach
the spine only through the `link_by_id` join on `parcel_id_local`. A null or
wrong key drops the entire assessor roll with no error anywhere.

**Choose the key empirically.** For each candidate id column, apply the
candidate conversions (`simple`, `pipe`, `no_conv`, the bundled per-county
instruction) and measure:

- population and distinct count (a key with far fewer distinct values than rows
  is degenerate — see below);
- **overlap with the incumbent statewide layer's own key**, which is what makes
  the two joinable.

**Three failure modes, all seen:**

1. **The bundled pattern does not fit the source.** `parcel_id_links.csv` holds
   a per-county pattern derived from the *statewide* layer's id. A county layer
   often formats the same PIN differently (dashed vs not, different width). The
   pattern then matches nothing and yields an all-null key. Since v-latest,
   `compute_parcel_id_local` guards this: it compares the conversion's loss
   against a plain `simple` conversion and falls back with a loud warning past
   `max_loss` (0.5). Trust the warning, but fix the root cause with an override
   row.
2. **The source id column is degenerate.** Falling back cannot invent precision.
   NC OneMap's `ALTPARNO` in Pamlico is a 4-digit block code — 140 distinct
   values for 17,109 parcels. `simple` reproduces that faithfully and the
   duplicate guard waves it through, because the duplicates come from the
   source. A *populated but degenerate* key is worse than a null one: it looks
   healthy. Fix by pointing at a different column.
3. **The id column is empty for that county.** Craven and Cumberland NC have no
   `ALTPARNO` at all in the statewide layer.

**The fix for 2 and 3 is the same**: add a row to
`src/openplaces/recipes/US/_all/parcel/_all/US_parcel_id-overrides.csv`:

```
admin_id,source_id,kind,pattern,conv,tolerance,source
US-NC-PM,nconemap,parcel,,simple,,parcel_id_admin3
```

- A blank `source_id` matches **every** source at that admin unit; an exact
  `source_id` row wins over it. If a blank-source row exists and you add a new
  county recipe there, **you must add an exact-source row too**, or your recipe
  inherits a redirect to a column it does not publish (Carteret NC).
- `source` redirects which raw column feeds the key — usually
  `parcel_id_admin3` (the county parcel number, `PARNO`) when the assessor id
  is empty or truncated.
- `kind` is `parcel` for parcel layers, `tax` for assessment/transaction rolls.

**Verify the key against the county roll after ingest** — an overlap below ~95%
means you picked the wrong column or conversion.

---

## Phase 6 — Write the recipe

See [`ingest_recipes.md`](ingest_recipes.md) for the YAML shape, the download
options, transformations and partitioning, and [`../README.md`](../README.md)
for path/naming/style conventions.

The one thing to carry over from this phase: **write down what you measured**.
The recipe header is the only place a future reader learns that a field was
checked and rejected. State populations with numbers ("968/1000 sampled"), not
adjectives.

Record the aggregates, never the records they came from. Sampling a layer puts
real parcel ids, addresses and owner names in front of you; none of them belong
in a tracked file. Fabricate any example value, preserving its format — see
*Schema, never records* in [`../README.md`](../README.md).

---

## Phase 7 — Ingest and reconcile

```python
op.ingest(get_recipe_by_id(rid), admin_ids=[admin_id], reprocess=True, verbose=True)
```

Then verify **exactly**, not by eyeball:

| check | how |
|---|---|
| row count | parquet rows vs `returnCountOnly` on the server |
| key quality | `parcel_id_local` population + distinct count + overlap with the county roll / statewide layer |
| the attribute you came for | count of `> 0` in parquet vs `where FIELD > 0` on the server |
| dtypes | ids are strings, dates are dates (not epoch integers) |

A small deficit (tens of rows) is usually export-snapshot lag and is fine;
report it rather than hiding it. A large one means the wrong layer, a subset,
or the bulk-vs-REST divergence from Phase 3.

---

## Phase 8 — Downstream: what actually needs re-running

Dependency order (derive it from `RecipeDAG`, do not guess):

```
property spine  →  footprint spine  →  parcel spine  →  enrich  →  curate parcels  →  curate footprints
```

- The **footprint spine consumes parcel ingests**; the **parcel spine consumes
  the footprint spine**. Adding a parcel source invalidates both.
- **`US_footprint-cheer-2026` links curated parcels** via `link_curated_entity`
  (`use_group_combined`, `land_use_class`, `manufactured_home_community`). New
  parcel sources make curated footprints stale — re-curate them.
- **Do not re-run an enrichment whose reference data does not cover your
  state.** `US_parcel_parcel-placeslab-fmv2026` covers 8 MA counties and
  PA-Delaware only. Running it elsewhere writes an evidence file holding just
  `_join_id`/`parcel_id`, and `merge_enrichments` *skips a missing* enrichment
  but *raises on one that exists and lacks a column* — turning a clean skip into
  a hard curate failure for every county. **A zero-second stage is not a fast
  stage.** Recovery: delete the empty evidence parquet, re-run curate alone.

Check an enrichment's `reference_parcel_recipe_id` / `image_recipe` coverage
before including it.

---

## Trap checklist (all observed, none hypothetical)

- Same-named county in another state passing a title search.
- Hub export missing the joined CAMA attributes (Wilson NC).
- Hub 202/404 stub saved as the data file.
- `arcgis_rest_scraper` count query inheriting `f=geojson`; most servers ignore
  `returnCountOnly` there and return a FeatureCollection with no `count`.
  (Fixed in-repo by forcing `f=json`; if you see `KeyError: 'count'`, that fix
  regressed.)
- Bundled id pattern nulling the whole key (Hertford/Northampton/Pamlico NC).
- Degenerate id column producing a populated-but-useless key (Pamlico NC).
- Blank-`source_id` override row hijacking a new county recipe (Carteret NC).
- Esri epoch-ms dates; integer `YYYYMMDD` dates; sentinel dates.
- Declared-but-empty schema fields (DeKalb GA; Greene NC's
  `ADJUSTED_SALE_PRICE` at 0%).
- Address-keyed "parcel" layers with duplicate parcels (Wilson NC: 3,644
  distinct PINs per 4,000 rows) — dedupe before per-parcel counts.
- Placeholder rows (`UPDATE IN PROGRESS`, all-null shells) — keep them, they
  get a null key and never join, but expect them in row counts.
- Assuming a source's data is unrestricted because it is publicly
  downloadable (it may not be — check for a terms page before assuming).

---

## Definition of done

1. Recipe validates: loads, every column in the registry, description ≤25
   words, comments ≤72 chars, and the source's terms recorded (`license`
   at minimum — `'unknown'` if a search found no terms page).
2. Ingested output reconciles exactly against the live server on rows and on
   the headline attribute.
3. `parcel_id_local` populated, non-degenerate, and overlapping the county roll
   / statewide layer above ~95%.
4. Downstream stages re-run in dependency order, with curated output verified
   against the ingest figures.
5. Header comment records every rejected alternative and every measured number,
   so the next agent does not re-litigate it.
6. Anything left unmapped for want of a registry attribute is raised explicitly
   with the user, not quietly dropped.

See also: [`../README.md`](../README.md) for shared conventions,
[`ingest_recipes.md`](ingest_recipes.md) for writing the recipe itself.
