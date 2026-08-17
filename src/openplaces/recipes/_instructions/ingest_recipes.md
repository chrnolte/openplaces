# Writing an ingest recipe

`stage: ingest` (the default — the key is usually omitted). One recipe reads
one source and writes one entity or dataset per admin unit, applying column
mappings, type casts and the registry's checks. No cross-source logic happens
here; that is harmonize's job.

Read [`../README.md`](../README.md) for naming and style. If the source is new
to the repo, read [`acquiring_sources.md`](acquiring_sources.md) first.

## Shape

```yaml
# Header comment: the source and its endpoint, why this layer rather than
# the alternatives, every measured fact relied on, and every field
# deliberately left unmapped with the reason.

description: >-                    # 25 words max
  Pitt County, NC parcel roll (land/building values, year built, heated area,
  owner, situs, most recent sale) from the county's ArcGIS parcel layer.

admin_id: US-NC-PI

entity:                            # or `dataset:` for a non-entity theme
  entity_type: parcel              # parcel | property | transaction | footprint | ...
  source:
    source_id: pittcounty
    portal_url: "https://..."      # human landing page
    download_url: "https://..."    # preferred: one bulk file
  version: "2026"

uncompressed_file_name: "pitt_parcels.geojson"

columns:                           # registry name: SOURCE_FIELD
  parcel_id_assessor: PARCELNUMBER
  last_sale_price: SalesPrice

parcel_id_local:                   # the join key; see below
  source: parcel_id_assessor
  kind: parcel

save_to:
  data_dir: core
```

## Getting the file

Three mutually exclusive routes, in order of preference:

| key | use when |
|---|---|
| `download_url` | a single bulk file exists (zip, gdb, geojson, csv) |
| `download_url_source` + `download_url_source_regex` | the real link must be scraped from a landing page (per-year files with unpredictable ids) |
| `download_url_scraper: arcgis_rest_scraper` + `scraper_options.layer_url` | only when no bulk file exists |

`compressed_file_name` / `uncompressed_file_name` name what lands on disk;
when `uncompressed_file_name` is set, the download is saved under that name
regardless of the URL's shape, so a query-string URL is fine.

Prefer bulk — REST paging costs the county's server hundreds of sequential
requests. But **verify the bulk export carries the same attributes as the REST
layer** before committing to it; see the equivalence test in
[`acquiring_sources.md`](acquiring_sources.md). One NC county's export had
every CAMA field null with a perfectly plausible feature count.

## Reading the file

- `layer:` — which layer of a multi-layer gdb/gpkg.
- `encoding:` — set `utf-8-sig` for files with a byte-order mark, or the first
  column name arrives as `﻿OBJECTID`.
- `csv_dtype:` — `str` for everything, or a `{FIELD: str}` map. **Always set
  this for all-digit id columns**, or pandas infers an integer, breaks the
  registry's string type and destroys leading zeros.
- `sheet_name` / `header: none` / `names:` — for spreadsheets without a header
  row, name columns positionally using canonical registry names directly; no
  `columns:` block is then needed.
- `query:` — a pandas query applied before parsing, e.g. to drop a leaked
  header row.
- `drop_duplicates:` — an explicit column subset. A bare full-row dedupe keeps
  rows that differ only in a column you do not care about.

## Transformations

Run after column mapping, on the mapped names (or raw names, for `concat`
inputs). Common patterns:

```yaml
transformations:
  - type: string                   # assemble book/page from two fields
    operation: concat
    inputs: [DEED_BOOK, DEED_PAGE]
    args: {sep: "-"}
    output: last_sale_book_page

  - type: unary                    # null a sentinel BEFORE parsing
    operation: null_if_equal
    input: last_sale_date
    args: {value: "1901/01/02"}
    output: last_sale_date

  - type: unary
    operation: to_datetime
    input: last_sale_date
    output: last_sale_date
```

Useful ops: `null_if_equal`, `to_numeric`, `to_datetime`, `parse_currency`,
`resolve_century` (2-digit years), `split_take`, `concat`, `zfill`.

### Dates are the most common silent corruption

- **Esri `f=geojson` returns date fields as epoch milliseconds**, and nothing
  casts them at ingest — `last_sale_date` lands as `1345161600000.0`. Prefer a
  string-formatted date field if the source has one; otherwise document the
  units.
- **Integer `YYYYMMDD`** read as a datetime becomes nanoseconds since 1970.
  Map a clean year field instead.
- **Sentinel dates** (`1901-01-02`, `1900-01-01`, `01/1900`) are common. Spot
  them by how sharply they spike against neighbouring dates, and null them
  before parsing. Keep the rows — usually only the date is a placeholder.

## `parcel_id_local`

The join key everything downstream depends on. Value fields and
`last_sale_price` are **not** in the parcel spine's `keep_columns`; they reach
it only through the `link_by_id` join on this key. A null or degenerate key
drops the whole table silently.

```yaml
parcel_id_local:
  source: parcel_id_assessor       # which mapped column feeds it
  kind: parcel                     # parcel | tax (selects the bundled conv)
```

Per-county conversions come from `geo/parcel_id_links.csv`. When that default
does not fit a source, add a row to
`US/_all/parcel/_all/US_parcel_id-overrides.csv` rather than editing the
bundled table. Full treatment of the three failure modes — misfitting pattern,
degenerate source column, empty column — is in
[`acquiring_sources.md`](acquiring_sources.md#phase-5--the-join-key-parcel_id_local).

## Partitioning and output

- `download_by:` — `admin_level`, or `partition: year | year_month | table |
  tile_id | latlon_tile`.
- `process_by:` — chunking granularity: `admin_level` plus either
  `admin_id_column` (filter rows by an in-data column) or `file_pattern` (one
  pre-split file per admin unit inside a shared download).
- `save_to:` — `data_dir` (`core` for entities you keep, `cache` for
  intermediates such as transactions, `share` for deliverables),
  `admin_level`, optional `retention`.
- `join_partitions_by:` — left-join per-table partitions into one file per
  admin unit.
- `additional_layers:` — secondary entities extracted from the same source
  file (a property table alongside a parcel table).

## Verify before calling it done

Reconcile **exactly** against the source, not by eyeball:

| check | how |
|---|---|
| rows | parquet rows vs `returnCountOnly` on the server |
| the attribute you came for | count `> 0` in parquet vs `where FIELD > 0` on the server |
| key quality | `parcel_id_local` population, distinct count, overlap with the county roll / statewide layer (expect >95%) |
| dtypes | ids are strings; dates are dates, not epoch integers |

A deficit of tens of rows is usually export-snapshot lag — report it. A large
one means the wrong layer, a subset, or a bulk export missing its join.
