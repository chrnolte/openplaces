# Recipe authoring guide

Conventions every recipe in this tree follows, and an index of the
stage-specific guides. Read this before adding or editing a recipe.

For the *architecture* — what each stage does, how the spines relate, what
the registry is for — see `AGENTS.md`. This guide is about how to write the
YAML, and about the traps that have actually cost time.

## Guides

| Guide | When to read it |
|---|---|
| [`_instructions/acquiring_sources.md`](_instructions/acquiring_sources.md) | Adding a county or state the repo does not cover yet: finding a source, validating it is the right one, choosing bulk vs REST |
| [`_instructions/ingest_recipes.md`](_instructions/ingest_recipes.md) | Writing a `stage: ingest` recipe (parcel, property, transaction, footprint) |
| [`_instructions/harmonize_recipes.md`](_instructions/harmonize_recipes.md) | Writing or extending a `stage: harmonize` pipeline |
| [`_instructions/enrich_recipes.md`](_instructions/enrich_recipes.md) | Writing a `stage: enrich` recipe that produces evidence |
| [`_instructions/curate_recipes.md`](_instructions/curate_recipes.md) | Writing a `stage: curate` recipe that produces a canonical entity |

`_instructions/` holds Markdown only. Recipe discovery globs `*.yaml`
exclusively (`rglob('*.yaml')` and `**/{entity_type}/*/*/*.yaml`), so nothing
here is visible to `get_recipe_by_id` or the DAG.

## Where a recipe lives

```
{admin_id_path}/{entity_type_or_theme}/{source}/{version}/{recipe_id}.yaml
```

The recipe id encodes its own path:
`{admin_id}_{entity_type_or_theme}-{source}-{version}[_{filename}]`.

```
US/NC/PI/_all/parcel/pittcounty/2026/US-NC-PI_parcel-pittcounty-2026.yaml
US/NC/_all/parcel/nconemap/2025/US-NC_parcel-nconemap-2025.yaml
US/_all/parcel/spine/2026/US_parcel-spine-2026.yaml
```

Sidecar CSVs (crosswalks, keyword tables, id overrides) sit beside their
recipe and are named for it, e.g.
`US_parcel-openplaces-2026_land-use-keywords.csv`.

## `source_id`

Name the **publisher**, not the geography, and default to `{county}county`
(`pendercounty`, `lenoircounty`, `carteretcounty`). Deviate when the
publisher genuinely differs, and say why in the header comment:

- `rockymountnc` — the City of Rocky Mount publishes Nash County's parcels
- `nhcgov`, `bladenco` — the county's own domain differs from its name

## Column names come from the registry

Every key under `columns:` (and every `output:` of a transformation) must
exist in `core/attribute_registry.csv`. The registry is the canonical
vocabulary; a recipe never invents a name.

If a source has something genuinely useful with no registry equivalent,
**leave it unmapped and say so loudly in the header**, then raise the
registry question. Do not force it into a nearby field — a mis-homed value
is worse than an absent one because nothing downstream can tell it is wrong.
Real examples currently parked this way:

- Pitt's `SALE_TYPE` (`SLTYLND`/`SLTYPKG`) describes what a sale conveyed,
  not a document type, so it is not `doc_type`.
- Bedroom / bathroom / room counts, heating, basement and garage flags are
  common in NC assessor layers and have no registry home yet.

Check a field's **values**, not its name, before mapping it. Sampson's
`USE_DESC` reads like land use and is actually building style (`RANCH`,
`DOUBLE WIDE MOHO|STORAGE`); its land use lives in `SEG_TYPE_D`.

## Schema, never records

A recipe describes a source's **schema** — column names, field codes,
category vocabularies. It must never carry the source's **records**. See the
privacy section of `AGENTS.md`; this is the recipe-side application of it.

Fine to quote, because they are schema:

- column names (`NAD83_PIN`, `SALE_PRICE`)
- field-code vocabularies (`RES`, `FRM`, `MAN-HOU`, `SLTYLND`/`SLTYPKG`)
- sentinel placeholders (`0` for "no year recorded", `1901-01-02` for
  "date not recorded")
- aggregate statistics (`968/1000 sampled records`, `17,086 distinct keys`)

Not fine, because they are records — **fabricate these**, preserving the
format, which is the only thing the example is for:

- parcel ids, account numbers, deed book/page references
- house numbers, addresses, owner names
- a sale date or price attached to a particular parcel

```
# bad  — copied from a live record
# Join key: NC_PIN ('3578-31-2699'), 98.4% populated.

# good — same format, fabricated
# Join key: NC_PIN ('9999-99-9999'), 98.4% populated.
```

The population figure is the informative part and is an aggregate; the id is
only there to show the shape.

## Check the source's terms before you write the recipe

Every recipe's `source:` block should set `license`, `terms_url`, and
`redistribution_restricted` once you have looked. "I didn't check" is not the
same as "no restriction exists" — if a reasonable search turns up no explicit
terms or license page, set `license: 'unknown'` and
`redistribution_restricted: null`, not `false`. Most U.S. government
open-data portals genuinely have no formal terms page, and recording that is
a real answer, not an empty field.

A source with unusually restrictive terms (non-commercial only, no
redistribution without permission, no bulk caching) is not a reason to skip
the recipe. It is a reason to write the restriction down, so the next person
or agent does not have to rediscover it and does not build something
downstream that assumes the data is unrestricted.

```yaml
# bad — silently assumes the data is free to use however
source:
  source_id: gadm
  portal_url: "https://gadm.org/"

# good — same source, restriction recorded
source:
  source_id: gadm
  portal_url: "https://gadm.org/"
  license: "non-commercial, no redistribution without permission"
  terms_url: "https://gadm.org/license.html"
  redistribution_restricted: true
```

## Style

- `description:` — 25 words maximum. It is a label, not documentation.
  Everything else goes in the header comment or `notes`.
- Comments wrap at 72 characters; content at 88. Long URLs are the accepted
  exception.
- Use `|` unions in any Python you touch (`isinstance(x, Foo | Bar)`), and
  run `ruff format` / `ruff check` on it.
- No `─`/`===` divider lines in comments.

## Write down what you measured

The header comment is the only place a future reader learns that a field was
checked and rejected. State numbers, not adjectives:

> `NOTE: no year_built field in this layer's schema (verified). Land and
> building value are populated (968/1000 and 667/1000 sampled records
> respectively), so land/building value is included as a bonus.`

Record every alternative you rejected and why — the sibling layer that was a
subset, the field that was 0% populated, the bulk export that was missing its
CAMA join. That is what stops the next agent re-litigating it.

Measured facts are counts, rates and distinct-value totals. They are
aggregates, so they are safe to record — unlike the record values they were
computed from (see *Schema, never records* above).

## Before you call it done

1. The recipe loads: `get_recipe_by_id(rid)`.
2. Every mapped column is in the registry.
3. `description` ≤ 25 words; comments ≤ 72 chars.
4. The source's terms were checked and recorded (`license` at minimum).
5. The ingested output reconciles against the source (see
   `_instructions/ingest_recipes.md`).
6. Downstream stages that consume it have been re-run — a new parcel source
   invalidates both spines and both curated outputs.
