# `admin_codes` — how administrative identifiers are made

An `admin_id` is a hierarchical identifier for one administrative unit:
`US-NC-MO` is Mecklenburg County, `US-AK-BE-AN` is Aniak. It is the
primary key of the admin-unit table and the join key of anything
aggregated across admin units, and it also appears in directory paths,
filenames and column names — so it has to be both recognizable and short.

The spine ships as committed CSVs under
`recipes/_all/admin/spine/2026/`. **Nothing in this directory has to run
for the package to work.** It exists so those files can be reproduced.

## Reading an identifier

| module | what it does |
|---|---|
| `candidates.py` | generates code candidates from a name's own grammar |
| `assign.py` | solves one sibling group as a weighted assignment |
| `derive.py` | picks the width, applies anchors and overrides |
| `frame.py` | mints a whole DataFrame; handles pinning and duplicates |
| `anchors.py` | published codes and reviewed overrides |
| `registry.py` | reads the committed spine |
| `audit.py` | invariants the spine must satisfy |
| `coverage.py` | how much of the world is covered |
| `build.py` | rebuilds the population weights and re-mints |

## Only the present spine

The spine is the registry of record, and only its current version is
kept. There are no snapshots of earlier vintages and no old-to-new
tables: a re-mint rewrites the spine in place, and nothing resolves a
retired identifier.

A re-mint can **recycle** identifiers: a string survives and names a
*different* unit. `US-NC-HY` was Hyde and is now Haywood. So a file keyed
on admin ids that outlives a re-mint is regenerated from its source's
own codes (a FIPS, a DANE or INSEE code, the unit's name within its
parent), never migrated by string substitution. For data files that
carry the national code beside the id, `rekey.rekey_files` is that
regeneration: it joins the code to the present spine and rewrites the id
column in place, footers and read-only bits intact.

```python
from openplaces.io.admin_codes.rekey import rekey_files
rekey_files(4, 'US', 'census_subdivision_id', apply=False)   # report
rekey_files(4, 'US', 'census_subdivision_id', apply=True)    # rewrite
```

Recipes are keyed on admin ids too, through their directory and
`admin_id:` field. After a re-mint, check each county-scoped recipe's
path against the spine by the county's name; a recipe under a retired
or recycled id is moved to the unit's present id.

## What the spine may carry

Only values its sources allow to be published. A unit's national code
(`admin{N}_id_admin1`) is kept only for the countries and levels listed
in `admin-spine-2026_code-sources.csv`, each naming the license-clean
recipe that supplies it. GADM's codes, alternative names and
native-script names were removed on 2026-09-11, because GADM does not
allow redistribution. `tests/io/test_admin_spine_provenance.py` holds
the line.

None of these columns takes part in the mint, which runs with pinning
off, so removing them moved no identifier.

## Rebuilding the spine

Prerequisite: ingest the population raster once.

```python
import openplaces as op
op.ingest('population-ghsl-r2023a')
```

Then, from `openplaces.io.admin_codes import build`:

```python
# 1. Zonal-sum the raster over each unit's own polygon, coarsest first.
for level in (2, 3, 4):
    build.build_population(level)

# 2. Override where a country's own layer knows its units better than
#    the world geometry does. New England towns are level 3 and come
#    from each state's own layer; the national admin4 layer excludes
#    them. The table is population-overrides.csv, which
#    rebuild.rebuild_spine reads; the world geometry (geoBoundaries,
#    build.GEOMETRY_RECIPE) adds one row per country-level its licence
#    sidecar marks permissive, derived at run time by
#    rebuild.geometry_overrides, and a national row wins over it.
for state in ('US-CT', 'US-MA', 'US-ME', 'US-NH', 'US-RI', 'US-VT'):
    build.build_population_from_entity(
        f'{state}_admin-census-2025_admin3', state, level=3,
        # Connecticut renumbered its counties into planning regions in
        # 2022, so join on state + subdivision, not the whole GEOID.
        key=lambda g: f'{g[:2]}{g[-5:]}',
    )
build.build_population_from_entity('CO_admin-dane-2025_admin3', 'CO', level=3)
for recipe, country in [
    ('GB_admin-ons-2024_admin4', 'GB'),
    ('DE_admin-gisco-2024_admin4', 'DE'),
    ('FR_admin-gisco-2024_admin4', 'FR'),
    ('US_admin-census-2025_admin4', 'US'),
]:
    build.build_population_from_entity(
        recipe, country, level=4, join_column='admin4_id_admin1'
    )

# 3. Apportion each parent's shortfall. Parents before children.
build.fill_population_gaps(3)
build.fill_population_gaps(4)

# 4-5. Repeat until the mint stops moving (two or three passes).
build.repair_zero_weights()
build.remint_spine(apply=True, backup_dir=...)   # outside the package
```

## Why each step is shaped the way it is

Each of these was a bug before it was a rule.

**Place geometry rows by name, never by "the id is live".** The admin
geometry carries more than one identifier vintage at once. Its `JP-TO`
row names Tokyo; the live `JP-TO` is Tokushima. Trusting the id credits
Tokyo's 13.7 million people to a town of 720,000.

**Name and type cannot separate a level change from a status change.**
Worcester County's polygon sits on the id the *town* of Worcester now
holds, and both are called "Worcester" — but Colombia's departments read
"Commissiary" in an older vintage and are the same units. There is no
general rule; `build.GEOMETRY_MISMATCHED` lists the affected units by
name and supplies them from an unambiguous layer.

**Never emit a zero weight.** A residual split hands out zero when the
covered children already account for the parent's total, which happens
when a matched polygon is too large — two communes once absorbed
Alpes-Maritimes and left 161 others at zero. Zero loses every tie it
enters, so a geometry artefact becomes a permanent handicap. Unknown must
mean "competes as typical".

**Weight by raw population, not its logarithm.** An identifier is read by
a person, so a tie-break should minimize the people who meet an
unreadable code, and that is linear in population. Log10 compresses a
110-fold difference into 1.44-fold, less than the rank penalty — under it
Cook County, King County and Somerville all lost their obvious codes to
much smaller neighbours.

**Sanity-check the total, not the coverage.** The level-2 table summing
to 7.79 billion against a true ~7.79 billion is what proved the zonal
step correct. Coverage percentages looked healthy while Tokyo's
population sat on Tokushima.

## Verifying a rebuild

```python
from openplaces.io.admin_codes.audit import audit_spine
audit_spine(levels=(2, 3, 4))     # format, orphans, duplicates, widths
```

`pytest tests/io/test_admin_codes_build.py` asserts the invariants that
matter: every unit weighted, none weighted zero, and the mint being a
fixed point.
