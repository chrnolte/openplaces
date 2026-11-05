.. openplaces

.. _cheer_pipeline:

How the inventory is built
==========================

The inventory is built county by county and then pooled into a region.
Each county passes through the four stages every ``openplaces``
dataset passes through (ingest, harmonize, enrich, curate). The
regional bundle is then written by the delivery, which also fills the
gaps. This page follows one county through the stages. The reasons
behind the main choices are on :doc:`decisions`, the gap filling is
described on :doc:`imputation`, and every step with its reason is
reviewed in the :ref:`step review <cheer_step_review>`.

.. toctree::
   :maxdepth: 1

   decisions
   imputation
   step-review

.. mermaid::

   flowchart TD
     subgraph ingest [Ingest]
       R[County tax rolls and parcels] --- O[Building footprints: OBM, Microsoft, NC state layer]
       O --- P[Points: NSI, Overture addresses, FEMA]
     end
     subgraph harmonize [Harmonize]
       FG[Footprint geospine: one footprint per building] --> BG[Building geospine and spine: points and parcels linked]
       BG --> PS[Parcel geospine and spine: roll values on the lot]
     end
     subgraph curate [Curate]
       PC[Parcel curate: use classes, clean values] --> BC[Building curate: values apportioned, occupancy voted]
       BC --> FC[Footprint curate: buildings carried onto footprints]
     end
     ingest --> harmonize --> curate
     FC --> D[Delivery: pool counties, withhold, impute, write the bundle]

Three entities
~~~~~~~~~~~~~~

Three kinds of row meet in the pipeline, and each means one thing (see
:ref:`entities <entities>`):

- a **parcel** is a unit of land as the county draws it;
- a **building** is one structure;
- a **footprint** is a building's shape as drawn from above.

Values and occupancy are decided for each building, because that is
what the sources describe. The bundle ships the footprint, because that
is the shape wind and water act on; a townhome row is one footprint but
several buildings, and the footprint carries their sum. A tax roll's rows are
properties (what a sale conveys), which are linked to the parcels they
sit on.

Ingest
~~~~~~

Ingest downloads each source and stores it as received, with its own
codes. Two rules apply to every source:

- **Every parcel table is split into lots and properties.** A row
  stacked on a shared lot polygon (a condominium unit, for example)
  is a property, not a parcel.
- **Personal columns never ship.** Owner names and mailing addresses
  never reach a bundle.

The sources for a CHEER county are its parcel layer and tax roll (from
the county, or the state's statewide parcel service), three footprint
layers, NSI points, Overture address points, FEMA's USA Structures and,
in North Carolina and Texas, an earlier CHEER inventory.

Harmonize
~~~~~~~~~

Harmonize builds one table per entity (its spine) and the links between
them. It chooses no values; it only says which records belong
together.

**Footprints.** Footprints from the three layers are merged in a fixed
order: OpenBuildingMap first, then Microsoft, then (in North Carolina)
the state's footprints. A lower-ranked footprint enters only where it
overlaps no footprint already accepted. Very small footprints (under
10 m2) and long thin duplicates are dropped. Where a roll records a
building on a parcel that no layer draws, the parcel polygon stands in
for it (its ``geometry_source`` reads ``parcel.<county>``). Stacked
condominium parcels are drawn as one footprint per building.

**Parcel links.** Every footprint is overlaid on the parcels. A footprint
keeps the parcels it overlaps substantially, so a building straddling a
lot line does not count twice.

**Buildings.** Each footprint becomes a building. NSI points and Overture
address points are joined to the building they fall in, or to the
nearest building on the same parcel. On a parcel with several
buildings, the ones with an address point are marked ``primary`` and
the others (garages, sheds, barns) ``secondary``; without address
points, the ones with an NSI point, else the largest. A large building
without an address point stays primary.

**Parcels.** The roll's values (land, improvement, total, year built,
use code) are joined to the parcels by the parcel number.

**Area identifiers.** Each footprint is stamped with the census tract,
block group, block, ZIP code area and county subdivision it lies in.

Enrich
~~~~~~

Enrichment adds evidence that is not in the ingested tables. For the
CHEER regions this is the earlier CHEER inventory's roof shape,
foundation, construction type and story count. The pipeline can also
classify roof shapes and story counts from satellite and street-level
imagery (BRAILS++); that is off by default, because it is costly, and
the imagery is fetched in memory and never stored.

Curate
~~~~~~

Curation chooses one value per attribute and records where it came
from. It runs three times.

**Parcel curate.** Use codes are read into classes (single-family,
manufactured home park, townhome, condominium, vacant, ...). Values the
roll states impossibly (negative improvement values, for example) are
set to missing.

**Building curate.** This is where most decisions are made.

- *Structure value.* A parcel's improvement value is shared among its
  primary buildings by their area on the parcel, weighted by floors. A secondary building carries 0, with
  ``structure_value_zero_reason`` set to ``secondary``.
- *Single-source attributes.* Year built, stories, dwellings,
  foundation and the others are each taken from a fixed order of
  sources: the first source that has a value supplies it.
- *Occupancy.* A base class comes first, from NSI, FEMA and the roll's
  use code (read as a class through the NSI class most common among the
  county's buildings with that code). Weighted votes then decide in
  turn: one dwelling or several (Overture's address count, the
  assessor's description, NSI's unit count); the residential class (RV
  dwelling, manufactured home, multi-family, single-family, secondary),
  where the footprint's shape, the assessor's description, NSI, FEMA and
  the parcel's park status each add a fixed weight; and, for
  multi-family buildings, the height band from the story count. Only
  evidence that reaches the building through its location votes. A last
  vote gives a manufactured home its number of sections.
- *Zero values.* Where the roll gives a building no value of its own,
  a reason is recorded (see :doc:`columns`).

**Footprint curate.** The building table is carried onto the footprints.
Where one footprint holds several buildings, counts and values add up,
classes list each building's value, and ``year_built`` is the earliest.

Delivery
~~~~~~~~

The delivery pools the region's counties into one bundle:

1. It refuses curated files older than their inputs.
2. A footprint delivered by two neighboring counties is cut at the
   county line, and each piece keeps its own county's values.
3. Values from sources whose terms do not allow redistribution, and
   share-alike values that cannot be published together with the
   footprints' licence, are withheld (their ``_source`` reads
   ``withheld``). The team bundle keeps the values cleared for the team.
4. The gaps are filled by models trained on the recorded values (see
   :doc:`imputation`).
5. The four tables, the QGIS map and the licence notice are written.

Running it yourself
~~~~~~~~~~~~~~~~~~~

The whole chain runs from the recipe tree
(:gh-file:`src/openplaces/recipes/`) with ``snakemake``, one job
per stage, recipe and county, plus one delivery job per region. For a
few counties, the notebook :gh-file:`notebooks/examples/US_curate_footprints.ipynb`
drives the same stages step by step. The recipes hold every parameter:

- :gh-file:`US_building-openplaces-2026 <src/openplaces/recipes/US/_all/building/openplaces/2026/US_building-openplaces-2026.yaml>`
  (building curate: values, occupancy, imputation inputs);
- :gh-file:`US_footprint-cheer-2026 <src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml>`
  (footprint curate and delivery).
