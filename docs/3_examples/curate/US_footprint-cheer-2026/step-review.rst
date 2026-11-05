.. openplaces

.. _cheer_step_review:

Step review: why each step exists
=================================

Every step that builds the :ref:`CHEER footprint inventory <cheer_footprints>`, with the reason it exists and whether that reason still holds.

.. contents:: Table of contents
   :local:
   :depth: 2

The inventory is not made in one pass. It is built by a long chain of
small steps, each of which does one thing: join a tax roll to its
lots, drop a duplicate footprint, choose a building's year of
construction. This annex lists every one of those steps, says why it
is there, and says whether that reason still holds. It was written on
2026-11-03 by reading the code (main branch, commit 500b09b9) and
checking the results in small North Carolina and Texas counties.

How to read this annex
~~~~~~~~~~~~~~~~~~~~~~

**The sources.** The inventory combines:

- the county's **tax roll** (the assessor's records: land and building
  values, year built, use code) and its **parcel map** (the lots as the
  county draws them), from the county or from the state's statewide
  parcel service;
- **building footprints** drawn from imagery: OpenBuildingMap (**OBM**),
  **Microsoft**'s building footprints and, in North Carolina, the
  state's own footprints (**NC DPS**);
- **points** that each describe one building: the USACE National
  Structure Inventory (**NSI**), FEMA's USA Structures (**FEMA**) and
  Overture's address points (**Overture**);
- the earlier CHEER inventory (**CHEER v0**), for roof shape,
  foundation, construction type and story count.

**Recipes and steps.** Each table is built by a *recipe*: a
configuration file that lists *steps* in the order they run, with
their settings. Each section below names its recipe and links to it.
The number after a step (#12) is its position in that recipe, and the
step's name (``derive_indicators``) is the name of the function that
runs it, so an engineer can find it in the code.

**The tables.** Every county is built in four stages (see
:doc:`pipeline`):

- *Assembling* (the "geospine" and "spine" recipes). Each kind of
  thing gets one table: lots (parcels), building footprints,
  buildings and tax-roll accounts (properties). The geospine holds the
  shapes, the spine the attributes. These steps collect what every
  source says and record which records belong together; choosing
  one value is left to curation.
- *Enriching*. Evidence from outside sources is added.
- *Choosing* (the "curate" recipes, named ``openplaces`` or ``cheer``).
  Here one value per attribute is chosen from the evidence, and its
  source recorded.
- *Delivering*. The counties of a region are pooled into one bundle,
  values whose terms do not allow sharing are withheld, and the gaps
  are filled by models (the *imputation*).

**Votes.** Several attributes, such as a building's occupancy class, are
chosen by a *vote*: each source proposes a class with a fixed weight,
and the class with the highest total wins. The weights are set in the
recipe.

**County rules.** Where one county's data needs its own treatment, the
rule is written in a small recipe for that county (a *county patch*),
which changes one step for that county only.

**Verdicts.** An entry without a verdict holds: the reason is right,
and the step does what it should. Other entries end with one of three
verdicts:

- **holds, reworded**: the reason is right and the step is fine, but
  its explanation in the code is out of date in a detail (an old file
  name, an undated count). The bundle stays the same;
- **weakened**: the case the step guards against no longer occurs, or
  is now handled elsewhere;
- **open**: not yet verified; the entry says what would settle it.

**Where steps are listed.** Steps are grouped by recipe, in the order
the build runs the recipes, and within a recipe in the order they run.
Steps that matter only in a few counties or outside North Carolina and
Texas say so in their reason. Steps that run but feed no column of the
bundle (addresses, for example) are listed briefly at the end of their
recipe.


Summary
~~~~~~~

Verdicts per recipe, in build order. An entry without a verdict
holds.

.. list-table::
   :header-rows: 1
   :widths: 40 12 12 12 12

   * - Recipe
     - Holds
     - Holds, reworded
     - Weakened
     - Open
   * - Building footprints (footprint geospine)
     - 3
     - 3
     - 0
     - 1
   * - Buildings and the points linked to them (building geospine)
     - 4
     - 1
     - 1
     - 0
   * - Building attributes (building spine)
     - 6
     - 1
     - 3
     - 0
   * - Footprint attributes (footprint spine)
     - 1
     - 0
     - 0
     - 0
   * - Tax-roll accounts (property spine)
     - 3
     - 2
     - 1
     - 0
   * - Lots and their values (parcel geospine)
     - 6
     - 6
     - 2
     - 1
   * - Lot-level evidence (parcel spine)
     - 12
     - 2
     - 0
     - 0
   * - Construction details of tax-roll accounts (property curate)
     - 5
     - 0
     - 0
     - 0
   * - Classifying each lot (parcel curate)
     - 10
     - 8
     - 2
     - 1
   * - Deciding each building's attributes (building curate)
     - 21
     - 18
     - 1
     - 1
   * - Footprint curate: carrying buildings onto footprints
     - 2
     - 1
     - 2
     - 0
   * - Footprint curate: pooling and withholding (delivery)
     - 10
     - 2
     - 1
     - 0
   * - Footprint curate: filling the gaps (imputation)
     - 15
     - 9
     - 2
     - 0
   * - Footprint curate: writing the files (delivery)
     - 1
     - 1
     - 0
     - 1
   * - **Total (173 steps)**
     - **99**
     - **54**
     - **15**
     - **5**

Steps by recipe
~~~~~~~~~~~~~~~

One section per recipe, in the order a county is built: building
footprints first, then the buildings and the points linked to them,
then the tax-roll accounts and the lots (which need the buildings to
count them), then the choosing stages, and last the regional bundle
and its imputation.

Building footprints (footprint geospine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_footprint-geospine-2026 <src/openplaces/recipes/US/_all/footprint/geospine/2026/US_footprint-geospine-2026.yaml>`.
This recipe draws the footprints: one shape per building as seen
from above, which is the row the delivered inventory ships.

Merge the footprint layers in a fixed order
   :Step: ``resolve_spine`` (#1)
   :Does: Combines building footprints from three layers:
      OpenBuildingMap (OBM) first, then Microsoft's building footprints,
      then, in North Carolina, the state's own footprints (NC DPS). An
      footprint from a later layer is added only where it barely overlaps
      a footprint already accepted and is not a long, thin copy of one.
      A Microsoft footprint is also left out where most Microsoft
      footprints around it have no match in the other layers, a sign that
      Microsoft's older imagery shows buildings since torn down.
      Footprints under 10 m2 are dropped, and FEMA's footprints are never
      used.
   :Why: Each building should be drawn once. OBM comes first because the
      maintainer found its footprints more complete and better than
      Microsoft's US layer (2026-11-03). FEMA's footprints are the lowest
      quality the project reads (maintainer, 2026-10-16).
   :Verdict: open. No reason is recorded for ranking Microsoft's
      footprints above North Carolina's own. Comparing both against an
      independent set of footprints in one North Carolina county would
      settle it.

Link each footprint to the lots it stands on
   :Step: ``link_to_reference``, parcels (#2)
   :Does: Lays the footprints over the county's parcel map and records
      which lots each footprint overlaps. An overlap counts if it is at
      least 10 m2 and at least a sixth of the footprint's largest
      overlap. Where a footprint is shifted so that it touches a row of
      lots one after another, it is linked to its main lot only.
   :Why: Almost every later step uses this link: to tell main buildings
      from sheds, to split a lot's value among its buildings, and to
      find condominium buildings. Without the size rules, a house that
      grazes the neighbor's lot line would count on two lots.
   :Verdict: holds, reworded. One setting in the recipe asks to sort
      the lots by a measure that does not exist yet at this point, so
      the step sorts them by lot area instead, which is the sensible
      order. The link itself is not affected.

Draw one footprint per condominium building
   :Step: ``consolidate_condo_cluster_footprints`` (#3)
   :Does: Finds groups of small, touching lots that are condominium
      units of one building, and replaces the footprints drawn over them
      with one footprint per building.
   :Why: In some counties each condominium unit is drawn as its own
      small lot. Without this step, those unit lots would get no
      footprint, or a stand-in footprint each, and one building would
      appear as many.

Add a stand-in footprint where the roll values a building no layer draws
   :Step: ``infer_spine_additions`` (#4)
   :Does: Where the roll gives a lot a building value but no footprint
      layer draws a building on it, uses the lot's own shape as the
      footprint. This is skipped where the lot's building value per
      hectare is low for its kind of land use.
   :Why: The roll says a building exists that the footprint layers miss.
      The stand-in
      keeps its value in the inventory; its ``geometry_source`` (the
      source of the footprint) reads ``parcel.<source>``.

Remove footprints that cover each other
   :Step: ``resolve_overlaps`` (#5)
   :Does: Where two footprints overlap by at least half of the smaller
      one's area, drops the smaller one.
   :Why: The merge in the first step tolerates small overlaps, and
      some larger ones still get past it. The same building should not
      appear twice.

Compute each footprint's center point and area
   :Step: ``derive_geometry_attributes`` (#6)
   :Does: Writes the center point and area of each footprint. The area is
      left empty on stand-in footprints taken from a lot's shape.
   :Why: Done once, after the footprints are final, so every later step
      uses the same numbers. A lot's area is not a building's area.
   :Verdict: holds, reworded. The explanation in the code says this
      runs right after the merge; it runs after overlaps are removed,
      which is the right place.

Record the census areas each footprint lies in
   :Step: ``link_geographic_ids`` (#7)
   :Does: Writes the municipality, census tract, block group, block and
      ZIP code area (ZCTA) of each footprint's center point.
   :Why: Done once and passed on to the buildings and lots, so all
      tables agree on which area a building is in.
   :Verdict: holds, reworded. The recipe's comment says the lots take
      these areas from this table directly; they take them through the
      building table. The result is the same.

.. rubric:: County patches

One county patch changes this recipe:
:gh-file:`US-NC-PIT_footprint-geospine-2026 <src/openplaces/recipes/US/NC/PIT/_all/footprint/geospine/2026/US-NC-PIT_footprint-geospine-2026.yaml>`.
It was not reviewed step by step.

Buildings and the points linked to them (building geospine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_building-geospine-2026 <src/openplaces/recipes/US/_all/building/geospine/2026/US_building-geospine-2026.yaml>`.
This recipe makes the building table: one row per structure, with
the points that describe it (NSI's building points and Overture's
address points) attached. Values and classes are decided for
buildings and later added up onto the footprints. Today each
footprint becomes one building; splitting a footprint that holds
several buildings, such as a townhome row, is planned.

Make each footprint one building
   :Step: ``resolve_spine`` (#1)
   :Does: Creates one building for every footprint. A footprint that
      holds several buildings, such as a townhome row, stays one
      building: the pipeline does not yet split footprints into their
      buildings (planned). The step also skips other building layers.
   :Why: One building per footprint is the first stage of the building
      table. Adding other layers would pull in the earlier CHEER
      inventory (CHEER v0) as extra buildings, duplicating ones already
      drawn.
   :Verdict: holds, reworded. The recipe's opening comment describes an
      older arrangement in which the footprint table was derived from the
      building table. Since 2026-10-26 the delivered inventory is built
      from the building table's final values. The step itself is fine.

Keep the footprint's identifier on each building
   :Step: ``adopt_source_entity_id`` (#2)
   :Does: Stores the identifier of its footprint on each building.
   :Why: The building values are later added up onto the footprints, and
      this identifier says which footprint each building belongs to.

Compute each building's center point and area
   :Step: ``derive_geometry_attributes`` (#3)
   :Does: Writes the center point and area of each building.
   :Why: The building's own measurements, used by the steps below.

Reuse the footprints' links to lots
   :Step: ``load_links`` (#4)
   :Does: Copies the links between footprints and lots made above, under
      each building's identifier.
   :Why: The overlay of footprints on lots is done once, and every later
      step reads the same links.

Attach NSI's building points to buildings
   :Step: ``link_to_reference``, NSI points (#5)
   :Does: Links each point of the National Structure Inventory (NSI) to
      the building it falls in; failing that, to the nearest building
      within 10 m; failing that, to the nearest within 100 m on the same
      lot. A second rule sets aside points that repeat a point from a
      higher-ranked NSI source at the same building.
   :Why: NSI is the main building-level evidence for occupancy and
      value.
   :Verdict: weakened. The link works (99.0% of NSI points linked in
      Pender, 98.4% in Galveston). The duplicate rule sets aside no
      point in North Carolina or Texas: it looks for two older NSI
      source labels that the 2026 NSI points there do not carry
      (Pender and Galveston, 2026-11-03).

Attach Overture's address points to buildings
   :Step: ``link_to_reference``, Overture addresses (#6)
   :Does: Links Overture's address points to the building within 10 m
      (50 m on the same lot), and counts unit addresses (such as
      apartment numbers) at one building as its dwellings.
   :Why: Address points show where people live and how many homes a
      building holds.

Building attributes (building spine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_building-spine-2026 <src/openplaces/recipes/US/_all/building/spine/2026/US_building-spine-2026.yaml>`.
This recipe adds what is known about each building before any
value is chosen: its footprint and lots, whether it is a main
building or an outbuilding, and its points' information, side by
side.

Start the building attribute table
   :Step: ``load_geospine`` (#1)
   :Does: Reads the buildings with their point and lot links back in,
      without the shapes.
   :Why: The steps that only add attributes can be rerun without
      redoing the map overlays.

Record which footprint each building belongs to
   :Step: ``link_entities_by_id`` (#2)
   :Does: Writes a table that pairs each building with its footprint.
   :Why: The delivered inventory adds the building values up onto the
      footprints through this table.
   :Verdict: holds, reworded. The explanation in the code says this
      runs with the footprint recipes; it runs with the building recipes,
      as the recipe's own comment says. The result is the same.

Tell main buildings from sheds and garages
   :Step: ``classify_footprint_priority`` (#3)
   :Does: On a lot with several buildings, marks those with an Overture
      address point "primary" and the rest "secondary". Two exceptions
      keep a building primary without an address point: a building with
      an NSI point that is much larger than the largest addressed one
      (at least twice as large, three times in North Carolina), and,
      beside such a building, other buildings of 355 m2 or more whose
      NSI point counts two or more dwellings. Where no
      building has an address point, those with an NSI point are
      primary; else the largest.
   :Why: Only primary buildings share the lot's building value from the
      roll. Without this, a detached garage would take part of the
      house's value.

Attach the lot's and the points' information to each building
   :Step: ``reconcile_attributes`` (#4)
   :Does: Copies to each building its lot's number, land-use group and
      address, and the information of its NSI and Overture points, each
      in a column named after its source.
   :Why: Every source's information is kept side by side as evidence.
      Which value the building gets is decided later, in the building
      curate.

Choose one address per building
   :Step: ``reconcile_addresses`` (#5)
   :Does: Takes the address from the lot, else from Overture. Where a
      second source agrees with it, that source fills the address's
      missing parts.
   :Why: Each building should carry one standard address.

Choose one ZIP code per building
   :Step: ``reconcile_postal_code`` (#6)
   :Does: Takes the ZIP code from Overture's address, else the census
      ZIP code area the building lies in.
   :Why: A building without an Overture address still gets a ZIP-like
      value.

Fill a missing city name from the ZIP code
   :Step: ``impute_postal_city`` (#7)
   :Does: Fills a missing city with the postal service's preferred city
      for the building's ZIP code, and marks the value as filled by
      openplaces.
   :Why: Each ZIP code has one preferred city name.

.. rubric:: Steps that do not reach the bundle

- An address key for buildings (``derive_address_id_local``, #8). No later step reads it. *Weakened.*
- Building permits matched by address and by parcel number
  (``link_by_id``, permits, #9 and #10). The permits come
  from a source whose terms do not allow redistribution; they are used
  only to check the inventory and are removed before delivery.
  *Weakened*: the recipe says the parcel-number match takes precedence
  where both match, but on current data the address match does.

Footprint attributes (footprint spine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_footprint-spine-2026 <src/openplaces/recipes/US/_all/footprint/spine/2026/US_footprint-spine-2026.yaml>`.
This recipe starts the footprint's attribute table. It reads only
the footprint geospine, so the build may run it at any time after
that recipe; it is listed here, before the curate recipes that
read it.

Start the footprint attribute table from the merged footprints
   :Step: ``load_geospine`` (#1)
   :Does: Reads the merged footprints and their lot links back in, without
      the shapes, as the starting point for the footprint's attributes.
   :Why: It keeps the footprint's identifier, ``footprint_id``, which is
      the identifier the delivered inventory and the footprint-level
      extra sources use.

Tax-roll accounts (property spine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_property-spine-2026 <src/openplaces/recipes/US/_all/property/spine/2026/US_property-spine-2026.yaml>`.
This recipe builds the property table: one row per account on the
county's tax roll (a house on its lot is one account, a condominium
unit is another). Some values reach the inventory only through this
table: in Galveston, the year of construction and the living area are
recorded only on the roll's accounts.

It reads only the roll, so the build may run it at any time before
the parcel geospine, the first recipe that reads it.

Collect every account from every tax roll
   :Step: ``union_spine_sources`` (#1)
   :Does: Puts all property records of the county into one table: the
      accounts on each county tax roll, and the condominium units that
      were found stacked on a shared lot in the parcel layers.
   :Why: Each account should be one row, whichever file it came from,
      so later steps find every account in one place.
   :Verdict: holds, reworded. The list of sources in the recipe's
      opening comment is out of date. The step itself is fine.

Add roll detail tables to their tax accounts
   :Step: ``link_by_id``, roll detail tables (#2)
   :Does: Where a county publishes extra tables beside its tax roll (for
      example the year built and living area of each account), adds
      those columns to the matching account.
   :Why: Such tables describe accounts that are already in the roll.
      They must add columns, never extra rows, or one property would be
      counted twice. Galveston publishes them; in the NC counties
      checked, the step leaves the data unchanged.

Identify each account by its assessor's account number
   :Step: ``assign_entity_ids`` (#3)
   :Does: Gives each property the account number its assessor issued,
      made unique by adding the county, and merges rows that carry the
      same number.
   :Why: The assessor's number is the identifier the county itself uses,
      so two records of the same account become one row, not two.

.. rubric:: Steps that do not reach the bundle

- Unit addresses for tax accounts (``reconcile_addresses``, #4). *Holds, reworded*: the label that records where each
  address came from names the kind of record, not the source.
- The city of each tax account (``impute_postal_city``, #5). Acts in Massachusetts only.
- An address key for tax accounts (``derive_address_id_local``,
  #6). *Weakened*: the key leaves out the town, so it
  never matches the parcels' key and cannot do its job.

Lots and their values (parcel geospine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_parcel-geospine-2026 <src/openplaces/recipes/US/_all/parcel/geospine/2026/US_parcel-geospine-2026.yaml>`.
This recipe builds the parcel table: one row per lot as the county
draws it, with the roll's values (land, building and total value, year
built, land-use code) joined on. Before it runs, every parcel file is
split when it is read in: a row stacked on the same lot shape as
others (a condominium unit, for example) becomes a property record,
not a lot, so land and value are not counted twice. This held on every
North Carolina and Texas parcel file checked.

Combine the parcel maps covering the county
   :Step: ``resolve_spine`` (#1)
   :Does: Stacks every parcel map covering the county (the county's
      own, the statewide one) in a fixed order of preference. A lot from
      a less preferred map is added only where it barely overlaps lots
      already accepted. Each lot keeps the original codes and values of
      the map it came from.
   :Why: The county should have one parcel map. The original codes are
      kept so the lot still has them if joining it to the roll fails.

Compute each lot's center point and area
   :Step: ``derive_geometry_attributes`` (#2)
   :Does: Writes each lot's center point (``lat``, ``long``) and area in
      hectares (``area_ha``).
   :Why: Done once, so later steps reuse the same numbers.

Join the roll's values to the lots by parcel number
   :Step: ``link_by_id``, standardized parcel id (#4)
   :Does: Joins every parcel file covering the county to the lots by
      parcel number, and writes the values the inventory uses: land,
      building and total value, and the land-use codes. A file whose
      numbers match at least half of the lots replaces existing values;
      a file that matches fewer only fills empty cells.
   :Why: The source closest to the county should decide each value.
   :Verdict: open. In Hyde County the statewide file matched half of
      the lots and replaced the county's own values on 3,965 of 7,698
      lots (2026-11-01). Which figures are right has not been checked;
      Hyde's published roll totals for its tax year would settle it.

Join again on parcel numbers without punctuation
   :Step: ``link_by_id``, punctuation-free id (#5)
   :Does: Repeats the join with punctuation and capitalization removed
      from the parcel numbers, filling only cells still empty.
   :Why: North Carolina's statewide parcel number and a county's own
      number often differ only in punctuation.
   :Verdict: holds, reworded. A newer safeguard also refuses a
      shortened number where it would merge two numbers the county
      kept apart, so filling only empty cells is no longer the only
      protection. The step itself is fine.

Match the statewide NC parcel layer on a second parcel number
   :Step: ``link_by_id``, PIN column (#6)
   :Does: Joins North Carolina's statewide parcel layer to the county's
      parcels a second time, using the state's parcel identification
      number (PIN) instead of its usual parcel number.
   :Why: In four eastern NC counties (Bladen, Gates, Northampton,
      Pasquotank), the state layer's usual parcel number is a tax
      account number, which never matches the county's parcel number.
      Without this second join, those parcels would get no values from
      the state layer.
   :Verdict: holds, reworded. The recipe states how many parcels this
      recovers, but without a date, so the number cannot be checked
      against a particular build.

Join each county roll and keep its latest tax year
   :Step: ``link_by_id``, county rolls (#7)
   :Does: Joins each county's tax roll (its assessor table) to its
      lots, keeps the latest tax year where the roll lists several, adds
      up the roll's values per lot, and writes the tax year next to the
      values the roll supplied.
   :Why: A value and its year should come from the same record.
   :Verdict: holds, reworded. In Galveston the roll states only a total
      value, so the tax year 2026 is written on lots whose land and
      building values are the statewide 2025 figures (2026-11-01). The
      delivered inventory takes its appraisal year from a separate
      table, so it is not misdated.

Link NSI's points to lots
   :Step: ``link_to_reference``, NSI points (#13)
   :Does: Links each point of the National Structure Inventory (NSI) to
      its lot, for the lot's land-use vote (the parcel curate below
      explains the vote).
   :Why: NSI says what kind of building stands on the lot.
   :Verdict: holds, reworded. The recipe's comment says the footprint
      table makes the same link; it is now made in the building table.
      The step itself is fine.

Link FEMA's footprints to lots
   :Step: ``link_to_reference``, FEMA footprints (#14)
   :Does: Lays FEMA's USA Structures footprints over the lots and keeps
      overlaps of at least 10 m2.
   :Why: The FEMA occupancy class covering most of a lot is one input to
      the lot's land-use vote.

Count and measure the buildings on each lot
   :Step: ``summarize_footprint_morphology`` (#15)
   :Does: Counts each lot's buildings, and its small, long and narrow
      buildings (the shape of a manufactured home), and adds up their
      areas.
   :Why: The lot's land-use vote reads these counts: many narrow
      buildings on one lot point to a manufactured home park, for
      example.

.. rubric:: Steps that do not reach the bundle

- Census areas for each parcel (``link_geographic_ids``, #3). The inventory uses each footprint's own census areas
  instead. *Holds, reworded*: the recipe says each parcel's areas come
  from its own center point; they are taken from its buildings.
- Number of tax accounts per parcel (``link_by_id``, property count,
  #8). Skipped in Texas, and no step that builds the
  inventory reads it. *Weakened.*
- Common land of townhome groups (``detect_shared_land_groups``, #9): finds the shared lot that belongs to a row of
  townhomes.
- Condominium building groups (``detect_condo_building_clusters``,
  #10). A later step on the footprints finds these groups
  again for itself, so this result is not used. *Weakened.*
- Common areas joined to their lots (``link_common_areas_by_contiguity``,
  #11). Acts in New Hanover County only.
- Parcel addresses from Overture (``link_to_reference``, Overture
  addresses, #12). *Holds, reworded*: the recipe's
  explanation names the wrong recipe.

Lot-level evidence (parcel spine)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_parcel-spine-2026 <src/openplaces/recipes/US/_all/parcel/spine/2026/US_parcel-spine-2026.yaml>`.
This recipe adds to each lot the evidence that the lot's land-use vote
reads: its land-use description, and what NSI and FEMA say stands on
it.

Start the lot attribute table
   :Step: ``load_geospine`` (#1)
   :Does: Reads the lots and their links back in, without the shapes.
   :Why: The steps that only add attributes can be rerun without
      redoing the map overlays.

Combine the lot's land-use group and subgroup
   :Step: ``derive_use_classes`` (#9)
   :Does: Joins each lot's land-use group and subgroup into one value,
      in two versions: one that keeps the county's raw code where no
      translation into words exists, and one with translated words only.
   :Why: A raw code still groups lots of the same kind. Only words can
      be matched against keywords such as "mobile home", so the keyword
      rules read the second version.
   :Verdict: holds, reworded. The recipe's comment gives out-of-date
      counts, and a blank land-use group in North Carolina's statewide
      file is recorded as coming from that file. The values are
      unaffected.

Record the NSI class with the most value on the lot
   :Step: ``reconcile_attributes``, NSI (#10)
   :Does: Writes, for each lot, the NSI occupancy group whose points on
      the lot add up to the largest building value.
   :Why: One input to the lot's land-use vote.

Record the FEMA class covering most of the lot
   :Step: ``reconcile_attributes``, FEMA (#11)
   :Does: Writes, for each lot, the FEMA occupancy group and height that
      cover most of its area.
   :Why: One input to the lot's land-use vote.

Link each roll account to its lot
   :Step: ``link_entities_by_id`` (#12)
   :Does: Writes a table that pairs each property account with the lot
      it lies on, by matching numbers, plus the condominium units found
      stacked on a lot when the parcel file was read in.
   :Why: The roll describes accounts, the map draws lots; this table
      says which account belongs to which lot.
   :Verdict: holds, reworded. The recipe's comment says it matches
      numbers once; the code tries up to three ways of matching, and
      labels numbers shared by several accounts. The step itself is
      fine.

.. rubric:: Steps that do not reach the bundle

- The assessor's address as written (``rename_columns``, #2).
- Address evidence from Overture (``reconcile_attributes``, Overture,
  #3).
- The address of a parcel's main building
  (``attribute_dwelling_address``, #4).
- One address per parcel (``reconcile_addresses``, #5).
- The parcel's ZIP code (``reconcile_postal_code``, #6).
- The parcel's city (``impute_postal_city``, #7).
- An address key for matching sales to parcels
  (``derive_address_id_local``, #8).
- Sale counts and prices (``link_by_id``, transactions, #13). Only New Hanover and Pitt counties have sales records here.
- Estimated number of units per parcel (``estimate_property_counts``,
  #14).

Construction details of tax-roll accounts (property curate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_property-openplaces-2026 <src/openplaces/recipes/US/_all/property/openplaces/2026/US_property-openplaces-2026.yaml>`.
This recipe translates the roll's foundation and construction
descriptions into classes for each tax-roll account.

.. rubric:: Steps that do not reach the bundle

- Foundation and construction descriptions on tax accounts (all five
  steps of
  :gh-file:`US_property-openplaces-2026 <src/openplaces/recipes/US/_all/property/openplaces/2026/US_property-openplaces-2026.yaml>`).
  They translate the roll's foundation and construction descriptions
  into classes, but wrote no values in the CHEER counties checked, and
  the building step does not read them.

Classifying each lot (parcel curate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_parcel-openplaces-2026 <src/openplaces/recipes/US/_all/parcel/openplaces/2026/US_parcel-openplaces-2026.yaml>`.
This recipe chooses one value per attribute for each lot (parcel, a
piece of land as the county draws it): its land-use class and its
land, improvement and total values. The building steps further down
read these choices.

Create empty NSI and FEMA columns where they are missing
   :Step: ``declare_columns``, NSI and FEMA (#1)
   :Does: Adds empty columns for the building class that NSI (the
      National Structure Inventory) and FEMA's USA Structures give a
      parcel, where no such column exists yet.
   :Why: Later steps expect these columns in every county. In New
      England, NSI and FEMA cannot be matched to towns, so the columns
      would otherwise be missing and the later steps would fail.

Add up the roll's accounts on each lot
   :Step: ``aggregate_from_entities``, tax-roll accounts (#2)
   :Does: The county's tax roll (assessor records) lists accounts, and
      one lot can hold several (the units of a condominium, for
      example). This step adds them up onto their lot: areas and values
      are summed, and the year of construction is the earliest. Where a
      condominium lot has no improvement value of its own, it takes the
      sum of its units' values.
   :Why: Account details are kept on the account table until this
      point, so that a lot only gets what its accounts say together.
      Without the sum, a condominium lot would show no building value.

Make sure every lot has the unit-count columns
   :Step: ``declare_columns``, unit-type counts (#4)
   :Does: Adds empty columns for the number of manufactured homes,
      travel trailers, park models and buildings on a lot, where the
      county's roll does not state them.
   :Why: The building steps read these counts in every county. An empty
      column means "not stated"; a missing column would stop the run.

Create empty land-use description columns where they are missing
   :Step: ``declare_columns``, use labels (#5)
   :Does: Adds empty columns for the county's land-use description where
      none exists.
   :Why: Later steps expect these columns. Some Virginia counties and
      cities publish no land-use code at all.

Treat negative property values as unknown
   :Step: ``null_out_of_range`` (#8)
   :Does: Empties any land, improvement or total value below zero.
   :Why: North Carolina's statewide parcel layer computes the
      improvement value as total minus land. On farmland taxed at its
      use value rather than its market value, that difference can be
      negative. A negative value is not a real value, and the later
      step that splits a parcel's value among its buildings assumes no
      value is negative. Measured 2026-11-03 in Duplin County: 6,845 of
      43,424 parcels.

Drop placeholder parcels for water and roads
   :Step: ``exclude_by_value`` (#9)
   :Does: Removes rows whose parcel number is a placeholder for a water
      body ("WATER") or a road right-of-way ("ROW").
   :Why: These rows are not pieces of land that anyone owns or pays tax
      on. Such placeholders were found in nine counties.
   :Verdict: holds, reworded. The recipe says these rows carry no
      land-use code and no value. Some now carry a land-use code, and
      one row in Carteret County carries a value, which the step
      therefore drops.

Drop lots smaller than 1 m2
   :Step: ``filter_numeric_at_least`` (#10)
   :Does: Removes lots with an area under 1 square meter.
   :Why: Such slivers are drawing errors. Their value per area would be
      meaningless and would distort any per-area figure.

Give every lot a building class in NSI's terms
   :Step: ``impute_from_group_statistic`` (#11)
   :Does: For each land-use description used in the county, finds the
      building class that the National Structure Inventory (NSI, a
      federal point layer of buildings) gives most often on lots with
      that description. Every lot with the description gets that class,
      marked as filled in by openplaces.
   :Why: Gives every lot a building class in NSI's vocabulary, including
      lots where no NSI point falls. The maintainer decided (2026-10-17)
      to keep this everywhere except Massachusetts and Harris County,
      TX. The class is a fill taken from NSI, not a statement of the
      roll.

Prepare the evidence for the land-use vote
   :Step: ``derive_indicators`` (#12)
   :Does: Computes the inputs of the land-use vote below: the class
      that keywords in the land-use description point to, the class of
      the building style, how large the lot's building footprints are
      compared with other lots of the same use, and how much of the lot
      the footprints cover.
   :Why: Each input is computed once. Every threshold then sits in one
      place, the vote, and can be checked there.
   :Verdict: holds, reworded. One of these inputs is not used by any
      later step, and the explanation in the recipe lists fewer
      excluded words than the keyword table holds. The step itself is
      fine.

Decide each lot's land-use subtype by vote
   :Step: ``resolve_by_vote``, land use (#13)
   :Does: Runs a vote: several sources each propose a class with a
      weight, and the class with the highest total wins. Here the
      sources are keywords in the land-use description, the NSI and
      FEMA classes (FEMA's USA Structures, a federal building layer),
      the shape of the building footprints and the split of the lot's
      value between land and buildings. There are 13 possible outcomes,
      such as vacant or manufactured home park.
   :Why: One vote assigns all the land-use subtypes, so they cannot
      contradict each other.
   :Verdict: holds, reworded. Two by-products (a vacancy score and a
      flag for review) are not used by any later step.

Give unclaimed lots a default class
   :Step: ``reconcile_land_use`` (#14)
   :Does: A lot that the vote left without a class gets one from a
      second, equal-weight vote of three sources: the NSI class filled
      in above, the class of the NSI points on the lot and the FEMA
      class.
   :Why: Every lot needs a class, also where no keyword rule applies.
   :Verdict: holds, reworded. The explanation in the code calls the
      vote by an old name. Because the first of the three sources is
      itself filled from NSI, NSI can cast two of the three votes.

Estimate missing land values on built lots
   :Step: ``impute_land_value`` (#17)
   :Does: Where a lot with a building has a land value of 0 or none,
      takes the total value minus the improvement value. Where that is
      not possible, it estimates the land value from the value per area
      of nearby lots on the same street and in the same city. Estimated
      values are marked as filled in by openplaces.
   :Why: Without this, the building would carry the land's value into
      the inventory. The method was chosen by testing it on lots whose
      values were known but held back (2026-09-22).
   :Verdict: holds, reworded. The planning document the recipe cites
      has moved.

.. rubric:: County patches

Northampton County, NC: improvement value from total minus land
   :Recipe: :gh-file:`US-NC-NOR_parcel-openplaces-2026 <src/openplaces/recipes/US/NC/NOR/_all/parcel/openplaces/2026/US-NC-NOR_parcel-openplaces-2026.yaml>`
   :Does: Sets each parcel's improvement value (the value of its
      buildings) to its total value minus its land value.
   :Why: Northampton's own parcel layer gives only one combined value,
      and the statewide layer gives an improvement value of 0 on every
      parcel. Without this rule, no Northampton building would have a
      value from the roll.

Harris County, TX: commercial classes from the county's building codes
   :Recipe: :gh-file:`US-TX-HAR_parcel-openplaces-2026 <src/openplaces/recipes/US/TX/HAR/_all/parcel/openplaces/2026/US-TX-HAR_parcel-openplaces-2026.yaml>`
   :Does: Replaces the parcel's broad land-use group with one read from
      Harris County's own building type code.
   :Why: Texas's state property class F1 puts offices, hospitals, shops
      and warehouses into one group. Harris's own code tells them apart.
   :Verdict: open. Not checked on Harris County data.

.. rubric:: Steps that do not reach the bundle

- Those foundation and construction classes carried to the parcel
  (``aggregate_from_entities``, foundation and construction, #3).
- Land-market inputs for valuation research (``merge_enrichments``,
  #6). *Holds, reworded*: one of the columns it is set to
  copy never exists, so that line has no effect.
- Each lot's share of shared common land, as an area
  (``apportion_common_land``, #7).
- A story count per parcel from building heights
  (``derive_stories_from_height``, #15). The building
  step works out its own story counts. *Weakened.*
- Parcel values per hectare (``derive_metrics``, #16).
  *Holds, reworded*: the recipe says the step also computes the
  parcel's area; it does not.
- Storing some parcel columns in a compact format
  (``cast_categoricals``, #18). The file is written with
  those columns as plain text anyway, so the step has no effect on the
  file. *Weakened.*
- The order of the parcel table's columns (``order_columns``, #19). *Holds, reworded*: the recipe's explanation names a step
  that has since been renamed.

Deciding each building's attributes (building curate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_building-openplaces-2026 <src/openplaces/recipes/US/_all/building/openplaces/2026/US_building-openplaces-2026.yaml>`.
This recipe chooses one value per attribute for each building: its
values, year built, stories, dwellings and occupancy class. A building
is one structure; on a lot with several, the main buildings are called
primary and the garages, sheds and barns secondary.

Settings shared by the occupancy votes
   :Step: settings at the top of the recipe
   :Does: Sets how each source's building classes translate into the
      inventory's classes, which four sources vote on the occupancy
      class (NSI, FEMA, the lot's class and Overture's address points,
      the last at half weight), and the size and shape thresholds for
      manufactured homes that several later steps reuse.
   :Why: One set of rules serves every county; regional differences
      come from the data, not from separate rules.
   :Verdict: holds, reworded. The recipe calls one input a FEMA class
      of the building. It is the most common FEMA class on the lot,
      the same for every building on it.

Create empty evidence columns where they are missing
   :Step: ``declare_columns`` (#1)
   :Does: Adds 23 empty columns for evidence from NSI, FEMA and Overture
      (Overture's address points) where they do not exist yet.
   :Why: Later steps expect these columns. In New England, NSI, FEMA and
      Overture are not matched to buildings, so the columns would be
      missing. In North Carolina and Texas, the columns already exist.

Copy the lot's decisions onto its buildings
   :Step: ``link_curated_entity`` (#2)
   :Does: Copies 13 of the lot's attributes onto each building on it
      (the building's main lot, where it overlaps several): the land-use
      class, the land-use descriptions, the NSI class filled in for the
      lot, the flag for manufactured home communities and the roll's
      unit counts.
   :Why: The building votes read what was decided for the lot rather
      than deciding it again.
   :Verdict: holds, reworded. The recipe names the wrong step as the
      reader of the community flag.

Copy optional roll details onto buildings
   :Step: ``link_curated_entity``, optional columns (#3)
   :Does: Copies the building style, story count and living area where
      the roll states them; elsewhere the columns stay empty.
   :Why: Only some counties publish these details.

Share the lot's values among its buildings
   :Step: ``apportion_curated_values`` (#4)
   :Does: Splits the lot's improvement value (the assessed value of its
      buildings) among its primary buildings, in proportion to each
      building's area on the lot weighted by its number of floors.
      Secondary buildings get none. The land and total values go whole
      to the main building. The building's year built is the average
      of the roll's year built on the lots it stands on (usually one).
   :Why: A building's value is its share of the lot's value, so the
      value is never counted twice. The floor weighting keeps a
      one-story annex from taking a share meant for a tall building.
   :Verdict: holds, reworded. The recipe says the dwelling count is
      split with the same weights as the value; the code differs in two
      details.

Keep the lot's undivided values beside the shares
   :Step: ``link_curated_entity``, whole values (#5)
   :Does: Copies the lot's full improvement and land values onto each
      building, next to the building's share.
   :Why: The manufactured-home test compares a building's value with
      the lot's total.

List every lot a building overlaps
   :Step: ``collect_link_ids`` (#6)
   :Does: Writes the ids of all lots a building overlaps into one column.
   :Why: A building that straddles a lot line can then be traced to
      every lot it touches.

Gather the remaining evidence
   :Step: ``merge_enrichments`` (#7)
   :Does: Adds the evidence from other sources, each in its own column:
      roof shape, stories, construction, garage and foundation from the
      earlier CHEER inventory (CHEER v0) and from image classification
      (BRAILS), year built from North Carolina's statewide building
      layer (NC DPS), height and floors from Overture, and the height
      FEMA measured by lidar.
   :Why: All evidence is in place before any value is chosen, so every
      choice can be checked against every source.
   :Verdict: holds, reworded. A comment points to an entry "above" that
      is below it.

Write the county's federal code
   :Step: ``derive_admin_attribute`` (#8)
   :Does: Writes ``county_fips``, the county's five-digit federal (FIPS)
      code.
   :Why: The federal code stays stable even if openplaces renames its
      own area codes.

Choose the roll or NSI for values, area by area
   :Step: ``select_value_source_by_admin_unit`` (#9)
   :Does: In each municipality, keeps the roll's building values where
      at least half of the main buildings have one. Elsewhere it uses
      NSI's value for every building in that area.
   :Why: Whether a roll is usable depends on the local roll. Taking the
      roll for one building and NSI for its neighbor would mix two
      different measures of value.

Ignore address points on vacant lots
   :Step: ``suppress_where``, address points on vacant lots (#10)
   :Does: Removes the dwelling count from Overture's address points on
      lots classed as vacant.
   :Why: An address point on an empty lot does not make it a home; the
      lot's land-use class is more reliable there.

Ignore tiny shares of a roll's dwelling count
   :Step: ``suppress_where``, roll dwellings (#11)
   :Does: Where a roll's dwelling count is divided among a parcel's
      buildings, removes any building's share of half a dwelling or
      less.
   :Why: Half a dwelling is not a dwelling. No NC or TX roll gives a
      dwelling count, so this acts only in Massachusetts.

Choose the number of dwellings, and keep NSI's costs
   :Step: ``reconcile_values``, dwellings and NSI costs (#12)
   :Does: Takes ``n_dwellings`` (the number of housing units) from the
      roll, else from Overture's address points, else from NSI. Copies
      NSI's replacement cost and depreciated replacement cost into
      columns of their own.
   :Why: The roll's unit count is more reliable than counting address
      points. NSI's costs measure what rebuilding would cost, not the
      market value, so they are kept apart from it.

Turn Overture's address count into a class
   :Step: ``resolve_by_vote``, Overture class (#13)
   :Does: Classes a building Multi-Family where Overture has two or
      more address points in it, and Single-Family where it has one.
   :Why: This becomes Overture's half-weight vote in the occupancy vote.

Record no address point as zero
   :Step: ``fill_missing_numeric`` (#14)
   :Does: Writes 0 where Overture has no address point in a building.
   :Why: Later steps compare counts; a gap would be read as unknown
      rather than as none.

Compute area and value per area
   :Step: ``derive_metrics`` (#15)
   :Does: Writes ``area_m2`` (the footprint's area in square meters) and
      the value per square meter.
   :Why: Later checks compare buildings by size and by value per area.

Remove values from condominium slivers
   :Step: ``suppress_where``, condominium slivers (#16)
   :Does: Removes the value of a condominium footprint smaller than 20 m2.
   :Why: Condominium units are separate parcels stacked on one
      building. Where a tiny footprint sits on those unit parcels, it is
      not the building the units' values describe, and it should not
      receive them.
   :Verdict: holds, reworded. The count of such footprints in the recipe
      is out of date: on 2026-11-01 there were 193 across North Carolina
      and Texas.

Prepare the evidence for the building votes
   :Step: ``derive_indicators`` (#17)
   :Does: Computes the inputs of the votes below: shape measures of the
      footprint (length to width, width, length of wall shared with a
      neighbor), each source's class in the inventory's terms, and the
      classes that keywords in the land-use description, the building
      style and the section count point to.
   :Why: This step only computes values; every threshold sits in a
      vote, where it can be checked.
   :Verdict: holds, reworded. Some counts in the recipe's explanation
      are undated, and a planning document it cites has moved.

Fill a missing dwelling count from the building class
   :Step: ``impute_n_dwellings`` (#18)
   :Does: Where a building has a class (single-family, duplex, and so
      on) but no dwelling count, fills the count from a published table
      of typical counts per class.
   :Why: Meant to fill dwelling counts left empty after all sources were
      combined.
   :Verdict: weakened. It filled no building in North Carolina or Texas
      (2026-11-01): wherever NSI gives a class, it also gives a count.

Vote on the base occupancy class
   :Step: ``impute_occupancy_type`` (#19)
   :Does: A weighted vote of NSI, FEMA, the lot's class and Overture
      chooses a first occupancy class; the class with the most weight
      wins, and a tie goes to NSI.
   :Why: Sources that agree can outvote a single one that disagrees.
   :Verdict: holds, reworded. The lot's class is itself filled from NSI
      (on 99.1% of Pender County lots, 2026-11-01), so NSI in effect
      votes twice.

Correct the class where the lot's land-use description is unambiguous
   :Step: ``resolve_occupancy`` (#20)
   :Does: Where the county's land-use description for the lot contains
      a reviewed keyword (for example "double wide" for a manufactured
      home or "townhouse" for multi-family) and the vote chose a
      different class, the keyword wins. The step also records where
      the sources disagreed, for review.
   :Why: Some descriptions are certain enough to overrule the vote.
   :Verdict: holds, reworded. The step also reads land-use codes the
      county never translated into words, which the recipe says it
      should not. In six counties checked this changed no building's
      class (2026-11-03).

Record whether the roll calls the building single- or multi-family
   :Step: ``derive_indicators``, roll statement (#21)
   :Does: Copies the roll's class for the building into a separate
      column, but only where the roll says Single-Family or
      Multi-Family.
   :Why: One safeguard in the manufactured-home vote asks "does the roll
      say this is a site-built house?". Keeping that answer in its own
      column lets a county whose roll never states a building type
      (Halifax, NC) switch off that one safeguard, while the rest of the
      vote still uses the roll.

Choose the year built
   :Step: ``reconcile_values``, year built (#22)
   :Does: Takes ``year_built`` (the year of construction) from the
      roll, else from NC DPS, else the median year of NSI's buildings in
      the census block. Years before 1500 or after 2035 are ignored.
   :Why: The assessor's record comes first, a record of the building
      itself next, and an average for the block last.

Estimate stories from measured height
   :Step: ``derive_stories_from_height`` (#23)
   :Does: Divides the height FEMA measured by lidar by 3.5 m to get a
      story count.
   :Why: Gives a story count where no record states one. A smaller
      height per story (3.05 m) counted pitched roofs as a second story.
   :Verdict: holds, reworded. The code's explanation still gives 3.05 m
      as the default.

Choose stories, roof shape, foundation and construction type
   :Step: ``reconcile_values``, stories, roof, foundation, construction (#24)
   :Does: Takes each attribute from the first source in a fixed list
      that has a value. Stories: image classification (BRAILS), NSI,
      FEMA's lidar height, CHEER v0. Roof shape: BRAILS, CHEER v0.
      Foundation: NSI, CHEER v0. Construction type: CHEER v0, NSI.
   :Why: The lists follow how reliable each source proved in tests.
   :Verdict: holds, reworded. Several numbers in the recipe's
      explanation are undated.

Take story counts from the roll where the roll is consistent
   :Step: ``adopt_stories_by_floor_area_fit`` (#25)
   :Does: County by county, fills a building's missing number of stories
      from the roll, but only where the roll's story counts agree with
      its own floor areas on at least 70% of the buildings tested.
   :Why: Counties define a story differently. Checking the counts
      against the roll's own floor areas keeps a county's counts only
      where they are consistent. Only 9 of 139 NC and TX
      counties have a roll story count (2,721 buildings, 2026-11-01).

Estimate how likely each building is a manufactured home
   :Step: ``classify_manufactured_homes`` (#26)
   :Does: For each county, fits a model that estimates the probability
      that a building is a manufactured home from its footprint's shape
      and its neighbors' shapes. The model learns from buildings whose
      land-use description says manufactured home. Those buildings get
      a probability of 1.
   :Why: The vote below weighs a probability; it does not take a class
      from this step as given.
   :Verdict: holds, reworded. The recipe says the probability reads no
      text, but buildings with the keyword are set to 1.

Decide one dwelling or several
   :Step: ``resolve_by_vote``, one dwelling or several (#27)
   :Does: A vote of Overture's address counts, keywords in the land-use
      description, NSI and FEMA decides whether a building holds one
      dwelling or several.
   :Why: How many homes a building holds is a different question from
      what kind of building it is, and different sources are good at
      each.
   :Verdict: holds, reworded. The analysis script the recipe cites is
      not in the repository. The scores against the field survey of
      2026-10-25 (eastern North Carolina) match what the recipe says.

Decide the occupancy class
   :Step: ``resolve_by_vote``, occupancy class (#28)
   :Does: Scores RV Dwelling, Manufactured Home, Multi-Family,
      Single-Family and Secondary from the footprint's shape, keywords,
      NSI, FEMA, the building's share of the lot's value and signs of a
      manufactured home park. A building no class claims keeps its base
      class.
   :Why: Decides the type of building, within the limits set by the
      one-or-several dwellings decision.
   :Verdict: holds, reworded. Two of the park signals come from the same
      decision about the lot, which the recipe does not say; planning
      documents it cites have moved.

Count the manufactured homes on each lot
   :Step: ``derive_group_count`` (#29)
   :Does: Counts the main manufactured homes on each lot.
   :Why: The second pass below needs the count.

Rank manufactured homes on a lot by size
   :Step: ``derive_group_rank`` (#30)
   :Does: Numbers the manufactured homes on each parcel from largest to
      smallest.
   :Why: Where a roll says a lot holds, say, two mobile homes, the two
      largest footprints keep that class and the smaller ones do not. This acts only where a roll gives such counts: a few Texas
      appraisal districts, and Harnett and Wake counties in NC.
   :Verdict: holds, reworded. Two homes of equal size are ordered by the
      building's identifier, not the footprint's as the recipe says. The
      two identifiers are the same today, so the order is the same.

Reclass small secondary manufactured homes as sheds
   :Step: ``resolve_by_vote``, second pass (#31)
   :Does: Turns a secondary building classed Manufactured Home into
      Secondary where it is smaller than 40 m2.
   :Why: Small secondary footprints that took the class from the lot
      were almost all sheds.

Split multi-family buildings by height
   :Step: ``resolve_by_vote``, height bands (#32)
   :Does: Splits Multi-Family into Low-Rise (up to 3 stories), Mid-Rise
      (4 to 7) and High-Rise (8 or more).
   :Why: These are the height bands of HAZUS, FEMA's loss model.

Decide the number of sections of a manufactured home
   :Step: ``resolve_by_vote``, section count (#33)
   :Does: Decides whether a manufactured home has 1, 2 or 3 sections,
      from a keyword in the roll ("double wide", weight 3) and the
      footprint's width and length-to-width ratio (weight 1 each).
   :Why: Most manufactured homes have more than one section, and each
      section has nearly the same width.
   :Verdict: holds, reworded. The recipe says 45 delivered counties;
      there are now 44, and its numbers are undated.

Flag manufactured home communities
   :Step: ``flag_manufactured_home_communities`` (#34)
   :Does: Flags lots with four or more manufactured homes, after the
      votes above.
   :Why: The lot steps run before the building classes are final, so
      they cannot count them.
   :Verdict: holds, reworded. The code's explanation names an imagery
      input that does not exist.

Flag values too high for the building's height
   :Step: ``flag_unconfirmed_height`` (#35)
   :Does: Flags a building whose value per area is too high for the
      number of floors measured (``height_unconfirmed``). The value
      itself is not changed.
   :Why: The project's web map draws each building's height from its
      value per area. A building drawn tall without a measured height to
      back it up is shown see-through.
   :Verdict: open. The floor counts come from Overture's heights, whose
      origin was checked only in Boston. Checking their sources in North
      Carolina and Texas would settle it. The flag affects few buildings
      (Pender 11, Galveston 44, 2026-11-01).

Record why a building's value is 0
   :Step: ``derive_structure_value_zero_reason`` (#37)
   :Does: Where ``structure_value`` (the building's market value) is 0,
      writes the first reason that applies: a secondary building,
      personal property (such as a manufactured home taxed apart from
      the land), exempt from tax, valued with another building, not on
      the roll, or no value stated.
   :Why: A 0 is a statement, not a gap, and the gap-filling models treat
      each reason differently (maintainer, 2026-10-31 and 2026-11-02).

Store classes as categories
   :Step: ``cast_categoricals`` (#38)
   :Does: Stores the class columns as categories.
   :Why: Smaller files, faster to load.

Store counts and years as whole numbers
   :Step: ``cast_integers`` (#39)
   :Does: Rounds ``year_built``, ``n_stories`` (stories), ``n_dwellings``
      and ``n_sections`` (manufactured-home sections) and stores them as
      whole numbers.
   :Why: Some sources give fractions: NSI's story counts are averages
      and its dwelling counts midpoints of a range.

Drop working columns and order the rest
   :Step: ``order_columns`` (#40)
   :Does: Drops 34 working columns, among them building permit evidence
      from a source whose licence does not allow publishing it, and puts
      the rest in a fixed order.
   :Why: Restricted evidence is never published, and working columns
      are of no use on their own.
   :Verdict: holds, reworded. The recipe names diagnostic scripts that
      are not in the repository.

.. rubric:: County patches

Two county patches change this recipe in NC:
:gh-file:`US-NC-HAL_building-openplaces-2026 <src/openplaces/recipes/US/NC/HAL/_all/building/openplaces/2026/US-NC-HAL_building-openplaces-2026.yaml>`
and
:gh-file:`US-NC-PIT_building-openplaces-2026 <src/openplaces/recipes/US/NC/PIT/_all/building/openplaces/2026/US-NC-PIT_building-openplaces-2026.yaml>`.
They were not reviewed step by step.

.. rubric:: Steps that do not reach the bundle

- A report of lots whose roll value is not shared among their large
  buildings (``report_lot_value_sharing``, #36). It
  changes no value. *Holds, reworded*: a parcel value that openplaces
  estimated is counted as if the roll had stated it.

Footprints and the delivered bundle (footprint curate)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recipe: :gh-file:`US_footprint-cheer-2026 <src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml>`.
Its steps carry the building table onto the footprints; its
``share:`` and ``impute:`` settings drive the delivery and the
imputation that follow.

Carrying buildings onto footprints
""""""""""""""""""""""""""""""""""

The earlier steps chose one value per attribute for each building. This
recipe turns that building table into the table CHEER receives: one row
per building footprint.

Put each footprint's buildings on one row
   :Step: ``aggregate_from_entities`` (#1)
   :Does: Collects the buildings that stand under each footprint onto one
      row. Classes are listed together, counts are combined, and
      ``year_built`` (the year of construction) is the earliest of
      them. The footprint keeps its own shape, area and census area
      codes.
   :Why: The inventory has one row per footprint, and the values were
      decided for buildings. Where one footprint holds several
      buildings (a row of townhomes, for example), their counts and
      values are added up.
   :Verdict: holds, reworded. The recipe's explanation of why the
      building values replace the footprint's own is out of date since
      2026-10-28. The step itself is fine.

Restore category column types
   :Step: ``cast_categoricals`` (#2)
   :Does: Stores the class columns again as categories (a compact
      format for columns with few distinct values).
   :Why: Combining the buildings creates new columns, which lose that
      format.

Restore whole-number column types
   :Step: ``cast_integers`` (#3)
   :Does: Stores counts such as stories and dwellings as whole numbers
      again.
   :Why: Combining the buildings returns them as decimal numbers.

Remove restricted columns and order the rest
   :Step: ``order_columns`` (#4)
   :Does: Removes the columns that the building step was told never to
      pass on, then puts the columns in their standard order.
   :Why: Building-permit evidence comes from a source whose terms do not
      allow publication, and it must never ship.
   :Verdict: weakened. Today the list has no column left to remove, because the
      building step already removed every column on it. It stays as a
      second safeguard.

.. rubric:: Steps that do not reach the bundle

- The accuracy check against the field survey (the ``validation:``
  settings of the footprint recipe). *Weakened*: it has not yet run under
  this recipe's current name, and where Overture has no address point
  on a building, it counts that as Overture saying the building is
  single-family. See :ref:`cheer_footprints_validation`.

Pooling and withholding (delivery)
""""""""""""""""""""""""""""""""""

The delivery pools the curated counties of a region into one bundle,
withholds what may not be shared, fills the gaps (the imputation, next
section) and writes the files (the section after it). Its settings are
in the ``share:`` block of the same recipe, and the function
``export_delivery`` runs it. The stages are listed in the order they
run; each is numbered D1 to D16 so the reviewer notes can refer to it.

Know which counties make up each region
   :Step: delivery stage D1, regions and paths
   :Does: Reads the list of counties in each delivered region from one
      shared table
      (:gh-file:`admin-regions-2026.csv <src/openplaces/recipes/_all/admin/regions/2026/admin-regions-2026.csv>`).
      For regions that also get a team bundle, it adds that second
      bundle. Each bundle is written to a folder named after its region.
   :Why: Delivery, maps and analysis all read the same county list, so
      the lists cannot drift apart.
   :Verdict: holds, reworded. The delivery documentation
      (:gh-file:`src/openplaces/io/delivery/README.md` and
      :gh-file:`src/openplaces/io/delivery/__init__.py`) says eastern
      North Carolina has 45 counties; the table, and the bundle, have
      44.

Refuse to ship out-of-date county files
   :Step: delivery stage D2, freshness check
   :Does: Stops the export if any county's curated file is older than
      one of the files it was built from.
   :Why: A county file built before its inputs changed carries old
      values. This once reached a user downstream.

Find every source with sharing conditions
   :Step: delivery stage D3, bundle sources
   :Does: Lists, for every county in the region, each input whose terms
      restrict sharing or require that derived data be shared under the
      same licence ("share-alike").
   :Why: A county's tax roll enters only that county's tables, so the
      search has to go county by county.

Never ship owner names or other personal columns
   :Step: delivery stage D4, personal-column refusal
   :Does: Refuses a recipe that asks to ship an owner, buyer or seller
      column, and never reads such a column into the bundle.
   :Why: Personal information never leaves the computer it was
      processed on.

Read only the needed columns, and check the row keys
   :Step: delivery stage D5, first read and index guard
   :Does: Reads only the columns the bundle ships and those the gap
      models need. Stops if the rows are keyed by row number instead of
      by footprint.
   :Why: Reading fewer columns keeps memory use low. Row numbers restart
      at 0 in every county, so pooled counties would share keys; this
      once lost rows in another delivery.

Cut footprints that cross a county line
   :Step: ``slice_at_unit_lines`` (delivery stage D6)
   :Does: Where two neighboring counties both deliver the same footprint,
      keeps one copy, cuts it at the county line, and gives each piece
      the values of its own county.
   :Why: Each county is processed on its own, so a building on the line
      would otherwise ship twice, with two assessors' values.

Remove repeated footprints
   :Step: delivery stage D7, duplicate ids
   :Does: Where the same footprint identifier appears twice, keeps the
      copy with the most values.
   :Why: A safety net for cross-county footprints the previous stage did
      not pair up.

Withhold what a restricted source supplied
   :Step: delivery stage D8, withholding rules 1 to 4
   :Does: Drops rows whose footprint came from a source that may not be
      shared, and empties each value such a source supplied. The
      matching ``_source`` column (which names where each value came
      from) then reads ``withheld``.
   :Why: Only what a restricted source supplied is removed, so one
      county's terms do not hold back the rest of the bundle.

Withhold values derived from restricted values
   :Step: delivery stage D9, withholding of derived values
   :Does: In a county where a restricted source supplied values, also
      empties the values openplaces calculated from them.
   :Why: A value calculated from a restricted value would reveal it.
      The maintainer decided this on 2026-10-23 and 2026-10-24.

Withhold a per-area value with its total
   :Step: delivery stage D10, withholding of per-area values
   :Does: Empties a value per square meter wherever the value itself is
      withheld.
   :Why: Multiplied by the footprint's area, the per-area value gives the
      withheld value back.

Withhold columns whose inputs were withheld
   :Step: delivery stage D11, withholding by declared inputs
   :Does: Some columns are calculated from the tax roll but do not say
      where their values came from. The recipe lists the inputs of each
      such column, and this stage empties the column wherever one of
      those inputs was withheld.
   :Why: Without that list, the delivery cannot tell that these columns
      hold restricted values.
   :Verdict: weakened. One listed input, the roll's building group, is
      removed earlier in the pipeline, so this stage never acts on the
      column that flags where sources disagree about a building's class
      (``occupancy_type_conflict``). Checked on the v0.7 eastern North
      Carolina bundle (2026-10-27). Fixed on main on 2026-11-03 (commit
      7888931e): the stage now reads inputs that reach the bundle, so
      the v0.8 export withholds these columns.

Final check for restricted values
   :Step: delivery stage D12, restricted-cell backstop
   :Does: Stops before writing if any value from a restricted source is
      still present.
   :Why: A last check on the values themselves, whatever the earlier
      stages did.

Keep incompatible share-alike licences apart
   :Step: delivery stage D13, share-alike check
   :Does: From the public bundle, withholds values under a share-alike
      licence that differs from the footprints' licence, and refuses to
      write a public bundle that still mixes two such licences.
   :Why: The footprints are under the Open Database License (ODbL), some
      county values under Creative Commons CC BY-SA. Both require
      derived data to carry their own licence, so they cannot be
      published together. A team bundle is never published, so it may
      keep both.
   :Verdict: holds, reworded. The delivery documentation
      (:gh-file:`src/openplaces/io/delivery/README.md`) lists fewer
      licence versions than the code checks.

Filling the gaps (imputation)
"""""""""""""""""""""""""""""

Where no source has a value, models trained on the recorded values fill
it in ("imputation"). This runs inside the delivery, after withholding,
for both CHEER regions; its settings are in the ``impute:`` block of the
same recipe. Stages are numbered I1 to I26.

Fill only after withholding
   :Step: imputation stage I1, imputation after withholding
   :Does: Fills missing values on the pooled region after every
      restricted value has been withheld. A filled cell's ``_source``
      reads ``<version>+open+imputed``, for example
      ``building-imputation-v0.8+open+imputed``.
   :Why: A model that never sees a restricted value cannot pass one on
      into a filled cell.

Choose what the models may learn from
   :Step: imputation stage I2, model inputs
   :Does: The models learn from the bundle's own evidence and from
      census data (American Community Survey (ACS) home values and
      income levels, population density, distance to towns). They do
      not use regional land-market values.
   :Why: The land-market values come from a source held back pending a
      licence review, and restricted evidence never reaches a shipped
      value.
   :Verdict: holds, reworded. A comment in the code gives a different
      reason (that the models do as well without these values), which
      the study it cites does not support. The licence is the reason.

Discard impossible story counts
   :Step: imputation stage I3, validity repairs
   :Does: Empties story counts above 60, above 10 on building classes
      that are low-rise by definition (such as single-family homes and
      manufactured homes), or below 1, and fills them like any other gap.
   :Why: Counts like these are errors, not observations.

Learn years and values from tax rolls only
   :Step: imputation stage I4, labels from rolls only
   :Does: The models learn years of construction and structure values
      from tax rolls only, never from NSI (the National Structure
      Inventory).
   :Why: NSI's years are estimates for an area, not records of the
      building, and its values measure a different thing (replacement
      cost, not market value).

Treat each kind of zero value as it deserves
   :Step: imputation stage I5, zero reasons
   :Does: A structure value of 0 comes with a reason. Zeros that mean
      "valued elsewhere" (as personal property, as tax-exempt, or
      included in another building's value) are filled like missing
      values. Only zeros that mean "the roll values no building" teach
      the model when a value is 0. All other zeros are kept.
   :Why: Some zeros say the building has no value; others only say the
      value was recorded somewhere else. The maintainer set these rules
      on 2026-10-31 and 2026-11-02.
   :Verdict: holds, reworded. A zero marked "included elsewhere" on a
      lot where another building carries the value keeps its 0 (see I6),
      though two comments say it is refilled. The step behaves as
      intended.

Do not count a lot's value twice
   :Step: imputation stage I6, lot-value guard
   :Does: Does not fill a dwelling with no value on a lot where another
      building already carries the roll's value.
   :Why: The roll's value already covers that dwelling. Filling it would
      count the same value twice.

Fill zero values on small homes
   :Step: imputation stage I7, small dwellings
   :Does: Treats a stated value of 0 on a residential building under
      120 m2 as missing, and fills it.
   :Why: Such a building is often a manufactured home missing from the
      roll, classed as single-family, with only its land on the roll.

Tell listed manufactured homes from unlisted ones
   :Step: imputation stage I8, manufactured-home listing
   :Does: Marks each manufactured home as listed on the roll, unlisted,
      or unclear. An unlisted home's value is filled; a listed home's
      value teaches the model.
   :Why: In North Carolina and Texas, a manufactured home on land its
      owner does not own is taxed as personal property and missing from
      the real-estate roll.

Bring all values to one year
   :Step: imputation stage I9, common year
   :Does: Scales each county's roll values from the year they were
      appraised to 2025, using the county house price index of the
      Federal Housing Finance Agency (FHFA), and ships the result as
      ``structure_value_adjusted``.
   :Why: Counties reappraise in different years, so their values are
      not comparable without it. A published index is used.

Warn when a county's roll looks mispriced
   :Step: imputation stage I10, roll-level check
   :Does: Compares each county's median single-family roll value (after
      the scaling above) with its census median home value, and warns
      when the ratio is outside 0.3 to 1.3.
   :Why: A wrong appraisal year inflates every value the model learns
      from. In v0.5, Pender County's 2026 reappraisal was read as 2019,
      which inflated its values by 71%.
   :Verdict: holds, reworded. An earlier v0.8 build compared the wrong
      column (NSI's depreciated cost instead of the roll value). The
      check only warns, and the v0.8 export uses the corrected check.

Scale filled values to the county's census home value
   :Step: imputation stage I11, ACS price anchor
   :Does: In a county with no recorded building values of its own, the
      filled values come from models trained on nearby counties (donor
      counties). This step rescales them so that the typical filled home
      relates to the county's Census (American Community Survey, ACS)
      median home value the way the donor counties' roll values relate
      to theirs. It acted in 8 NC and 7 TX counties in v0.7 (2026-10-27).
   :Why: Price levels differ between counties, so a donor county's
      prices do not carry over. The ratio between a county's roll values
      and its census home value was expected to carry over better.
   :Verdict: weakened. In a test that hid 14 eastern NC counties'
      values, the typical county's miss was 18.7% without the rescaling
      and 19.0% with it (2026-10-28). Whether to keep it is a decision
      for the maintainer.

Borrow values from nearby counties
   :Step: imputation stage I12, donors by distance
   :Does: Trains each county's model on its own values plus those of
      other counties, weighted by distance (weights fall off over about
      250 km; counties beyond 400 km are not used).
   :Why: A county with few recorded values learns from its neighbors.
      The maintainer chose distance as the only weight (2026-10-16).
   :Verdict: holds, reworded. The recipe quotes 17.5%, which is from a
      tuning run. The delivered v0.7 model misses a typical county's
      values by 19.0% (eastern North Carolina, 14 counties held out,
      2026-10-28).

Ignore a county's own values where they are too few
   :Step: imputation stage I13, sparse own labels
   :Does: Where a county's recorded values cover under 30% of its
      buildings, the model for that county learns from donor counties
      only, not from the county's own few values.
   :Why: Values that sparse are usually not a fair sample: they tend to
      be the buildings a recent record reached, such as new
      construction. It acted on ``year_built`` (the year of
      construction) in 3 NC and 6 TX counties in v0.7 (2026-10-27).

Fill attributes in a fixed order
   :Step: imputation stage I14, chain order
   :Does: Fills the attributes one after another in a fixed order (area,
      year, stories, construction, foundation, roof, garage, sections,
      dwellings, structure value, then NSI's two cost measures). Each
      attribute's model can read the values filled before it.
   :Why: A filled attribute helps predict later ones: a building's year
      helps predict its value, for example.
   :Verdict: holds, reworded. The recipe gives no reason for this
      particular order.

Draw classes at random from their probabilities
   :Step: imputation stage I15, class draws
   :Does: For a missing class or count, the model gives each possible
      answer a probability, and one answer is drawn at random with those
      probabilities.
   :Why: Always choosing the most likely answer would overstate the
      most common class. Drawing keeps each county's mix of classes
      right.
   :Verdict: holds, reworded. An earlier v0.8 build used one random
      number per building for all attributes with the same gaps, which
      tied the filled construction type to the filled foundation type.
      Each county's mix of classes was not affected. The v0.8 export
      draws each attribute separately.

Predict values in two parts
   :Step: imputation stage I16, two-part value model
   :Does: First predicts whether a building's value is 0. If not, it
      predicts the value and adds a random spread drawn from how far
      similar predictions missed. Values are capped at $500 million.
   :Why: Loss models need both the totals and the spread of values
      right. Filling every building with its expected value alone was
      tested and rejected.

Cap implausibly high filled values
   :Step: imputation stage I17, value per area cap
   :Does: Lowers a filled value whose value per square meter is above
      the highest 0.1% of recorded values for its class. Its ``_source``
      then also reads ``+clipped``.
   :Why: Such a value means the model is predicting outside the range
      of its data.

Value manufactured homes with their own model
   :Step: imputation stage I18, manufactured-home value and park factor
   :Does: Values unlisted manufactured homes with a model trained on
      listed ones. Homes on a parcel with five or more homes (a park)
      are scaled down by a park factor.
   :Why: A home missing from the roll is valued the way assessors value
      the homes they list. Homes in a park are worth less where the
      household does not own the land.

Fill NSI's cost measures
   :Step: imputation stage I19, NSI cost measures
   :Does: Fills NSI's replacement cost and depreciated replacement cost
      where NSI has no point, learning from the county's own NSI values,
      and keeps the depreciated cost below the replacement cost.
   :Why: These costs ship in their own columns and are never mixed with
      the market value from the roll (maintainer, 2026-10-28).
   :Verdict: holds, reworded. A comment in the code names a setting
      that does not exist. The step is fine.

Train the models on incomplete rows
   :Step: imputation stage I20, gap-trained models
   :Does: Before training, hides inputs on half the training rows, in
      the same pattern as the buildings that need filling.
   :Why: A building missing one attribute usually misses others too. A
      model trained only on complete rows would meet gaps it never saw.

Make the filled values reproducible
   :Step: imputation stage I22, seed
   :Does: One random seed (2026) drives every random choice. The licence
      notice records it with the version of the code.
   :Why: With the same seed, inputs and code, a rerun gives the same
      filled values.
   :Verdict: holds, reworded. A rerun also needs the exact code state
      recorded in the notice. For v0.7 that code included changes never
      saved to the shared repository, so only the copy on the build
      machine can reproduce it.

Never ship NSI's year or value
   :Step: imputation stage I23, NSI never ships as year or value
   :Does: Treats years and values that came from NSI as missing, fills
      them, and empties any that are not filled.
   :Why: NSI's value is not reliable for a single building, and its
      years are estimates for an area.

Write the filled values into the bundle
   :Step: imputation stage I24, overlay and adjusted value
   :Does: Writes the filled values into the bundle, then computes
      ``structure_value_adjusted`` (the value scaled to 2025) and
      ``appraisal_year`` from the values that ship.
   :Why: One place writes the filled values, so each is written
      once.

Build the team bundle from the public one
   :Step: imputation stage I25, team bundle
   :Does: Takes the public bundle's filled values and puts the team's
      own roll values back only where the public bundle withheld them.
   :Why: No model ever learns from a value that is cleared for the team
      only, so team-only values cannot leak into the public bundle.
   :Verdict: holds, reworded. In the cells put back, the team gets the
      roll's own value, without the zero-value and manufactured-home
      rules above (I5, I8).

State the accuracy in the recipe
   :Step: imputation stage I26, accuracy statement in the recipe
   :Does: States how accurate the method is, next to the method.
   :Why: A reader needs the accuracy together with the method.
   :Verdict: weakened. The recipe's figures are from v0.6 (a typical
      county's values missed by 12.8%, 2026-10-24). The delivered v0.7
      misses by 19.0% (eastern North Carolina, 14 counties held out,
      2026-10-28). Structure value and year built were not rescored on
      v0.8.

.. rubric:: Steps that do not reach the bundle

- Realistic gaps for testing the models (imputation). The models are
  tested by hiding one county's recorded values and filling them again.
  Each test building is given the same missing inputs as a building in
  that county that really lacks the value, so the test is as hard as
  the real gaps. This is used only in those tests, never in a
  delivery.

Writing the files (delivery)
""""""""""""""""""""""""""""

After the imputation, the delivery writes the bundle.

Write the main files
   :Step: delivery stage D14, file writes
   :Does: Writes the inventory table, the point file and the footprint
      file, after all withholding.
   :Why: A user loads only the file needed. Only this stage writes to
      the shared folder, so only checked files reach it.

Write the evidence file and the licence notice
   :Step: delivery stage D15, evidence file and notice
   :Does: Writes the evidence file (every source's own value) and the
      licence notice ``*_LICENSE.txt``, then moves all files into place
      at once.
   :Why: All files share one row key. A failed run leaves the previous
      bundle untouched. The notice carries the sources' conditions with
      the data.
   :Verdict: holds, reworded. The notice does not yet explain the
      ``clipped`` marker (see I17) or the NSI cost columns, and it calls
      the manufactured-home value model the county's own, though that
      model also learns from other counties.

Add a map and a read-me to the shared folder
   :Step: delivery stage D16, map and read-me of a shipped folder
   :Does: Adds a QGIS map, county footprints and a read-me file to the
      region's folder on the shared drive.
   :Why: A recipient who does not run openplaces needs them to open and
      understand the bundle.
   :Verdict: open. The scripts that make these files are not part of
      the openplaces package, so a shipped folder cannot be rebuilt from
      the package alone.
