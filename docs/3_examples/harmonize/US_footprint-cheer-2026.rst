.. openplaces

.. _cheer_footprints:

CHEER building footprints
=========================

This page walks through the CHEER footprint harmonization pipeline
end to end, explaining what each step does and why.

The recipe driving this pipeline is ``US_footprint-cheer-2026.yaml``.

It produces a county-level GeoParquet file of building footprints enriched with parcel attributes, NSI occupancy data, and Overture address points.


What this produces
------------------

For each county in the CHEER study area (initially coastal North Carolina and
Texas), the pipeline produces one parquet file containing:

- Every building :ref:`footprint <footprints>`: geometry from the best available source(s).
- A ``footprint_role`` label (primary / secondary / unknown) that identifies
  the main structure on each parcel.
- Parcel-sourced attributes: assessed improvement value, land value, address,
  use class (``purpose_group`` / ``purpose_subgroup``), and year built.
- NSI attributes: occupancy type, structure value, number of stories, and a
  block-median year-built estimate.
- Overture address attributes: street address, unit count, and postal code.
- Derived columns: footprint area in m², value per m², and a reconciled
  occupancy group.

The pipeline runs at county level (admin level 3) and saves output to the
``core`` data directory.


Precursor datasets
------------------

Before harmonization, the following datasets must be ingested. The
harmonization notebook (see `Running the pipeline`_ below) handles this
automatically:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Recipe ID
     - Description
   * - ``US_admin-census-2021_admin3``
     - US county boundaries (used to allocate Microsoft footprints)
   * - ``footprint-obm-2025``
     - Global footprints from OpenBuildingsMap (OBM)
   * - ``US_footprint-microsoft-v2``
     - US footprints from Microsoft's ML building detection model
   * - ``{state}_footprint-{source}-{version}``
     - State-specific footprint dataset, auto-discovered per state
   * - ``US_footprint-fema-2023``
     - US footprints from FEMA USA Structures
   * - ``{state}_parcel-{source}-{version}``
     - State parcel data, auto-discovered per state
   * - ``US_building-nsi-2022``
     - FEMA National Structure Inventory (NSI) building points
   * - ``dwelling-overture-2025``
     - Overture Maps dwelling address points

Each ``ingest()`` call downloads and caches the data for the requested county
IDs; subsequent runs skip already-cached files unless ``reprocess=True``.


Running the pipeline
--------------------

The harmonization notebook is at
``notebooks/03_harmonize/footprints/harmonize_footprints.ipynb``.
For a single county (Brunswick, NC):

.. code-block:: python

   from openplaces.io.harmonizer import harmonize

   harmonize(
       'US_footprint-cheer-2026',
       admin_ids=['US-NC-BS'],
       reprocess=True,
       verbose=True,
   )

The notebook also exports a standalone Python script that can be run on a
cluster with multiple counties in parallel.


Pipeline walkthrough
--------------------

The recipe's ``pipeline:`` list declares nine steps, grouped into four phases.
Each step is a registered function that receives the shared ``HarmonizeState``
object, modifies it, and returns it. The state carries a ``spine``
(the output GeoDataFrame being built up), ``references`` (loaded source
tables), ``crosswalks`` (spine ↔ reference join tables), and ``overlays``
(geometry-bearing overlay results).


A. Build the footprint spine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1 — Merge footprints from multiple sources**

``resolve_spine``

The spine is the primary GeoDataFrame that all later steps extend. It starts
empty and is filled by merging footprints from four sources in order of
descending priority:

1. **OBM** (OpenBuildingsMap) — a global open dataset.
2. **Microsoft** — US-wide ML-detected footprints.
3. **State-specific** — any state-scoped footprint ingest recipe found via
   ``auto_discover: true``. For example, a county in North Carolina will
   automatically pick up ``US-NC_footprint-nconemap-2024`` if that recipe
   exists.
4. **FEMA** — USA Structures, used as a last-resort fallback.

Each source after the first is compared against what is already in the spine.
A candidate footprint is added only if its *intersection-over-union* (IoU)
with every existing spine footprint is below ``overlap_iou_max = 0.02``. IoU
is the ratio of the intersection area to the union area of two polygons; a
value above 0.02 means the two footprints share more than 2 % of their
combined area and are treated as the same building. Footprints below
``min_area_m2 = 10`` m² are dropped before merging.

Because IoU-based deduplication can miss displaced thin-rectangle footprints
(mobile homes or trailers that appear in two sources as parallel rectangles
shifted sideways rather than overlapping), the *elongated-duplicate filter*
checks pairs of elongated footprints separately. Two footprints are considered
duplicates when they are both elongated (aspect ratio ≥ 2.5), their long axes
are aligned within 15°, their long-axis projections overlap by at least 50 %,
and their lateral separation is less than twice their average width. Candidates
that fail this test are dropped even though their IoU is below the threshold.

After this step, ``state.spine`` is a GeoDataFrame with columns ``geometry``
and ``source`` (the label of the source dataset, e.g. ``'obm'``,
``'microsoft'``, ``'nconemap'``, or ``'fema'``).


B. Attribute sources to the spine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 2 — Link footprints to parcel records**

``link_to_reference`` (parcel, spatial_overlay)

Parcels carry assessed value, land use class, address, and year-built
information that is not present in any footprint source. This step builds a
crosswalk between footprints and parcels by running a *polygon identity
overlay*: each footprint is intersected with all parcels it touches, producing
a row for every (footprint, parcel) pair.

The crosswalk is a MultiIndex DataFrame indexed by ``(footprint_id,
parcel_id)`` with columns:

- ``area_intersection_m2`` — area of the footprint-parcel intersection.
- ``iou`` — intersection-over-union of the two polygons.
- ``fraction_of_largest`` — this intersection's area as a fraction of the
  largest intersection for the same footprint.

Two thresholds keep the crosswalk clean: secondary parcel links whose
``fraction_of_largest`` is below 1/6 are dropped (a footprint's corner
clipping a distant parcel should not be attributed), and intersections smaller
than 10 m² are removed. After filtering, a footprint may link to:

- **One parcel** (most common) — ``link = 'unique parcel'``.
- **Multiple parcels** — ``link = 'multi-parcel footprint'`` (e.g., a large
  commercial building spanning a property boundary).
- **No parcel** — ``link = 'no parcel'`` (footprint lies outside mapped parcel
  coverage).

The parcel dataset is also aggregated by ``geo_id`` to collapse stacked
duplicate records (some assessor files include multiple rows per parcel for
different tax classes), and an ``improvement_value_per_ha`` column is
computed for use in the next step.

**Step 3 — Add synthetic footprints for unmatched parcels**

``infer_spine_additions``

Some parcels with buildings on them have no footprint in any source — rural
structures, manufactured housing, and buildings in areas with poor satellite
coverage are common examples. This step adds a synthetic footprint for each
such parcel that is likely to have a structure.

Two criteria must both be met:

1. The parcel's ``purpose_group`` has a mean footprint count per parcel
   (across the county) of at least ``n_per_group_min = 0.2`` — groups where
   only 1 in 5 parcels typically has a footprint are not expected to have
   missing footprints.
2. The parcel's ``improvement_value_per_ha`` exceeds the 5th percentile of
   matched parcels in the same purpose group — parcels with negligible
   improvement value are unlikely to contain a building.

Parcels that pass both tests receive a synthetic footprint whose geometry is
the parcel polygon itself. The footprint is assigned
``source = 'parcel.{source_id}'`` and is appended to ``state.spine``.
A reference to the inferred footprints is also stored in
``state.metadata['inferred_from_{recipe_id}']`` so that the attribute step
can handle them correctly (no area-weighting is needed because each inferred
footprint occupies exactly one parcel).

**Step 4 — Clean overlapping footprint geometries**

``resolve_overlaps``

Adding inferred parcel-shaped footprints can introduce geometry overlaps
where the parcel polygon extends over an existing footprint from OBM or
Microsoft. This step calls ``clean_polygons()`` to remove self-intersections
and degenerate geometry, then ``resolve_overlapping_polygons()`` to trim any
remaining pairwise overlaps, keeping ``state.spine`` topologically clean.

**Step 5 — Link footprints to NSI building points**

``link_to_reference`` (NSI, spatial_point)

The National Structure Inventory (NSI) is a FEMA dataset of building points
with occupancy class (``purpose_subgroup``), structure replacement value, year
built, and number of stories. Unlike parcel records, NSI classifies structures
individually rather than by land-use zoning, making it a valuable crosscheck
on parcel-derived use classes.

NSI records are points; footprints are polygons. The four-pass proximity
method (Lochhead et al. 2026, Table 3) handles imprecise geocoding:

1. **Pass 1 — containment**: ``sjoin(predicate='within')``. Points inside a
   footprint are matched directly.
2. **Pass 2 — inner proximity (10 m)**: unmatched points within 10 m of a
   footprint edge (typical GPS error) are matched to the nearest footprint.
3. **Pass 3 — outer proximity (100 m), same-parcel constraint**: unmatched
   points within 100 m are matched to the nearest footprint *on the same
   parcel*. The parcel constraint prevents linking a point to a footprint
   across the street.
4. **Pass 4 — unbounded fallback**: disabled in this recipe
   (``unbounded_proximity_m`` defaults to 0).

After linking, each point appears at most once in the crosswalk (the
highest-quality match is kept when passes produce duplicates). The result is
stored as a flat DataFrame in ``state.crosswalks['US_building-nsi-2022']``.

**Step 6 — Link footprints to Overture address points**

``link_to_reference`` (Overture dwelling, spatial_point)

Overture Maps dwelling points are address-geocoded locations of individual
housing units. Unlike NSI (which represents one structure per record),
a multi-unit building may have several Overture points — one per unit.

Two additional options are enabled for this join:

- ``dedup_addresses: true`` — when a base address (street + house number)
  has both a building-level record and unit-specific records (e.g., ``Apt 1``,
  ``Apt 2``), the building-level record is dropped. Each unit record is tagged
  ``n_dwelling_units = 1`` for downstream summation.
- ``aggregate_multipoint: true`` — after linking, multiple dwelling points
  matched to the same footprint are collapsed into one row. Their
  ``n_dwelling_units`` values are summed, giving the total residential unit
  count for that footprint.

The same three proximity passes as for NSI are used (inner 10 m, outer 50 m
same-parcel). The resulting crosswalk has one row per footprint that has at
least one attributed dwelling point.

**Step 7 — Classify each footprint as primary or secondary**

``classify_footprint_role``

On parcels with more than one footprint (house, garage, shed, etc.), only the
primary structure should receive the parcel's full land value and assessed
improvement value. This step assigns each footprint one of three roles:

- **primary** — the main structure on the parcel.
- **secondary** — an accessory structure (garage, outbuilding, etc.).
- **unknown** — footprint not linked to any parcel.

The classification follows Lochhead et al. (2026, Table 4), using point
evidence to resolve ambiguity on multi-footprint parcels:

1. If any footprint on the parcel has a *dwelling point* (Overture) linked to
   it, those footprints are ``primary``; the others are ``secondary``.
2. Otherwise, if any footprint has an *NSI building point* linked to it, those
   are ``primary``; the others are ``secondary``.
3. If no footprint on the parcel has any point evidence, all are
   ``secondary`` (the pipeline cannot determine which is the main structure).
4. A footprint that is the *sole* footprint on its parcel is always
   ``primary`` regardless of point evidence.
5. Footprints not linked to any parcel are ``unknown``, except those with
   dwelling-point evidence, which are promoted to ``primary``.

The result is stored as a Categorical column ``spine['footprint_role']``.


C. Select values for features with disagreement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 8 — Aggregate and reconcile attributes from all sources**

``reconcile_attributes``

Multiple sources provide overlapping information. This step aggregates all
reference attributes onto the spine and resolves conflicts.

*Parcel attributes (polygon reference)*

Parcel values are distributed across footprints proportionally to intersection
area. For a footprint that overlaps two parcels, ``improvement_value`` and
``n_dwelling_units`` are each split by ``area_fraction`` (the footprint's
share of the total intersection area for each parcel). ``land_value`` and
``address`` are assigned from the largest-intersection parcel only, and
``land_value`` is further restricted to ``primary`` footprints — accessory
structures do not receive a share of the land value.

When a parcel has multiple footprints and at least one has dwelling-point
evidence, the area fractions for footprints *without* dwelling evidence are
zeroed out before the split. This means only dwelling-linked footprints
receive ``improvement_value`` and ``n_dwelling_units`` from the parcel, even
if the non-dwelling-linked footprints are geometrically larger (Lochhead et
al. 2026, Table 4).

Columns written with suffix ``_parcel``:
``purpose_group_parcel``, ``purpose_subgroup_parcel``,
``openplaces_group_parcel``, ``improvement_value_parcel``,
``land_value_parcel``, ``year_built_parcel``, ``address_parcel``,
``n_dwelling_units_parcel``, ``overlap_fraction_parcel``,
``n_other_footprints_parcel``.

*Parcel purpose group → openplaces group mapping*

Parcel assessors use locally-defined land-use codes in ``purpose_group`` that
vary by state and county and do not map directly to NSI occupancy classes. To
bridge this gap, the step derives ``openplaces_group_parcel`` for every
footprint by building a county-wide majority-vote lookup from the
co-occurrence of parcel ``purpose_group`` and NSI ``openplaces_group`` labels
on matched footprints:

1. For every footprint that has both a parcel link and an NSI link, the pair
   *(purpose_group, openplaces_group)* is recorded.
2. Across all such pairs in the county, the most common NSI group for each
   parcel purpose group is selected as the representative label.
3. This lookup is applied back to every footprint via its
   ``purpose_group_parcel`` value — including footprints that have no NSI
   match of their own.

The effect is that NSI's structure-level occupancy knowledge propagates
indirectly to parcels that NSI does not cover, using county-level
co-occurrence to normalize heterogeneous assessor codes into a shared
vocabulary.

*Overture dwelling attributes (point reference)*

``n_dwelling_units_overture`` — the count of attributed dwelling points
(from ``aggregate_multipoint``). Address columns written with suffix
``_dwelling_overture``: ``address_street_dwelling_overture``,
``address_number_dwelling_overture``, ``postal_code_dwelling_overture``,
``city_dwelling_overture``.

*NSI attributes (point reference)*

Columns written with suffix ``_building_nsi``:
``purpose_subgroup_building_nsi``, ``openplaces_group_building_nsi``,
``structure_value_building_nsi``, ``year_built_block_median_building_nsi``,
``n_stories_building_nsi``.

*Priority selection*

After all sources are aggregated, a priority rule selects the final value for
features where sources disagree, using ``.bfill(axis=1)`` across suffixed
columns in order:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Final column
     - Priority order
     - Rationale
   * - ``purpose_subgroup``
     - NSI → parcel
     - NSI classifies structures individually; parcel class is land-use zoning
   * - ``n_dwelling_units``
     - Overture → parcel
     - Overture provides explicit unit counts; parcel is an estimate
   * - ``year_built``
     - parcel → NSI
     - Tax-record year built is preferred; NSI has only block-median fallback
   * - ``improvement_value``
     - parcel only
     - No competing source

The final columns ``purpose_subgroup``, ``n_dwelling_units``, ``year_built``,
and ``improvement_value`` each contain the first non-null value across their
source-suffixed counterparts.


D. Fill gaps
~~~~~~~~~~~~~

**Step 9 — Derive additional columns and fill remaining gaps**

``infer_attributes``

The last step derives additional columns from the data assembled so far:

- ``m2`` — footprint area in square metres, computed from the polygon
  geometry.
- ``improvement_value_parcel_per_area`` / ``structure_value_building_nsi_per_area``
  — value per m², computed as the value column divided by ``m2``. Useful for
  detecting data-entry errors and for cross-county comparisons.
- ``openplaces_group_combined`` — reconciles ``openplaces_group_parcel``
  (derived from the majority-vote lookup in step 8) with
  ``openplaces_group_building_nsi``. When both are present and agree, a single
  label is used; when they differ, the combined label is
  ``'{parcel_group} | {nsi_group}'`` for transparency.
- ``n_dwelling_units`` — fills remaining nulls using a lookup table
  (``_OCC_UNITS``) that maps NSI occupancy classes to expected unit counts,
  e.g. ``'Single Family' → 1``, ``'Multi-Family (2 units)' → 2``,
  ``'Multi-Family (50+ units)' → 51``. This provides a fallback unit count
  for footprints that have a ``purpose_subgroup`` but no direct count from
  Overture or parcel data.

After this step, string columns whose names match entries in the attribute
registry are cast to ``pd.Categorical`` dtype to reduce memory usage.


Key design decisions
---------------------

**IoU deduplication vs. nearest-neighbor merge**

Merging footprints from several sources naively by nearest footprint would
combine physically distinct buildings (e.g., two row houses) when a closer
match from a different source is on the other side of a wall. IoU
deduplication avoids this: a candidate footprint is kept if and only if it
does not substantially overlap an existing one, so geometrically distinct
buildings are always added regardless of how close they are.

**Area-weighted value distribution**

When a single footprint spans two parcels (e.g., a commercial building on a
split lot), the parcel's improvement value cannot be assigned to just one.
Distributing proportionally to intersection area produces a consistent and
auditable allocation that sums back to the full parcel total.

**Dwelling > building evidence hierarchy**

NSI records one entry per structure; Overture records one point per
*residential unit*. When both sources are present for a multi-footprint
parcel, dwelling-point evidence is considered stronger because it explicitly
locates an occupied housing unit. This prevents a garage (which may have its
own NSI record as an accessory structure) from being classified as primary.

**Inferred footprints as a fallback**

Rather than dropping parcels with no footprint source coverage, the pipeline
creates a synthetic entry using the parcel polygon. This ensures that high-
value structures that are absent from all footprint datasets still appear in
the output with their parcel attributes, even if the geometry is less precise.


Key reference
--------------

Lochhead M, Zsarnóczay Á, Deierlein G (2026) Exposure matters: a synthesis framework for high-resolution building inventory development. International Journal of Disaster Risk reduction 139 (2026): 106148 `<https://doi.org/10.1016/j.ijdrr.2026.106148>`_
