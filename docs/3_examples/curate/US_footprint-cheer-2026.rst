.. openplaces

.. _cheer_footprints:

U.S. footprint inventory (CHEER)
================================

.. image:: ../../1_overview/concepts/images/footprint_building_dwelling_urban.png
  :width: 400
  :alt: Illustration of footprints, buildings, and dwellings
  :align: right

This recipe creates a footprint-level building inventory for hurricane damage and exposure modeling in the U.S.

It first resolves building :ref:`footprints <footprints>` from multiple polygon sources (OpenBuildingMap, Microsoft, and local datasets), then enriches each footprint with data from linked parcels, the National Structure Inventory (NSI), Overture :ref:`dwelling <dwellings>` addresses, and FEMA's USA Structures.

It also integrates deep-learning-based recognition of roof shapes and story counts from Google Satellite and Street View imagery (BRAILS++).

The pipeline is currently being tested in Florida, North Carolina, and Texas.

* **Curation recipe**: :gh-file:`src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml`
* **Companion notebook**: :gh-file:`notebooks/examples/US_curate_footprints.ipynb`

This work is supported by NSF's Coastal Hazards, Economic Prosperity & Resilience hub (`CHEER <https://www.drc.udel.edu/cheer/>`_). 

Output columns
~~~~~~~~~~~~~~

This recipe produces one :file:`.parquet` file per county.

Each row represents a deduplicated footprint (or parcel fallback).

Canonical attributes
--------------------

``footprint_id``
    A unique identifier for each footprint: its 11-digit `openlocationcode <https://github.com/google/open-location-code>`_ (duplicate codes receive a numeric suffix, e.g., :input:`-1`, :input:`-2`, to guarantee uniqueness).
``geometry``
    The spatial polygon outlining the footprint, or the parcel polygon for fallback records.
``geometry_source``
    Source of the geometry: :input:`obm` (OpenBuildingMap), :input:`microsoft`, local state sources (e.g., :input:`nconemap`), or :input:`parcel.<source>` for parcel-shaped fallbacks representing unlocated structures.
``occupancy_type``
    Canonical occupancy class.

    Multi-Family structures are split into HAZUS height bands (Low-Rise: 1-3 stories, Mid-Rise: 4-7 stories, High-Rise: 8+ stories) based on ``n_stories``. Multi-Family rows with no story count keep the plain ``Multi-Family`` label.
``value``
    Reconciled structure value in USD. Prioritizes the parcel's improvement value (apportioned across its primary footprints by floor-area share), falling back to the NSI structure replacement value.
``year_built``
    Reconciled construction year. Prioritizes assessor parcel records over NSI block-median fallbacks.
``n_stories``
    Reconciled number of stories. Prioritizes street-level imagery predictions (BRAILS++), then measured-height-derived counts (FEMA), and lastly NSI modeled block-median counts.
``n_dwellings``
    Reconciled count of dwelling units. Prioritizes Overture geocoded address counts over NSI structure counts, falling back to occupancy-class imputation.
``height``
    Reconciled building height in meters (where available from the footprint source, e.g. OpenBuildingMap).
``roof_shape``
    Reconciled roof structure classification (e.g., Gable, Hip, Flat) from visual models.
``m2``
    Calculated footprint area in square meters.
``priority_on_parcel``
    Structural role: :input:`primary` (main structure), :input:`secondary` (accessory structure), or :input:`unknown` (unlinked to parcel).

Provenance sidecars
-------------------
Indicates which input dataset or curation rule set decided the final canonical value (later steps override earlier ones). The values contain short method/source strings (e.g., ``parcel``, ``nsi``, ``overture``, ``imputed``, ``geometry``, ``dwellings``, ``secondary``, ``park``, ``keyword``, ``manufactured_home``, ``brails-2026``):

* ``occupancy_type_source``
* ``value_source``
* ``year_built_source``
* ``n_stories_source``
* ``n_dwellings_source``
* ``roof_shape_source``

Parcel-derived evidence (assessor)
----------------------------------
Attributes inherited from the primary parcel linked to the footprint.

``parcel_id``
    Globally unique ID of the primary linked parcel, used to join parcels back onto footprints.
``parcel_id_local`` / ``parcel_id_local_all``
    Locally cross-comparable assessor ID of the primary linked parcel, and combined IDs of all intersected parcels (populated only if more than one distinct ID exists). Not unique across administrative units, so it is not used as a join key.
``use_group_combined_parcel`` / ``use_group_combined_parcel_all``
    Primary and combined land-use description strings (populated only if more than one distinct use exists).
``land_use_class_parcel``
    Curated parcel land-use classification mapped into 11 potential classes (e.g., Vacant, Manufactured Home, Retail, Office).
``occupancy_type_parcel``
    Assessor-proposed occupancy based on property keywords/groups.
``occupancy_type_footprint_fema``
    Each parcel's dominant FEMA USA-Structures occupancy class by overlap area (remapped via the ``US_footprint-fema-2023_occupancy-type-remap`` crosswalk). Used as second-ranked evidence in the base occupancy cascade.
``land_value_parcel``
    Assessor land valuation. Kept in full on primary footprints and set to missing (null) on secondary footprints.
``improvement_value_parcel``
    Assessor improvement valuation. Apportioned across the parcel's primary footprints by floor-area share.
``year_built_parcel``
    Construction year recorded in assessor tax records.
``address_parcel``
    Property street address from assessor records (where the parcel geometry source provides it).
``overlap_fraction_parcel``
    Fraction of footprint area intersecting this parcel.
``area_intersection_m2_parcel``
    The overlap area in square meters between the footprint and the dominant parcel.
``n_dwellings_parcel``
    Area-distributed parcel dwelling count from the harmonized spine (may be empty if assessor rolls only exist in the curated parcel lane).

National Structure Inventory evidence (NSI)
--------------------------------------------
Structure-level point attributes matched to the footprint.

``n_buildings_nsi`` / ``building_id_nsi``
    Count and ID of matched NSI structure records.
``n_dwellings_nsi``
    Summed NSI unit counts per footprint (excluding records flagged to skip upward correction). Serves as a fallback source for ``n_dwellings``.
``occupancy_type_building_nsi`` / ``occupancy_type_building_nsi_all``
    NSI occupancy class (single matches, and combined list of matches if more than one distinct class exists).
``group_building_nsi`` / ``group_building_nsi_all``
    NSI occupancy group mapped to openplaces vocabulary (single matches, and combined list of matches if more than one distinct group exists).
``structure_value_building_nsi``
    NSI-calculated replacement value of the structure.
``year_built_block_median_building_nsi``
    Median year built for the associated census block.
``source_building_nsi``
    Underlying source database of the NSI record (e.g., :input:`Model`).

Overture dwelling and address evidence
--------------------------------------
Residential unit address points matched from Overture.

``n_dwellings_overture``
    Total count of geocoded residential unit points linked, representing total dwelling units summed across linked address points (address deduplication and multipoint aggregation enabled). Unmatched footprints or those on vacant-use parcels are filled with 0.
``address_street_dwelling_overture`` / ``address_number_dwelling_overture``
    Geocoded street name and house number components.
``postal_code_dwelling_overture`` / ``city_dwelling_overture``
    Postal code and city components.

Diagnostics and special metrics
-------------------------------
Flags and intermediate calculations used for curation and quality control.

``n_parcels_per_footprint``
    Number of parcels intersected by the footprint.
``n_footprints_per_parcel``
    Number of footprints on the associated parcel (includes synthetic fallbacks).
``occupancy_type_conflict``
    Summary of conflicting occupancy labels across inputs (formatted :input:`"nsi: X | fema: Y | parcel: Z | overture: W"`). This column is only populated where two or more evidence sources disagree. Non-residential classes are collapsed to :input:`Non-Residential` before comparison, so only residential (or residential-vs-non-residential) disagreements are surfaced.
``occupancy_type_review``
    Flag (:input:`True`) for low improvement-value parcel shares indicating potential manufactured homes needing inspection.
``value_per_area`` / ``improvement_value_parcel_per_area`` / ``structure_value_building_nsi_per_area``
    Calculated values divided by footprint area (USD/m²).
``manufactured_home_community``
    Boolean flag indicating whether the parcel is part of a manufactured home community (having more than 3 final Manufactured Home footprints).
``n_manufactured_homes_per_parcel``
    The total number of curated Manufactured Home footprints residing on the same parcel.

Processing pipeline
~~~~~~~~~~~~~~~~~~~

.. image:: US_footprint-cheer-pipeline.png
  :width: 300
  :alt: Illustration of pipeline
  :align: right

The creation of the CHEER dataset proceeds through the four core stages of the ``openplaces`` pipeline: **Ingest** (twice: once for entities, once for images), **Harmonize**, **Enrich**, and **Curate**.

Stage 1: ingest
---------------

This stage downloads and extracts raw geometry and reference point datasets:

1. **Footprint polygon datasets**: Downloads and unzips raw footprints from OpenBuildingMap (OBM), Microsoft, and state/local layers.
2. **Parcel datasets**: Gathers property tax assessor geometry and tax rolls from local/state agencies.
3. **Reference layers**: Downloads structure point databases (National Structure Inventory; NSI), geocoded residential address points (Overture), and FEMA USA Structures footprints (used for parcel-level occupancy evidence).

Stage 2: harmonize
------------------

This stage merges geometries and links datasets to build the core footprint spine (``US_footprint-spine-2026``) and parcel spine (``US_parcel-spine-2026``):

1. **Footprint spine harmonization** (``US_footprint-spine-2026``): Creates the spatial exposure units by merging and deduplicating footprint boundaries, linking them to parcels, and attaching point reference data:

   * **Geometry deduplication**: Merges footprints in priority order (OBM first, then Microsoft, then state-specific sources). Footprints smaller than 10 m² are dropped. Candidate shapes are added to the spine only if their Intersection-over-Union (IoU) with existing shapes is < 0.02 and they do not fall within a size-scaled exclusion buffer around large existing spine footprints (area >= 250 m²; buffer distance scales with the square root of the area: :input:`keep-out distance` = 2 m + 0.5 * sqrt(area)).
   * **Elongated-footprint filter**: Aspect ratios >= 2.5 with aligned axes (within 15°), longitudinal overlap >= 50%, and lateral spacing < 2x width are deduplicated to prevent parallel-shifted footprint representations (common for manufactured homes) from appearing twice.
   * **Parcel spatial overlay**: Intersects footprints with parcels via a polygon identity overlay. Intersections under 10 m² or minor slivers (< 1/6 of the footprint's largest parcel intersection) are filtered. Systematic spatial displacements between footprint and parcel layers are corrected by snapping chain-displaced footprints to their dominant parcel when minor overlaps are small and land on neighbors with their own buildings.
   * **Synthetic fallbacks**: For parcels where assessor records indicate a structure exists but no footprint is detected, synthetic "footprint" rows are created using the parcel boundary geometry (labeled :input:`parcel.<source>`). Overlaps against detected footprints are spatial-trimmed. These fallback polygons are included in the parcel footprint count, are backfilled with the ID (``parcel_id``) and local ID (``parcel_id_local``) of their source parcel, and are excluded from morphology metrics.
   * **Reference point linking**: Associates structure-level point evidence with the footprint spine using a tiered proximity join (`Lochhead et al. 2026`_):

     - *Containment*: Point is within the footprint.
     - *Inner Proximity*: Points within 10 meters of an edge are assigned to the closest footprint.
     - *Outer Proximity*: Points within 100 meters (NSI) or 50 meters (Overture) are assigned to the closest footprint *on the same parcel*, ensuring points aren't misaligned across street boundaries.
     - *Duplicate Resolution*: Colocated duplicate points from low-rank sources (such as outdated ESRI or HAZUS/NSI-2015 records sharing a building ID or location cell with a higher-ranking source) are flagged during the link step and excluded when NSI evidence is merged onto the footprint spine.

   * **Role classification**: Footprints on multi-structure parcels are classified to identify primary vs. accessory structures:

     - Footprints with Overture dwelling points are :input:`primary`.
     - Otherwise, footprints with NSI building points are :input:`primary`.
     - If no point evidence exists, all footprints on the parcel default to :input:`secondary`.
     - Sole footprints on a parcel are always classified as :input:`primary`.
     - Footprints not linked to any parcel are classified as :input:`unknown`, unless they carry dwelling-point evidence, in which case they are promoted to :input:`primary`.
     - Synthetic, parcel-derived fallback geometries representing unlocated structures are always classified as :input:`primary`.

   * **Raw attribution**: Compiles all raw joined variables into source-suffixed evidence columns in the intermediate spine.

2. **Parcel spine harmonization** (``US_parcel-spine-2026``): Creates the parcel-level matching baseline by compiling land boundaries, merging tax assessor records, and linking point evidence:

   * **Boundary baseline**: Merges discovered statewide and local parcel geometry layers into a unified spatial spine.
   * **Assessor records merge**: Joins county and local assessment tables by matching local ID keys and standardizing property use codes.
   * **Reference data enrichment**: Joins building structures (NSI), dominant building groups, and FEMA occupancy, and counts footprint morphology features (such as primary and small elongated footprints) on each parcel. It resolves colocated duplicate NSI points using the same rules (flagging low-rank twins) to exclude them from parcel aggregates.

Stage 3: ingest images
----------------------

This stage fetches external imagery required for deep-learning visual classification, querying the Google API using footprint geometries from the harmonized spine:

1. **Satellite imagery**: Scrapes per-building Google Satellite images at Zoom level 20 (``image-googlesatellite-z20``) via the BRAILS++ scraper.
2. **Street View imagery**: Downloads street-level Google Street View photos (``image-googlestreetview-2026``).

Stage 4: enrich
---------------

This stage runs deep learning models (BRAILS++) on the ingested imagery to predict visual building attributes (`Cetiner et al. 2025`_):

1. **Roof shape prediction**: Runs neural network classifiers on the scraped satellite imagery to predict ``roof_shape``.
2. **Story count detection**: Runs detectors on the Street View photos to infer ``n_stories``.

Stage 5: curate
---------------

This stage curates the spines into clean, canonical datasets:

1. **Parcel curation** (``US_parcel-openplaces-2026``): Derives parcel-level occupancy groups, calculates relative footprint area z-scores, and classifies land use (e.g., Manufactured Home Park, Vacant, Retail, Office, Townhome) using weighted voting. Townhome classification is driven by structural and dwelling footprint indicators (spanned parcels or multiple units) combined with assessor use keywords.
2. **Footprint curation** (``US_footprint-cheer-2026``): Integrates curated parcels, corrected address counts, reconciled attribute priorities, base occupancy imputation, visual model prediction, and weighted voting to resolve final footprint occupancy classes. This footprint curation process executes the following steps:

   * **Curated parcel linking**: Curated attributes from the parcel curation lane (including ``improvement_value``, ``land_value``, ``year_built``, ``use_group_combined``, ``group_parcel``, ``manufactured_home_park``, ``occupancy_type_footprint_fema``, ``n_stories_footprint_fema``, and ``land_use_class``) are joined by matching the footprint's parcel reference (``parcel_id``) to the curated parcel's own globally unique ID (``parcel_id``). Note that the joined curated-parcel columns overwrite any raw harmonized ``_parcel`` evidence columns, initially repeating the undivided per-parcel totals across every footprint on the parcel before they are apportioned in the next step.
   * **Parcel value apportionment (apportion_parcel_values)**: Splits ``improvement_value_parcel`` across a multi-footprint parcel's dwelling-linked (or, absent those, all) primary footprints by floor-area share, while keeping ``land_value_parcel`` whole on the primary footprint only.
   * **Address evidence correction (suppress_where)**: Suppresses (nulls) ``n_dwellings_overture`` counts on parcels classified as :input:`Vacant` by the parcel land-use vote, filtering out pre-construction or platted address points that do not represent physical buildings.
   * **Implied Overture occupancy (resolve_by_vote)**: Resolves the transient implied occupancy type (``occupancy_type_dwelling_overture``) from corrected Overture counts alone (Single-Family for <= 1 dwelling, Multi-Family for >= 2).
   * **Attribute reconciliation**: Canonical attributes are resolved from competing evidence (such as prioritizing Overture dwelling counts over NSI, assessor construction year over NSI block-median, and assessor improvement value over NSI replacement value).
   * **Address count zero-filling (fill_missing_numeric)**: Fills missing or suppressed ``n_dwellings_overture`` counts with 0.
   * **Compute footprint metrics (derive_metrics)**: Calculates structural metrics like footprint area (``m2``) and structural improvement value per area metrics for all value columns (e.g., ``value_per_area``).
   * **Dwelling-count imputation (impute_n_dwellings)**: Fills missing residential unit counts when no source evidence exists.
   * **Base occupancy imputation**: Assigns base occupancy class via a weighted consensus vote across present evidence columns (NSI, FEMA, parcel, Overture), falling back sequentially to geometry and single-family dwelling checks where no evidence exists.
   * **Keyword corrections (resolve_occupancy)**: Refines occupancy classes using property-use keywords from ``parcel-occupancy-keywords.csv``.
   * **Merge visual predictions (merge_enrichments)**: Merges predicted building attributes from deep-learning visual classifiers.
   * **Story count reconciliation (reconcile_values)**: Resolves the canonical ``n_stories`` attribute, prioritizing street-level imagery predictions (``n_stories_brails``), then measured-height-derived counts (``n_stories_footprint_fema``), and lastly NSI modeled counts (``n_stories_building_nsi``).
   * **Manufactured-home probability scoring (classify_manufactured_homes)**: Scores manufactured home probabilities using rulesets.
   * **Occupancy voting**: Resolves ambiguous Manufactured Home vs. Multi-Family cases using a weighted point vote scoring system.
   * **Height-band splits**: Splits Multi-Family classes into HAZUS height bands (Low-Rise: 1-3, Mid-Rise: 4-7, High-Rise: 8+) based on ``n_stories``.
   * **Flag communities (flag_manufactured_home_communities)**: Flags parcels containing more than 3 final Manufactured Home footprints.
   * **Cast year built to integer (cast_integers)**: Converts year built columns to nullable integer type.
   * **Diagnostics & formatting**: Computes final diagnostic columns (e.g., value-per-area), flags occupancy conflicts, casts categoricals, and orders output columns.

Technical Annex
~~~~~~~~~~~~~~~

.. toctree::

   annexes/US_footprint-cheer-2026-annex.rst


Key references
~~~~~~~~~~~~~~

.. _Lochhead et al. 2026:

Lochhead M, Zsarnoczay A, Deierlein G (2026) Exposure matters: a synthesis framework for high-resolution building inventory development. International Journal of Disaster Risk Reduction 139 (2026): 106148. `doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_

.. _Cetiner et al. 2025:

Cetiner, B., McKenna, F., Yi, S.-ri, Wang, B., & Manousakis, I. V. (2025). BRAILS++ (v4.2.0). Zenodo. `doi:10.5281/zenodo.17797364 <https://doi.org/10.5281/zenodo.17797364>`_


