.. openplaces

.. _cheer_footprints_annex:

Technical annex
===============

A step-by-step description of the data pipeline that produces the canonical :ref:`CHEER footprint inventory <cheer_footprints>`, spanning from raw data ingestion to the final curation stage.

.. contents:: Table of contents
   :local:
   :depth: 3


Stage 1: ingest
---------------

This stage downloads and extracts raw footprint, parcel, and reference point datasets:

Precursor ingestion
~~~~~~~~~~~~~~~~~~~

1. **Administrative boundaries**

   Downloads US Census administrative boundaries (``US_admin-census-2021_admin2``, ``US_admin-census-2021_admin3``, and ``US_admin-census-2021_admin4``).

   *Dependency*: Foundational step. Microsoft footprint allocation requires Admin 2 (state) and Admin 3 (county) units, and image scraping recipes operate at Admin 4 (townships).

   *Function*: :func:`openplaces.io.ingester.ingest`

2. **OBM tile index and linking**

   Downloads OBM tile geometries (``tile-obm-2025``) and generates a precomputed tile-to-admin boundary link overlay.

   *Dependency*: Requires administrative boundaries to resolve the spatial overlay.

   *Function*: :func:`openplaces.geo.link.create_entity_link`

Core dataset ingestion
~~~~~~~~~~~~~~~~~~~~~~

3. **Base footprints**

   Downloads raw building geometries from OpenBuildingMap (OBM) 2025 (``footprint-obm-2025``), Microsoft v2 (``US_footprint-microsoft-v2``), and auto-discovered state/local GIS footprint layers.

   *Dependency*: Tile-partitioned footprints (OBM) depend on the tile index link to resolve tiles for the requested admin units.

   *Function*: :func:`openplaces.io.ingester.ingest`

4. **FEMA occupancy footprints**

   Downloads FEMA USA Structures footprints (``US_footprint-fema-2023``). These are ingested for parcel-occupancy linkage, not as a footprint-spine geometry source (due to intersection/IoU errors).

   *Dependency*: Spatial inputs are limited to requested admin unit IDs.

   *Function*: :func:`openplaces.io.ingester.ingest`

5. **Parcels and assessor rolls**

   Gathers assessor geometry and property tax rolls from state/local GIS agencies.

   *Function*: :func:`openplaces.io.ingester.ingest`

6. **Reference point databases**

   Downloads the National Structure Inventory (NSI) 2022 point database (``US_building-nsi-2022``) and Overture 2025 dwelling address points (``dwelling-overture-2025``).

   *Function*: :func:`openplaces.io.ingester.ingest`


Stage 2: harmonize
------------------

This stage merges geometries and links datasets to build the core entities.

Footprint spine harmonization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recipe: ``US_footprint-spine-2026``

1. **Merge source footprints**

   Combines raw footprint geometries from OBM, Microsoft, and local layers. It filters out shapes below the :input:`min_area_m2` threshold (10.0 m²), resolves duplicates using the :input:`overlap_iou_max` threshold (0.02), and filters parallel-shifted shapes using aspect ratio and angle tolerances via :input:`elongated_aspect_min` (2.5), :input:`elongated_angle_tol` (15.0), and :input:`elongated_long_overlap_min` (0.5) with lateral separation ratio :input:`elongated_lateral_sep_ratio` (2.0). To protect larger structures, it enforces keep-out buffers for features exceeding :input:`buffer_min_area_m2` (250.0 m²) with a keep-out distance calculated as: :input:`keep-out distance` = :input:`buffer_base_m` (2.0 m) + :input:`buffer_area_scale` (0.5) * sqrt(:input:`area_m2`).

   *Function*: :func:`openplaces.io.harmonizer.spine.resolve_spine`

2. **Intersect footprints with parcels**

   Performs a spatial identity overlay between footprints and parcels based on the :input:`join` method (``spatial_overlay``) and :input:`source_geometry_type` (``mixed_type_footprint``) for the :input:`entity_type` ``parcel``. It filters out minor intersections under :input:`area_intersection_m2_min` (10 m²) or slivers less than :input:`min_fraction_of_largest` (0.1667) of the largest parcel intersection, sorting by :input:`sort_by` (``area_intersection_m2``). To resolve systematic spatial displacements between footprint and parcel layers that inflate parcel counts, it snaps chain-displaced footprints to their dominant parcel when all minor overlaps are below the :input:`chain_fraction_max` (0.75) threshold and land on neighbor parcels that have their own building.

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

3. **Infer synthetic fallback footprints**

   Generates synthetic footprint geometries for the :input:`entity_type` ``parcel`` using parcel boundaries where tax assessor records indicate a structure exists but no footprint is detected by spatial sources. It uses the thresholds :input:`n_per_group_min` (0.2) and :input:`value_per_ha_quantile` (0.05).

   *Function*: :func:`openplaces.io.harmonizer.links.infer_spine_additions`

4. **Trim overlapping boundaries**

   Adjusts and trims geometry boundaries to resolve all remaining overlaps in the spine, preventing intersecting footprint representations.

   *Function*: :func:`openplaces.io.harmonizer.links.resolve_overlaps`

5. **Link building structure points**

   Connects structure-level point evidence from the NSI database using a tiered containment, :input:`proximity_m` (10 m) inner proximity, or :input:`far_proximity_m` (100 m) outer proximity join (:ref:`Lochhead et al. 2026 <Lochhead et al. 2026>`). The join is configured as :input:`join` (``spatial_point``) for the :input:`source_geometry_type` (``single_building_point``) using the :input:`recipe_id` (``US_building-nsi-2022``) with a remapping crosswalk specified by :input:`remap_id` (``US_building-nsi-2022_occupancy-type-remap``). It resolves and flags colocated duplicate points from low-rank sources (grouping by `building_id_ubid` and labeling low-rank twins from ``ESRI`` and ``HAZUS/NSI-2015`` as ``'colocated low-rank source'`` via :func:`~openplaces.io.harmonizer.links.flag_duplicate_points`) to be excluded from downstream aggregates.

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

6. **Link dwelling address points**

   Integrates Overture geocoded residential address points from the :input:`recipe_id` (``dwelling-overture-2025``) using a tiered proximity join configured via :input:`join` (``spatial_point``) for the :input:`source_geometry_type` (``single_dwelling_point``). It uses a proximity range from :input:`proximity_m` (10 m) to :input:`far_proximity_m` (50 m), with :input:`aggregate_multipoint` (:input:`true`) and address deduplication enabled via :input:`dedup_addresses` (:input:`true`).

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

7. **Classify structural role**

   Determines whether a footprint represents a primary or secondary structure on multi-building parcels based on NSI and Overture matches, using the :input:`entity_type` ``parcel``. Synthetic, parcel-derived fallback geometries are always classified as ``'primary'``.

   *Function*: :func:`openplaces.io.harmonizer.attributes.classify_footprint_priority`

8. **Package raw variables**

   Aggregates all joined source evidence from NSI, Overture, and parcels into intermediate columns on the footprint spine using the configured list of :input:`sources`. Unhandled numeric columns (such as NSI's ``n_stories`` and ``area_sqft``) are aggregated using the attribute registry's default function so they are carried onto the spine. Point reference records flagged by the duplicate resolution (where ``duplicate_resolution`` is non-null) are filtered out and excluded from all aggregates (match counts, sums, means, and value-weighted picks).

   *Function*: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

Parcel spine harmonization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Recipe: ``US_parcel-spine-2026``

1. **Establish parcel boundary baseline**

   Merges discovered statewide and local parcel geometry layers into a unified spatial spine, automatically discovering sources with the :input:`auto_discover` (:input:`true`) flag for the :input:`entity_type` ``parcel`` and keeping the configured list of :input:`keep_columns`.

   *Function*: :func:`openplaces.io.harmonizer.spine.resolve_spine`

2. **Merge assessor tax records**

   Discovers and joins county/local assessment tables by matching local ID keys, applying custom attribute remapping crosswalks. It automatically discovers and links tables for the :input:`entity_type` ``parcel`` using :input:`auto_discover` (:input:`true`).

   *Function*: :func:`openplaces.io.harmonizer.links.link_by_id`

3. **Standardize property use codes**

   Constructs a combined property use description and maps it to normalized use classifications.

   *Function*: :func:`openplaces.io.harmonizer.attributes.derive_use_classes`

4. **Associate building points to parcels**

   Joins NSI point data from the :input:`recipe_id` (``US_building-nsi-2022``) using a spatial overlay and proximity boundaries configured via :input:`join` (``spatial_point``), :input:`source_geometry_type` (``single_building_point``), :input:`proximity_m` (10 m), and :input:`far_proximity_m` (100 m). It resolves and flags colocated duplicate points from low-rank sources (grouping by `building_id_ubid` and flagging ``ESRI`` and ``HAZUS/NSI-2015`` twins via :func:`~openplaces.io.harmonizer.links.flag_duplicate_points`) to exclude them from downstream aggregates.

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

5. **Identify dominant building group**

   Resolves and summarizes NSI building attributes to find the modal building group per parcel using the specified :input:`sources`. Point records flagged by the duplicate resolution step are excluded from these summaries.

   *Function*: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

6. **Integrate FEMA footprint occupancy**

   Links FEMA footprints to parcels via spatial overlay from the :input:`recipe_id` (``US_footprint-fema-2023``) using the :input:`join` (``spatial_overlay``) method for the :input:`source_geometry_type` (``mixed_type_footprint``) and filters out minor intersections under :input:`area_intersection_m2_min` (10 m²).

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

7. **Extract dominant FEMA occupancy**

   Reconciles FEMA occupancy types from the :input:`sources` using the remapping crosswalk specified by :input:`remap_id` (``US_footprint-fema-2023_occupancy-type-remap``).

   *Function*: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

8. **Summarize footprint morphology**

   Counts total, primary, and small elongated footprint features on each parcel to feed downstream land-use classification, and tracks the maximum parcels spanned and maximum dwellings contained by any single footprint on the parcel. It links footprints from :input:`footprint_recipe_id` (``US_footprint-spine-2026``) matching on :input:`on` (``parcel_id``), filtering by :input:`small_area_max_m2` (185 m²), :input:`elongated_aspect_min` (2.0), and :input:`min_overlap_m2` (10 m²), computing ``max_dwellings_per_footprint`` and ``max_parcels_per_footprint`` for townhome and multi-family detection.

   *Dependency*: Has a **hard dependency** on Footprint Spine Harmonization, as it queries the completed footprint spine.

   *Function*: :func:`openplaces.io.harmonizer.attributes.summarize_footprint_morphology`


Stage 3: ingest images
----------------------

This stage fetches imagery required for deep-learning visual classification:

* **Satellite imagery** (``image-googlesatellite-z20.yaml``): Scrapes zoom-level 20 Google Satellite tiles using footprint geometries via BRAILS++.
* **Street View imagery** (``image-googlestreetview-2026.yaml``): Downloads street-level Google Street View photos and depth maps.

*Dependency*: Requires the completed footprint geometries from Stage 2 to calculate image scrape bounds.


Stage 4: enrich
---------------

This stage runs deep learning models (BRAILS++) to predict visual building attributes:

1. **Infer roof shape**

   Runs BRAILS++ deep learning classifiers on satellite imagery to predict roof shape, using visual models on the footprint spine (:ref:`Cetiner et al. 2025 <Cetiner et al. 2025>`).

   *Dependency*: Requires satellite imagery from Stage 3.

   *Function*: :func:`openplaces.io.enricher.attributes.classify_roof_shape`

2. **Detect story counts**

   Uses computer vision detectors on street-level photos to estimate floors of living area, predicting story height (:ref:`Cetiner et al. 2025 <Cetiner et al. 2025>`).

   *Dependency*: Requires Street View photos from Stage 3.

   *Function*: :func:`openplaces.io.enricher.attributes.detect_n_stories`


Stage 5: curate
---------------

This stage curates the spines into clean, canonical datasets.

Parcel curation
~~~~~~~~~~~~~~~

Recipe: ``US_parcel-openplaces-2026``

This stage curates the parcel spine to produce clean assessor attributes:

1. **Impute parcel occupancy group**

   Assigns groups based on NSI modal counts per use code using :input:`group_column` (``use_group_combined``), :input:`value_column` (``group_building_nsi``), the Mode statistic via :input:`statistic` (``mode``), and saving to :input:`output` (``group_parcel``).

   *Function*: :func:`openplaces.io.curator.imputers.impute_from_group_statistic`

2. **Score relative footprint area**

   Calculates log-space z-scores to assist Vacant and Townhome rule classification by measuring how anomalously small the largest footprint is relative to other parcels sharing the same assessor use code. It uses :input:`group_column` (``use_group_combined``), :input:`value_column` (``max_footprint_area_m2``), :input:`output` (``footprint_area_log_zscore``), :input:`transform` (``log1p``), and :input:`statistic` (``zscore``).

   *Function*: :func:`openplaces.io.curator.inferers.score_relative_to_group`

3. **Classify parcel land use**

   Assigns parcel land-use classes via weighted voting. It runs multi-indicator voting rules to identify Manufactured Home Park, RV Park, Standalone Manufactured Home, Townhome, Vacant, etc., saving to :input:`output` (``land_use_class``) with flags mapped to :input:`flag_column` (``manufactured_home_park``) and :input:`flag_class` (``Manufactured Home Park``), scoring columns via :input:`score_columns` (Vacant: ``land_use_vacancy_score``), review tracking on :input:`review_column` (``land_use_review``) with a margin of :input:`review_margin` (1.0) and the configured :input:`rules`. In particular, Townhome classification uses the new morphology indicators ``max_parcels_per_footprint`` and ``max_dwellings_per_footprint`` combined with assessor keyword matches to detect shared-footprint row houses.

   *Function*: :func:`openplaces.io.curator.inferers.classify_parcel_land_use`

4. **Reconcile default land use**

   Determines default land-use class for parcels not claimed by specific rules via consensus voting across three group-vocabulary columns (NSI, FEMA, and parcel use groups).

   *Function*: :func:`openplaces.io.curator.reconcilers.reconcile_land_use`

5. **Derive story count from height**

   Approximates the story count (``n_stories_footprint_fema``) as ``height / 3.05``, rounded and floored at one story, using the LiDAR-derived FEMA footprint height.

   *Function*: :func:`openplaces.io.curator.inferers.derive_stories_from_height`

6. **Standardize data categories**

   Casts string columns to pandas Categorical types to optimize storage footprint and query performance.

   *Function*: :func:`openplaces.io.curator.formatters.cast_categoricals`

7. **Format curated parcel schema**

   Enforces a standard column order on the final curated parcel schema.

   *Function*: :func:`openplaces.io.curator.formatters.order_columns`

Footprint curation
~~~~~~~~~~~~~~~~~~

Recipe: ``US_footprint-cheer-2026``

This stage curates the footprint spine, integrating parcel, imagery, and point evidence into the final exposure dataset. The curation recipe steps are chronologically executed, structured into functional sub-stages to highlight data-flow dependencies:

1. Input integration and value apportionment

   a. **Integrate clean assessor data**

      Matches each footprint in the spine to its corresponding parcel in the curated parcel lane using :input:`recipe_id` (``US_parcel-openplaces-2026``) and joins the configured :input:`columns` (improvement_value, land_value, year_built, use_group_combined, group_parcel, manufactured_home_park, occupancy_type_footprint_fema, n_stories_footprint_fema, and land_use_class). Note that the joined curated-parcel columns overwrite any raw harmonized ``_parcel`` evidence columns, initially repeating the undivided per-parcel totals across every footprint on the parcel before they are apportioned in the next step.

      *Dependency*: Requires the completed parcel curation lane.

      *Function*: :func:`openplaces.io.curator.evidence.link_curated_entity`

   b. **Apportion parcel values**

      Splits ``improvement_value_parcel`` across a multi-footprint parcel's dwelling-linked (or, absent those, all) primary footprints by floor-area share. ``land_value_parcel`` stays whole on the primary footprint only.

      *Dependency*: Requires curated parcel financial values integrated in Step 1.a.

      *Function*: :func:`openplaces.io.curator.evidence.apportion_curated_values`

   c. **Collect overlapping parcel IDs**

      Retains the full n:m footprint-parcel membership as a canonical column (`parcel_id_all`), pipe-joined with the dominant parcel first, from the overlay link sidecar.

      *Dependency*: Requires the spatial overlay from Harmonization.

      *Function*: :func:`openplaces.io.curator.evidence.collect_link_ids`

2. Dwelling and address evidence prep

   a. **Correct address evidence**

      Suppresses Overture dwelling unit counts on vacant parcels. It sets :input:`column` (``n_dwellings_overture``) to null if the :input:`condition_column` (``land_use_class_parcel``) matches the value :input:`condition_value` (:input:`Vacant`).

      *Dependency*: Requires curated parcel land-use classification integrated in Step 1.a.

      *Function*: :func:`openplaces.io.curator.reconcilers.suppress_where`

   b. **Determine implied Overture occupancy**

      Assigns a temporary occupancy class to :input:`target` (``occupancy_type_dwelling_overture``) based on the corrected Overture count using the configured :input:`decisions` rules.

      *Dependency*: Requires corrected address evidence from Step 2.a.

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

3. Core value and metric reconciliation

   a. **Select canonical values**

      Resolves conflicts between competing source attributes by selecting the canonical value from the prioritized lists in the :input:`priority` mapping (e.g. for dwelling counts, year built, and financial valuation).

      *Dependency*: Requires apportioned parcel values (Step 1.b) and implied Overture occupancy (Step 2.b).

      *Function*: :func:`openplaces.io.curator.reconcilers.reconcile_values`

   b. **Zero-fill address counts**

      Fills missing or suppressed Overture dwelling unit counts listed in :input:`columns` (``[n_dwellings_overture]``) with ``0`` and casts the column to integer.

      *Dependency*: Requires reconciled dwelling counts from Step 3.a.

      *Function*: :func:`openplaces.io.curator.imputers.fill_missing_numeric`

   c. **Compute footprint metrics**

      Calculates structural indicators such as footprint area in square meters from geometry and computes structural value-per-area metrics for all value columns. Note that ``value_per_area`` is kept in the final output.

      *Dependency*: Requires canonical values from Step 3.a and footprint geometries.

      *Function*: :func:`openplaces.io.curator.inferers.derive_metrics`

   d. **Impute missing residential units**

      Imputes residential unit counts when no matched source evidence exists based on the occupancy base class.

      *Dependency*: Requires reconciled dwelling counts from Step 3.a.

      *Function*: :func:`openplaces.io.curator.imputers.impute_n_dwellings`

4. Occupancy consensus and correction

   a. **Establish baseline occupancy class**

      Resolves a weighted consensus vote across present evidence columns (NSI, FEMA, parcel, Overture) where the heaviest class wins and Overture has a fractional (0.5) vote weight; (2) applies a geometry-based Manufactured Home fallback for long/narrow structures; (3) fills residential gaps using a single-family dwelling count check; and (4) assigns accessory structures to the Secondary class (excluding habitable structures in manufactured home parks).

      *Dependency*: Requires joined parcel occupancy groups (Step 1.a) and implied Overture occupancy (Step 2.b).

      *Function*: :func:`openplaces.io.curator.inferers.impute_occupancy_type`

   b. **Apply property-use keyword corrections**

      Refines baseline occupancy by correcting classes using property-use keywords from the ruleset (only rules marked reviewed can override the base class, and they never override Secondary). It also writes the ``occupancy_type_parcel`` column, sets the ``occupancy_type_review`` flag for low improvement-value shares, generates the ``occupancy_type_conflict`` summary, and outputs a county-level conflicts CSV report.

      *Dependency*: Requires baseline occupancy class from Step 4.a.

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_occupancy`

5. Imagery enrichment integration

   a. **Merge visual model predictions**

      Merges predicted building attributes from the configured :input:`recipes` (e.g. ``US_footprint_built-roof-shape-brails-2026`` and ``US_footprint_built-n-stories-brails-2026``) and their respective columns.

      *Dependency*: Requires completed Stage 4 (Visual enrichment).

      *Function*: :func:`openplaces.io.curator.evidence.merge_enrichments`

   b. **Reconcile story counts**

      Resolves conflicts between competing story count sources (street-level imagery predictions ``n_stories_brails``, FEMA height-derived counts ``n_stories_footprint_fema``, and NSI block-median counts ``n_stories_building_nsi``) by selecting the canonical value.

      *Dependency*: Requires merged visual predictions from Step 5.a.

      *Function*: :func:`openplaces.io.curator.reconcilers.reconcile_values`

6. Final occupancy voting and refinement

   a. **Score manufactured home probability**

      Computes a probability score for manufactured home classification based on the :input:`ruleset` (``parcel-occupancy-keywords.csv``) while setting the :input:`update_occupancy` parameter to :input:`false`.

      *Dependency*: Requires footprint metrics from Step 3.c.

      *Function*: :func:`openplaces.io.curator.inferers.classify_manufactured_homes`

   b. **Resolve occupancy by weighted vote**

      Resolves final occupancy class (specifically Manufactured Home vs. Multi-Family conflicts) to the :input:`target` column (``occupancy_type``) using the configured :input:`decisions` rules. Footprints under 20 m² are prevented from becoming Manufactured Home by a hard precondition, and the winning decision records its source label into ``occupancy_type_source``.

      *Dependency*: Requires reconciled dwelling counts (Step 3.a) and manufactured home probability (Step 6.a).

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

   c. **Split height bands**

      Splits the standard :input:`multi_family_class` (``Multi-Family``) into HAZUS height bands based on the reconciled number of stories using the configured :input:`bands`.

      *Dependency*: Requires final occupancy (Step 6.b) and reconciled story counts (Step 5.b).

      *Function*: :func:`openplaces.io.curator.inferers.refine_occupancy_height`

   d. **Flag manufactured home communities**

      Re-evaluates mobile home park boundaries and flags parcels containing more than :input:`min_homes` (3) final Manufactured Home footprints (i.e., 4 or more). It writes the count to ``n_manufactured_homes_per_parcel`` and the boolean flag to ``manufactured_home_community``.

      *Dependency*: Requires final occupancy classification from Step 6.b.

      *Function*: :func:`openplaces.io.curator.inferers.flag_manufactured_home_communities`

7. Schema standardization and formatting

   a. **Standardize data categories**

      Converts string columns to pandas Categorical types.

      *Function*: :func:`openplaces.io.curator.formatters.cast_categoricals`

   b. **Cast year built to integer**

      Rounds and casts the year of construction columns listed in :input:`columns` (``[year_built]``) to nullable integer data types.

      *Function*: :func:`openplaces.io.curator.formatters.cast_integers`

   c. **Clean up and order columns**

      Enforces standard column order and drops transient helper columns (including ``group_parcel``, ``occupancy_type_dwelling_overture``, ``occupancy_type_dwelling_overture_source``, ``occupancy_type_base``, ``p_manufactured_home``, ``manufactured_home_park``, and ``n_stories_brails_source``).

      *Function*: :func:`openplaces.io.curator.formatters.order_columns`
