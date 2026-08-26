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

   Downloads the National Structure Inventory (NSI) 2026 point database (``US_building-nsi-2026``) and Overture 2025 dwelling address points (``dwelling-overture-2025``).

   *Function*: :func:`openplaces.io.ingester.ingest`

7. **Census statistical geographies**

   Downloads Census tract, block group, block, and ZCTA5 boundaries (``US_tile-census-2025_tract``, ``_blockgroup``, ``_block``, ``_zcta5``) and county subdivisions (``US_admin-census-2025_admin4``).

   *Dependency*: Consumed by ``link_geographic_ids`` in Stage 2.

   *Function*: :func:`openplaces.io.ingester.ingest`

8. **Modeled building inventory** (regional, optional)

   Registers a precomputed building inventory as a ``building`` entity - for North Carolina, CHEER Inventory v0 (``US-NC_building-cheer-v0``), carrying modeled roof shape (three ranked classes with confidences), foundation, construction type, garage flag, and story count.

   The source is a CHEER/SimCenter deliverable rather than a public download, so the file is placed by hand in the recipe's external directory; the recipe declares only ``uncompressed_file_name``. It is read in one statewide pass and split into per-county outputs on save, because the parquet reader cannot chunk by an in-data admin column. The roof-shape column carries the non-classes ``Unknown`` and ``No Structure`` instead of nulls, which are nulled on ingest so they cannot travel downstream as if they were roof shapes.

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

5. **Derive geometry attributes**

   Computes each spine row's own centroid (``lat``/``long``) and area once, immediately after the geometry is final (synthetic fallbacks and overlap trimming included), so later harmonize and curate steps reuse them instead of recomputing.

   *Function*: :func:`openplaces.io.harmonizer.spine.derive_geometry_attributes`

6. **Assign containing-area identifiers**

   Stamps each footprint with the id of every reference polygon containing it: the level-4 admin unit (``admin4_id``), the Census county subdivision (``census_subdivision_id``), tract, block group, block, and the 5-digit ZCTA (``zcta5_id``). Each entry in the step's :input:`links` list names either an :input:`admin_level` or a :input:`recipe_id`, plus the :input:`id_column` to copy and the :input:`output_column` to write, so the reference set is recipe configuration rather than a fixed list.

   Two passes, in this order: a point-in-polygon test on the row's own centroid, then - only for rows that missed - an identity overlay of the row's actual geometry, keeping the reference polygon with the largest intersection. Rows matched by neither are left null, so a coverage gap stays visible instead of being masked. Unlike ``link_to_reference``, every configured reference is assumed to tile space without overlaps, so the result is exactly one containing polygon per row, never a many-to-many crosswalk.

   *Dependency*: Requires ``lat``/``long`` from Step 5. Runs on the footprint spine first so the parcel spine can inherit the result (see that recipe's ``inherit_from``).

   *Function*: :func:`openplaces.io.harmonizer.spine.link_geographic_ids`

7. **Link building structure points**

   Connects structure-level point evidence from the NSI database using a tiered containment, :input:`proximity_m` (10 m) inner proximity, or :input:`far_proximity_m` (100 m) outer proximity join (:ref:`Lochhead et al. 2026 <Lochhead et al. 2026>`). The join is configured as :input:`join` (``spatial_point``) for the :input:`source_geometry_type` (``single_building_point``) using the :input:`recipe_id` (``US_building-nsi-2026``) with a remapping crosswalk specified by :input:`remap_id` (``US_building-nsi-2026_occupancy-type-remap``). It resolves and flags colocated duplicate points from low-rank sources (grouping by `building_id_ubid` and labeling low-rank twins from ``ESRI`` and ``HAZUS/NSI-2015`` as ``'colocated low-rank source'`` via :func:`~openplaces.io.harmonizer.links.flag_duplicate_points`) to be excluded from downstream aggregates.

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

8. **Link dwelling address points**

   Integrates Overture geocoded residential address points from the :input:`recipe_id` (``dwelling-overture-2025``) using a tiered proximity join configured via :input:`join` (``spatial_point``) for the :input:`source_geometry_type` (``single_dwelling_point``). It uses a proximity range from :input:`proximity_m` (10 m) to :input:`far_proximity_m` (50 m), with :input:`aggregate_multipoint` (:input:`true`) and address deduplication enabled via :input:`dedup_addresses` (:input:`true`).

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

9. **Classify structural role**

   Determines whether a footprint represents a primary or secondary structure on multi-building parcels based on NSI and Overture matches, using the :input:`entity_type` ``parcel``. Synthetic, parcel-derived fallback geometries are always classified as ``'primary'``.

   *Function*: :func:`openplaces.io.harmonizer.attributes.classify_footprint_priority`

10. **Package raw variables**

   Aggregates all joined source evidence from NSI, Overture, and parcels into intermediate columns on the footprint spine using the configured list of :input:`sources`. Unhandled numeric columns (such as NSI's ``n_stories`` and ``area_sqft``) are aggregated using the attribute registry's default function so they are carried onto the spine. Point reference records flagged by the duplicate resolution (where ``duplicate_resolution`` is non-null) are filtered out and excluded from all aggregates (match counts, sums, means, and value-weighted picks).

   NSI's ``foundation_type`` is among the columns carried across, as ``foundation_type_building_nsi``. NSI assigns foundation from regional and flood-zone rules, so it is a prior rather than a per-building observation - but a stable one, which a per-building classifier is not guaranteed to be. That is why it leads the foundation reconciliation in curation.

   *Function*: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

Parcel spine harmonization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Recipe: ``US_parcel-spine-2026``

1. **Establish parcel boundary baseline**

   Merges discovered statewide and local parcel geometry layers into a unified spatial spine, automatically discovering sources with the :input:`auto_discover` (:input:`true`) flag for the :input:`entity_type` ``parcel`` and keeping the configured list of :input:`keep_columns`.

   *Function*: :func:`openplaces.io.harmonizer.spine.resolve_spine`

2. **Assign containing-area identifiers**

   Writes the same admin and Census identifiers the footprint spine resolves (``admin4_id``, ``census_subdivision_id``, tract, block group, block, ``zcta5_id``), but takes them second-hand where it can: :input:`inherit_from` names the footprint spine and a :input:`group_by` key (``parcel_id``), and each parcel adopts a value only where **every** footprint on it agrees. Disagreeing groups, and parcels with no footprint at all, are left null and resolved by the same two-pass centroid-then-overlay join the footprint spine uses. Disagreements are recorded in the run's metadata rather than silently averaged away.

   *Dependency*: Requires the completed footprint spine, which already runs first for :func:`~openplaces.io.harmonizer.attributes.summarize_footprint_morphology`.

   *Function*: :func:`openplaces.io.harmonizer.spine.link_geographic_ids`

3. **Merge assessor tax records**

   Discovers and joins county/local assessment tables by matching local ID keys, applying custom attribute remapping crosswalks. It automatically discovers and links tables for the :input:`entity_type` ``parcel`` using :input:`auto_discover` (:input:`true`).

   *Function*: :func:`openplaces.io.harmonizer.links.link_by_id`

4. **Standardize property use codes**

   Joins an ordered :input:`columns` list into the combined use label (``use_group_combined``) that the parcel land-use classifier groups and votes on. The default pair is ``use_group`` + ``use_subgroup``; this recipe appends ``building_style`` last.

   The reason is that in some counties the occupancy signal is in neither land-use field. Sampson County, NC is the measured case: its ``use_group`` is a land *segment* type (homesite, cropland, woodland) matching no land-use keyword at all, while its style column names the structure and identifies thousands of manufactured homes. Adding it took the share of parcels receiving a keyword class from 0% to 15%. Listed last, it only ever extends the assessor's own label rather than displacing it.

   The combined label is also the grouping key for cohort statistics such as ``footprint_area_log_zscore``, so a high-cardinality column added here fragments those cohorts - a reason to measure before adding another one.

   *Function*: :func:`openplaces.io.harmonizer.attributes.derive_use_classes`

5. **Associate building points to parcels**

   Joins NSI point data from the :input:`recipe_id` (``US_building-nsi-2026``) using a spatial overlay and proximity boundaries configured via :input:`join` (``spatial_point``), :input:`source_geometry_type` (``single_building_point``), :input:`proximity_m` (10 m), and :input:`far_proximity_m` (100 m). It resolves and flags colocated duplicate points from low-rank sources (grouping by `building_id_ubid` and flagging ``ESRI`` and ``HAZUS/NSI-2015`` twins via :func:`~openplaces.io.harmonizer.links.flag_duplicate_points`) to exclude them from downstream aggregates.

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

6. **Identify dominant building group**

   Resolves and summarizes NSI building attributes to find the modal building group per parcel using the specified :input:`sources`. Point records flagged by the duplicate resolution step are excluded from these summaries.

   *Function*: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

7. **Integrate FEMA footprint occupancy**

   Links FEMA footprints to parcels via spatial overlay from the :input:`recipe_id` (``US_footprint-fema-2023``) using the :input:`join` (``spatial_overlay``) method for the :input:`source_geometry_type` (``mixed_type_footprint``) and filters out minor intersections under :input:`area_intersection_m2_min` (10 m²).

   *Function*: :func:`openplaces.io.harmonizer.links.link_to_reference`

8. **Extract dominant FEMA occupancy**

   Reconciles FEMA occupancy types from the :input:`sources` using the remapping crosswalk specified by :input:`remap_id` (``US_footprint-fema-2023_occupancy-type-remap``).

   *Function*: :func:`openplaces.io.harmonizer.attributes.reconcile_attributes`

9. **Summarize footprint morphology**

   Counts total, primary, and small elongated footprint features on each parcel to feed downstream land-use classification, and tracks the maximum parcels spanned and maximum dwellings contained by any single footprint on the parcel. It links footprints from :input:`footprint_recipe_id` (``US_footprint-spine-2026``) matching on :input:`on` (``parcel_id``), filtering by :input:`small_area_max_m2` (185 m²), :input:`elongated_aspect_min` (2.0), and :input:`min_overlap_m2` (10 m²), computing ``max_dwellings_per_footprint`` and ``max_parcels_per_footprint`` for townhome and multi-family detection.

   *Dependency*: Has a **hard dependency** on Footprint Spine Harmonization, as it queries the completed footprint spine.

   *Function*: :func:`openplaces.io.harmonizer.attributes.summarize_footprint_morphology`


Stage 3: enrich
---------------

This stage produces entity-keyed evidence without selecting any canonical value. Two routes reach the same attributes: running the visual models here (Steps 1-2), or reading them off an inventory that already ran them (Step 3).

1. **Infer roof shape**

   Fetches a zoom-level 20 Google Satellite image per building (``image-googlesatellite-z20.yaml``) and runs a BRAILS++ classifier over it to predict roof shape (:ref:`Cetiner et al. 2025 <Cetiner et al. 2025>`). The image is held in memory for the inference call and never written to disk: Google's Static API policy prohibits storing or caching content.

   *Dependency*: Requires the completed footprint geometries from Stage 2, which define the area photographed.

   *Function*: :func:`openplaces.io.enricher.attributes.classify_roof_shape`

2. **Detect story counts**

   Fetches street-level Google Street View photos (``image-googlestreetview-2026.yaml``) and runs detectors over them to estimate floors of living area (:ref:`Cetiner et al. 2025 <Cetiner et al. 2025>`). The EfficientDet inference engine is not bundled (its upstream is LGPL-3.0, incompatible with this repository's license), so this step raises unless a replacement is supplied; ``n_stories`` is still produced from NSI and from a modeled inventory.

   *Dependency*: Requires the completed footprint geometries from Stage 2.

   *Function*: :func:`openplaces.io.enricher.attributes.detect_n_stories`

3. **Attach modeled inventory attributes**

   Recipe: ``US-NC_footprint_building-cheer-v0``

   Matches each footprint to the single reference building it overlaps most and copies that building's declared :input:`columns` across as ``{column}_building_cheer`` evidence.

   Ranking is by intersection-over-union, not raw intersection area. The two sides are independent renderings of the same buildings, and a large reference building clipping the corner of a small footprint shares more area with it than the correct small building does; only normalizing by the union rejects that. :input:`min_iou` defaults to a permissive 0.1, because roof-versus-wall outlines and building-versus-complex splits routinely put a correct pair well below a half.

   Two details worth knowing. First, the inventory's roof-shape confidences survive rows whose class was nulled at ingest; :input:`null_with` gates each confidence on its own class so a confidence never reads as certainty about a class that was dropped. Second, an admin unit the reference does not cover still gets the declared columns written as all-null - curate skips a missing evidence file, but treats a present one lacking a declared column as a recipe error, so a column-less table would poison every later curate for that unit.

   *Dependency*: Requires the ingested reference building entity (Stage 1) and a spine loaded with geometry (:input:`spine_geom` (:input:`true`)).

   *Function*: :func:`openplaces.io.enricher.buildings.enrich_footprints_from_reference_buildings`


Stage 4: curate
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

Recipe: ``US_footprint-openplaces-2026``

This stage curates the footprint spine, integrating parcel, imagery, and point evidence into the final exposure dataset. The curation recipe steps are chronologically executed, structured into functional sub-stages to highlight data-flow dependencies:

1. Input integration and value apportionment

   a. **Integrate clean assessor data**

      Matches each footprint in the spine to its corresponding parcel in the curated parcel lane using :input:`recipe_id` (``US_parcel-openplaces-2026``) and joins the configured :input:`columns` (use_group_combined, group_parcel, manufactured_home_community, group_footprint_fema, and land_use_class). Note that the joined curated-parcel columns overwrite any raw harmonized ``_parcel`` evidence columns.

      *Dependency*: Requires the completed parcel curation lane.

      *Function*: :func:`openplaces.io.curator.evidence.link_curated_entity`

   b. **Apportion parcel values**

      Joins property valuation and construction year columns from the curated parcel lane, then: (1) splits ``improvement_value_parcel`` across a multi-footprint parcel's dwelling-linked (or, absent those, all) primary footprints by floor-area share; (2) keeps ``land_value_parcel`` whole on the principal footprint only; and (3) assigns the average ``year_built_parcel`` to all linked footprints.

      *Dependency*: Requires the completed parcel curation lane.

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

   b. **Reconcile street addresses**

      Reconciles and harmonizes street addresses from competing source inputs (parcel assessor addresses and Overture dwelling address components), completing missing components (like state names derived from administrative units) and checking for consistency.

      *Dependency*: Requires integrated assessor data (Step 1.a) and Overture address evidence.

      *Function*: :func:`openplaces.io.curator.reconcilers.reconcile_addresses`

   c. **Zero-fill address counts**

      Fills missing or suppressed Overture dwelling unit counts listed in :input:`columns` (``[n_dwellings_overture]``) with ``0`` and casts the column to integer.

      *Dependency*: Requires reconciled dwelling counts from Step 3.a.

      *Function*: :func:`openplaces.io.curator.imputers.fill_missing_numeric`

   d. **Compute footprint metrics**

      Calculates structural indicators such as footprint area in square meters from geometry and computes structural value-per-area metrics for all value columns. Note that ``structure_value_per_area`` is kept in the final output.

      *Dependency*: Requires canonical values from Step 3.a and footprint geometries.

      *Function*: :func:`openplaces.io.curator.inferers.derive_metrics`

   e. **Derive named indicator columns**

      Computes the named precursor columns the occupancy votes score against: the metric minimum-bounding-rectangle ``aspect_ratio``; each occupancy source coerced to the canonical class vocabulary (``occupancy_type_nsi_class``, ``occupancy_type_fema_class``, and the reviewed-only ``occupancy_keyword_class``); and ``habitable_size_ratio``, each footprint's area relative to a locally derived habitable-size threshold. Indicator columns hold measurements or labels, never pre-thresholded booleans, as every cutoff lives in the vote decisions, so each threshold is stated exactly once in the recipe.

      *Dependency*: Requires footprint metrics from Step 3.d and joined parcel evidence (Step 1.a).

      *Function*: :func:`openplaces.io.curator.inferers.derive_indicators`

   f. **Impute missing residential units**

      Imputes residential unit counts when no matched source evidence exists, falling back to the first available raw ``occupancy_type*`` or ``use_subgroup*`` evidence column (such as ``occupancy_type_building_nsi``).

      *Dependency*: Requires reconciled dwelling counts from Step 3.a.

      *Function*: :func:`openplaces.io.curator.imputers.impute_n_dwellings`

4. Occupancy consensus and correction

   a. **Establish baseline occupancy class**

      Resolves a weighted consensus vote across present evidence columns (NSI, FEMA, parcel, Overture) where the heaviest class wins and Overture has a fractional (0.5) vote weight. The geometry-based Manufactured Home fallback, the single-family dwelling gap-fill, and the Secondary demotion this step once performed are retired: those calls now belong to the two-question vote in Section 6, where each is an explicit, weighted decision.

      *Dependency*: Requires joined parcel occupancy groups (Step 1.a) and implied Overture occupancy (Step 2.b).

      *Function*: :func:`openplaces.io.curator.inferers.impute_occupancy_type`

   b. **Apply property-use keyword corrections**

      Refines baseline occupancy by correcting classes using property-use keywords from the ruleset (only rules marked reviewed can override the base class, and they never override Secondary). It also writes the ``occupancy_type_parcel`` column, sets the ``occupancy_type_review`` flag for low improvement-value shares, generates the ``occupancy_type_conflict`` summary, and outputs a county-level conflicts CSV report.

      *Dependency*: Requires baseline occupancy class from Step 4.a.

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_occupancy`

5. Enrichment integration

   a. **Merge enrichment evidence**

      Merges predicted building attributes from the configured :input:`recipes` and their respective columns: ``US_footprint_built-roof-shape-brails-2026``, ``US_footprint_built-n-stories-brails-2026``, and - where it covers the admin unit - ``US-NC_footprint_building-cheer-v0``. A recipe with no evidence for the admin unit is skipped, so the inventory entry changes nothing outside North Carolina.

      ``record_source`` is set only on the roof-shape enrichment, which writes a direct canonical column with a competing source to reconcile against. The inventory's columns are single-source or already reconciled elsewhere, so a provenance sidecar there would be bookkeeping with nothing to record.

      The inventory's roof-shape ranks and confidences, construction type, and garage flag land directly on their canonical names. Its roof shape, foundation, and story count are deliberately kept under evidence names (``roof_shape_building_cheer``, ``foundation_type_building_cheer``, ``n_stories_building_cheer``) because each has a competing source reconciled in Step 5.b.

      *Dependency*: Requires completed Stage 4 (enrichment).

      *Function*: :func:`openplaces.io.curator.evidence.merge_enrichments`

   b. **Reconcile enrichment evidence into canonical values**

      Resolves three sets of competing sources by priority:

      ``n_stories``
         Street-level floor detection (``n_stories_brails``), then NSI's modeled count (``n_stories_building_nsi``), then the inventory's (``n_stories_building_cheer``) - last because it shares NSI's modeled lineage rather than being an independent observation.

      ``roof_shape``
         This repository's own per-image classification, then the inventory's. Both are BRAILS++ outputs, so this is a coverage cascade rather than a quality ranking: whichever actually ran for the admin unit wins.

      ``foundation_type``
         NSI first, the inventory only where NSI is silent. This is not a blanket judgement on the inventory - the two agree on about 65% of 1.57M buildings across 45 NC counties, and the inventory's classes are informative there. It guards against a failure mode NSI's regional prior cannot have: the inventory's per-county class shares swing from a few percent to over 90% Basement, and in New Hanover County 84% of buildings come back Basement with essentially no lift over the marginal - the class collapsed, on a coast where basements are near-absent.

      Known residue: roughly 2% of rows carry a joined value such as :input:`'Slab; Crawl'`, where several NSI points land on one footprint with differing foundations. ``reconcile_attributes`` joins every string column it is not specifically handling, honoring the attribute registry's per-attribute aggregation for numeric columns only - the same reason ``address_number_dwelling_overture`` and ``city_dwelling_overture`` carry joined values.

      *Dependency*: Requires merged evidence from Step 5.a.

      *Function*: :func:`openplaces.io.curator.reconcilers.reconcile_values`

6. Final occupancy voting and refinement

   a. **Score manufactured home probability**

      Computes a probability score for manufactured home classification based on the :input:`ruleset` (``parcel-occupancy-keywords.csv``) while setting the :input:`update_occupancy` parameter to :input:`false`.

      *Dependency*: Requires footprint metrics from Step 3.c.

      *Function*: :func:`openplaces.io.curator.inferers.classify_manufactured_homes`

   b. **Vote dwelling multiplicity**

      The first of two questions: does this footprint hold one dwelling or several? A weighted vote over Overture dwelling counts, assessor keywords, and the NSI/FEMA classes writes ``multi`` or ``single`` to the intermediate ``dwelling_multiplicity`` column. Every manufactured-home signal also scores as single-dwelling evidence here (a manufactured home is by definition one dwelling), which is precisely the information a flat three-way vote discarded by treating the classes as competitors.

      *Dependency*: Requires derived indicators (Step 3.e), reconciled dwelling counts (Step 3.a), and manufactured home probability (Step 6.a).

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

   c. **Vote occupancy class**

      The second question, gated by the first: among single-dwelling footprints, manufactured or site-built? Each decision carries a ``require`` list referencing ``dwelling_multiplicity``, so the three classes partition cleanly instead of competing across populations. Footprints under 20 m² are prevented from becoming Manufactured Home by a hard precondition, Single-Family is the residual of the single-dwelling group, accessory structures resolve to Secondary via the ``priority_on_parcel`` gate, and the winning decision records its source label into ``occupancy_type_source``.

      *Dependency*: Requires dwelling multiplicity (Step 6.b).

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

   d. **Split height bands**

      Splits ``Multi-Family`` into HAZUS height bands with a third vote over the reconciled story count: three overlapping story-count bounds plus earliest-listed-wins form a cascade, the pre-refinement class is snapshotted to ``occupancy_type_base``, and a footprint with no story count keeps plain ``Multi-Family``.

      *Dependency*: Requires final occupancy (Step 6.c) and reconciled story counts (Step 5.b).

      *Function*: :func:`openplaces.io.curator.reconcilers.resolve_by_vote`

   e. **Flag manufactured home communities**

      Re-evaluates mobile home park boundaries and flags parcels containing more than :input:`min_homes` (3) final Manufactured Home footprints (i.e., 4 or more). It writes the count to ``n_manufactured_homes_per_parcel`` and the boolean flag to ``manufactured_home_community``, deliberately overwriting the parcel-lane seed joined in Step 1.a (the final value supersedes the seed).

      *Dependency*: Requires final occupancy classification from Step 6.c.

      *Function*: :func:`openplaces.io.curator.inferers.flag_manufactured_home_communities`

7. Schema standardization and formatting

   a. **Standardize data categories**

      Converts string columns to pandas Categorical types.

      *Function*: :func:`openplaces.io.curator.formatters.cast_categoricals`

   b. **Cast year built to integer**

      Rounds and casts the year of construction columns listed in :input:`columns` (``[year_built]``) to nullable integer data types.

      *Function*: :func:`openplaces.io.curator.formatters.cast_integers`

   c. **Clean up and order columns**

      Enforces standard column order and drops transient helper columns (including ``group_parcel``, ``occupancy_type_dwelling_overture``, ``occupancy_type_dwelling_overture_source``, ``occupancy_type_base``, ``p_manufactured_home``, ``postal_city``, ``postal_zip5``, ``postal_city_acceptable``, ``postal_city_unacceptable``, ``postal_city_source``, and ``improvement_value_parcel_per_area``).

      *Function*: :func:`openplaces.io.curator.formatters.order_columns`
