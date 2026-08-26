.. openplaces

.. _cheer_footprints:

U.S. footprint inventory (CHEER)
================================

.. image:: ../../1_overview/concepts/images/footprint_building_dwelling_urban.png
  :width: 400
  :alt: Illustration of footprints, buildings, and dwellings (AI generated)
  :align: right

This recipe creates a footprint-level building inventory for hurricane damage and exposure modeling in the U.S.

It **harmonizes** building :ref:`footprints <footprints>` from multiple polygon sources (OpenBuildingMap, Microsoft, and local datasets), then enriches each footprint with data from linked parcels, the National Structure Inventory (NSI), Overture :ref:`dwelling <dwellings>` addresses, and FEMA's USA Structures.

The pipeline can also use deep-learning-based recognition of building features (roof shapes, story counts) from Google Satellite and Street View imagery (similar to BRAILS++). By default, these steps are turned off to avoid costly downloads and GPU use.

The pipeline is being tested in Central Florida, Eastern North Carolina, and Coastal Texas.

This work is supported by NSF's Coastal Hazards, Economic Prosperity & Resilience hub (`CHEER <https://www.drc.udel.edu/cheer/>`_).


Data description
~~~~~~~~~~~~~~~~

Two inventories have been shared with the CHEER team: an Eastern North Carolina delivery with 2,463,326 footprints (44 counties) and a Coastal Texas delivery with 4,118,698 footprints (42 counties).

These are fully processed **pre-imputation** inventories. They contain all available input data from the original sources, and some attributes have been pre-processed with careful, rule-based imputations (e.g., for condos, which often don't provide a separate measure of land value). However, the inventories still contain many empty values of key fields: a preliminary imputation run is expected to close this  soon (Askari).

In North Carolina, a legacy version of CHEER's v0 inventory supplies statewide roof shape, foundation, construction type, garage, and story evidence in North Carolina. (No other deep-learning-based recognition of roof shapes, story counts, etc. has been run.)

Each folder contains:

.. list-table::
   :header-rows: 1
   :widths: 30 12 12 46

   * - File
     - Eastern NC (44 counties)
     - Coastal TX (42 counties)
     - Holds
   * - :file:`{stem}.parquet`
     - 114 MB
     - 187 MB
     - The canonical table: one row per footprint, no geometry.
   * - :file:`{stem}_point.parquet`
     - 205 MB
     - 338 MB
     - The same canonical columns on centroid points, plus added columns for quick mapping in QGIS.
   * - :file:`{stem}_geo.parquet`
     - 312 MB
     - 497 MB
     - The footprint polygons.
   * - :file:`{stem}_evidence.parquet`
     - 356 MB
     - 529 MB
     - The full evidence behind the canonical values. All columns obtained from the original sources (some of them attributed / ascribed)
   * - :file:`{stem}_map.qgz`
     - 374 KB
     - 341 KB
     - A portable QGIS project over the bundle: centroid points on a
       neutral basemap by default, ready-made polygon views of every
       canonical classification, administrative outlines and labels,
       and web basemaps.
   * - :file:`{stem}_admin3_geo.parquet` /
       :file:`{stem}_admin4_geo.parquet`
     - 3.4 / 9.4 MB
     - 2.3 / 5.7 MB
     - County and county-subdivision outlines for the region.
   * - :file:`{stem}_LICENSE.txt`
     - 6 KB
     - 2 KB
     - The license and attribution terms the bundle ships under.

The four first :file:`.parquet` tables carry the same ``footprint_id`` index in the same row order, so any two of them join one-to-one.

To open the files by hand instead of going through the template map, use the :gui:`Load joined openplaces parquet files` processing algorithm in the toolbox. Pick any of the four files, choose boundary polygons or centroid points, and tick :gui:`Join evidence columns` when needed. Joining the full table will usually take a while (slow).

Processing pipeline
~~~~~~~~~~~~~~~~~~~

.. image:: US_footprint-cheer-pipeline.png
  :width: 300
  :alt: Illustration of pipeline
  :align: right

The creation of the CHEER dataset proceeds through the core stages of the ``openplaces`` pipeline: **Ingest**, **Harmonize**, **Enrich**, and **Curate**. Imagery has no ingest stage of its own: Google's Static API policy prohibits storing or caching content, so the enrichment steps that need imagery fetch it in memory and discard it.

For the precise step-by-step implementation rules, threshold parameters, and Python function bindings of each stage, see the :ref:`Technical Annex <cheer_footprints_annex>`.

Stage 1: ingest
---------------
Downloads, extracts, and resolves raw boundary, tile, geometry, and reference point datasets.
* **Precursor datasets**: Downloads U.S. Census administrative boundaries and resolves tile structures.
* **Core assets**: Downloads raw footprint boundaries (OBM, Microsoft, local layers), parcel assessor shapes and tax rolls, point references (NSI, Overture), Census tabulation geometries, and reference building inventories.

Stage 2: harmonize
------------------
Resolves geometries and links datasets to build the core footprint and parcel spines. Each spine is processed in two phases (geometry and attribute) to prevent reprocessing expensive spatial joins when only attributes change:
* **Footprint spine**: Merges footprint geometries in priority order, filters slivers and duplicate shapes, intersects footprints with parcels, stamps containing-area identifiers, links point databases via proximity joins, and assigns structural roles (primary vs. accessory).
* **Parcel spine**: Standardizes parcel boundaries and assessor rolls, links building structures, and calculates parcel-level morphology aggregates (footprint counts, elongated features) to assist curation rules.

Stage 3: enrich
---------------
Produces structure-level evidence without selecting final canonical values. This includes predicting roof shapes and story counts from imagery using BRAILS++ (fetched directly in-memory to comply with Google APIs policies), or attaching precomputed modeled attributes from reference building inventories.

Stage 4: curate
---------------
Applies rules, prioritizations, and voting logic to derive clean, reconciled canonical values:
* **Parcel curation**: Imputes property occupancy groups, scores relative footprint sizes, and runs multi-indicator voting rules to identify specific land-use classes (such as Manufactured Home Parks or Townhomes).
* **Footprint curation**: Integrates curated parcels, corrects Overture count anomalies on vacant land, resolves competing source values (addresses, year built, stories, values), and runs a final weighted consensus vote (specifically resolving Manufactured Home vs. Multi-Family conflicts).

  The curation recipe containing all decisions is in:

  :gh-file:`src/openplaces/recipes/US/_all/footprint/openplaces/2026/US_footprint-openplaces-2026.yaml`

Running the pipeline
~~~~~~~~~~~~~~~~~~~~

The pipeline can be run as a Jupyter notebook or as a ``snakemake`` flow:

- **Notebook**: :file:`notebooks/examples/US_curate_footprints.ipynb` drives ingest, harmonize, and curate for a chosen set of counties (best for few counties). It can be converted to a script to be deployed on a cluster.

  :gh-file:`notebooks/examples/US_curate_footprints.ipynb`

- **Snakemake**: :file:`workflow/Snakefile` implements the full workflow (one job per stage, recipe, and admin unit from the recipe tree, plus a terminal deliver job per declared region).


Output columns
~~~~~~~~~~~~~~

The columns below are the full schema. The inventory deliveries split them across four files: the compact set named in the recipe's ``share:`` block goes to the canonical file and to the centroid points, the geometry to :file:`_geo`, and everything else to :file:`_evidence`. Nothing is dropped; the four files together still carry every column documented here.

Canonical attributes
--------------------

``footprint_id``
    A unique identifier for each footprint: its 11-digit `openlocationcode <https://github.com/google/open-location-code>`_ (duplicate codes receive a numeric suffix, e.g., :input:`-1`, :input:`-2`, to guarantee uniqueness).
``geometry``
    The spatial polygon outlining the footprint, or the parcel polygon for fallback records.
``geometry_source``
    Source of the geometry: :input:`obm` (OpenBuildingMap), :input:`microsoft`, local state sources (e.g., :input:`nconemap`), or :input:`parcel.<source>` for parcel-shaped fallbacks representing unlocated structures.
``address``
    Reconciled street address. Prioritizes the assessor parcel address (where available), falling back to Overture dwelling address components, completed with state names derived from administrative units.
``occupancy_type``
    Canonical occupancy class.

    Multi-Family structures are split into HAZUS height bands (Low-Rise: 1-3 stories, Mid-Rise: 4-7 stories, High-Rise: 8+ stories) based on ``n_stories``. Multi-Family rows with no story count keep the plain ``Multi-Family`` label.
``n_sections``
    Number of factory-built sections a manufactured home shipped in: 1 for a single-wide, 2 for a multi-section (double-wide), 3 for a triple-wide.

    Decided by the same weighted vote as every other classification here, gated on ``occupancy_type == 'Manufactured Home'`` - so it is missing for every other class. Two lanes of evidence: the section count the assessor states in its own land-use text (exact, but present in only ten of the 44 delivered counties), and footprint shape, which is available everywhere. A section is a road-legal load, so its width is near-constant while its length is not; multi-section homes sit at a median footprint width of 8.85 m against 5.39 m for single-wides.

    Two nulls to tell apart, both of which mean "no claim", never "one section": the footprint is not a manufactured home, or its width and its elongation contradict each other, in which case neither decision reaches its threshold. Shape alone resolves 77.4% of manufactured homes at 86.8% accuracy, but multi-section is 82.2% of those rows, so the shape lane is worth only about +4.6 points over always answering "multi-section". Its value is that it finds single-wides at all - at 0.626 precision and 0.509 recall. Treat a single-wide from the shape lane as a useful flag rather than a fact, and take ``section_keyword_class`` from the ``_evidence`` file to restrict yourself to the rows the assessor labeled itself.
``structure_value``
    Reconciled structure value in USD. Prioritizes the parcel's improvement value (apportioned across its primary footprints by floor-area share), falling back to the NSI structure replacement value.
``year_built``
    Reconciled construction year. Prioritizes assessor parcel records over NSI block-median fallbacks. Assessor sentinel values outside 1500-2035 (rolls ship 1, 1000 and 8104 for "unknown") are treated as missing evidence, so selection falls through to NSI instead of shipping a sentinel as a year.
``n_stories``
    Reconciled number of stories. Prioritizes street-level imagery predictions (BRAILS++), then NSI modeled counts, then a modeled inventory's count - last because it shares NSI's modeled lineage rather than being an independent observation.
``n_dwellings``
    Reconciled count of dwelling units. Prioritizes Overture geocoded address counts over NSI structure counts, falling back to occupancy-class imputation.
``height``
    Reconciled building height in meters (where available from the footprint source, e.g. OpenBuildingMap).
``roof_shape``
    Reconciled roof structure classification (e.g., Gable, Hip, Flat) from visual models. Prioritizes this repository's own per-image classification, falling back to the CHEER Inventory v0 run of the same classifier. Both are BRAILS++ outputs, so this is a coverage cascade rather than a quality ranking: whichever actually ran for the county wins.
``roof_shape_confidence``
    Classifier confidence in ``roof_shape``.
``roof_shape_2`` / ``roof_shape_3``
    The classifier's second- and third-ranked alternative roof shapes, with ``roof_shape_confidence_2`` / ``roof_shape_confidence_3``. The top class is near-saturated, so the runner-up is what carries information when the leader is weak.
``foundation_type``
    Reconciled foundation classification. Prioritizes NSI, using Inventory v0 only where NSI is silent - not a blanket judgement on the inventory, but a guard against a failure mode NSI's regional prior cannot have: the inventory's per-county class shares swing widely, and in at least one coastal county where basements are near-absent the Basement class collapses onto most buildings and carries no information.
``construction_type``
    Structural construction material class (e.g., wood, masonry, steel, concrete).
``has_garage``
    Whether the building has a garage.
``m2``
    Calculated footprint area in square meters.
``priority_on_parcel``
    Structural role: :input:`primary` (main structure), :input:`secondary` (accessory structure), or :input:`unknown` (unlinked to parcel).

Containing-area identifiers
---------------------------

Identifiers of the areas each footprint falls in, assigned during harmonization (see :ref:`containing-area identifiers <containing_area_ids>`). They let the inventory be joined to any statistic published for those areas without repeating the spatial join.

``admin4_id``
    ``openplaces`` administrative unit at level 4 (town, city, county subdivision).
``census_subdivision_id``
    Raw U.S. Census county subdivision (COUSUB) code for that same level-4 unit.
``census_tract_id`` / ``census_blockgroup_id`` / ``census_block_id``
    U.S. Census tract, block group, and block.
``zcta5_id``
    5-digit ZIP Code Tabulation Area. A statistical approximation of a ZIP code's service area, not the authoritative ZIP; ``postal_code`` prefers a real address-parsed value and only falls back to this.

Provenance sidecars
-------------------
Indicates which input dataset or curation rule set decided the final canonical value (later steps override earlier ones). The values contain short method/source strings (e.g., ``parcel``, ``nsi``, ``overture``, ``imputed``, ``geometry``, ``dwellings``, ``secondary``, ``park``, ``keyword``, ``manufactured_home``, ``brails-2026``):

* ``address_source``
* ``occupancy_type_source``
* ``structure_value_source``
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

Modeled inventory evidence (CHEER Inventory v0)
------------------------------------------------
Attributes of the reference building each footprint overlaps most, retained as evidence rather than merged onto a canonical name because a competing source is reconciled against them. Present for North Carolina only.

``roof_shape_building_cheer``
    Inventory roof shape, second-ranked behind this repository's own per-image classification in ``roof_shape``.
``foundation_type_building_cheer``
    Inventory foundation class, second-ranked behind NSI in ``foundation_type``.
``n_stories_building_cheer``
    Inventory story count. Ranked last in ``n_stories``, behind direct street-level floor detection and NSI, because it shares NSI's modeled lineage rather than being an independent observation.

Diagnostics and special metrics
-------------------------------
Flags and intermediate calculations used for curation and quality control.

``n_parcels_per_footprint``
    Number of parcels intersected by the footprint.
``n_footprints_per_parcel``
    Number of footprints on the associated parcel (includes synthetic fallbacks).
``occupancy_type_conflict``
    Summary of conflicting occupancy labels across inputs (formatted :input:`"nsi: X | fema: Y | parcel: Z | overture: W"`). This column is only populated where two or more evidence sources disagree. Non-residential classes are collapsed to :input:`Non-Residential` before comparison, so only residential (or residential-vs-non-residential) disagreements are surfaced.
``address_conflict``
    Summary of conflicting street address values across input sources, only populated where two or more address sources disagree.
``occupancy_type_review``
    Flag (:input:`True`) for low improvement-value parcel shares indicating potential manufactured homes needing inspection.
``structure_value_per_area`` / ``structure_value_building_nsi_per_area``
    Calculated values divided by footprint area (USD/m²).
``manufactured_home_community``
    Boolean flag indicating whether the parcel is part of a manufactured home community (having more than 3 final Manufactured Home footprints).
``n_manufactured_homes_per_parcel``
    The total number of curated Manufactured Home footprints residing on the same parcel.


Validation and accuracy
~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::

   annexes/US_footprint-cheer-2026-validation.rst


Technical Annex
~~~~~~~~~~~~~~~

.. toctree::

   annexes/US_footprint-cheer-2026-annex.rst


References
~~~~~~~~~~

.. _Lochhead et al. 2026:

Lochhead M, Zsarnoczay A, Deierlein G (2026) Exposure matters: a synthesis framework for high-resolution building inventory development. International Journal of Disaster Risk Reduction 139 (2026): 106148. `doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_

   *Informs the evidence-then-reconcile staging and the footprint priority rules (their Tables 3 and 4).*

.. _Cetiner et al. 2025:

Cetiner, B., McKenna, F., Yi, S.-ri, Wang, B., & Manousakis, I. V. (2025). BRAILS++ (v4.2.0). Zenodo. `doi:10.5281/zenodo.17797364 <https://doi.org/10.5281/zenodo.17797364>`_

   *Provides the imagery classifiers behind the roof-shape and story-count evidence.*

.. _Khanal et al. 2025:

Khanal K, Kaza N, Hino M, Sebastian A (2025) Characterizing manufactured home parks in North Carolina: a computer vision based approach. EPB: Urban Analytics and City Science. `doi:10.1177/23998083251395471 <https://doi.org/10.1177/23998083251395471>`_

   *Independently corroborates the manufactured-home geometry thresholds and the three-unit community cutoff.*

.. _Tackie-Otoo et al. 2026:

Tackie-Otoo N O, Askari M, Hadinata P, Davidson R A, Taciroglu E, Hardy G (2026) Hurricane wind loss modeling using insurance claims data. Natural Hazards 122: 300. `doi:10.1007/s11069-026-08059-z <https://doi.org/10.1007/s11069-026-08059-z>`_

   *Motivates the curated attribute set and the separate Manufactured Home class: their wind-loss model consumes exactly these attributes.*
