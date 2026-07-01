.. openplaces

.. _cheer_footprints:

U.S. footprint inventory (CHEER)
================================

.. image:: ../../1_overview/concepts/images/footprint_building_dwelling_urban.png
  :width: 400
  :alt: Illustration of footprints, buildings, and dwellings
  :align: right

This recipe creates a footprint-level building inventory for U.S. hurricane damage and exposure modeling.

It resolves building :ref:`footprints <footprints>` from multiple geometry sources (OpenBuildingsMap, Microsoft, local sources), then enriches each footprint with data from linked parcels, the National Structure Inventory (NSI), and Overture :ref:`dwellings <dwellings>` (addresses). FEMA USA Structures is no longer a geometry source (its footprints caused merge errors); instead its building occupancy is attributed to parcels and reaches footprints as occupancy evidence.

This work is supported by NSF's Coastal Hazards, Economic Prosperity & Resilience hub (`CHEER <https://www.drc.udel.edu/cheer/>`_). It is currently tested for use in Florida, Massachusetts, North Carolina, and Texas.

The companion notebook is:

:gh-file:`notebooks/examples/US_curate_footprints.ipynb`.

The :ref:`recipe <recipes>` with instructions & thresholds is at:

:gh-file:`src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml`


What this recipe produces
~~~~~~~~~~~~~~~~~~~~~~~~~

Running this recipe for a U.S. region will produce a building inventory at the footprint level for each county (level 3 in the :ref:`administrative hierarchy <administrative_units>`) as a :file:`.parquet` file.

Each row is a footprint, deduplicated across sources. If parcel information points towards the presence of another structure missing in any footprint dataset, the "footprint" geometry is the parcel boundary.

The table includes:

- ``footprint_id``, the unique ID of the footprint: its 11-digit `openlocationcode <https://github.com/google/open-location-code>`_).
- ``geometry``: the footprint polygon, or a parcel polygon for inferred fallback records.
- ``geometry_source``: the geometry source used for the row (geometry's provenance sidecar): ``obm`` for OpenBuildingMap, ``microsoft``, ``fema``, ``nconemap``. A ``parcel.<source>`` value flags parcel boundaries with unlocated buildings. Because such parcel-shaped fallbacks are not real building outlines, they link to NSI buildings and Overture dwellings by strict containment (``within``) only — they are excluded from the proximity-based point allocation applied to true footprints.
- ``priority_on_parcel``: ``primary`` for the primary footprint on a parcel, ``secondary`` for others, ``unknown`` if no parcel data was available.
- Parcel-derived attributes, including land value, improvement value, address, land-use or purpose classes, year built, and sometimes dwelling units.
- NSI-derived attributes, including structure or occupancy class, structure value, number of stories, and block-median year built.
- Overture-derived dwelling and address attributes, especially dwelling-unit point evidence and address components.
- Reconciled final columns selected from competing source-specific columns. Each reconciled/inferred canonical variable carries a categorical ``{col}_source`` provenance sidecar — all sidecars are grouped together after the canonical columns — naming the source or decision that set the value, e.g. ``occupancy_type_source`` in ``{nsi, fema, parcel, geometry, dwellings, park, secondary, keyword, mobile_home}`` (``keyword`` from ``resolve_occupancy``'s reviewed override; ``mobile_home`` and a repeat of ``dwellings`` from ``resolve_by_vote``'s two decisions), ``value_source`` in ``{parcel, nsi}`` (the footprint's own reconciled value — see "Parcels" below for how this differs from a parcel's own directly-assessed ``value``), and ``n_dwellings_source`` in ``{overture, nsi, imputed}``.
- ``occupancy_type_base``: inferred occupancy, derived **NSI-first** across the recipe's ordered evidence — the footprint's own NSI occupancy where present, then FEMA's parcel-aggregated building occupancy, then the parcel ``group_parcel`` filling remaining gaps — followed by an elongated-small-footprint geometry signal for manufactured homes, ``n_dwellings`` single-family gap-fill, and a park-aware secondary rule (a habitable-size footprint on a ``manufactured_home_park`` parcel is ``Manufactured Home`` rather than ``Secondary``). Residential footprints collapse to ``Single-Family``, ``Multi-Family``, or ``Manufactured Home``; non-residential footprints keep their source category (e.g. ``Retail``, ``Hotel``, ``Church``); non-primary *residential* footprints not covered by the park rule become ``Secondary``. A reviewed parcel keyword override (e.g. ``MANUFACTURED`` → Manufactured Home) is applied on top of this base. Manufactured Home vs. Multi-Family is then decided by a weighted vote (``resolve_by_vote``) over four indicators — a low parcel-value share, a parcel mobile-home keyword, the parcel's NSI-derived group, and a footprint-morphology/imagery probability (``p_manufactured_home``, from ``classify_mobile_homes``) — against a single ``n_dwellings`` ≥ 2 Multi-Family vote. All class names, evidence columns, and thresholds come from the recipe ``occupancy`` config block (and the ``occupancy-class-map.csv`` label→class map), so the curator code carries no US/NSI terminology. This pre-height-split class is preserved as ``occupancy_type_base``; the canonical ``occupancy_type`` adds the height-band split below.
- ``occupancy_type_parcel``: the parcel-proposed occupancy — the keyword-ruleset proposal, falling back to the ``group_parcel`` class so generic residential parcels still carry a parcel occupancy (retained so disagreements with NSI can be reviewed).
- ``occupancy_type_conflict``: a categorical summary across every present occupancy evidence (NSI, FEMA, parcel, and any other source in ``occupancy.evidence``) for rows where two or more disagree, formatted ``"nsi: X | fema: Y | parcel: Z"`` (null when they agree), so conflicts can be categorized and inspected directly in QGIS. Every non-residential class is collapsed into a single ``Non-Residential`` bucket so the column stays low-cardinality and surfaces the residential disagreements that matter. The same disagreements are also written to an ``occupancy-conflicts.csv`` report ranked by frequency.
- ``occupancy_type_review``: ``True`` where the parcel improvement value is a small nonzero share of total value (``0 < improvement/(improvement+land) < 2.5%``) — a likely manufactured home flagged for inspection without changing the class.
- ``occupancy_type``: the canonical occupancy. A copy of ``occupancy_type_base`` in which ``Multi-Family`` is split into HAZUS height bands — ``Low-Rise Multi-Family`` (1–3 floors of living area), ``Mid-Rise Multi-Family`` (4–7), and ``High-Rise Multi-Family`` (8+) — wherever a merged ``n_stories`` value is available. The bands count floors of *living area*: floors that are open (e.g. a ground floor on stilts/pilotis, common in coastal flood zones) are not counted. All other classes (and Multi-Family rows lacking ``n_stories``) are carried over unchanged.
- ``manufactured_home_community``, ``n_manufactured_homes_per_parcel``: recomputed from the final footprint ``occupancy_type`` (after the vote and height split), flagging parcels with more than a threshold number of Manufactured Home footprints — richer than the pre-occupancy parcel-side pass of the same concept.

Ready-made QGIS categorized styles for ``occupancy_type`` (height-banded) and ``occupancy_type_base`` (un-banded) ship in ``src/openplaces/qgis/styles/``; apply one via Layer Properties → Symbology → Load Style after loading the layer.


Source datasets
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Source family
     - Main contribution
     - Common analytical use
   * - Footprints (detected)
     - Building footprint geometries with associated data
     - Exposure locations, structure area, hazard overlay, spatial joins
   * - Parcels (land ownership)
     - Assessor values, land-use classes, addresses, year built, parcel context
     - Valuation, land-use analysis, property-level aggregation
   * - National Structure Inventory (buildings)
     - Structure-level point records with occupancy class, structure value, stories, and block-median year built
     - Building-type classification, exposure modeling, crosswalks from parcel classes to structure classes
   * - Overture address points (dwellings)
     - Residential unit and address point evidence
     - Dwelling-unit estimation and primary-structure identification


Footprints
~~~~~~~~~~

No single national dataset provides complete, high-quality building footprint coverage for the United States. Open datasets differ in geographic focus, detection method, and vintage: OBM has global scope but variable density; Microsoft's ML-detected footprints give broad domestic coverage but are noisier in rural areas; state-specific sources often have the highest local accuracy but exist only for certain jurisdictions. The recipe therefore merges these sources rather than choosing one. (FEMA USA Structures was previously merged here too, but its footprints caused merge errors; it now contributes occupancy via parcels rather than geometry.)

The spine is built by merging sources in priority order — OBM first, then Microsoft, then any state-specific footprint recipe found via ``auto_discover: true`` (for example, ``US-NC_footprint-nconemap-2024`` in North Carolina). Footprints smaller than 10 m² are dropped before merging.

Each source after the first is compared against the existing spine. A candidate footprint is added only if its *intersection-over-union* (IoU) with every current spine footprint is below 0.02. IoU is the ratio of intersection area to union area; a value above 0.02 means the two footprints share more than 2 % of their combined area and are treated as the same building.

Because IoU can miss displaced thin-rectangle footprints, the recipe also applies an *elongated-footprint duplicate filter*. Two footprints are treated as duplicates when they are both elongated (aspect ratio ≥ 2.5), their long axes are aligned within 15°, their long-axis projections overlap by at least 50 %, and their lateral separation is less than twice their average width. This catches parallel shifted representations of manufactured homes and trailers that appear across two sources without enough polygon overlap to trigger the IoU threshold.

After merging, each spine row carries a ``geometry`` and a ``geometry_source`` column indicating which dataset contributed the footprint (e.g. ``'obm'``, ``'microsoft'``, or ``'nconemap'``).


Parcels
~~~~~~~

Parcel datasets contribute assessed value, land-use classification, address, and year-built information. Depending on the jurisdiction this can include land value, improvement value, mailing address, land-use or property-purpose classes, year built, and dwelling-unit estimates. Coverage, field completeness, valuation conventions, and land-use code definitions are not uniform across counties or states.

Join method
-----------

Parcels are polygons, so they are joined to footprints with a *polygon identity overlay*: each footprint is intersected with all parcels it touches, producing one row for every (footprint, parcel) pair. The crosswalk is a MultiIndex DataFrame indexed by ``(footprint_id, parcel_id)`` with columns ``area_intersection_m2``, ``iou``, and ``fraction_of_largest`` (this intersection's area as a fraction of the largest intersection for the same footprint).

Two thresholds keep the crosswalk clean: links whose ``fraction_of_largest`` is below 1/6 are dropped (a footprint's corner clipping a distant parcel should not be attributed), and intersections smaller than 10 m² are removed. After filtering, a footprint links to one parcel (``'unique parcel'``), multiple parcels (``'multi-parcel footprint'``), or none (``'no parcel'``).

Some parcels with buildings on them have no footprint in any source — new structures and buildings under trees are common examples. The recipe adds a synthetic footprint for each such parcel that is likely to contain a structure, using two criteria that must both be met: the parcel's ``use_group`` has a mean footprint count per parcel across the county of at least 0.2, and the parcel's ``improvement_value_per_ha`` exceeds the 5th percentile of matched parcels in the same use group. Parcels that pass both tests receive a synthetic footprint whose geometry is the parcel polygon, tagged ``source = 'parcel.{source_id}'``. After these inferred footprints are added, ``resolve_overlaps`` trims any geometry overlaps they introduce against existing OBM or Microsoft footprints.

Derived attributes
------------------

All parcel columns are written with the suffix ``_parcel``:

- ``use_group_parcel``, ``use_subgroup_parcel`` — direct from assessor land-use codes.
- ``group_parcel``, ``use_group_combined_parcel``, ``manufactured_home_park`` — these three are not produced by this recipe at all: they are canonical outputs of the *parcel* curation lane (``US_parcel-openplaces-2026``, run on the harmonized parcel spine ``US_parcel-spine-2026``) and reach footprints as parcel columns via ``link_curated_parcels`` (see Implementation walkthrough below). ``group_parcel`` in particular is computed by the generic ``infer_from_group_statistic`` curate step: per parcel use group, the most common NSI group (``statistic: mode``), propagating NSI's structure-level occupancy vocabulary to parcels with no direct NSI match.

  This parcel-side data depends on the parcel spine having ``use_group`` / ``use_subgroup`` populated for the admin unit, which ``US_parcel-spine-2026`` gets either directly from the admin's own statewide parcel geometry recipe when it provides these natively (as North Carolina's does), or, when it doesn't (as in Massachusetts), from an auto-discovered bundled roll via the same ``link_by_id`` auto-discovery mechanism that links local tax/assessor rolls onto the spine. Auto-discovery — rather than a hardcoded single-state roll — is what makes this generalize to admins like North Carolina without a per-state recipe change.
- ``improvement_value_parcel`` — split proportionally to intersection area. When any footprint on the parcel has dwelling-point evidence, area fractions for footprints *without* dwelling evidence are zeroed before the split, so improvement value accrues only to dwelling-linked footprints (Lochhead et al. 2026, Table 4). Further restricted to ``primary`` footprints: ``secondary`` footprints are left as NaN rather than 0, mirroring ``land_value_parcel`` so that no parcel financial value is assigned to accessory structures. This is footprint-side evidence for the canonical ``value`` (see "Attribute reconciliation" below) — distinct from a parcel's own directly-assessed ``value`` (land + improvement, as recorded by the assessor), which this recipe never attaches to footprints.
- ``land_value_parcel`` — assigned from the largest-intersection parcel only; further restricted to ``primary`` footprints to avoid assigning parcel land value to accessory structures.
- ``year_built_parcel`` — direct from assessor records.
- ``address_parcel`` — from the largest-intersection parcel.
- ``n_dwellings_parcel`` — area-weighted from the assessor unit count.
- ``overlap_fraction_parcel``, ``n_footprints_per_parcel`` — diagnostic columns recording the footprint's share of parcel area and the total number of footprints on the same parcel (``n_parcels_per_footprint`` records the reverse: how many parcels the footprint spans).


National Structure Inventory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The National Structure Inventory contributes structure-level point records with occupancy class, structure replacement value, number of stories, and block-median year built. Unlike parcel records, NSI classifies individual structures rather than land-use parcels, making it a useful crosscheck on jurisdiction-specific assessor codes and a source of structure-type information for multi-structure parcels.

Join method
-----------

NSI records are points; footprints are polygons. The four-pass proximity method (Lochhead et al. 2026, Table 3) handles imprecise geocoding:

1. **Containment** — ``sjoin(predicate='within')``. Points inside a footprint are matched directly.
2. **Inner proximity (10 m)** — unmatched points within 10 m of a footprint edge (typical GPS error) are matched to the nearest footprint.
3. **Outer proximity (100 m), same-parcel constraint** — unmatched points within 100 m are matched to the nearest footprint *on the same parcel*. The parcel constraint prevents linking a point to a footprint across the street.
4. **Unbounded fallback** — disabled in this recipe.

When multiple passes produce candidate matches for the same point, the highest-quality match is retained.

Derived attributes
------------------

All NSI columns are written with the suffix ``_building_nsi``:

- ``purpose_subgroup_building_nsi`` — NSI occupancy class (e.g. ``'Single Family'``, ``'Multi-Family (2 units)'``).
- ``group_building_nsi`` — mapped from occupancy class to the openplaces group vocabulary.
- ``structure_value_building_nsi`` — structure replacement value in USD.
- ``year_built_block_median_building_nsi`` — median year built for the census block, used as a fallback when parcel year built is absent.
- ``n_stories_building_nsi`` — number of stories.


Overture dwelling/address points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overture contributes geocoded residential unit and address evidence. Unlike NSI, which records one entry per structure, Overture records one point per residential unit: a ten-unit apartment building may have ten Overture points while NSI has one. Those points are used to count dwelling units and to help distinguish primary structures from accessory structures on multi-footprint parcels.

Join method
-----------

The same three proximity passes as NSI are used (containment, inner 10 m, outer 50 m same-parcel). Two additional options are enabled for this join:

- ``dedup_addresses: true`` — when a base address (street + house number) has both a building-level record and unit-specific records (e.g. ``Apt 1``, ``Apt 2``), the building-level record is dropped and each unit record is tagged ``n_dwellings = 1`` for downstream summation. This prevents double-counting units.
- ``aggregate_multipoint: true`` — multiple dwelling points matched to the same footprint are collapsed into one row with ``n_dwellings`` summed, giving the total residential unit count for that footprint.

Derived attributes
------------------

All Overture columns are written with the suffix ``_dwelling_overture``:

- ``n_dwellings_overture`` — total attributed dwelling points (after deduplication and aggregation).
- ``address_street_dwelling_overture``, ``address_number_dwelling_overture``, ``postal_code_dwelling_overture``, ``city_dwelling_overture`` — address components from the matched dwelling record.


Footprint role classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

On parcels with more than one footprint (house, garage, shed, etc.), only the primary structure should receive the full parcel land value and, when dwelling evidence is absent for some footprints, the full improvement value. Each footprint is assigned one of three roles:

- ``primary`` — the main structure on the parcel.
- ``secondary`` — an accessory structure.
- ``unknown`` — footprint not linked to any parcel.

The classification follows Lochhead et al. (2026, Table 4). Point evidence resolves ambiguity on multi-footprint parcels in this order:

1. If any footprint on the parcel has a linked *Overture dwelling point*, those footprints are ``primary``; the rest are ``secondary``.
2. Otherwise, if any footprint has a linked *NSI building point*, those are ``primary``; the rest are ``secondary``.
3. If no footprint on the parcel has any point evidence, all are ``secondary`` — the recipe cannot determine which is the main structure.
4. A footprint that is the *sole* footprint on its parcel is always ``primary``, regardless of point evidence.
5. Footprints not linked to any parcel are ``unknown``, except those with dwelling-point evidence, which are promoted to ``primary``.

Dwelling evidence is evaluated before building evidence because Overture explicitly locates an occupied residential unit, whereas an NSI record may represent an accessory structure such as a detached garage.

``priority_on_parcel`` is stored as a ``pd.Categorical`` column on the spine.


Attribute reconciliation and design rationale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple sources provide overlapping information for the same footprint. The harmonization recipe assembles all of it as source-suffixed evidence columns; the curation recipe's ``reconcile_values`` step then selects three canonical values, filling from left to right across the listed source columns in priority order:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Canonical column
     - Priority order
     - Rationale
   * - ``n_dwellings``
     - ``n_dwellings_overture`` → ``n_dwellings_nsi``; occupancy-class fallback for remaining nulls
     - Overture provides explicit counted units per dwelling point; NSI's per-structure unit count is the fallback (parcel dwelling counts are not in this priority list). Remaining nulls are filled by ``impute_n_dwellings`` from a lookup mapping NSI occupancy classes to expected unit counts (e.g. ``'Single Family' → 1``, ``'Multi-Family (50+ units)' → 51``).
   * - ``year_built``
     - ``year_built_parcel`` → ``year_built_block_median_building_nsi``
     - Tax-record year built is structure-specific and preferred; NSI's census-block median year built is now a genuine fallback for footprints with no parcel year built, rather than inspectable-only evidence.
   * - ``value``
     - ``improvement_value_parcel`` → ``structure_value_building_nsi``
     - The footprint's own reconciled value: the parcel's improvement value (split across footprints on that parcel) is preferred; NSI's structure replacement value is a competing fallback where no parcel value reached this footprint. Distinct from (but sharing a registry name with) a parcel's own directly-assessed ``value`` (land + improvement), which this recipe never attaches to footprints — see "Parcels" above.

``purpose_subgroup`` and parcel land value are intentionally left as suffixed evidence rather than collapsed into canonical columns: final occupancy is derived separately by the curation steps, and ``land_value_parcel`` is already restricted to ``primary`` footprints during harmonization. Inspect ``purpose_subgroup_parcel`` / ``purpose_subgroup_building_nsi`` and ``land_value_parcel`` directly for those concepts.

Source-suffixed columns (e.g. ``year_built_parcel``, ``year_built_block_median_building_nsi``) are retained in the output so users can inspect the evidence behind any final value and audit disagreements.

Several design choices underpin reliable reconciliation:

- **IoU deduplication rather than nearest-neighbor merging** ensures that geometrically distinct nearby buildings — such as two row houses sharing a wall — are each retained as separate records rather than collapsed.
- **Elongated-footprint duplicate filtering** catches displaced thin-rectangle footprints (manufactured homes, trailers) that appear as parallel shifted rectangles across sources without enough polygon overlap to trigger the IoU threshold.
- **Area-weighted value distribution** makes multi-parcel overlaps explicit and reproducible. When a single footprint spans two parcels, improvement value is distributed proportionally to intersection area, so allocations sum back to the full parcel total.
- **Dwelling evidence over building evidence** when classifying primary structures prevents a detached garage (which may carry its own NSI record) from being misclassified as the main structure on a multi-footprint parcel.
- **Inferred parcel-shaped footprints** preserve exposure and valuation evidence for structures absent from all footprint datasets, rather than silently dropping those parcels from the inventory.


Implementation walkthrough
~~~~~~~~~~~~~~~~~~~~~~~~~~

The work is split across two recipes. The harmonization recipe
(``US_footprint-spine-2026``) builds an *evidence-only* spine: it merges
footprint sources, links parcels, NSI, and Overture, and attributes their columns
as source-suffixed evidence (e.g. ``improvement_value_parcel``,
``occupancy_type_building_nsi``). Each step is a registered function that receives
the shared ``HarmonizeState`` (carrying ``spine``, ``references``, ``crosswalks``,
and ``overlays``), modifies it, and returns it.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Harmonize step
     - Role in the recipe
   * - ``resolve_spine``
     - Builds the footprint spine from OBM, Microsoft, and state-specific footprints, with IoU and elongated-footprint deduplication.
   * - ``link_to_reference``

       Parcels
     - Creates footprint-parcel overlay crosswalk and parcel-derived reference evidence.
   * - ``infer_spine_additions``
     - Adds inferred parcel-shaped footprint rows for likely missing structures.
   * - ``resolve_overlaps``
     - Trims overlaps introduced by inferred footprints.
   * - ``link_to_reference``

       NSI
     - Links NSI building points to footprints by four-pass proximity.
   * - ``link_to_reference``

       Overture
     - Links dwelling/address points, deduplicates address evidence, and aggregates multiple dwelling points per footprint.
   * - ``classify_footprint_priority``
     - Derives ``primary``, ``secondary``, and ``unknown`` values for ``priority_on_parcel``.
   * - ``reconcile_attributes``
     - Aggregates source columns into suffixed evidence columns (attribution only).

The curation recipe (``US_footprint-cheer-2026``) turns that evidence into
canonical values. Each step is a registered function receiving a ``CurateState``
(canonical GeoDataFrame in ``state.curated``).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Curate step
     - Role in the recipe
   * - ``link_curated_parcels``
     - Joins the curated parcel lane (``US_parcel-openplaces-2026``) onto each footprint by ``parcel_id_local``, overwriting the footprint's raw (empty, evidence-only) parcel columns: the property-merged assessor values, the refined ``use_group_combined_parcel``, the modal-NSI ``group_parcel``, the FEMA building occupancy, and the ``manufactured_home_park`` flag. Runs first because parcels are curated before footprints in the same admin run, so these are final before reconciliation and occupancy inference.
   * - ``reconcile_values``
     - Selects ``n_dwellings``, ``year_built``, and ``value`` from competing source columns by priority (first non-null wins).
   * - ``derive_metrics``
     - Computes ``m2`` and per-area ratios for ``value`` and the raw ``improvement_value*`` / ``structure_value*`` evidence columns.
   * - ``impute_n_dwellings``
     - Fills remaining null dwelling-unit counts from an occupancy-class lookup.
   * - ``infer_occupancy_type``
     - Derives ``occupancy_type_base`` from the recipe's ordered ``occupancy.evidence`` (NSI, then FEMA, then parcel), then a footprint-geometry rule, single-family dwelling gap-fill, and the secondary-on-parcel rule. The secondary rule is park-aware: on a ``manufactured_home_park`` parcel a habitable-size footprint (at least ``habitable_fraction`` of the average manufactured-home area) is ``Manufactured Home``, not ``Secondary`` — only sub-threshold sheds stay ``Secondary``. Vocabulary/columns/thresholds come from the ``occupancy`` config block.
   * - ``resolve_occupancy``
     - Applies the reviewed parcel keyword override only (value-share and dwelling-count class assignment moved to ``resolve_by_vote`` below); sets ``occupancy_type_parcel``, the ``occupancy_type_review`` flag, and the categorical ``occupancy_type_conflict`` summary across every present occupancy evidence column. Runs before enrichment (no ``n_stories`` dependency).
   * - ``merge_enrichments``
     - Joins image-derived ``roof_shape`` and ``n_stories`` evidence (canonical values win).
   * - ``classify_mobile_homes``
     - Evidence-only: estimates a manufactured-home probability (``p_manufactured_home``) from assessor keywords, footprint morphology, and imagery. Does not itself assign ``occupancy_type``.
   * - ``resolve_by_vote``
     - Decides Manufactured Home vs. Multi-Family by weighted indicator vote: a low parcel-value share, a parcel mobile-home keyword, the parcel's NSI-derived group, and ``p_manufactured_home`` each vote Manufactured Home (≥ 2 of 4 needed) against a single ``n_dwellings`` ≥ 2 Multi-Family vote.
   * - ``refine_occupancy_height``
     - Splits the multi-family class into the recipe-defined height bands, writing the canonical ``occupancy_type`` in place and keeping the un-banded, pre-vote class as ``occupancy_type_base``.
   * - ``flag_manufactured_home_communities``
     - Correction sidecar: recomputes ``manufactured_home_community`` and ``n_manufactured_homes_per_parcel`` from the final footprint occupancy (richer than the pre-occupancy parcel pass), flagging parcels with more than ``min_homes`` manufactured-home footprints.
   * - ``cast_categoricals``
     - Casts registry categoricals and the provenance sidecars to ``pd.Categorical``.
   * - ``order_columns``
     - Writes the final column order in bands: canonical/final variables (incl. ``m2``, by attribute-registry ``sort`` rank) ending with ``priority_on_parcel``; then every ``{col}_source`` provenance sidecar grouped together (ordered by their base column's rank); then source variables inherited from other entities (relational counts first, then the suffixed evidence); then flags and visualization-only columns (``occupancy_type_conflict``/``_review``, ``*_per_area``). ``geometry`` is last, preceded by ``geometry_source``. Rule-based, so no explicit column list is maintained. The recipe also uses this step's ``drop`` option to remove the transient ``group_parcel`` from the final output — its information now lives in ``occupancy_type_parcel``.



Running the recipe
~~~~~~~~~~~~~~~~~~

Activate the project environment before running development or data commands:

.. code-block:: bash

   conda activate openplaces

The recommended entry point is the example notebook:

:gh-file:`notebooks/examples/US_curate_footprints.ipynb`

The equivalent script is available for batch or cluster execution:

:gh-file:`scripts/examples/US_curate_footprints.py`

The notebook and script cover ingestion, harmonization, enrichment, and the
``US_footprint-cheer-2026`` curation recipe. Cached inputs and outputs are
reused on subsequent runs unless reprocessing is requested. The notebook
accepts ``--no_streetview`` to skip Google Street View entirely (its ingest
and the ``n_stories`` enrichment step that depends on it) — useful to avoid
its per-request billing while iterating.


Using the output
~~~~~~~~~~~~~~~~

Use ``geometry`` as the footprint polygon for spatial analysis, hazard overlay, and area calculations. For rows whose ``geometry_source`` indicates an inferred parcel source, the geometry is a parcel-shaped fallback and should not be interpreted as a precise building outline. Such rows are useful for exposure completeness but should be treated carefully in analyses that require roof shape, building dimensions, or parcel-independent geometry.

For most analyses, prefer the canonical curated columns: ``n_dwellings``, ``year_built``, and ``value`` (the footprint's own reconciled value, by source priority), the derived ``occupancy_type`` (height-banded) / ``occupancy_type_base``, and area or value-per-area fields (``m2``, ``value_per_area``). For land-use class and parcel land value, which are not collapsed into canonical columns, read the suffixed evidence directly — ``purpose_subgroup_parcel`` / ``purpose_subgroup_building_nsi`` and ``land_value_parcel``.

Parcel-derived columns reflect assessor data. Field completeness, valuation conventions, land-use code definitions, and year-built interpretation vary by jurisdiction and are not uniform across counties or states. Inspect source-suffixed columns when uncertainty matters: compare ``year_built_parcel`` against ``year_built_block_median_building_nsi``, or ``purpose_subgroup_parcel`` against ``purpose_subgroup_building_nsi``, to audit where sources agree or diverge. This is especially useful when analyzing valuation outliers, conflicting occupancy labels, or unexpectedly high dwelling-unit counts.

NSI and Overture are point-based sources. Their locations can be affected by geocoding imprecision, address ambiguity, or source coverage gaps. A missing NSI or Overture link for a footprint does not necessarily mean the structure has no occupants — it may reflect coverage or geocoding limitations rather than an absence of buildings or residents.

Interpret ``priority_on_parcel`` as a derived analytical label:

- ``primary`` means the footprint is treated as the main structure on its parcel, often supported by dwelling or building-point evidence.
- ``secondary`` means the footprint is treated as an accessory or non-primary structure on a parcel with another primary candidate.
- ``unknown`` means the recipe does not have enough parcel or point evidence to assign a primary/secondary role.

Final classes and values are harmonized estimates derived from multiple sources. They are not authoritative labels and should be validated before use in high-stakes parcel, valuation, or damage assessments.


Key reference
~~~~~~~~~~~~~

Lochhead M, Zsarnoczay A, Deierlein G (2026) Exposure matters: a synthesis framework for high-resolution building inventory development. International Journal of Disaster Risk Reduction 139 (2026): 106148. `doi:10.1016/j.ijdrr.2026.106148 <https://doi.org/10.1016/j.ijdrr.2026.106148>`_
