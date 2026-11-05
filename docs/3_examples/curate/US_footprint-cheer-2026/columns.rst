.. openplaces

.. _cheer_columns:

Variable dictionary
===================

This page starts with the columns a hurricane damage model reads, then
gives more detail on each of them (how it is chosen, its alternatives,
its helper columns and the evidence behind it), then the identifiers.
It ends with every column of the bundle, generated from the recipe
(:gh-file:`US_footprint-cheer-2026 <src/openplaces/recipes/US/_all/footprint/cheer/2026/US_footprint-cheer-2026.yaml>`)
and the attribute registry
(:gh-file:`attribute_registry.csv <src/openplaces/core/attribute_registry.csv>`),
so it always matches what a bundle ships. The ``_source`` columns that
say where each value came from are explained on
:doc:`provenance`.

.. toctree::
   :maxdepth: 1

   provenance

Key columns for damage modeling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are the building attributes that hurricane wind and flood loss
models take as input (for example
:ref:`Tackie-Otoo et al. 2026 <Tackie-Otoo et al. 2026>`). All are in
the canonical file, and where no source has a value, the models
described on :doc:`imputation` fill it.

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Column
     - What it is
     - Values
   * - ``occupancy_type``
     - The building's use
     - ``Single-Family``, ``Manufactured Home``, multi-family by height
       band, ``Secondary``, NSI's non-residential classes
   * - ``structure_value``
     - Market value of the building, net of land
     - USD at the county's appraisal date
   * - ``year_built``
     - Year of construction
     - A year
   * - ``n_dwellings``
     - Housing units in the building
     - 0, 1, 2, ...
   * - ``n_sections``
     - Sections of a manufactured home
     - 1 single-wide, 2 double-wide, 3 triple-wide
   * - ``area_m2``
     - Footprint area
     - m2
   * - ``n_stories``
     - Number of stories
     - 1, 2, 3, ...
   * - ``foundation_type``
     - Foundation
     - ``Slab``, ``Crawl``, ``Basement``, ``Solid wall``, ``Pier``,
       ``Pile``, ``Fill``
   * - ``construction_type``
     - Structural material
     - ``Wood``, ``Masonry``, ``Concrete``, ``Steel``, ``Manufactured``
   * - ``roof_shape``
     - Roof shape
     - ``Gable``, ``Cross Gable``, ``Complex Gable``, ``Hip``,
       ``Cross Hip``, ``Flat``, ``Mobile Homes``
   * - ``has_garage``
     - Whether the building has a garage
     - 1 or 0

The location is ``lat`` and ``long`` (the footprint's centroid) in the
canonical file, the centroid point in ``*_point.parquet`` and the
footprint's polygon in ``*_geo.parquet``.

Each key column has a ``_source`` column beside it
(``year_built_source``) that names where each value came from, and
says when a value was filled by a model (see :doc:`provenance`).

Occupancy, value, year built and dwellings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These columns combine the most sources: the county rolls, NSI, FEMA,
Overture's address points and the footprint's shape.

.. _cheer_col_occupancy:

Occupancy
---------

``occupancy_type``
    Residential classes are ``Single-Family``, ``Manufactured Home``,
    ``RV Dwelling`` and the multi-family classes, which are split into
    HAZUS height bands (``Low-Rise`` 1 to 3 stories, ``Mid-Rise`` 4 to
    7, ``High-Rise`` 8 or more); a multi-family building with no story
    count keeps the plain ``Multi-Family``. ``Secondary`` is a garage,
    shed or barn on a lot whose main building is elsewhere. Other
    classes are NSI's non-residential occupancy classes (``Retail``,
    ``Church``, ...). The class is decided by a vote among NSI, FEMA,
    the roll and the footprint's shape (see :doc:`pipeline`).

*Related columns:*

- ``land_use_class_parcel`` (canonical): the class of the lot the
  footprint stands on, from the county's use code. It keeps
  distinctions ``occupancy_type`` folds together: ``Townhome`` and
  ``Condominium`` lots hold multi-family or single-family buildings,
  and ``Manufactured Home Park`` lots hold manufactured homes.
- Evidence: each source's own class (``occupancy_type_building_nsi``,
  ``group_footprint_fema``, ``occupancy_type_parcel``),
  ``occupancy_type_conflict`` (where the sources disagree) and
  ``manufactured_home_community``.

.. _cheer_col_value:

Value
-----

The bundle carries three measures of a building's value. They answer
different questions and are not interchangeable.

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Column
     - Measures
     - Price date
   * - ``structure_value``
     - Market value of the building, net of land, as the county assessor
       states it
     - The county's appraisal date (``appraisal_year``)
   * - ``structure_value_adjusted``
     - The same value in 2025 dollars (FHFA county house price index)
     - 2025
   * - ``replacement_cost``
     - What rebuilding the building new would cost (NSI)
     - August 2025
   * - ``depreciated_replacement_cost``
     - Replacement cost less wear and age (NSI)
     - August 2025

``structure_value``
    A parcel's recorded improvement value, shared among its main
    buildings by floor area, or a filled value. No structure value in
    the bundle comes from NSI.

*Related columns:*

- ``appraisal_year`` (canonical): the year of the appraisal date the
  county's values reflect.
- ``structure_value_zero_reason`` (canonical): set only where
  ``structure_value`` is 0, and says why:

  - ``secondary``: a garage or shed, valued with the main building;
  - ``personal_property``: a manufactured home taxed as personal
    property, not on the real-property roll;
  - ``exempt``: a public or institutional building, or one the roll
    records as exempt, so the roll carries no taxable value for it;
  - ``included_elsewhere``: another building on the same lot holds the
    lot's value;
  - ``not_on_roll``: a building the footprint shows on a lot the roll
    calls vacant, often removed since the footprint was drawn;
  - ``no_value``: the roll states no value for the building.

  Some of these zeros are filled by the models (see
  :doc:`imputation`): the filled cell then holds a value and its
  ``_source`` says it was filled, while the reason stays, so you can
  take the roll's view back.
- ``improvement_value_parcel`` (canonical): the improvement value
  recorded for the lot, as shared to this building.
- ``structure_value_per_area`` (point file only): ``structure_value``
  divided by ``area_m2``, for quick maps.
- Evidence: the roll's whole-lot values
  (``improvement_value_parcel_whole``, ``land_value_parcel``,
  ``total_value_parcel``) and NSI's own values
  (``structure_value_building_nsi``,
  ``structure_value_full_building_nsi``).

.. _cheer_col_year_built:

Year built
----------

``year_built``
    The year the roll records, else a filled value. Where one footprint
    draws several buildings, it is the earliest of their years.

*Related columns (evidence):* the roll's year (``year_built_parcel``),
North Carolina's footprint layer (``year_built_footprint_ncdps``) and
NSI's year (``year_built_block_median_building_nsi``), which is the
median of the Census block, not a record of the building.

.. _cheer_col_dwellings:

Dwellings
---------

``n_dwellings``
    Housing units in the building: Overture's address points on it,
    else NSI's unit count, else a filled value.

*Related columns (evidence):* each source's count
(``n_dwellings_overture``, ``n_dwellings_nsi``, ``n_dwellings_parcel``).

.. _cheer_col_sections:

Manufactured-home sections
--------------------------

``n_sections``
    Only for manufactured homes: 1 for a single-wide, 2 for a
    double-wide, 3 for a triple-wide. Empty on every other class, and
    empty where the home's width and length point to different answers.
    A single-wide decided from the footprint's shape alone is a useful
    flag, not a fact.

*Related columns (evidence):* ``section_keyword_class``, the section
count the assessor states itself, and ``habitable_size_ratio``, the
footprint's size against a typical home in the county.

.. _cheer_col_area:

Area
----

``area_m2``
    The footprint's area. Where a footprint is a stand-in drawn from the
    lot (a building the roll records but no footprint layer draws), the
    lot's area would overstate the building, so the area is left to the
    models; ``area_m2_source`` says so.

Columns taken first from NSI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These columns come first from the National Structure Inventory (NSI),
else from the other sources named below, else from the models. Where
one footprint holds several buildings with different classes, a class
cell lists each, separated by a semicolon (``Slab; Crawl``).

.. _cheer_col_stories:

Stories
-------

``n_stories``
    The number of stories from NSI, else FEMA, else the earlier CHEER
    inventory, else a filled value.

*Related columns:*

- ``height_unconfirmed`` (canonical): true where the building's value
  per m2 of footprint is high enough to imply a tall building, but its
  story count does not explain it. Rare.
- Evidence: each source's count (``n_stories_parcel``,
  ``n_stories_building_nsi``, ``n_stories_footprint_overture``,
  ``n_stories_footprint_fema``, ``n_stories_building_cheer``) and the
  drawn heights (``height_footprint_overture``,
  ``height_footprint_fema``).

.. _cheer_col_foundation:

Foundation
----------

``foundation_type``
    NSI's foundation type, else the earlier CHEER inventory's, else a
    filled value.

*Related columns (evidence):* ``foundation_type_building_nsi`` and
``foundation_type_building_cheer``.

Columns taken first from the earlier CHEER inventory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These columns come first from the earlier CHEER inventory (CHEER
Inventory v0, see :doc:`research`), else from the sources named below,
else from the models. Class cells of a footprint with several buildings
list each class, separated by a semicolon (``Wood; Masonry``).

.. _cheer_col_construction:

Construction
------------

``construction_type``
    The earlier CHEER inventory's construction type, else NSI's, else a
    filled value.

*Related columns (evidence):* ``construction_type_building_cheer`` and
``construction_type_building_nsi``.

.. _cheer_col_roof:

Roof shape
----------

``roof_shape``
    The earlier CHEER inventory's roof shape, else a filled value. Where
    the optional recognition from imagery runs, its roof shape comes
    first.

*Related columns (evidence):* ``roof_shape_building_cheer``, and the
classifier's confidence and its second and third choices
(``roof_shape_confidence``, ``roof_shape_2``, ``roof_shape_3``).

.. _cheer_col_garage:

Garage
------

``has_garage``
    1 or 0, from the earlier CHEER inventory, else a filled value.

Identifiers and location
~~~~~~~~~~~~~~~~~~~~~~~~

``footprint_id``
    The footprint's id: the Open Location Code (plus code) of its
    centroid. Where two footprints would get the same code, the code is
    lengthened (a finer cell), so every id stays a location. Every file
    of the bundle carries it, in the same row order.

    Rarely, two neighboring counties both delivered one footprint. The
    delivery then cuts it at the county line, and each piece keeps its
    own county's values under a new id. The evidence file's
    ``footprint_id_parent`` names the footprint the pieces came from;
    add up the pieces' values to get the building's.
``admin3_id``, ``county_fips``
    The county, as an ``openplaces`` id (``US-NC-PEN``) and as its
    5-digit FIPS code (``37141``). Use ``county_fips`` to join Census,
    FEMA or other federal data.
``census_tract_id``, ``census_blockgroup_id``
    The Census areas the footprint's centroid lies in. More areas (block,
    ZIP code area, county subdivision) are in the evidence file.
``address``
    The street address, from the roll or Overture.
``lat``, ``long``
    The footprint's centroid (WGS84).

All columns
~~~~~~~~~~~

Every canonical column has a ``_source`` column beside it (see
:doc:`provenance`), unless the table says otherwise. The key columns
link to their explanation above. The evidence file's columns are
grouped by the source that supplied them.

.. The fragment is regenerated with
   python -m openplaces.flow.delivery_columns US_footprint-cheer-2026
   --bundle <canonical parquet of a delivered bundle>
   --described cheer_col_occupancy=occupancy_type
   --described cheer_col_value=structure_value
   --described cheer_col_year_built=year_built
   --described cheer_col_dwellings=n_dwellings
   --described cheer_col_sections=n_sections
   --described cheer_col_area=area_m2
   --described cheer_col_stories=n_stories
   --described cheer_col_foundation=foundation_type
   --described cheer_col_construction=construction_type
   --described cheer_col_roof=roof_shape
   --described cheer_col_garage=has_garage

.. include:: ../../_generated/US_footprint-cheer-2026_columns.rst
