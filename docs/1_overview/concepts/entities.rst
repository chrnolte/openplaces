.. openplaces

.. _entities:

Entities: the units of analysis
===============================

Entities are the units of analysis in ``openplaces``.

They refer to the fundamental building blocks of property information: parcels, buildings, transactions, etc.

In a table or DataFrame, entities are represented by **rows**: each row is a unique entity (e.g., a building).

Most data processed in ``openplaces`` is attributed to an entity.

Datasets organized by entities are covered here. For datasets that are not organized by entity (e.g., a global raster image, a text file, or a non-entity table), see :ref:`datasets <datasets>`.


Building blocks
~~~~~~~~~~~~~~~

The building blocks below are not levels of one hierarchy.

Each is the unit of a different record-keeper: the cadastre draws parcels, imagery draws footprints, engineers and hazard models see structures, the census and the postal system count dwellings, and the tax roll records properties.

On a detached house, they coincide.

Everywhere else, they diverge, which is why ``openplaces`` keeps one table per entity rather than a single building table with columns from each.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Entity
     - One row is
   * - :ref:`parcel <parcels>`
     - One unit of land as the cadastre draws it.
   * - :ref:`footprint <footprints>`
     - One building outline polygon. It may cover one building or several (a townhome row).
   * - :ref:`building <buildings>`
     - One structure. A townhome row drawn as a single footprint is several buildings; a condominium building holding many units is one building.
   * - :ref:`dwelling <dwellings>`
     - One housing unit. A single-family home is one dwelling; a multi-family building is one dwelling per unit.
   * - :ref:`property <properties>`
     - One unit of ownership as a tax roll records it: what a sale conveys. A single-family home on its lot is one property; a condominium unit is one property.
   * - :ref:`transaction <transactions>`
     - One recorded sale or conveyance.

The same text lives in ``openplaces.core.schema.ENTITY_DEFINITIONS``, and a test keeps the two in step.


.. _parcels:

Parcels
-------

Parcels are geo-referenced boundaries that describe a spatial unit of property: a lot of land.

Parcel data is most often created by local land surveyors and tax assessors, typically with the goal of covering all taxable property within a given administrative unit.

Each parcel is indexed by a geometry-derived :ref:`parcel_id (geo_id) <parcel_id>`, a stable fingerprint that identifies the same lot across data versions and sources.

A parcel is land only: a condominium unit has no parcel of its own, and a manufactured home may be owned apart from the lot it stands on.


.. _footprints:

Footprints
----------

Footprints are the boundaries of a building envelope as seen from space.

Footprints are usually produced from satellite imagery.

Because a footprint is what can be seen from above, a row of townhomes in New York is one footprint holding several buildings, each on its own parcel.

Some hazard models (e.g., for hurricane exposure) operate at the footprint level, such as :ref:`CHEER footprints <cheer_footprints>`.


.. _buildings:

Buildings
---------

.. image:: images/footprint_building_dwelling_single.png
  :width: 300
  :alt: Illustration of footprints, buildings, and dwellings
  :align: right

Buildings are human-built structures with a roof.

Buildings often constitute the largest share of a parcel's value.

Some buildings are separable from parcels (e.g., a manufactured home).

Hazard risk models often require:

- The location of buildings (e.g., for flood risk models).
- Structural properties, e.g., for earthquakes, hurricanes, and tornadoes.

A footprint may hold several buildings, and a building may hold many dwellings and many properties.


.. _dwellings:

Dwellings
---------

.. image:: images/footprint_building_dwelling_urban.png
  :width: 400
  :alt: Illustration of footprints, buildings, and dwellings
  :align: right

Dwellings are individual residential units within a building, such as:

- An apartment in a building.
- A unit in a two-family home.
- A condominium unit.
- A single-family home.

Address and census databases commonly refer to dwellings.

A dwelling is a housing unit regardless of its ownership: a rental apartment building is many dwellings and one property.


.. _properties:

Properties
----------

Properties are the assets (property rights) that are sold, valued, and taxed.

Ownership, not structure, draws the boundary: a rental apartment building is one property, and a condominium building of the same size is many.

The taxable property is the unit by which most tax assessors organize information, so a tax roll's rows are properties regardless of what each row describes.

A property need not be land or a building at all (e.g., a right-of-way).


.. _transactions:

Transactions
------------

Transactions are events in which one or more :ref:`properties <properties>` change full or partial ownership.

These typically take the form of sales or easements and are recorded in deeds or similar documents.

One transaction may convey several properties, and a single property may sell many times.

Transaction data, which may identify the seller, buyer, property, and date, is private in many countries.


Worked cases
------------

The table below counts rows per entity for common situations.

Every column differs from every other in at least one row, showing why each entity is modeled separately.

.. list-table::
   :header-rows: 1
   :widths: 40 12 12 12 12 12

   * - Situation
     - parcel
     - footprint
     - building
     - dwelling
     - property
   * - Vacant lot
     - 1
     - 0
     - 0
     - 0
     - 1
   * - Detached house on its lot
     - 1
     - 1
     - 1
     - 1
     - 1
   * - House with a detached garage
     - 1
     - 2
     - 2
     - 1
     - 1
   * - Townhome row, each home on its own lot
     - n
     - 1
     - n
     - n
     - n
   * - Condominium building
     - 1
     - 1
     - 1
     - n
     - n
   * - Rental apartment building
     - 1
     - 1
     - 1
     - n
     - 1
   * - Manufactured-home park, lots rented
     - 1
     - n
     - n
     - n
     - 1

In the manufactured-home park, the homes are personal property in most U.S. states and appear on a separate roll, if at all.


Spatial reference and partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two types of entities - :ref:`administrative units <administrative_units>` and :ref:`tiles <tiles>` - serve as spatial partitions for data storage and ingestion of other entities.


.. _entity_administrative_units:

Administrative units
--------------------

:ref:`Administrative units <administrative_units>` are a special type of entity. See the :ref:`section on administrative units <administrative_units>` to learn how they are defined and referred to.

All :ref:`recipes <recipes>` belong to an administrative unit (global, country, state, county, or similar). Many external dataset downloads are partitioned by administrative units (e.g., U.S. building footprints by state). Most datasets in ``openplaces`` are organized by administrative unit.

.. _tiles:

Tiles
-----

Tiles are fixed spatial grid cells that cover a geographic area.
They are used to partition global or large-area datasets into manageable download and processing chunks.

For instance, `OpenBuildingMap <https://gee-community-catalog.org/projects/obm/>`_ serves global building footprints by tile. `Global Forest Change <https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html>`_ serves global raster data by tile.
