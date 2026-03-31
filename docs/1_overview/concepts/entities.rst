.. openplaces

.. _entities:

Entities
========

Entities are the units of analysis in ``openplaces``.

They refer to the fundamental building blocks of property information: parcels, buildings, transactions, etc.

In a table or dataframe, entities are represented by **rows** (each row being an instance of an entity, e.g., a building).

Most data processed in ``openplaces`` is ultimately attributed to an entity.

Datasets organized by entities are covered in this section. For datasets that are not organized by entity (e.g. an image, a text, a non-entity table), see :ref:`datasets <datasets>`.


Building blocks
~~~~~~~~~~~~~~~

.. _parcels:

Parcels
-------

Parcels are geo-referenced boundaries that describe a spatial unit of property: a lot of land, a piece of Earth.

Parcel data is most often created by local land surveyors and tax assessors, typically with the goal of covering all taxable property within a given administrative unit.


.. _buildings:

Buildings
---------

Buildings are human-built structures with roofs.

They are most often identified by their **footprint** as observed in high-resolution satellite or aerial imagery.
Some detailed tax assessor datasets also collect building information separately from property information (e.g. in the case of condominiums or multi-unit structures).


.. _dwellings:

Dwellings
---------

Dwellings are individual residential units within a building or parcel. Examples: a single apartment in a multi-family building; a condominium (many dwelling in one building); a single-family home (1 dwelling = 1 building).

Dwellings sit below the parcel and property level: a single parcel may contain one building that contains many dwellings.
They are relevant when data sources distinguish individual housing units (e.g. from assessor CAMA records or building permit data).


.. _properties:

Properties
----------

Unit at which property is assigned and taxed.

This is the unit by which tax assessors organize information: the taxable property.

In practice, properties can refer to any collection of subsets of the above: a property can be only about parcels (and not include the manufactured home on it), only about dwellings (a condominium), about both (a multi-apartment complex), or be about a different type of right (e.g., right of way).


.. _transactions:

Transactions
------------

Transactions are events in which one or more :ref:`properties` change full or partial ownership, typically in the form of a sale or easement.

They are recorded in deeds or similar property documents and identify the seller, buyer, the property, and the date.


Spatial reference and partitioning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two types of entities — :ref:`administrative units <administrative_units>` and :ref:`tiles <tiles>` — serve as spatial partitions for data storage and ingestion of other entities.

.. _entity_administrative_units:

Administrative units
--------------------

:ref:`Administrative units <administrative_units>` are a special type of entity. See the :ref:`section on administrative units <administrative_units>` to learn how they are defined and referred to.

All :ref:`recipes <recipes>` belong to an administrative unit (global, country, state, county or similar). Many external dataset downloads are partitioned by administrative units (e.g., US building footprints by state). Most datasets in ``openplaces`` are organized by administrative unit.

.. _tiles:

Tiles
-----

Tiles are fixed spatial grid cells that cover a geographic area.
They are used to partition global or large-area datasets into manageable download and processing chunks.

For instance, `OpenBuildingMap <https://gee-community-catalog.org/projects/obm/>`_ serves global building footprints by tile. `Global Forest Change <https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html>`_ serves global raster data by tile.
