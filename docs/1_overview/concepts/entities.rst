.. openplaces

.. _entities:

Entities
========

Entities are the fundamental building blocks of property information.

Most data we process in ``openplaces`` is ultimately attributed to one of these entities.


.. _parcels:

Parcels
~~~~~~~

Parcels refer to geo-referenced parcel boundaries that describe a spatial unit of property: a lot of land, a piece of Earth.

Parcel data is most often created by local land surveyors and tax assessors, usually with the goal to cover all taxable property within a given administrative unit / jurisdiction.


.. _properties:

Properties
~~~~~~~~~~

Unit at which property is assigned (and taxed).

For single-unit parcels (e.g. single-family homes) and vacant lots, the property *is* the parcel and everything on it.

For multi-unit parcels (e.g. an apartment complex), parcels and properties are not identical and need to be treated separately.


.. _transactions:

Transactions
~~~~~~~~~~~~

Transactions are events in which one or more :ref:`properties` change full or partial ownership, often in the form of a sale or easement. They are recorded in deeds or similar property documents. They identify the seller, buyer, the property, and the date.


.. _buildings:

Buildings
~~~~~~~~~

Buildings refer to any human-built structures with roofs.

Buildings are often identified by their building **footprint**, as observed in high-resolution satellite data. Some detailed tax assessor datasets also collect building information separately from property information (e.g. in the case of multi-unit houses, condominiums, etc.).


Administrative units
~~~~~~~~~~~~~~~~~~~~

:ref:`Administrative units <administrative_units>` are a special type of entity. For more, see the :ref:`section on administrative units <administrative_units>`.