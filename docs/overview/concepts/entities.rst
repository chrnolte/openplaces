.. openplaces

.. _entities:

Entities
========

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

Transactions are events in which a **property** changes hands, often in the form of a sale. They are recorded in deeds or similar property documents. They identify the seller, buyer, the property, and the date.


.. _buildings:

Buildings
~~~~~~~~~

Buildings refer to any human-built structures with roofs.

Buildings are often identified by their building **footprint**, as observed in high-resolution satellite data. Some detailed tax assessor datasets also collect building information separately from property information (e.g. in the case of multi-unit houses, condominiums, etc.).

.. _administrative_units:

Administrative units
~~~~~~~~~~~~~~~~~~~~

Administrative units are the geographic regions at which property data generation is organized.

``openplaces`` comes with a globally usable hierarchical system for administrative unit identifiers, principally derived from the `Global Administrative Database <https://gadm.org/>`_.

- Level 0 (``admin0``): countries

  Identified mostly by ISO Alpha-2 code, with gaps filled.

  - ``'US'`` for United States
  - ``'CO'`` for Colombia

- Level 1 (``admin1``): states, departments

  - ``'US-MA'`` for the state of Massachusetts, U.S.
  - ``'CO-AN'`` for the department of Antioquia, Colombia

- Level 2 (``admin2``): counties, municipalities, etc.

  - ``'US-MA-MI'`` for Middlesex county, Massachusetts, U.S.
  - ``'CO-AN-ME'`` for the municipality of Medellín, Antioquia, Colombia

- Level 3 (``admin3``): cities, towns, subdivisions, etc.

  *Not globally supported: requires custom recipe for country/region*

  - ``'US-MA-MI-SO'`` for the town of Somerville, Middlesex, Massachusetts, U.S.
