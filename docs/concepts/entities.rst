.. openplaces

Entities
========


Parcels
~~~~~~~

Parcels refer to geospatial parcel boundaries that define a spatial unit of property (a lot of land).

Parcel data is usually created by local land surveyors and tax assessors for all land within a given administrative unit or jurisdiction.


Properties
~~~~~~~~~~

Unit at which property is assigned and taxed.

For single-unit parcels (e.g. single-family homes) and vacant lots, the property *is* the parcel and everything on it.

For multi-unit parcels (e.g. an apartment complex), parcels and properties are not identical and need to be treated separately.


Transactions
~~~~~~~~~~~~

Transactions are events in which a **property** changes hands, often in the form of a sale. They are recorded in deeds or similar property documents. They identify the seller, buyer, the property, and the date.


Buildings
~~~~~~~~~

Buildings are structures on land, often identified as a footprint derived from high-resolution satellite data, or from a tax assessor dataset on buildings.


Administrative units
~~~~~~~~~~~~~~~~~~~~

Administrative units are the geographic regions at which property data generation is organized.

``openplaces`` comes with a globally usable hierarchical system for administrative unit identifiers, principally derived from the `Global Administrative Database <https://gadm.org/>`_.

- Level 0 (``admin0``): countries

   Identified mostly by ISO Alpha-2 code, with gaps filled.

   ``'US'`` for United States
   ``'CO'`` for Colombia

- Level 1 (``admin1``): states, departments

   ``'USMA'`` for the state of Massachusetts, U.S.
   ``'COAN'`` for the department of Antioquia, Colombia

- Level 2 (``admin2``): counties, municipalities, etc.

   ``'USMAMI'`` for Middlesex county, Massachusetts, U.S.
   ``'COANME'`` for the municipality of Medellín, Antioquia, Colombia
