.. openplaces

Parcel data
===========

Parcel data refer to any dataset that contains geospatial property boundaries (vector data, usually 2D polygons) maintained by government agencies for the purpose of property tax assessment.

Parcel data can come with a wide range of attributes used by the tax assessor to collect property-specific information.


Understand your parcel data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

What does the data need to be split into parcels, properties, and sales?

-  How does the dataset distinguish parcels and properties?

   Multiple properties can exist on a single parcel (e.g. multi-unit housing, condos). Parcel datasets handle that in a variety of ways: some duplicate parcel geometries for each property, leading to duplicates. Others provide a database with separate layers/tables for parcels and properties.

   Is there a unique property identifier for every row in the data table?

   Is there a separate table of properties linked to parcels via a parcel identifier?

-  How does the dataset distinguish parcels and transactions?

   Some parcel datasets contains last sale data.

   Are they provided as a separate table or as part of the parcel/property data?

-  Are there empty geometries? What do you want to do with them?

