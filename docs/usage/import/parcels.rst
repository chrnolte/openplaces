.. openplaces

.. _import_parcels:

Parcel data
===========

Parcel data refer to any dataset that contains geospatial property boundaries (vector data, usually 2D polygons) maintained by government agencies for the purpose of property tax assessment.

Parcel data can come with a wide range of attributes used by the tax assessor to collect property-specific information.


Understand your parcel data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multiple entities?
------------------

Parcel datasets are usually a merger of multiple sources of data. Frequently, attributes (columns) refer to different :ref:`entities` (rows, e.g. parcels, buildings, units within buildings, sales transactions, etc.).

-  For instance, multiple :ref:`properties` can exist on a single :ref:`parcel`: duplexes, apartments, condominium associations, manufactured home parks, high-rises, etc.

   Parcel datasets handle this multi-entity problem in different ways. Some contain one row per property, but then duplicate parcel attributes and geometries for each. Multi-table datasets (e.g. geodatabases :file:`.gdb`) often provide separate tables for parcels (often called geometries or GIS) and properties (often called tax roll, assessment, or similar).

-  Some parcel datasets provide last sales prices, sometimes multiple ones (in multiple columns), which refer to :ref:`transactions`

Which columns of your parcel data refer to:

-  parcels (e.g., boundary, parcel ID)?

-  property (e.g., tax value)

-  buildings (e.g., roof type)?

-  units in buildings (e.g. number of bathrooms)?

-  transactions (e.g. sales prices)?

Does your dataset provide these data in separate tables or as columns in one large (merged) table?

If the dataset is merged, can you de-compose the data into its original entities (i.e., a separate table for each)? Is there a unique property identifier for every row in the parcel data table? Does it appear to refer to land parcels (property boundaries) or to taxable properties?

Geometries
----------

Are there invalid geometries that need fixing (e.g. with a zero-buffer)?

Are there empty geometries? What do you want to do with them?

Attributes
----------

What attributes exist?

Can you map or translate all of them to the `variable dictionary <https://docs.google.com/spreadsheets/d/1_m3d3l8ngTYXXI7onRNsz0zau6UVejIn1rF1IgxF0ns/edit?gid=369370336#gid=369370336>`_?

If not, which columns would you like to keep and add to the variable dictionary?

Administrative referencing
--------------------------

Is it easy to break down the larger parcel dataset by administrative subdivisions (e.g., a county in the United States) for subsequent data processing?

For instance, is the data already provided by subdivision, or is there a column that allows you identify the administrative subdivision?
