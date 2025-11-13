.. openplaces

Parcel data
===========

Parcel data refer to any dataset that contains geospatial property boundaries (vector data, usually 2D polygons) maintained by government agencies for the purpose of property tax assessment.

Parcel data can come with a wide range of attributes used by the tax assessor to collect property-specific information.


Understand your parcel data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internal structure
------------------

Parcel datasets often blend several sources of data that refer to distinct entities related to property (parcels, buildings, units within buildings, transactions)

Multiple units properties can exist on a single parcel (e.g. multi-unit housing, condos). Parcel datasets handle that in a variety of ways: some duplicate parcel geometries for each property, leading to duplicates. Others provide a database with separate tables for parcels and properties.

Which columns of your parcel data refer to parcels (e.g. boundary), buildings (e.g. number of stories), property units (e.g. number of bedrooms), and transactions (e.g. sales prices)?

Does your dataset provide these data in separate tables?

If your dataset blends these sources, is there a unique property identifier for every row in the parcel data table? Does it appear to refer to land parcels (property boundaries) or to taxable properties?

(How) does the dataset distinguish parcels and transactions?

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

Is it easy to break down the larger parcel dataset by administrative subdivisions (e.g., a county in the United States) for faster processing? For instance, is the data already provided by subdivision, or is there a column that allows you identify the administrative subdivision?
