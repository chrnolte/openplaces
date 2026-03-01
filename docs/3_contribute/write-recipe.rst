
.. _writing_recipes:

Writing recipes
===============

Recipes are the instructions that allow ``openplaces`` to ingest new data into its data structure.

By :ref:`contributing <contribute>` recipes, you permit others to reproduce your work.

Recipe can be found in :file:`src/openplaces/recipes`.


Data ingestion recipes
~~~~~~~~~~~~~~~~~~~~~~

Recipes that manage data ingestion: file handling, data formatting, and gentle preprocessing (no major edits to data).


Dataset description and source
------------------------------

.. attribute:: admin_id

   Admin ID of the (top-level) :ref:`administrative unit <administrative_units>` for which the dataset provides data.

   :input:`NULL`
      global, e.g., the Global Administrative Database

   :input:`US`
      United States, e.g., US Microsoft Building Footprints

   :input:`US-MA`
      Massachusetts, U.S., e.g., MassGIS tax parcels

   :input:`US-MA-MI`
      Middlesex county, Massachusetts, United States
   
   :input:`US-MA-MI-SO`
      City of Somerville, Massachusetts, United States

.. attribute:: entity

   :ref:`Entity <entities>` of the dataset: the "thing" that every row of the dataset table refers to.

   .. attribute:: entity_type

      Type of entity, e.g. :ref:`admin <administrative_units>` unit, :ref:`property <properties>`, :ref:`parcel <parcels>`, :ref:`building <buildings>`, :ref:`transaction <transactions>`.

   .. attribute:: source

      Source of the data

      .. attribute:: portal_url

         URL of the landing page of the portal that offers and explains the data

      .. attribute:: download_url

         URL to download the data directly.

         If :attr:`download_by` is also set, :attr:`download_url` can include placeholders to be resolved, e.g.:

         :input:`https://nsi.sec.usace.army.mil/downloads/nsi_2022/nsi_2022_{admin2_id_admin1}.gpkg.zip`

      .. attribute:: download_url_source

         URL of page from which the download URL can be extracted.

      .. attribute:: download_url_source_regex

         String pattern (regular expression) that permits the extraction of the download URL from the HTML content of :attr:`download_url_source`

   .. attribute:: version

      Version of the dataset (year or version number)


File handling
-------------

.. attribute:: download_by

   Instructions to download data that is provided in smaller download partition, commonly by administrative subdivision (state, county, etc.).

   .. attribute:: admin_level

      Level of the administrative breakdown for download partitions

   .. attribute:: admin_key_transform
    
      Rules to transform keys before substituting into download URL.

      *In development: only one transformation is currently supported (remove_spaces)*

.. attribute:: compressed_file_name

   Filename of the compressed file (usually in ``external`` folder, see :ref:`directory_structure`).

   Providing this argument allows the skipping of downloads if file is found.

   Can include placeholders (substituted by :attr:`download_by`).

   Can include wildcards (will search for files matching pattern).

.. attribute:: uncompressed_file_name

   Filename of uncompressed file, ready to be read.

   If :attr:`compressed_file_name` is set, the uncompressed file will be in the :file:`heap` folder

   If there is no :attr:`compressed_file_name`, the uncompressed file is assumed to be the original download, to be found in the :file:`external` folder.

   Can include placeholders (substituted by :attr:`download_by`).

   Can include wildcards (will search for files matching pattern).


Reading
-------

.. attribute:: layer

   Layer to be read from the file.

   Geodatabases (:file:`.gdb`) and geopackages (:file:`.gpkg`) can have multiple layers.


.. attribute:: process_by

   Instructions to process the data file in smaller processing chunks, commonly by administrative subdivision (reading a state geodatabase by county).

   .. attribute:: admin_level
      :no-index:

      Level of the administrative breakdown for processing chunks.

   .. attribute:: admin_id_column

      Original name of the column in the input data table that contains the information needed to split the dataset into processing chunks (commonly an administrative identifier, but can be a census block).

   .. attribute:: admin_id_crosswalk

      Instructions to crosswalk Admin IDs from the input dataset to ``openplaces`` Admin IDs (e.g. :input:`'US-MA-MI'`).


Columns
-------

.. attribute:: columns

   Dictionary of columns in the original dataset to keep and (optionally) rename.

   Provided as ``key: value`` pairs:

   ``openplaces_column_name``: ``original_column_name``

   Example (from Global Administrative Database):

   .. code-block:: yaml

      columns:
        name: NAME_1
        type: ENGTYPE_1 
        admin1_name: COUNTRY

.. attribute:: keep_unnamed_columns

   Set to :input:`True` to keep columns not named in :attr:`columns`.

   Includes columns that were generated during index creation (e.g. :attr:`create_index`).


Space saving
------------

.. attribute:: columns_to_categorical

   Columns to be cast to categorical (saving space in memory and on disk).


Filters
-------

.. attribute:: query

   Query to filter the dataframe. 

   Will be passed to pd.DataFrame.query.

   Use to remove unwanted geometries

   Example: :input:`"type != 'Water body'"` in GADM.


Indexing
--------

.. attribute:: set_index

   Name of column to be used as the index.

.. attribute:: create_index

   Function to give the GeoDataFrame an index

   Example:

   .. code-block:: yaml

      create_index:
        function: "openplaces.geo.vector.add_geo_id_index"
        args:
          name: footprint_id

   .. attribute:: function

      ``openplaces`` function to use for creating the index.

      Example:
         :input:`"openplaces.geo.vector.add_geo_id_index"`

   .. attribute:: args

      Arguments passed to :attr:`function` (e.g., the name of the index)

   .. attribute:: method

      Apply function identified via string.

      *Deprecated: move to* :attr:`function`

.. attribute:: index_function

   Like :attr:`create_index` but without arguments.

   *Deprecated: move to* :attr:`create_index.function`

.. attribute:: sort_index

   Set to :input:`True` to sort the dataset by its index (after index creation).


Data transformations
--------------------

.. attribute:: transformations
   
   Operations to perform on data to create new derivative columns

   Can use any ``openplaces`` transformations (see ``openplaces.io.transform``).

   Example: extracting the first five digits from the US census block ID to get county FIPS code.

   .. code-block::

      transformations:
        - output: admin3_id_admin1
          type: string
          operation: substring
          input: census_block_id
          args:
            start: 0
            end: 5


Attribution to administrative units
-----------------------------------


.. attribute:: admin_id_crosswalk
   :noindex:

   Add ``openplaces`` Admin ID column using a crosswalk from input Admin IDs to ``openplaces`` Admin IDs.

   .. attribute:: admin_level
      :noindex:

      Administrative level of crosswalk, e.g. :input:`3`.

   .. attribute:: admin_id_column
      :noindex:

      Input data column to crosswalk, e.g. :input:`admin3_id_admin1`.

   .. attribute:: admin_recipe_id
      :noindex:

      Recipe to derive the crosswalk, e.g. :input:`"US_admin-nhgis-2020_admin3"`.


.. attribute:: overlay_admin_ids

   Assign administrative IDs to rows using spatial overlays.

   .. attribute:: admin_level
      :no-index:

      Level of administrative units to assign to geometries

   .. attribute:: admin_recipe_id
      :no-index:

      Recipe from which to obtain geometries for administrative units
      (commonly an in-country recipe, as it needs to be sufficiently
      precise to allocate, e.g., building footprints correctly.)

      Example: :input:`'US_admin-nhgis-2020_admin3'` for US counties.


Saving
------

.. attribute:: cache_by

   Save data in cache in chunks.

   Argument is not needed if processing already happens at that level (because :attr:`download_by` or :attr:`process_by` is set)

   .. attribute:: admin_level
      :no-index:

      Admin level at which to save the data in cache.

.. attribute:: cache_filename

   Name of filename to save in cache.

   This is only needed if the filename cannot be derived from the reference (admin_id, entity, etc.) alone. Use if source file has multiple layers or if `layer` has a value.