
.. _writing_recipes:

Writing recipes
===============

Recipes are the instructions that allow ``openplaces`` to ingest new data into its data structure.

By :ref:`contributing <contribute>` recipes, you permit others to reproduce your work.

Recipes can be found in :gh-file:`src/openplaces/recipes`.


Data ingestion recipes
~~~~~~~~~~~~~~~~~~~~~~

Recipes that manage data ingestion: file handling, data formatting, and gentle preprocessing (no major edits to data).


Dataset description and source
------------------------------

.. attribute:: description

   One-sentence summary of the dataset: what it is, what it covers, and
   where it comes from. Keep it to 25 words or less.

   The :ref:`recipe catalog <recipe_catalog>` is generated from this field
   (parsed as reStructuredText) at every documentation build, so every
   recipe should have one. Put longer commentary in :attr:`notes`, and keep
   ``#`` comments for short, key-specific implementation notes.

   Example:

   .. code-block:: yaml

      description: >-
        Massachusetts registry of deeds transactions, crawled from
        masslandrecords.com.

.. attribute:: notes

   Long-form prose about how the recipe obtains the data: what the source
   publishes, access quirks (portals, scrapers, formats), and any
   conventions the recipe follows.

   Written as a YAML block scalar (``notes: |``). Optional, and not shown
   in the recipe catalog.

.. attribute:: admin_id

   Admin ID of the (top-level) :ref:`administrative unit <administrative_units>` for which the dataset provides data.

   :input:`NULL`
      Global

   :input:`US`
      United States

   :input:`US-MA`
      Massachusetts, U.S.

   :input:`US-MA-MI`
      Middlesex county, Massachusetts, United States.

   :input:`US-MA-MI-SO`
      City of Somerville, Middlesex county, Massachusetts, United States.

.. attribute:: entity

   :ref:`Entity <entities>` of the dataset: the "thing" that every row of the dataset table refers to.

   Use :attr:`entity` when each row in the data is a fundamental building block of ``openplaces`` — a parcel, building, admin unit, etc.

   Use :attr:`dataset` instead when the data is not itself an entity type (see :attr:`dataset` below).

   .. attribute:: entity_type

      Type of entity, e.g. :ref:`admin <administrative_units>`, :ref:`parcel <parcels>`, :ref:`property <properties>`, :ref:`building <buildings>`, :ref:`transaction <transactions>`, :ref:`tile <tiles>`.

   .. attribute:: source

      Source of the data.

      .. attribute:: portal_url

         URL of the landing page of the portal that offers and explains the data.

      .. attribute:: download_url

         URL to download the data directly.

         If :attr:`download_by` is also set, :attr:`download_url` can include placeholders to be resolved, e.g.:

         :input:`https://nsi.sec.usace.army.mil/downloads/nsi_2022/nsi_2022_{admin2_id_admin1}.gpkg.zip`

      .. attribute:: download_url_source

         URL of page from which the download URL can be extracted.

      .. attribute:: download_url_source_regex

         String pattern (regular expression) that extracts the download URL from the HTML content of :attr:`download_url_source`.

         Placeholders in the pattern (e.g. ``{admin3_name}``) are substituted with the current partition value before matching.

   .. attribute:: version

      Version of the dataset (year, date, or version number).

.. attribute:: dataset

   :ref:`datasets` provide attributes that will be linked to :ref:`entities <entities>`.

   Datasets can come in many formats: tables, vectors, rasters, XML. What distinguishes them from :ref:`entities <entities>` is that datasets are *not yet* organized by entity: the rows don't refer to the entity yet (and therefore require some linkage algorithm).

   .. attribute:: theme

      Short descriptor of the data theme, e.g. :input:`landcover-annual`.

   .. attribute:: source
      :no-index:

      Same sub-attributes as the :attr:`source` of entities.

   .. attribute:: version
      :no-index:

      Version of the dataset (year, date, or version number).

   .. attribute:: is_raster

      Set to :input:`True` for raster datasets, so the file is not treated as a table or vector layer (default).


File handling
-------------

.. attribute:: download_by

   Instructions for downloading data that is provided in partitioned chunks (by administrative subdivision, tile, or year). Not needed if the data is provided as a single file.

   .. attribute:: admin_level

      Level of the administrative breakdown for download partitions:

      | :input:`2` for state-level files
      | :input:`3` for county-level files

   .. attribute:: admin_key_transform

      Rules to transform admin key values before substituting them into the download URL (e.g. remove spaces from county names).

      Example (NC building recipe, which uses the county name in the URL):

      .. code-block:: yaml

         admin_key_transform:
           admin3_name: remove_spaces

   .. attribute:: partition

      Type of non-admin partition. Supported values:

      :input:`year`
         Download one file per year. Requires :attr:`first` and :attr:`last`.

      :input:`year_month`
         Download one file per calendar month. Requires :attr:`first` and
         :attr:`last` as ``YYYYMM`` values.

      :input:`tile_id`
         Download one file per tile. Requires :attr:`tile_recipe_id`.

   .. attribute:: first

      First year (or ``YYYYMM`` month) to download when :attr:`partition` is
      :input:`year` or :input:`year_month`.

   .. attribute:: last

      Last year (or ``YYYYMM`` month) to download when :attr:`partition` is
      :input:`year` or :input:`year_month`.

   .. attribute:: tile_recipe_id

      Recipe ID for the tile index when :attr:`partition` is :input:`tile_id`.

      Example: :input:`tile-obm-2025`.

   Example (raster dataset partitioned by year):

   .. code-block:: yaml

      download_by:
        partition: year
        first: 1986
        last: 2024

   Example (building footprints partitioned by tile):

   .. code-block:: yaml

      download_by:
        partition: tile_id
        tile_recipe_id: tile-obm-2025

.. attribute:: compressed_file_name

   Filename of the compressed file (usually in the :file:`external` folder; see :ref:`directory_structure`).

   Providing this skips re-downloading when the file is already present.

   Can include placeholders substituted by :attr:`download_by` (e.g. ``{admin2_id_leaf}``) and wildcards (the ingester will search for a matching file).

.. attribute:: uncompressed_file_name

   Filename of the uncompressed file, ready to be read.

   If :attr:`compressed_file_name` is set, the uncompressed file is expected in the :file:`heap` folder after extraction.

   If there is no :attr:`compressed_file_name`, the uncompressed file is treated as the original download in the :file:`external` folder.

   Can include placeholders and wildcards.

.. attribute:: encoding

   Character encoding of the source file, passed directly to the underlying reader (e.g. :input:`latin-1`).

   Only needed when the file is not UTF-8.

.. attribute:: process_by

   Instructions to process a large data file in smaller chunks, typically by administrative subdivision (e.g. reading a state-wide geodatabase county by county to avoid loading it entirely into memory).

   .. attribute:: admin_level
      :no-index:

      Level of the administrative breakdown for processing chunks.

   .. attribute:: admin_id_column

      Name of the column in the input data that identifies the processing chunk (commonly a county FIPS code or similar admin identifier).

   .. attribute:: admin_id_transformation

      Optional transformation to apply to the values in :attr:`admin_id_column` before building the crosswalk.

      Useful when the source column contains a partial identifier that needs to be modified to match ``openplaces`` Admin IDs (e.g. prefixing a 3-digit county code with a state FIPS code to produce a 5-digit FIPS).

      Uses the same transformation dict format as :attr:`transformations`.

      Example (Wisconsin parcels, which stores a 3-digit county code that needs a ``55`` prefix):

      .. code-block:: yaml

         admin_id_transformation:
           output: admin3_id_admin1
           type: string
           operation: add_prefix
           args:
             prefix: "55"

   .. attribute:: admin_id_crosswalk

      Instructions to map values from the source admin column (after any :attr:`admin_id_transformation`) to ``openplaces`` Admin IDs, so that rows can be filtered by admin chunk.

      Two forms are supported:

      **Recipe shorthand** — points to a prebuilt crosswalk table:

      .. code-block:: yaml

         admin_id_crosswalk:
           recipe_id: "US-MA_parcel-massgis-2025_admin4-crosswalk"

      **Dynamic crosswalk** — built at runtime from an admin recipe:

      .. code-block:: yaml

         admin_id_crosswalk:
           admin_level: 3
           admin_id_column: admin3_id_admin1
           admin_recipe_id: "US_admin-census-2021_admin3"

      In the dynamic form:

      .. attribute:: admin_level
         :no-index:

         Administrative level of the crosswalk target, e.g. :input:`3` for counties.

      .. attribute:: admin_id_column
         :no-index:

         Column name in the input data (or produced by :attr:`admin_id_transformation`) to match against the crosswalk.

      .. attribute:: admin_recipe_id
         :no-index:

         Recipe from which to derive the crosswalk, e.g. :input:`"US_admin-census-2021_admin3"`.

   .. attribute:: use_spatial_index

      Set to :input:`True` to use a spatial index on admin geometries to determine which rows fall within the current processing chunk, instead of using a crosswalk column.

      Requires admin geometries to be loaded (admin level and recipe are read from this same ``process_by`` block).

   .. attribute:: use_spatial_mask

      Set to :input:`True` to clip the data to the geometry of the current admin chunk instead of just filtering rows by a crosswalk column.

      Useful when source data does not have a reliable admin ID column and rows must be assigned by spatial containment.


Reading
-------

.. attribute:: layer

   Layer to read from the file.

   Geodatabases (:file:`.gdb`) and GeoPackages (:file:`.gpkg`) can contain multiple layers. Use this to identify which one to read.

   Also used as a cache key when :attr:`additional_layers` references the same source file.


Columns
-------

.. attribute:: columns

   Dictionary of columns to keep from the source data, with optional renaming.

   Provided as ``openplaces_column_name: original_column_name`` pairs.

   Only the listed columns are retained (plus any columns generated by index creation or admin attribution); use :attr:`keep_unnamed_columns` to retain additional columns.

   Example (from the Global Administrative Database):

   .. code-block:: yaml

      columns:
        name: NAME_1
        type: ENGTYPE_1
        admin1_name: COUNTRY

   The left-hand names should follow the openplaces canonical attribute naming
   convention:

   - **snake_case**, with role prefixes such as ``owner_``, ``tax_``,
     ``last_sale_``, ``source_``; ``{qualifier}_value`` for monetary fields;
     ``_code`` twins for coded categoricals; ``m2`` for square metres.
   - **Do not prefix a column with the dataset's own entity type, except for
     identifiers.** In a ``parcel`` dataset, ``parcel_id`` / ``parcel_id_admin1``
     (identifiers) are fine, but the parcel's own type is ``type`` (not
     ``parcel_type``) and its map number is ``mapnumber`` (not
     ``parcel_mapnumber``). Prefixes naming a *different* entity (e.g.
     ``owner_name``) are fine.
   - **Do not abbreviate words** — ``value`` not ``val``, ``subdivision`` not
     ``subd``, ``market`` not ``mkt``, ``number`` not ``no``.
   - **``n_``** is the prefix for counts (``n_dwellings``, ``n_sales``).

   The authoritative, entity-scoped list of canonical
   names lives in ``src/openplaces/core/attribute_registry.csv`` and is queryable
   in code via ``openplaces.core.attribute_registry.get_attributes(entity_type=...)``.
   It spans every entity type and pipeline stage (a blank ``entity_type`` means
   the name is shared). Reuse an existing canonical name wherever one fits; add a
   new registry row when introducing a genuinely new attribute.

   For positional or headerless sources, name columns only **once**: assign the
   canonical names directly in the positional spec (``fixed_width:`` field names,
   or ``names:`` together with ``header: none``) and omit ``columns:`` entirely.
   The ``columns:`` block only renames and reorders an already-read table, so it is
   redundant when the positional spec already carries the final names. Keep a
   source-style working name only for a field that feeds a ``transformations:``
   step rather than becoming an output column directly.

.. attribute:: keep_unnamed_columns

   Set to :input:`True` to retain all columns not listed in :attr:`columns`.

   Also retains columns generated during index creation (e.g. by :attr:`create_index`).

.. attribute:: parcel_id_local

   Compute a standardized, locally cross-comparable parcel matching key at
   ingest, so parcel, tax, and transaction datasets in the same locality can be
   linked without re-deriving ids. ``parcel_id_assessor`` is the raw source id;
   ``parcel_id_local`` is the standardized key derived from it.

   .. code-block:: yaml

      parcel_id_local:
        source: parcel_id_assessor   # raw column to standardize
        kind: parcel                 # parcel | tax (selects the default conversion)
        admin_id_column: admin4_id   # optional: per-row admin unit (e.g. MA towns)
        instruction:                 # optional source-supplied override
          "US-NC-NE": {pattern: ..., conv: ...}

   The conversion is admin-unit-specific: a recipe ``instruction`` if given,
   else the bundled default table (``geo/parcel_id_links.csv``, keyed by
   ``admin_id`` with separate ``parcel`` and ``tax`` conversions), else
   ``simple`` (uppercase, keep alphanumerics). The engine
   (:func:`openplaces.geo.ids.compute_parcel_id_local`) is hardened so the
   conversion never adds duplicates over those already in
   ``parcel_id_assessor`` — it falls back to ``simple`` then the raw id. The
   harmonizer ``link_by_id`` step then joins datasets on ``parcel_id_local``
   (attaching attributes, or counting transactions per parcel). The methodology
   is adapted from the ZTRAX/parcel APN-matching workflow.


Data cleaning
-------------

.. attribute:: null_value_strings

   List of strings in the source data that should be interpreted as missing values (``None``).

   Applied to all non-ID columns before any other preprocessing.

   Example: :input:`["NA", "NOT APPLICABLE", "NOT PROVIDED"]`

.. attribute:: query

   Pandas query string to filter rows.

   Passed to ``pd.DataFrame.query()``. Use to remove unwanted geometries before further processing.

   Example: :input:`"type != 'Water body'"` (used in GADM to drop ocean polygons).


Space saving
------------

.. attribute:: columns_to_categorical

   Columns to cast to the ``pandas`` ``Categorical`` dtype, reducing memory and on-disk size.

   Each entry is either a plain column name (unordered categorical) or a single-key dict with an explicit category list (ordered categorical, from lowest to highest):

   .. code-block:: yaml

      columns_to_categorical:
        - city                           # unordered
        - zip                            # unordered
        - height_source: [H, HBET, HHT]  # ordered: H < HBET < HHT

   Ordered categoricals enable ``prefer_higher`` in :attr:`keep_overlapping_polygons`.


Overlap handling
----------------

.. attribute:: keep_overlapping_polygons

   Policy for resolving pairs of polygons that overlap substantially (intersection-over-union > 0.5).

   If omitted, overlapping polygons are reported as a warning but both are kept.

   .. attribute:: prefer_higher

      Name of an (ordered) categorical column. When two polygons overlap, the one with the higher category value is kept. Useful for preferring footprints derived from better data sources.

      Example (OBM buildings, preferring height sources that are more reliable):

      .. code-block:: yaml

         keep_overlapping_polygons:
           prefer_higher: height_source


Data transformations
--------------------

.. attribute:: transformations

   List of operations to derive new columns from existing ones.

   Each transformation is a dict with ``output``, ``type``, ``operation``, ``input``, and optionally ``args``.

   Refer to ``openplaces.io.transform`` for available operations.

   Example (extract the first five characters of a census block ID to get the county FIPS code):

   .. code-block:: yaml

      transformations:
        - output: admin3_id_admin1
          type: string
          operation: substring
          input: census_block_id
          args:
            start: 0
            end: 5


Indexing
--------

.. attribute:: set_index

   Name of an existing column to use as the index.

   Raises an error if the column contains duplicates.

.. attribute:: create_index

   Define an index using a function and optional arguments.

   .. attribute:: function

      Dotted path to an ``openplaces`` function that adds an index to the GeoDataFrame.

      Must start with ``openplaces.`` (enforced to prevent arbitrary code execution).

      Example:

      .. code-block:: yaml

         create_index:
           function: "openplaces.geo.ids.add_openlocationcode_index"
           args:
             name: footprint_id

   .. attribute:: args

      Keyword arguments passed to :attr:`function`.

.. attribute:: sort_index

   Set to :input:`True` to sort the dataset by its index after creation.

.. attribute:: drop

   List of index values to drop after the index has been created.

   Use to remove known bad rows that cannot be excluded by a :attr:`query` filter.


Attribution to administrative units
------------------------------------

Two mechanisms assign ``openplaces`` Admin IDs to rows.

Use :attr:`admin_id_crosswalk` when the source data already contains an admin identifier column, and :attr:`overlay_admin_ids` when Admin IDs must be derived from spatial geometry.

.. attribute:: admin_id_crosswalk
   :noindex:

   Add an ``openplaces`` Admin ID column by joining from a crosswalk between source admin identifiers and ``openplaces`` Admin IDs.

   Distinct from :attr:`process_by.admin_id_crosswalk`: that crosswalk is used to *split* the data during reading; this one *attributes* rows to admin units in the output.

   .. attribute:: admin_level
      :noindex:

      Administrative level of the target Admin IDs, e.g. :input:`3`.

   .. attribute:: admin_id_column
      :noindex:

      Column in the input data to join on, e.g. :input:`admin3_id_admin1`.

   .. attribute:: admin_recipe_id
      :noindex:

      Recipe from which to derive the crosswalk, e.g. :input:`"US_admin-census-2021_admin3"`.

.. attribute:: overlay_admin_ids

   Assign Admin IDs to rows by spatial overlay — intersecting geometries with admin unit boundaries.

   Use when the source data has no reliable admin ID column and spatial containment must be used instead.

   .. attribute:: admin_level
      :no-index:

      Level of administrative units to assign to geometries.

   .. attribute:: admin_recipe_id
      :no-index:

      Recipe that provides the admin unit geometries.

      Should use a high-resolution in-country boundary dataset for accurate allocation (e.g. building footprints require county-level precision).

      Example: :input:`'US_admin-census-2021_admin3'` for US counties.


Saving
------

.. attribute:: save_to

   Controls where and how output is written.

   .. attribute:: data_dir

      Name of the ``openplaces`` data directory in which to save output.

      Must be one of the registered directory names (e.g. :input:`cache`, :input:`core`, :input:`out`).

      Defaults to :input:`cache` if omitted.

   .. attribute:: admin_level
      :no-index:

      Admin level at which to split output into separate files.

      Not needed if splitting is already determined by :attr:`download_by` or :attr:`process_by`.

      Example: set to :input:`3` to write one parquet file per county.

      .. note::

         When this level is **coarser** than :attr:`download_by: admin_level` or attr:`process_by: admin_level` (e.g. save at county=3 while processing by town=4), the ingester automatically aggregates the intermediate files to the coarser level and deletes the intermediates, including any empty directories that remain.

   .. attribute:: filename

      Override the auto-generated output filename stem.

      The auto-generated name is derived from the reference (admin ID, entity, source, version).
      Use this when the default is ambiguous — for example, a recipe that produces
      multiple files for different admin levels, or a dataset that needs a fixed conventional name.

      Example: :input:`admin2` (used in admin recipes to produce :file:`admin-gadm-4~1_admin2.parquet`).

.. _aggregate_by:

Aggregating partitions
----------------------

.. attribute:: aggregate_by

   Roll per-partition output files (e.g. one parquet per month) up into fewer,
   non-redundant files after ingestion.

   The aggregated file's parquet footer records the partition IDs it contains
   under the key ``openplaces:partitions``. With ``reprocess=False``, the
   ingester reads this footer to skip partitions that are already integrated —
   the per-partition files do not need to be kept for re-runs to be
   incremental.

   .. attribute:: single_file

      If :input:`true`, combine every partition into one dataset-wide
      :file:`..._all.parquet` per saved admin unit, and produce no per-group
      files.

   .. attribute:: partition
      :no-index:

      Roll-up granularity when :attr:`single_file` is not set, e.g.
      :input:`year` to combine the twelve months of each year into one
      per-year file.

   .. attribute:: how

      Merge policy when the aggregated file already exists (named after the
      ``how`` argument of ``geopandas.overlay``):

      :input:`union` (default)
         Integrate the new partitions into the existing file, de-duplicating
         by full row. Re-adding an unchanged partition is a no-op, and rows
         that are only present in the existing file are never deleted — safe
         when partition files may be missing or out of sync.

      :input:`replace`
         Overwrite the file with only the partitions of the current run. Use
         together with a full-range reprocess, or the file narrows to those
         partitions.

      Both policies first verify that the new batch contains no internal
      duplicate rows (which indicate a source or processing bug) and raise an
      error if it does. Row identity includes the index whenever a frame
      carries a meaningful one (anything other than a default integer range
      index), so rows with equal values but distinct keys are not duplicates.

      One ``union`` caveat: if the source retroactively edits a field of an
      already-integrated record, both row versions remain in the aggregated
      file until a full reprocess with :input:`replace`.

   .. attribute:: keep_partitions

      If :input:`true`, retain the per-partition files after aggregation
      (default is to delete them). Used, for example, to keep raw monthly
      scrape files in the downloads directory as the source of truth.

   Example (Wisconsin transactions: months are intermediate, one combined
   output):

   .. code-block:: yaml

      aggregate_by:
        single_file: true

   Example (Massachusetts registry scrape: monthly files are retained, new
   rows are integrated):

   .. code-block:: yaml

      aggregate_by:
        single_file: true
        keep_partitions: true


Additional layers
-----------------

.. attribute:: additional_layers

   List of additional tables to ingest from the same source file, each producing a separate output for a different entity type.

   For example, a Massachusetts parcel geodatabase contains both a parcel polygon layer and a property attribute table; ``additional_layers`` ingests both in a single recipe.

   Each entry in the list is a layer spec dict. The full recipe for each layer is built by merging the primary recipe with the layer spec:

   - **Per-table keys** are taken from the layer spec when present; otherwise they are removed from the merged recipe. These include:

     ``entity``, ``layer``, ``columns``, ``keep_unnamed_columns``, ``set_index``, ``create_index``, ``index_function``, ``drop``, ``query``, ``null_value_strings``, ``transformations``, ``columns_to_categorical``, ``encoding``, ``save_to``, ``overlay_admin_ids``
   - **Shared keys** are always inherited from the primary recipe. These include:

     ``admin_id``, ``download_by``, ``compressed_file_name``, ``uncompressed_file_name``

   - **``process_by``** is inherited from the primary recipe unless the layer spec sets it explicitly. Use ``process_by: null`` in the layer spec to disable chunking for that table.
   - ``additional_layers`` cannot be nested.

   ``entity`` is required in every layer spec.

   Example: Massachusetts parcels. Primary layer is parcel polygons; additional layer is the property attribute table.

   .. code-block:: yaml

      layer: L3_TAXPAR_POLY
      process_by:
        admin_level: 4
        admin_id_column: TOWN_ID
        admin_id_crosswalk:
          recipe_id: "US-MA_parcel-massgis-2025_admin4-crosswalk"
      columns:
        parcel_id_admin2: LOC_ID
        polygon_type: POLY_TYPE

      additional_layers:
        - layer: L3_ASSESS
          entity:
            entity_type: property
            source:
              source_id: massgis
            version: 2025
          columns:
            property_id_assessor: PROP_ID
            parcel_id_admin2: LOC_ID
            total_value: TOTAL_VAL
          columns_to_categorical:
            - city
            - zip
   
   For the full recipe, see:

   :gh-file:`src/openplaces/recipes/US/MA/_all/parcel/massgis/2025/US-MA_parcel-massgis-2025.yaml`

