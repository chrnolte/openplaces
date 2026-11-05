.. openplaces

.. _cheer_provenance:

Provenance flags
================

Every canonical column ``x`` has a column ``x_source`` beside it. It
names the source that supplied the value, or the rule that decided it.
Use it to keep only recorded values, to compare sources, or to leave out
what the models filled.

.. code-block:: python

   # Year built as the county roll records it, without filled values.
   recorded = inventory[inventory['year_built_source'] == 'parcel']

   # Every structure value openplaces filled.
   filled = inventory['structure_value_source'].str.endswith('+imputed', na=False)

Reading a token
~~~~~~~~~~~~~~~

A token is one or more parts:

- ``a+b`` joins every piece of evidence that decided the value. In a
  vote, these are the indicators that agreed on the winning class.
  For example, ``shape_width+shape_aspect`` in ``n_sections_source``
  says the footprint's width and its length-to-width ratio decided the
  number of sections; ``nsi+fema`` in ``occupancy_type_source`` says
  NSI and FEMA both voted for the class.
- ``a/b`` lists the sources that agreed on the winning class in the
  first, base occupancy vote (NSI, FEMA, the roll). For example,
  ``nsi/fema/parcel`` says all three gave the same class, and
  ``nsi/parcel`` says NSI and the roll agreed while FEMA had none or
  another.
- ``parcel.<source>`` is a value from the parcel layer of that source:
  a county (``parcel.pendercounty``) or a statewide service
  (``parcel.nconemap``). For ``geometry_source``, it marks a building
  that a roll records but no footprint layer draws: the parcel polygon
  stands in for it.
- A token ending in ``+imputed`` was filled by ``openplaces``.
- An empty ``_source`` goes with an empty value.

Tokens you will meet
~~~~~~~~~~~~~~~~~~~~

Sources
-------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Token
     - Source
   * - ``parcel``, ``parcel.<source>``
     - The county tax roll or parcel layer
   * - ``nsi``
     - The National Structure Inventory (USACE)
   * - ``overture``, ``dwelling_overture``
     - Overture address points
   * - ``fema``
     - FEMA's USA Structures
   * - ``cheer``
     - The earlier CHEER building inventory (roof shape, foundation,
       construction type, stories)
   * - ``ncdps``
     - North Carolina's statewide building footprints (and the year built
       they carry)
   * - ``obm``, ``microsoft``, ``microsoftglobal``
     - Footprint layers (``geometry_source`` only)
   * - ``condo_cluster.parcel.<source>``
     - A footprint drawn for a stacked condominium building from the
       parcels of its units (``geometry_source`` only)

Fills and withholding
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Token
     - Meaning
   * - ``building-imputation-v0.8+open+imputed``
     - Filled by the delivery's models (see :doc:`imputation`). ``open``:
       the models saw no withheld value.
   * - ``building-imputation-v0.8+clipped+open+imputed``
     - Filled, then held to the range of value per m2 of the building's
       occupancy.
   * - ``parcel.<group>+imputed``
     - A parcel value the roll lacks, estimated during curation from
       other parcels of the same group (the group is named in the
       token, for example ``city__is_residential``).
   * - ``withheld``
     - A source supplied a value that its terms keep out of this bundle.
       The cell is empty, or holds a filled value whose token says so.
       The licence notice counts these per source.

Rules
-----

Occupancy and the other voted columns name the evidence that decided
them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Token part
     - Evidence
   * - ``dwellings``
     - The vote on one dwelling or several: one gives Single-Family,
       several give Multi-Family
   * - ``priority``
     - The building is a secondary building on its parcel
   * - ``overture_count``, ``nsi_units``
     - The number of addresses or units on the building
   * - ``keyword``
     - The assessor's use description names the class
   * - ``keyword_probability``
     - The manufactured-home model (footprint shape, neighbors and the
       assessor's description) gives a probability of 0.75 or more
   * - ``morphology``, ``shape_band``, ``small_shape``
     - The footprint's size and shape
   * - ``no_improvement_value``
     - The parcel's improvement value is near zero, as for a home taxed
       as personal property
   * - ``parcel_park``, ``parcel_community``
     - The parcel is a manufactured home park or community
   * - ``height_band``
     - The multi-family height band from the story count
   * - ``shape_width``, ``shape_aspect``, ``parcel.keyword``
     - Width and elongation of a manufactured home, and the assessor's
       section count (``n_sections``)
   * - ``reconciled``, ``usaddress``
     - Address: assembled from several sources; parsed into its parts
