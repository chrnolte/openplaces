.. openplaces

.. _cheer_footprints_validation:

Validation and accuracy
=======================

How the canonical :ref:`CHEER footprint inventory <cheer_footprints>` is scored against its two out-of-band references, with the current results.

.. contents:: Table of contents
   :local:
   :depth: 2

Two references score the inventory: a hand-labeled survey (North
Carolina) and Shovels building-permit records (both regions). Permit agreement is not ground truth; 8 to 14 percent of even the strongest permit matches disagree with every source.

Methods
~~~~~~~

**Survey.** The CHEER summer-scholars survey holds 1,370 buildings
across ten eastern North Carolina counties, each classified by hand
from imagery and street view. Points link to curated footprints by
address first and by distance as a fallback (15 m cap, matched to the
observed pin offsets), with address ties broken toward the parcel's
primary structure. Scores cover the residential three-class problem
over all linked points.

**Permits.** Shovels permit records aggregate per parcel (the mode of
permits that name an occupancy), link to parcels by assessor id in
North Carolina and by point-in-parcel in the Texas metros (whose CAD
ids do not match the statewide parcel layer), and fan out to the
parcel's footprints. Footprints the inventory labels Secondary are
excluded from the denominator: a house permit fanning onto its garage
is linkage noise, not classification error (Harris County measures
92.5 percent agreement on single-footprint parcels against 49.9
percent on multi-footprint ones, driven by Single-Family to Secondary
rows). Construction years compare the inventory's ``year_built``
against the most recent permit record's year; permit years are partly
assessor-derived, so that comparison is not fully independent.

**Reading the matrices.** Rows are the reference class and columns
the class a source asserted. Beside the three residential classes,
Secondary counts outbuildings, Non-residential counts any other
asserted class, and No class counts rows where the source asserted
nothing. Producer's accuracy (recall) is the share of a reference
class's answered rows that a source classed correctly; consumer's
accuracy (precision) is the share of rows a source assigned to a class
that belong to it. A Secondary or Non-residential assertion counts as
an answer, and as a miss.

**NSI on survey points.** NSI's No class on survey points is partly a
linkage effect: an NSI point counts only where it falls on the
footprint the survey point linked to. Of the 108 surveyed manufactured
homes with no NSI point on the linked footprint, 104 have an NSI point
within 100 m (median distance 29 m), and 29 of those nearest points are
classed Manufactured Home. NSI also classes 46 surveyed New Hanover
manufactured homes as Professional Technical Services, which the
matrices count as Non-residential.

Occupancy type against the survey (North Carolina)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Generated replacement pending: once the validate job has written
   this table's confusion CSVs, replace the hand-typed tables of this
   section with:

   .. include:: _generated/cheer-eastern-nc_US_footprint-openplaces-2026_occupancy-survey.rst

Overall accuracy 0.756, macro-F1 0.783, a residential class on 1,292
of the 1,370 points.

.. list-table::
   :header-rows: 1

   * - class
     - support
     - precision
     - recall
     - F1
   * - Single-Family
     - 420
     - 0.674
     - 0.800
     - 0.731
   * - Multi-Family
     - 586
     - 0.894
     - 0.703
     - 0.787
   * - Manufactured Home
     - 364
     - 0.865
     - 0.796
     - 0.829

.. list-table:: Confusion matrix (rows: hand labeled, columns: inventory)
   :header-rows: 1

   * - hand label (row)
     - Single-Family
     - Multi-Family
     - Manufactured Home
     - unassigned
   * - Single-Family
     - **335**
     - 26
     - 45
     - 14
   * - Multi-Family
     - 131
     - **412**
     - 0
     - 43
   * - Manufactured Home
     - 31
     - 23
     - **289**
     - 21

.. list-table:: Per county
   :header-rows: 1

   * - county
     - points
     - accuracy
     - Manufactured Home recall
   * - Carteret (CAR)
     - 154
     - 0.487
     - 0.889
   * - Pender (PEN)
     - 38
     - 0.605
     - 0.286
   * - Beaufort (BEA)
     - 215
     - 0.660
     - 0.795
   * - Dare (DAR)
     - 24
     - 0.750
     - 0.000
   * - Robeson (ROB)
     - 232
     - 0.797
     - 0.773
   * - Brunswick (BRU)
     - 205
     - 0.810
     - 0.684
   * - Halifax (HAL)
     - 376
     - 0.822
     - 0.806
   * - Johnston (JOH)
     - 79
     - 0.911
     - 0.938
   * - New Hanover (NHA)
     - 46
     - 0.978
     - 0.978

Occupancy type against permits (North Carolina)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Generated replacement pending: once the validate job has written
   this table's confusion CSVs, replace the hand-typed tables of this
   section with:

   .. include:: _generated/cheer-eastern-nc_US_footprint-openplaces-2026_permit-occupancy.rst

Agreement 93.7 percent on 136,666 scored footprints in 39 counties;
96.6 percent on single-footprint parcels.

.. list-table::
   :header-rows: 1

   * - class
     - support
     - precision
     - recall
     - F1
   * - Single-Family
     - 113,208
     - 0.979
     - 0.949
     - 0.964
   * - Multi-Family
     - 6,543
     - 0.693
     - 0.793
     - 0.740
   * - Manufactured Home
     - 16,915
     - 0.809
     - 0.907
     - 0.856

Occupancy type against permits (Texas)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Generated replacement pending: once the validate job has written
   this table's confusion CSVs, replace the hand-typed tables of this
   section with:

   .. include:: _generated/cheer-coastal-tx_US_footprint-openplaces-2026_permit-occupancy.rst

Agreement 94.9 percent on 357,969 scored footprints in the eight
permit-covered metro counties; 96.7 percent on single-footprint
parcels. The 2026-08-25 Overture corroboration guard raised agreement
from 93.8 percent and removed 36 percent of the Single-Family-read-as-
Multi-Family conflicts.

.. list-table::
   :header-rows: 1

   * - class
     - support
     - precision
     - recall
     - F1
   * - Single-Family
     - 331,323
     - 0.993
     - 0.955
     - 0.974
   * - Multi-Family
     - 10,629
     - 0.554
     - 0.932
     - 0.695
   * - Manufactured Home
     - 16,017
     - 0.644
     - 0.822
     - 0.722

Year built against permits
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. Generated replacement pending: once the validate job has written
   this table's confusion CSVs, replace the hand-typed tables of this
   section with:

   .. include:: _generated/cheer-eastern-nc_US_footprint-openplaces-2026_permit-year-built.rst
   .. include:: _generated/cheer-coastal-tx_US_footprint-openplaces-2026_permit-year-built.rst

.. list-table::
   :header-rows: 1

   * - region
     - footprints with both years
     - inventory coverage on permit rows
     - median absolute error
     - within 1 year
     - within 5 years
     - within 10 years
   * - North Carolina
     - 167,936
     - 98%
     - 0
     - 95.0%
     - 95.6%
     - 96.2%
   * - Texas
     - 481,905
     - 98%
     - 0
     - 84.4%
     - 87.9%
     - 90.9%

Manufactured homes, stratified
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 706 Single-Family/Manufactured-Home survey points split by the
manufactured share of each building's census block, self excluded, to
score the case neighborhood-based signals fail on.

.. list-table::
   :header-rows: 1

   * - stratum
     - points
     - inventory F1
     - NSI alone F1
   * - pure manufactured (>=80%)
     - 19
     - **0.973**
     - 0.690
   * - mixed (20 to 80%)
     - 218
     - **0.823**
     - 0.482
   * - pure site-built (<=20%)
     - 461
     - **0.787**
     - 0.244
   * - all
     - 706
     - **0.814**
     - 0.390

Improvement paths
~~~~~~~~~~~~~~~~~

- The survey residual concentrates in Multi-Family read as
  Single-Family (131 of 334 errors, 20 of them the Overture guard's
  deliberate precision trade). Keyword-vocabulary coverage is the lever
  that moves it without giving back the permit-side gain.
- Texas Multi-Family permit precision (0.554) mixes classification
  with residual fan-out on multi-footprint parcels; rescoring after a
  per-footprint permit linkage would separate the two.
- The largest North Carolina permit conflict is Single-Family read as
  Manufactured Home (3,496 rows); the NAIP detector handoff targets
  exactly this class.
- Texas years trail North Carolina's (84 versus 95 percent within one
  year) because only North Carolina has an original per-building year
  source (NCDPS) ranked above NSI's census-block median; an equivalent
  Texas source would close the gap.
