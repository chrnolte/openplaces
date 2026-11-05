.. openplaces

.. _cheer_imputation:

Filling the gaps (imputation)
=============================

No source covers every building. Where none has a value, the delivery
fills the cell with a model trained on the values the sources do
record. A filled cell says so: its ``_source`` reads
``building-imputation-v0.8+open+imputed``.

This page describes what is filled, how, and how well. The method
follows the research of Mohammad Askari (UCLA): he chose the learner
(gradient-boosted trees), the evaluation (testing on whole counties
held out, with the held-out buildings as incomplete as the real gaps),
the validity repairs and the gap-trained boosting, and contributed the
code for the last three.

What is filled
~~~~~~~~~~~~~~

Footprint area (for parcel-shaped stand-ins), year built, stories,
construction type, foundation type, roof shape, garage, number of
sections (manufactured homes only), number of dwellings, structure
value, and NSI's two cost measures (replacement cost and depreciated
replacement cost). Occupancy is not filled by the models; the votes
described in :doc:`pipeline` give every building a class.

The models are trained and applied per region, on the publishable
bundle, after every withheld value has been removed from it. That is
what ``open`` in the token means: no withheld value reaches a model,
so none can reach a filled cell. The team bundle reuses the same
filled values and puts back its own values where the publishable
bundle had withheld them.

Labels: what the models learn from
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Recorded values only.** Year built and structure value are learned
  from the county rolls. NSI's year and value are model inputs, never
  labels: NSI's years are area lookups, and its value is a cost, not a
  market value. No year built or structure value in the bundle comes
  from NSI.
- **Implausible values are emptied first.** A story count above 60, or
  above 10 on a single-family, manufactured or low-rise building, is
  removed and filled like any other gap.
- **Zeros are read through their reason** (``structure_value_zero_reason``):

  .. list-table::
     :header-rows: 1
     :widths: 25 75

     * - Reason
       - Treatment
     * - ``no_value``
       - Kept as 0. The only zeros the model learns from.
     * - ``secondary``
       - Kept as 0 (a garage or shed valued with the main building).
     * - ``not_on_roll``
       - Kept as 0. Most such buildings were removed since the
         footprint was drawn.
     * - ``included_elsewhere``
       - Kept as 0: another building on the lot holds the roll value,
         and filling this one would count the lot twice.
     * - ``personal_property``, ``exempt``
       - Filled: the roll is silent about the building, not stating
         that it has no value.

- **Small "single-family" zeros are filled.** A ``no_value`` zero on a
  single-family or multi-family building under 120 m2 is filled. Most
  such buildings are manufactured homes held as personal property and
  classed single-family for lack of a use code.
- **Manufactured homes.** A manufactured home the roll does not list
  as real property (personal property, or a home in a park of five or
  more) is valued from the county's listed manufactured homes of the
  same size and age, and a park home is scaled by the ratio at which
  assessors value the park homes they do list.

How the models work
~~~~~~~~~~~~~~~~~~~

- **One model per county for year built and structure value**, trained
  on the county's own labels plus those of other counties, weighted by
  the distance between county centers (a Gaussian with a 250 km
  bandwidth, up to 400 km). A county whose own labels cover
  less than 30% of its buildings is filled from the other counties
  only: such labels are the buildings a recent record happened to
  reach, not a sample.
- **One regional model for each other attribute**, filled in a fixed
  order so that later attributes can use earlier ones.
- **Classes and counts are drawn** from the predicted probabilities,
  not set to the likeliest value, so the county's class shares stay
  right.
- **Structure value is modeled in two parts**: whether it is zero, and
  if not, its logarithm, corrected for the bias of the back
  transformation and scaled so the training totals match. A filled
  value is drawn from the spread of the model's errors, not set to its
  average. A value outside its occupancy's range of value per m2 is
  held to that range (``+clipped`` in its token).
- **A common year.** County rolls are appraised in different years
  (2018 to 2026 in North Carolina). Values are moved to 2025 with the
  FHFA county house price index before fitting and moved back after.
  The bundle ships both: ``structure_value`` at the appraisal year and
  ``structure_value_adjusted`` in 2025 dollars.
- **Counties without value labels** take their price level from the
  Census median home value (ACS).
- **NSI's cost measures** are filled per county from the county's own
  NSI points. The depreciated cost is the filled replacement cost times
  a filled depreciation ratio, so it never exceeds it.
- **Reproducible.** Every random step starts from a recorded seed
  (2026): the same inputs and code give the same values.

Model inputs are the evidence columns of the bundle (NSI, Overture,
FEMA, the earlier CHEER inventory, the footprint's shape), the location,
and Census levels by tract and block group (median home value, year
built, income, population density) and distances to places. The Census
inputs are read at delivery; the bundle does not carry them.

How well it works
~~~~~~~~~~~~~~~~~

Accuracy was measured by holding out whole counties and filling them as
if they had no labels, with the held-out buildings as incomplete as the
real gaps. The first three rows come from the cheer-eastern-nc region;
no comparable figure exists yet for cheer-coastal-tx. The version and
date are given for each row, because the method changed between
versions.

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Attribute
     - Version, date
     - Result (whole counties held out)
   * - Structure value
     - v0.7, 2026-10-28
     - County totals 19.0% low at the median county; typical miss
       19.0%. NSI's value as a stand-in: 42.8% high.
   * - Year built
     - v0.7, 2026-10-28
     - Off by 14.9 years on average (NSI: 16.6); county means within
       a year at the median county (NSI: 6.5 years too late).
   * - Other attributes
     - v0.6, 2026-10-24
     - Better than always guessing the most common class (macro F1:
       construction 0.34 against 0.18, roof 0.32 against 0.06).
   * - NSI cost measures
     - v0.8 inputs, NC and TX
     - Replacement cost: total bias -1.3%, typical miss 4.2% (NC);
       -3.4% and 3.7% (TX). Depreciated: -2.0% and 4.1% (NC); -3.1%
       and 3.4% (TX).

Year built and structure value have not been rescored since v0.7.
Counties whose roll gives no values at all are the ones filled from
other counties only; their totals are more likely too low than too
high. The counts of filled cells for each column are listed in each
bundle's licence notice.
