.. openplaces

.. _cheer_decisions:

Design decisions
================

The choices below shape the values in the bundle most. Each gives what
was decided, why, and the alternative it was chosen over. Every step of
the pipeline, with its reason and whether that reason still holds, is in
the :ref:`step review <cheer_step_review>`.

Evidence first, then one value
------------------------------

**Decision.** The harmonize stage collects each source's value side by
side without choosing among them; curation then chooses one value per attribute
and records the source of each. The evidence file keeps every source's
value.

**Why.** A choice can be revisited without collecting the evidence
again, and a user who prefers another source can take it from the
evidence file. The staging follows the synthesis framework of
:ref:`Lochhead et al. (2026) <Lochhead et al. 2026>`.

The building decides, the footprint ships
-----------------------------------------

**Decision.** Values and classes are decided for buildings; the bundle
carries them on the footprints.

**Why.** The sources describe buildings: a tax roll, NSI and FEMA give
a building its use, age and value. A hazard model needs the footprint: it
is the shape that wind and water act on. The two can differ, because a
row of townhomes is one footprint but several buildings, each with its
own owner and value. Deciding each value for a building and then adding
the buildings up onto their footprint (counts and values summed, the
earliest year built) gives every footprint the total of what stands
under it. Only the footprint table is published.

Footprint sources in a fixed order
----------------------------------

**Decision.** OpenBuildingMap footprints are taken first, then
Microsoft's, then North Carolina's statewide footprints. A lower-ranked
footprint enters only where it overlaps no footprint already accepted.

**Why.** OpenBuildingMap's footprints were found more complete and better
than the US Microsoft layer (maintainer, 2026-11-03). A fixed order,
rather than a score per footprint, keeps the choice predictable and easy
to check.

Structure value from the roll, shared by floor area
---------------------------------------------------

**Decision.** A building's structure value is its parcel's recorded
improvement value, shared among the parcel's main buildings in
proportion to the area of each on the parcel, weighted by its number
of floors where Overture states a story count or a height. Garages and
sheds carry 0.

**Why.** County assessors value the improvements on a parcel, not each
building. Copying the parcel's whole value onto each of its buildings
inflated regional totals 1.23 to 1.69 times on earlier CHEER bundles.
The floor weight keeps a one-story annex from taking a tall building's
value. A secondary building is usually valued with the main one; giving
it a share would take value from the house.

**Alternative.** NSI's modeled value for every building. NSI's value is
a replacement cost, not a market value; filling held-out counties with
it put county totals 42.8% above the roll (cheer-eastern-nc,
2026-10-24). It ships in its own columns instead.

A 0 is a statement
------------------

**Decision.** A recorded 0 passes through to the building, with a reason
(``structure_value_zero_reason``). A 0 counts as the roll's statement
only where the parcel has a land value and its total is within $100 of
it; otherwise the cell is missing.

**Why.** Many zeros are true: a shed valued with the house, a home taxed
as personal property, an exempt church. Others are a roll's silence. The
reason lets a user, and the models, tell them apart (see
:doc:`imputation` for which reasons are filled).

Occupancy by weighted vote
--------------------------

**Decision.** Each building's class is decided by votes in which every
source that reaches the building through its location adds a fixed
weight: the footprint's shape, NSI, FEMA, Overture's address count, the
roll's class and the assessor's description. One vote decides whether
a building holds one dwelling or several; the next decides its class.

**Why.** No source is right everywhere. NSI's classes are modeled; the
roll's codes differ by county; a shape cannot tell a small house from a
manufactured home on its own. Votes let the sources that are reliable
for one question decide it: Overture's address counts identify
multi-family buildings, the assessor's description and the shape
identify manufactured homes. The weights were set against the CHEER
Summer Scholars survey and building permits (see the
:ref:`validation annex <cheer_footprints_validation>`).

**Alternative.** A fixed order of sources for occupancy, as for the
other attributes. It cannot combine weak signals that agree, which is
how most manufactured homes are found.

Manufactured homes as their own class
-------------------------------------

**Decision.** Manufactured homes are a class of their own, with their
number of sections, and RV dwellings are told apart from them.

**Why.** Wind-loss models treat manufactured homes separately
(:ref:`Tackie-Otoo et al. 2026 <Tackie-Otoo et al. 2026>`). Many are
missing from the real-property roll, so finding them takes the
footprint's shape and the neighborhood as well as the roll; the shape
thresholds agree with :ref:`Khanal et al. (2025) <Khanal et al. 2025>`.

The earliest year of a footprint
--------------------------------

**Decision.** Where one footprint holds several buildings,
``year_built`` is the earliest of their years.

**Why.** An average states a year in which none of them was built.

Withheld values stay withheld
-----------------------------

**Decision.** Values from sources whose terms do not allow
redistribution, and anything computed from them, are withheld from the
publishable bundle. Values from a share-alike licence other than the
footprints' are withheld the same way, so the bundle can be published
under one licence. A withheld structure value is filled by the models,
which never saw it, and never by NSI's value.

**Why.** The bundle can then be shared under the terms its notice
states. The team bundle restores the values cleared for the team.

Footprints cut at county lines
------------------------------

**Decision.** A footprint two neighboring counties both delivered is cut
at the county line, and each piece keeps its own county's values. A
piece under 10 m2 or 5% of the footprint is not cut.

**Why.** Each county's roll values the part of the building on its
side. Keeping both whole footprints would count the building twice;
keeping one would drop the other county's value.

Imputation
----------

The choices behind the filled values (labels from rolls only, draws
instead of the likeliest value, a two-part value model, donor counties
by distance, a common year and the Census price anchor) are on
:doc:`imputation`. Alternatives that were tested and set aside:

- Values modeled with a Poisson or gamma loss instead of the two-part
  model: county totals were close with Poisson (+0.6%), but errors per
  building were large (cheer-eastern-nc, 2026-10-21).
- The expected value per building instead of a draw: it multiplies
  each value by the chance that it is zero, which gave Wake's filled
  single-family homes a median of $4,840.
- NSI's value as the price level for counties without labels: NSI is
  near the roll on the coast and 1.3 to 2.7 times it inland (typical
  county miss 20.1%, against 12.8% with the Census anchor; v0.6,
  2026-10-24).
