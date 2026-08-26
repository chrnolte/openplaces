.. openplaces

.. _admin_identifiers:

Administrative identifiers: rules and generation
=================================================

An ``admin_id`` is a hierarchical database key used to reference an administrative unit. For example, ``US-NC-ME`` represents Mecklenburg County, North Carolina, while ``US-MA-SOM`` represents the city of Somerville, Massachusetts. 

In addition to acting as a join key across datasets, administrative identifiers appear directly in directory paths, filenames, exported bundles, and column names. As a result, they must balance two conflicting goals:

1. **Recognizability**: A reader seeing the code should be able to intuitively guess the administrative unit without a lookup table.
2. **Conciseness**: Because directory paths are built by concatenating identifier segments, codes must remain short to avoid path-length limitations (such as Windows' MAX_PATH limit).

This section details the core principles, current implementations, empirical simulations, open design questions, and operational decision rules governing the generation of administrative identifiers.


Core principles
~~~~~~~~~~~~~~~

Any administrative identifier system in ``openplaces`` must satisfy four foundational properties:

- **Derivability**: Identifier segments must be derived from the unit's own name in a predictable manner, allowing readers to guess codes forwards and confirm them backwards.
- **Stability**: Code assignments must remain stable under subsequent spine generation runs. Adding or removing a unit should not trigger a cascade of changes to unrelated siblings.
- **Never Zero**: Assignment algorithms must not yield zero weights or invalid null/zero identifiers.
- **Historical Resolvability**: Retired or superseded codes must always resolve to the correct unit through a lookup system, ensuring historical continuity.


Current implementation
~~~~~~~~~~~~~~~~~~~~~~

The active identifier spine (v2026) operates under the following rules:

Default length and widening
---------------------------
Codes are two characters by default (e.g., matching ``[A-Z][A-Z0-9]{1}``). A sibling group (all children of a given parent unit) is **widened** to three characters if and only if:

* The parent unit has **more than 250 children**, or
* **More than 25%** of the generated codes under that parent would be opaque, **and** the parent has **at least 10 children**.

Requiring a minimum of 10 children prevents spurious widening of tiny sibling groups due to simple arithmetic quirks (e.g., a two-unit group cannot logically suffer from code-space crowding).

Tie-breaking and weighting
--------------------------
When multiple siblings contend for the same candidate code, conflicts are resolved using raw population weights (sourced from GHS-POP 2020). The assignment problem is solved globally for the sibling group to maximize total name recognizability. Raw population weighting ensures that larger, more populous units receive their most intuitive first-choice codes, minimizing the number of users who encounter opaque identifiers.

Deference to published conventions
----------------------------------
The generation pipeline adopts published national or international code lengths (e.g., ISO 3166-2) when the published scheme maps to the corresponding administrative level and unit counts require it. However, a reviewed override set (e.g., for Colombian departments) outranks inferred conventions.


Simulation and empirical findings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spine metrics
-------------
The 2026 spine contains:

* **Level 2 (States/Regions)**: 3,654 units
* **Level 3 (Counties/Districts)**: 48,695 units
* **Level 4 (Municipalities/Towns)**: 218,651 units

Population coverage is **100%** across all three levels. The global population totals 7,788,558,830, matching the true 2020 epoch figure of ~7.79 billion. Name-to-code derivability rates are:

* **Level 2**: 98.0%
* **Level 3**: 99.3%
* **Level 4**: 96.1%

Saturation of the two-character space
-------------------------------------
While a two-character alphanumeric space theoretically holds ``26 x 36 = 936`` codes (or 676 purely alphabetical codes), candidate names cluster heavily in practice due to shared prefixes (e.g., *San*, *Santa*, or regional suffixes). 

Synthetic group simulations show that the candidate pool saturates near **600 distinct candidates**. When candidates-per-unit falls toward 2.0, the assignment solver is forced into opaque fallbacks:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Sibling units
     - Opaque share (2-char)
     - Distinct candidates
     - Candidates per unit
   * - 50
     - 4.0%
     - 325
     - 6.5
   * - 100
     - 10.0%
     - 401
     - 4.0
   * - 150
     - 16.4%
     - 448
     - 3.0
   * - 200
     - 21.0%
     - 486
     - 2.4
   * - 250
     - 26.3%
     - 508
     - 2.0
   * - 300
     - 31.1%
     - 521
     - 1.7
   * - 400
     - 38.3%
     - 542
     - 1.4
   * - 500
     - 49.5%
     - 560
     - 1.1
   * - 936
     - 62.4%
     - 599
     - 0.6

In comparison, existing published subdivision schemes tolerate a median of **41.9%** non-derivable codes, indicating that the 25% widening threshold yields significantly more readable identifiers than current international standards.


Open questions: the collaborator ballot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ruleset contains six active design votes and two structural questions open for discussion and refinement among collaborators.

Vote 1: Width trigger
---------------------
What criteria should trigger a sibling group to widen from two to three characters?

* **Option A (Opacity Only)**: Widen a group when more than X% of codes would be opaque.
* **Option B (Sibling Count Only)**: Widen when a parent unit has more than N children.
* **Option C (Hybrid)**: Widen when either count exceeds N or opacity exceeds X% (with a minimum group size of 10). *Currently implemented.*

**Trade-off**: A pure count rule is highly predictable, but it fails to address small, linguistically crowded groups. For example, Donetsk Oblast in Ukraine has only 48 raions but runs 60.4% opaque at two characters due to masculine/feminine spelling pairs.

Vote 2: Sibling count cutoff (N)
--------------------------------
At what group size should widening occur?

* **N = 200**: Widens 0.5% of parents; Texas counties widen. Opacity capped at ~21%.
* **N = 250**: Widens 0.4% of parents; Texas counties widen. Opacity capped at ~26%. *Currently implemented.*
* **N = 300**: Texas counties stay at 2 characters. Opacity capped at ~31%.
* **N = 500**: Texas counties stay at 2 characters. Opacity capped at ~50%.

Vote 3: Deference to published conventions
------------------------------------------
Should we adopt national code lengths automatically?

* **Option A**: Always defer to the published length.
* **Option B**: Defer only when the published scheme covers the matching level and the unit count requires it; reviewed overrides take precedence. *Currently implemented.*

**Trade-off**: Option A can over-widen. For example, adopting the ISO 3166-2:GB convention forces three characters onto the four home nations (e.g., ``GB-ENG``) when two characters are ample.

Vote 4: Skipping non-governing levels
-------------------------------------
Should we omit administrative levels that exist on paper but govern nothing (e.g., county governments in Connecticut or Massachusetts)?

* **Option A**: Keep every level everywhere to maintain structural uniformity.
* **Option B**: Skip levels that do not govern, keeping them only as named groupings. *Currently implemented.*

**Trade-off**: Option B shortens Somerville's identifier from ``US-MA-MI-SO`` to ``US-MA-SOM`` and removes a directory level. The cost is that census tracts no longer nest perfectly within towns.

That cost is smaller than it first appears, and it is worth stating precisely, because this is the sentence a future reader would use to justify reversing the decision. A tract nests in a *county* everywhere, New England included: what a New England tract lacks is one town, not one county. So the tract keeps a clean join, to its county, and tract-to-town is a linkage nothing currently asks for. What the skipped level does cost is that a county-keyed source can no longer assume level 3 means counties. That is resolved by keying on the code's ``county`` *segment* rather than on an admin level (see :mod:`openplaces.io.admin_codes.segments`), which is also what survives a government reorganizing: Connecticut replaced its eight counties with nine planning regions in 2022, and the segment outlived the level.

Vote 5: Tie-breaking by population weight
-----------------------------------------
Should population determine which unit wins its first-choice candidate code?

* **Option A**: Equal weights; ties resolve alphabetically.
* **Option B**: Raw population-weighted, applied once at minting. *Currently implemented.*

**Trade-off**: Under equal weighting, the global optimization solver systematically awards obvious codes to smaller units with longer names (which have fewer fallback options), leaving major population centers with opaque codes (e.g., Cook County, IL receiving ``CK`` instead of ``CO``). Raw population weighting aligns codes with human expectations.

Vote 6: Permanence
------------------
Should identifiers be permanent once minted?

* **Option**: Once assigned, a unit's code is locked. If a unit disappears, its code is retired and reserved forever, rather than recycled. *Proposed as non-negotiable.*

**Trade-off**: Re-minting a spine from scratch or altering tie-breaks recycles identifiers (e.g., the string ``CO-CA`` naming Caldas before the rebuild and Cauca after). Recycling requires downstream migrations to resolve codes by unit name rather than by direct string substitution.

Vote 7: Overrides for non-Latin short forms
-------------------------------------------
How should we handle languages whose customary abbreviations cannot be derived from Latin romanizations?

* **Option**: Use reviewed override tables (e.g., ``code-overrides.csv``) rather than algorithmic generation. *Currently implemented.*

**Trade-off**: In South Korea, Gyeonggi-do contracts to Gyeonggi (``KR-GG``), but Jeollabuk-do contracts to Jeonbuk (``KR-JB``) due to pronunciation rules. Algorithmic Latin string rules cannot capture these shifts; they must be supplied as data.

Vote 8: Adopting native short-form systems
------------------------------------------
Should we adopt native short-form systems when formatting allows?

* **Option**: Yes, if they are stable, widely known, and fit ``[A-Z][A-Z0-9]{1,2}``. If they fail format rules (e.g., Germany's single-letter Kfz codes like ``M`` for Munich) or structural 1:1 mappings, store them as aliases. *Currently implemented.*


Decision rules and failure preventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During spine generation and data harmonization, the following operational rules must be followed:

1. **Resolve geometry rows by unit name, never by ID**
   * *Failure prevented*: Crediting a populous unit's data to a smaller unit when IDs are recycled (e.g., the historical ID ``JP-TK`` named Tokyo, but the live ID ``JP-TK`` names Tokushima; joining by ID would credit Tokyo's 13.7M population to a town of 720,000).
2. **Handle level vs. status changes explicitly**
   * *Failure prevented*: Heuristics misinterpreting level promotions (e.g., Worcester Town replacing Worcester County) as name changes. Mismatched units must be listed explicitly in ``build.GEOMETRY_MISMATCHED``.
3. **Never emit a zero weight**
   * *Failure prevented*: Geometry artifacts (such as a bounding box overlapping sibling boundaries) leaving units with zero population weights, permanently handicapping them in tie-resolutions. Fall back to the level-median instead.
4. **Weight by raw population, not its logarithm**
   * *Failure prevented*: Logarithmic compression reducing population differences below the tie-break solver's rank penalty, causing Cook County, King County, and Somerville to lose their obvious codes to tiny neighbors.
5. **Sanity-check the total population, not the coverage**
   * *Failure prevented*: Overlooking regional mapping bugs because data coverage flags report a healthy 100%. Total population sums must be verified against global epochs.
6. **Join by the source's own administrative code, never spatially**
   * *Failure prevented*: Inefficiencies and edge-matching errors during data loading.
7. **Expect code renumbering under stable names**
   * *Failure prevented*: Joins failing when countries reorganize subdivisions (e.g., Connecticut replacing counties with planning regions in 2022). Match on state and subdivision name.
8. **Anchor rename regexes**
   * *Failure prevented*: Corrupting substrings within unrelated country prefixes. Use ``(?<![A-Z-])`` (e.g., preventing a replacement for ``CO-XXX`` from matching inside Burkina Faso's ``BF-CO-SAN``).
9. **Recipe ID is not an admin ID**
   * *Failure prevented*: Word-boundary regexes failing to match because of trailing underscores (e.g., ``CO-ANT_parcel-catastro-2018``). Directory, filename, and referencing ``recipe_id`` must be changed together.
10. **Assign discovery by specificity, not coverage**
    * *Failure prevented*: Empty merges where a broad recipe "wins" units it holds no rows for, hiding geometry gaps.


Human judgment calls
~~~~~~~~~~~~~~~~~~~~

Three processes cannot be automated and require human intervention:

- **Identifying override scopes**: Researching and choosing which countries require custom administrative boundary ingests because their default geometries are empty (e.g., Colombia's municipalities).
- **Evaluating code-change acceptability**: Reviewing whether code recycling is permissible based on which data bundles have already shipped to users.
- **Hand-sourcing missing values**: Manually supplying population data (recorded in ``admin-spine-2026_population-overrides.csv``) for units with no polygon geometries (e.g., Bogotá, England, San Andrés).
