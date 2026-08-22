.. _no_personal_data:

Keeping personal data out
=========================

``openplaces`` processes property records containing information about identifiable people. The data pipeline is built to process this information locally on your own system. The repository itself must never store personal data. It is public, and anything committed to it is published permanently, even if it is later deleted.

This page explains the standards that your contributions must meet and how to prevent personal data from entering the repository.

The rule
--------

Never commit real personal information into any tracked file. This includes source code, recipe YAML files, tests, fixtures, notebooks (including cell outputs), and documentation.

Do not commit real names, street addresses tied to a named individual, phone numbers, personal email addresses, government identification numbers, or comparable identifiers. This rule applies to all datasets in all jurisdictions. It covers a single illustrative value copied from a real record just as much as a bulk export.

Recipes describe a source's schema (column names and field codes). They must never describe its records.

What counts as a record
-----------------------

Not every real-world string is personal data. Excluding all geographic references makes testing meaningless.

Postal codes (such as ZIP codes in the US) and their associated municipalities or canonical place names are public reference data. So are county names, census geography identifiers, and state codes. Tests that assert on a postal-code-to-city mapping require real pairs to be effective.

An address becomes a record when you combine a house number and a street name. This combination points to one dwelling and can resolve to the people in it through public assessor rolls. Parcel identifiers and tax account numbers behave the same way because they resolve to a single property in a jurisdiction.

Draw the line at the street address rather than excluding all geographical terms. Fabricating the number and street allows the surrounding reference data to remain usable.

Where personal data slips in
----------------------------

These are the most common entry points for personal data:

**Test fixtures**
A test needs an address that parses, leading a developer to paste a real address from a debugging session. This is the single most common route. The value is usually surrounded by other real values (like a real city or postal code), which makes the address identifiable.

**Docstrings**
An example in a module or function docstring shows what a parsed address looks like. These are particularly problematic because they ship inside the installed package rather than just the repository.

**Recipe comments**
A comment explains why a particular parcel needs special handling, naming the parcel or its real owner.

**Notebook outputs**
A dataframe preview contains an owner-name column. Notebooks in this repository are committed with outputs stripped using ``nbstripout`` via ``.gitattributes``. Do not disable this tool for any notebook that processes entity data.

Writing a fixture that tests the code
-------------------------------------

A fabricated address must keep the shape of a real address. Otherwise, the parser under test cannot exercise the code path you want to check. Keep the following components:

* A house number or building number
* A street name and any local street-type indicators (suffixes or prefixes)
* A city or municipality name
* A state or department code
* A valid local postal code format

Fabricate the house number and street name. The city, state, and postal code can be real. Using a real, large city keeps the postal-code-to-city reference data internally consistent. This ensures a test that resolves one from the other remains meaningful. An address naming a street that does not exist at that location points to no property and no person.

**US example**
::

    100 Sample St, Chicago, IL 60601

**European/International example**
::

    Musterstrasse 42, 10115 Berlin, Germany

If your test asserts on parsed components, make sure the substitute satisfies every assertion. For example, a two-token street will fail a test that expects three.

Parcel identifiers
~~~~~~~~~~~~~~~~~~

The same rule covers local parcel numbers, cadastral identifiers, and tax account numbers (such as APNs or PINs in the US). These resolve to a specific property in a jurisdiction, even if they do not contain a person's name. Do not paste real identifiers into recipes, tests, or docstrings.

Fabricate them in the source's real format (using the same digit count and separators) so that code and comments describing that format stay correct:
::

    parcel_id_local: '111111111111'
    parcel_id_local: '000-00-000-0000'

Before you commit
-----------------

Search your staged changes rather than relying on memory:
::

    git diff --cached | grep -nE "[0-9]{1,6} [A-Za-z'-]+ (ST|RD|AVE|DR|LN|CT|BLVD|PKWY|WAY|HWY|PL|TER|CIR)\b" -i
    git diff --cached | grep -nE "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    git diff --cached | grep -nE "\b[0-9]{3}[-.][0-9]{3}[-.][0-9]{4}\b"

The first pattern targets typical English street suffixes. If you are working with non-English data, adapt the regex to match local street indicators (such as *Calle*, *Avenida*, *Rue*, *Boulevard*, *Strasse*, or *Weg*) and local word orders where the house number follows the street name.

For example, to scan for German street address patterns:
::

    git diff --cached | grep -nE "[A-Za-zÄäÖöÜüß'-]+(str|straße|weg|gasse|allee|platz)\s+[0-9]{1,4}\b" -i

A pre-commit hook enforcing these patterns is the planned long-term solution. Run these manual checks in the meantime.

If you find personal data committed
-----------------------------------

Stop. Do not build on top of it, and do not quietly fix it forward. Editing the value at ``HEAD`` leaves the data in the git history, where it remains accessible.

Report the issue to a maintainer. Purging data from history is a decision for a human because it rewrites every commit identifier from the affected commit onward. This requires coordination among all developers who have cloned the repository.

If a history rewrite is necessary
---------------------------------

These are the common failure modes encountered during a history rewrite. Each of them can bypass a naive check that only inspects ``HEAD``.

**Back up first as a mirror**
Use ``git clone --mirror`` to preserve all references and objects. If the rewrite goes wrong, this mirror is your only way to restore the repository.

**Editing HEAD does not fix history**
Replacing a value in the current checkout leaves every prior commit untouched. Only a history rewrite removes it from the repository history.

**Replacement rules are case-sensitive**
Enumerate every case variant that occurs in the history. A value in an uppercase data fixture, a title-case docstring, and a lowercase test assertion require three separate rules. Use case-preserving rules instead of a single case-insensitive rule. Collapsing both sides of a case-comparison test to the same string leaves the test passing while testing nothing.

**A value can span a line break**
A docstring that wraps an address across two lines will not match a single-line literal. Search for the individual components rather than the whole address block.

**Bare numeric rules need word boundaries, and boundaries are not enough**
Before replacing a number, check where else those digits occur. A five-digit postal code can appear as a substring of a longer identifier in a reference table, where a plain literal rule would silently corrupt hundreds of thousands of rows. A word-bounded regex fixes that case, but if the bare number is also a legitimate standalone value, scope the rule to the punctuation around it (for example, matching it only when quoted). Check the shape of a value's surroundings rather than just its digits.

**A word that is also a place name needs scoping**
Street names are frequently also town names. Replacing the bare word corrupts the administrative reference tables. Scope the rule to the street form instead.

**Order rules longest and most specific first**
Ensure full-address forms are replaced before the bare street or city fallbacks process any remaining items.

**Cover every published branch**
A rewrite driven from a single-branch clone leaves the value active on all other published branches. Enumerate the remote references and confirm each one is covered.

**Blob-level replacement is blind**
This approach can corrupt valid strings. After rewriting, re-run the test suite and the linter, and re-read the affected fixtures to confirm their assertions still describe the substituted values.

**A rewrite does not reach everything**
Forks are separate repositories with their own copies of the history. Existing clones can re-push the old history. Hosting providers retain unreferenced objects until they garbage-collect. Some references (such as pull-request refs on GitHub) are server-side and cannot be deleted by pushing. Removing those requires contacting the host's support team and the fork owners. Treat the rewrite as necessary but not sufficient.
