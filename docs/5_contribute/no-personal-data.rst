.. _no_personal_data:

Keeping personal data out
=========================

``openplaces`` processes property records, which are full of information about
identifiable people. The pipeline is built to handle that data on your own
disk. The repository is not: it is public, and anything committed to it is
published permanently, whether or not it is later deleted.

This page is the standard your contribution has to meet, and the checks that
catch the common ways personal data slips in.

The rule
--------

Never commit real personal information into any tracked file: source code,
recipe YAML, tests, fixtures, notebooks (including cell outputs), or docs.

That means no real names, no addresses tied to a named individual, no phone
numbers, no personal email addresses, no government identification numbers,
and no comparable identifiers — for any dataset, in any jurisdiction. It
applies to a single illustrative value copied from a real record just as much
as to a bulk export.

Recipes describe a source's *schema*: its column names and field codes. They
never describe its *records*.

What makes a value a record
---------------------------

Not every real-world string is personal data, and treating them all as though
they were makes tests meaningless.

A **ZIP code and its USPS-preferred city are public reference data.** So are
county names, census geography identifiers, and state codes. Tests that assert
on a ZIP-to-city mapping need real pairs, or they assert nothing.

What turns those into a record is the **house number and street name**. That
combination points at one dwelling and, through any assessor roll, at the
people in it. Parcel identifiers and account numbers behave the same way: they
resolve to one property in one jurisdiction.

So the line is drawn at the street address, not at every value that happens to
exist. Neutralize the number and street, and the surrounding reference data
stays usable.

Where personal data actually gets in
------------------------------------

In rough order of how often it happens:

**Test fixtures.** A test needs an address that parses, so a real one gets
pasted in from whatever the author was debugging. This is the single most
common route, and the resulting value is usually surrounded by other real
values — a real city, a real ZIP — which is what makes it identifiable.

**Docstrings.** An example in a module or function docstring showing what a
parsed address looks like. These are worse than test fixtures because they ship
inside the installed package, not just the repository.

**Recipe comments.** A note explaining why a particular parcel needs special
handling, naming the parcel.

**Notebook outputs.** A dataframe preview containing an owner-name column.
Notebooks in this repository are committed with outputs stripped, via
``nbstripout`` wired through ``.gitattributes``. Never disable that for a
notebook that touches entity data.

Writing a fixture that still tests something
--------------------------------------------

A fabricated address has to keep the *shape* of a real one, or the parser under
test stops exercising the code path you care about. Keep:

- a numeric house number
- a street name plus a street-type suffix
- a city
- a state code
- a five-digit postal code

What has to be invented is the **house number and street**. The city, state and
postal code can be real, and a real, large city is often the better choice: it
keeps the postal-code-to-city reference data internally consistent, so a test
that resolves one from the other still means something. An address naming a
street that does not exist at that location points at no property and no
person::

    100 Sample St, Chicago, IL 60601

If your test asserts on parsed components, make sure the substitute satisfies
every assertion — a two-token street will fail a test that expects three.

Parcel identifiers
~~~~~~~~~~~~~~~~~~

The same rule covers parcel numbers. An APN, PIN, or tax account number
resolves to exactly one property in one jurisdiction, so a real one is a record
even though it contains no name. Do not paste them into recipes, tests, or
docstrings.

Fabricate them in the source's real format — same digit count, same separators
— so that code and comments describing that format stay correct::

    parcel_id_local: '111111111111'
    parcel_id_local: '000-00-000-0000'

Before you commit
-----------------

Search your own diff, not just the files you remember touching::

    git diff --cached | grep -nE "[0-9]{1,6} [A-Za-z'-]+ (ST|RD|AVE|DR|LN|CT|BLVD|PKWY|WAY|HWY|PL|TER|CIR)\b" -i
    git diff --cached | grep -nE "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    git diff --cached | grep -nE "\b[0-9]{3}[-.][0-9]{3}[-.][0-9]{4}\b"

A pre-commit hook enforcing these patterns is the durable version of this
check; the manual form is what to run until one is installed.

If you find personal data already committed
--------------------------------------------

Stop. Do not build on top of it, and do not quietly fix it forward — editing
the value at ``HEAD`` leaves it in the history, where it stays reachable.

Report it to a maintainer. Purging it from history is a decision for a human,
because it rewrites every commit identifier from the affected commit onward and
requires coordinating everyone who has cloned the repository.

If a history rewrite becomes necessary
--------------------------------------

These are the failure modes that a rewrite hits in practice. Every one of them
survives a naive "does the string still appear at HEAD" check.

**Back up first, as a mirror.** ``git clone --mirror`` keeps every reference and
every object. If the rewrite goes wrong it is the only way back.

**Editing HEAD does not fix history.** Replacing a value in the current
checkout leaves every prior commit untouched. Only a history rewrite removes it
from the repository.

**Replacement rules are case-sensitive.** Enumerate every case variant that
occurs anywhere in history, not just the spelling you happened to see. A value
in an uppercase data fixture, a title-case docstring, and a lowercase test
assertion are three separate rules. Prefer case-preserving rules over one
case-insensitive rule: collapsing both sides of a case-comparison test to the
same string leaves the test passing while testing nothing.

**A value can span a line break.** A docstring that wraps an address across two
lines matches no single-line literal. Search for the parts, not just the whole.

**Bare numeric rules need word boundaries, and boundaries are not enough.**
Before replacing a number, check where else those digits occur. A five-digit
postal code can appear as a *substring* of a longer identifier in a reference
table, where a plain literal rule would silently corrupt hundreds of thousands
of rows. A word-bounded regex fixes that case — but if the bare number is also
a legitimate standalone value somewhere, scope the rule to the punctuation
around it instead, for example matching it only when quoted. Check the shape of
a value's surroundings, not just its digits.

**A word that is also a place name needs scoping.** Street names are frequently
also town names. Replacing the bare word corrupts the administrative reference
tables; scope the rule to the street form instead.

**Order rules longest and most specific first**, so full-address forms are
replaced before the bare street or city fallbacks fire on any stragglers.

**Cover every published branch.** A rewrite driven from a single-branch clone
leaves the value live on every other published branch. Enumerate the remote's
references and confirm each one is covered.

**Blob-level replacement is blind.** It will happily corrupt a string that
mattered. After rewriting, re-run the test suite and the linter, and re-read
the affected fixtures to confirm their assertions still describe the
substituted values.

**A rewrite does not reach everything.** Forks are separate repositories with
their own copies. Existing clones can re-push the old history. Hosting
providers retain unreferenced objects until they garbage-collect, and some
references — such as pull-request refs on GitHub — are server-side and cannot
be deleted by pushing. Removing those requires contacting the host's support
and the fork owners. Treat the rewrite as necessary but not sufficient.
