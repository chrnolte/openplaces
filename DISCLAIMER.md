# Disclaimer

This document supplements — and does not replace or modify — the Apache License,
Version 2.0 under which `openplaces` is distributed (see [LICENSE.md](LICENSE.md)).
It sets out the intended scope of the software, what it does and does not do with
personal data, and the disclaimers of warranty and liability that go with that. It
is not legal advice, to you or about you; if you need advice about your own use of
`openplaces` or the data you process with it, consult a qualified lawyer in your
jurisdiction.

## 1. What this software is

`openplaces` is a data-processing engine and set of research tools for integrating
property, land, and environmental data. It is published as source code and
machine-readable "recipes" (instructions for downloading and transforming
third-party data). The project itself does not operate a service, does not
collect data from anyone, and does not host, sell, or distribute any dataset. Any
data ever produced by `openplaces` exists only on the infrastructure of whoever
chooses to run it, using their own downloads, credentials, and storage.

## 2. No personal data in this repository — ever

The source code, recipe YAML files, tests, fixtures, notebooks, and documentation
tracked in this repository and shipped in the published package do not, and must
never, contain real personal information: names of identifiable individuals,
addresses tied to a named person, phone numbers, personal email addresses,
government ID numbers, or comparable identifiers. This holds for every dataset
and every jurisdiction, with no exceptions, including for values that might seem
harmless as a documentation example. Recipes describe the *schema* of a source
(field names, codes, data types) — never its records. Any illustrative values that
appear in code, tests, or docs are fabricated.

This is a standing engineering rule for every human contributor and every AI
coding agent working in this codebase (see `AGENTS.md`). If personal data is ever
found to have been committed to this repository in error, please report it
privately (see Section 8) rather than opening a public issue that repeats it; the
maintainers will remove it and purge it from git history.

## 3. Third-party and public-record data

Datasets that `openplaces` recipes can download typically originate from
government agencies (e.g., county assessors, national statistical offices) or
other third parties, under those sources' own terms. The maintainers make no
representation about the accuracy, completeness, currency, or legal status of any
such upstream dataset, and disclaim responsibility for it. Anyone using a recipe
is responsible for confirming they have a lawful basis to obtain and process that
source's data — including any personal data it may contain — under that source's
terms of use and applicable law.

## 4. You are the data controller for what you process

Some public-record sources that `openplaces` can connect to (for example, some
property or land ownership records) include personal information, such as an
owner's name. If you run `openplaces` to ingest, harmonize, enrich, or curate a
dataset that includes information about identifiable individuals, *you* — not the
`openplaces` maintainers or contributors — are the party responsible for that
processing under whatever privacy law applies to you. That may include, depending
on where you and the data subjects are located, frameworks such as the EU/UK GDPR,
the U.S. CCPA/CPRA and comparable state laws, Brazil's LGPD, Canada's PIPEDA,
Australia's Privacy Act 1988, New Zealand's Privacy Act 2020, South Africa's
POPIA, or similar data-protection regimes elsewhere. A record being published by a
government as "public" does not automatically exempt it from these obligations in
every jurisdiction — notably, it generally does not under EU/UK GDPR. The
`openplaces` maintainers and contributors write general-purpose software; they do
not process, store, or have access to your data, and are not a controller,
processor, or joint controller of it.

## 5. Not advice, and not for automated decisions about individuals

Outputs of `openplaces` — including any modeled, imputed, or inferred attributes —
are provided for research purposes only. They are not legal, financial, tax,
insurance, valuation, or other professional advice. They should not be used,
without independent human review and your own compliance assessment, to make or
inform a decision about a specific identifiable individual's eligibility for
credit, insurance, housing, employment, or a similar benefit — activities that may
be separately regulated, e.g. under the U.S. Fair Credit Reporting Act, the U.S.
Fair Housing Act, Article 22 of the EU GDPR on automated decision-making, or
equivalent rules elsewhere.

## 6. No warranty; limitation of liability

Supplementing, not replacing, the warranty and liability terms of the Apache
License, Version 2.0: `openplaces`, and any data, models, or outputs produced with
it, are provided "AS IS," without warranty of accuracy, completeness, or fitness
for a particular purpose. To the maximum extent permitted by applicable law, the
maintainers, contributors, Boston University, and project funders disclaim all
liability for any claim, damage, or loss — including claims alleging privacy
violations, misappropriation, defamation, or regulatory penalties — arising from a
third party's use, misuse, or configuration of the software or of data processed
with it.

## 7. No government endorsement

Development of this codebase has been supported in part by U.S. federal research
grants from the National Science Foundation (NSF) and the National Aeronautics and
Space Administration (NASA). Nothing in this document or in the software
constitutes an endorsement by, or reflects an official position of, those
agencies, Boston University, or any other funder.

## 8. Ownership and hosting of this repository (under review)

Copyright ownership of the contents of this repository — as between individual
contributors and any institution to which they are affiliated, including Boston
University — has not yet been finally determined and is currently under
institutional review (see the Governance section of the [README](README.md)).
This repository is presently hosted under a contributor's personal GitHub
account rather than an institutional one; that hosting arrangement is also
under review and may change. Nothing in this repository, including any past or
present copyright notice, should be read as a final resolution of ownership,
custody, or hosting rights by any party.

## 9. Reporting a concern

To report personal data mistakenly present in this repository, or any other
privacy concern about the `openplaces` codebase itself, contact
**contact@openplaces.io** or use GitHub's private security-advisory reporting for
this repository — please do not post the data itself in a public issue.

This disclaimer does not create an obligation for the maintainers to monitor,
audit, or respond to concerns about data produced by someone else's deployment of
`openplaces`; such data is outside the maintainers' custody and control, and
concerns about it should be directed to whoever operates that deployment.
