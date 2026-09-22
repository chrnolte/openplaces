# Contributing to openplaces

Contributions are welcome: a recipe for a county or country the catalog does
not cover yet, a fix, a test, a documentation page. This file is the short
version. The documentation has the long one:

- [Install](https://docs.openplaces.io/en/latest/2_get-started/install.html)
  and configure a development environment.
- [Contributor workflow](https://docs.openplaces.io/en/latest/5_contribute/contributor-workflow.html):
  branches, formatting, pre-commit hooks, pull requests.
- [Writing recipes](https://docs.openplaces.io/en/latest/5_contribute/writing-recipes.html),
  and the stage guides under `src/openplaces/recipes/_instructions/`.
- [No personal data](https://docs.openplaces.io/en/latest/5_contribute/no-personal-data.html).

## Before you open a pull request

1. Work on a branch, one feature per branch.
2. Run the checks CI runs:

   ```bash
   conda activate openplaces
   ruff format --check src/ tests/
   ruff check src/ tests/
   pytest
   ```

3. **No personal data, ever.** No real names, addresses tied to a person,
   phone numbers, personal emails or government ids in code, recipes, tests,
   fixtures, notebooks or docs. Recipes describe a source's schema, never its
   records; example values are fabricated. Notebooks are committed with
   their outputs stripped.
4. **Record a source's terms.** A recipe for a new source states its
   `license` and `terms_url`, and sets `redistribution_restricted`,
   `resale_restricted` or a `usage_requirement` where the terms say so. If
   the terms restrict publishing the source's schema or products derived from
   it, say so in the pull request before adding the recipe.
5. **Attribute third-party code.** Code adapted from another project names
   its source, authors and licence in the file's own header and in `NOTICE`.
   Permissive licences (MIT, BSD, Apache-2.0) are fine; ask before adding
   anything under a copyleft licence.
6. **Sign off every commit** (next section).

## Sign-off: the Developer Certificate of Origin

`openplaces` is licensed under the Apache License 2.0. To keep it clear that
every contribution can be distributed under that licence, each commit
carries a `Signed-off-by` line, by which you certify the
[Developer Certificate of Origin](DCO.md): in short, that you wrote the
contribution or otherwise have the right to submit it under the project's
licence. You keep the copyright in your contribution. No paperwork is
involved. The maintainer's own commits are not signed off: the certificate
is a contributor's statement to the project.

```bash
git commit -s -m "Add a parcel recipe for Example County"
```

`-s` appends the line from your git identity, so set a real name and an
address you are willing to see in a public history:

```bash
git config user.name "Your Name"
git config user.email "your.email@example.org"
```

Forgot to sign off? `git commit --amend -s` fixes the last commit, and
`git rebase --signoff main` fixes a whole branch.

If you contribute as part of your employment or studies, make sure your
employer or institution allows it: clause (a) of the certificate is your
statement that you have the right to submit the work.

Commits made with the help of an AI assistant are signed off by the person
who reviewed and submits them. Assistants are not listed as co-authors.

## Where to start

- A county, state or country without a parcel, property or transaction
  recipe: the coverage maps in the documentation show the gaps, and
  `src/openplaces/recipes/_instructions/` explains how to find and validate
  a public source.
- A land-use code without a crosswalk: about fifteen recipes map a
  `use_group_code` that nothing translates yet.
- Anything labelled `good first issue` on GitHub.

Questions: open an issue, or write to contact@openplaces.io. To report
personal data found in the repository, use GitHub's private security
advisory instead of a public issue.
