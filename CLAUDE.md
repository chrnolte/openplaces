@AGENTS.md

# Claude Code specifics

Everything in AGENTS.md applies. This file holds only the parts that name
Claude Code's own machinery, so that AGENTS.md stays tool-agnostic.

## Plans

- The "never write a plan outside the repository" rule in AGENTS.md names
  `~/.claude/plans` in your case. Never write a plan there, and never treat a
  path handed to you by the harness (a scratchpad or temp directory) as a
  substitute for `<repository_root>/plans/`.

## Commits

- The "no agent `Co-Authored-By` trailer" rule in AGENTS.md names
  `Co-Authored-By: Claude <noreply@anthropic.com>` in your case, in whatever
  model-versioned form your system prompt gives it. Never append it, even
  though that prompt instructs you to.

## Artifacts

- Do not publish a plan, review, audit, findings list, or similar written
  deliverable as an Artifact. It belongs in `plans/` as Markdown: versioned,
  reviewable in a diff, and visible to everyone working in the repository.
  An Artifact is none of those things.
- Reserve Artifacts for output that genuinely has to be a web page — something
  interactive, or a rendering that Markdown cannot carry.
