#!/usr/bin/env bash
# qsub for the SCC profile. Snakemake appends the jobscript path.
#
# The job is charged to the SCC project the run belongs to: the one
# whose space the checkout sits in (/project/<name>/...,
# /projectnb/<name>/..., or the same under /restricted). A run from
# /project/places/... is charged to places, one from
# /project/openplaces/... to openplaces. OPENPLACES_SCC_PROJECT
# overrides it.
#
# -V passes the submitting environment, so `python -m openplaces...` on
# the compute node is the activated env's python with the same
# PYTHONPATH. Logs go to OPENPLACES_SGE_LOG_DIR, which the Snakefile
# sets to the configured logs directory.
set -euo pipefail

project="${OPENPLACES_SCC_PROJECT:-}"
if [ -z "$project" ]; then
  # /project/<name> or /projectnb/<name>: "projectnb?" would read
  # "projectn" plus an optional "b" and never match /project/.
  if [[ "$PWD" =~ ^(/restricted)?/project(nb)?/([^/]+) ]]; then
    project="${BASH_REMATCH[3]}"
  else
    echo "qsub_submit.sh: cannot tell the SCC project from $PWD;" \
      "set OPENPLACES_SCC_PROJECT" >&2
    exit 1
  fi
fi

log_dir="${OPENPLACES_SGE_LOG_DIR:-$PWD/.snakemake/sge}"
mkdir -p "$log_dir"
exec qsub -terse -V -P "$project" -o "$log_dir/" "$@"
