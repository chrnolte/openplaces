#!/usr/bin/env bash
# qsub for the SCC profile. Snakemake appends the jobscript path.
#
# Usage: qsub_submit.sh MEM_MB THREADS [qsub args...] JOBSCRIPT
#
# Memory is requested as cores. On the SCC, mem_per_core selects nodes
# with at least that much memory per core rather than reserving it, so
# passing a job's total there sent a 26 GB job to wait for the rare
# large-memory nodes and left a 106 GB one never starting. A job instead
# takes max(THREADS, ceil(MEM_MB / 8000)) slots on nodes with at least
# 8 GB per core, at most OPENPLACES_SCC_MAX_SLOTS (default 16, 128 GB):
# beyond that the memory model is extrapolating, not measuring.
# Measured 2026-11-01 on US-FL: property spines reserved up to 267 GB, peaked 16.7 GB.
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

mem_mb=$1
threads=$2
shift 2
max_slots="${OPENPLACES_SCC_MAX_SLOTS:-16}"
slots=$(( (mem_mb + 7999) / 8000 ))
[ "$slots" -lt "$threads" ] && slots=$threads
[ "$slots" -gt "$max_slots" ] && slots=$max_slots

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
exec qsub -terse -V -P "$project" -o "$log_dir/" \
  -pe omp "$slots" -l mem_per_core=8G "$@"
