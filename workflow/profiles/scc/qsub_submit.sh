#!/usr/bin/env bash
# qsub for the SCC profile. Snakemake appends the jobscript path.
#
# Usage: qsub_submit.sh MEM_MB THREADS RUNTIME_MIN RULE [qsub args...] JOBSCRIPT
#
# Memory is requested as cores. On the SCC, mem_per_core selects nodes
# with at least that much memory per core rather than reserving it, so
# passing a job's total there sent a 26 GB job to wait for the rare
# large-memory nodes and left a 106 GB one never starting. A job that
# fits in 3 GB per thread (OPENPLACES_SCC_ANY_NODE_MB_PER_CORE) fits on
# any node, so it takes THREADS slots and no selector: the selector
# confined small jobs to the 8 GB nodes, where they queued behind the
# large ones. A larger job takes max(THREADS, ceil(MEM_MB / 8000))
# slots on nodes with at least 8 GB per core, at most
# OPENPLACES_SCC_MAX_SLOTS (default 16, 128 GB): beyond that the
# memory model is extrapolating, not measuring.
# Measured 2026-11-01 on US-FL: property spines reserved up to 267 GB, peaked 16.7 GB.
# Measured 2026-11-02 on SCC: 1,052 of 1,068 nodes have at least 3 GB per core, 431 at least 8 GB.
#
# RUNTIME_MIN becomes h_rt. RULE is the rule name, which Snakemake
# gives as GROUP for a group job (several jobs of one family run in
# one submission, see group-components in config.yaml). Snakemake
# sums its jobs' runtimes into the group's, and each is a stage
# ceiling (240 minutes for a harmonize job), not an estimate: twenty
# of them asked for 80 hours, past the 12 hours that let a job run on
# any SCC node. A group's h_rt is therefore capped at
# OPENPLACES_SCC_MAX_GROUP_RUNTIME_MIN (default 720). An ungrouped job
# keeps its own, so a batched ingest can still ask for more.
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
runtime=$3
rule=$4
shift 4
max_slots="${OPENPLACES_SCC_MAX_SLOTS:-16}"
any_node_mb="${OPENPLACES_SCC_ANY_NODE_MB_PER_CORE:-3000}"
if [ "$mem_mb" -le $(( threads * any_node_mb )) ]; then
  slots=$threads
  selector=()
else
  slots=$(( (mem_mb + 7999) / 8000 ))
  [ "$slots" -lt "$threads" ] && slots=$threads
  selector=(-l mem_per_core=8G)
fi
[ "$slots" -gt "$max_slots" ] && slots=$max_slots

max_group_runtime="${OPENPLACES_SCC_MAX_GROUP_RUNTIME_MIN:-720}"
if [ "$rule" = "GROUP" ] && [ "$runtime" -gt "$max_group_runtime" ]; then
  runtime=$max_group_runtime
fi

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
# h_rt is 0:<minutes>:00: SGE rejects "<minutes>:00" (720:00 is "not a
# time value") and reads 0:720:00 as 12 hours.
exec qsub -terse -V -P "$project" -o "$log_dir/" \
  -pe omp "$slots" ${selector[@]+"${selector[@]}"} -l "h_rt=0:${runtime}:00" "$@"
