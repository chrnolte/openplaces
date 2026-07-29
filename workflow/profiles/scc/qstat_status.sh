#!/usr/bin/env bash
# Map an SGE job id to a Snakemake status (running / success / failed).
# Used by cluster-generic-status-cmd in the SCC profile.

jobid="$1"

if qstat -j "$jobid" >/dev/null 2>&1; then
    echo running
    exit 0
fi

# Job left the queue: consult qacct for the exit status. qacct can lag a
# few seconds behind qstat on the shared accounting file; treat a missing
# record as still running so Snakemake polls again rather than failing.
acct=$(qacct -j "$jobid" 2>/dev/null)
if [ -z "$acct" ]; then
    echo running
    exit 0
fi

exit_status=$(echo "$acct" | awk '/^exit_status/ {print $2; exit}')
failed=$(echo "$acct" | awk '/^failed/ {print $2; exit}')

if [ "$exit_status" = "0" ] && [ "$failed" = "0" ]; then
    echo success
else
    echo failed
fi
