#!/bin/bash
#
#  Build, start and run the lsolve_J0 resolution study.
#  Usage: ./run_all.sh [N016 N032 ...]   (default: all four)
#  N016-N064 use 1+1+4 cores and run concurrently; N128 uses 8 cores and runs after them
#  (at most 8 cores in use at any time).
#  Each run writes its log to <run>/run.log.
#
cd "$(dirname "$0")" || exit 1
runs=${@:-N016 N032 N064 N128}

one_run() {
    ( cd "$1" && pc_build > build.log 2>&1 && pc_start > start.log 2>&1 \
        && pc_run > run.log 2>&1 && echo "$1: done" ) || echo "$1: FAILED (see $1/*.log)"
}

small=""; large=""
for r in $runs; do [ "$r" = N128 ] && large=$r || small="$small $r"; done
for r in $small; do one_run "$r" & done
wait
[ -n "$large" ] && one_run "$large"
