#!/usr/bin/env bash
# Sequential benchmark queue: runs the first not-yet-done line of bench/queue.txt, one at a time,
# each wrapped in calibration probes; lines can be appended while the queue runs.
# Done/failed status is kept in results/bench/queue.done ("<lineno> <exit> <start> <end>").
cd "$(dirname "$0")/.."; mkdir -p results/bench; touch results/bench/queue.done
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
while true; do
  n=0; job=""
  while IFS= read -r line; do
    n=$((n+1)); [[ -z "$line" || "$line" == \#* ]] && continue
    grep -q "^$n " results/bench/queue.done && continue
    job="$line"; break
  done < bench/queue.txt
  [[ -z "$job" ]] && { echo "QUEUE_EMPTY $(date +%T)" >> results/bench/queue.log; break; }
  start=$(date +%s)
  echo "=== [$n] $(date '+%F %T') $job" >> results/bench/queue.log
  taskset -c 16 python3 bench/probe.py "pre:$n" >> results/bench/probe.log 2>&1
  bash -c "$job" >> results/bench/queue.log 2>&1; rc=$?
  taskset -c 16 python3 bench/probe.py "post:$n" >> results/bench/probe.log 2>&1
  echo "$n $rc $start $(date +%s)" >> results/bench/queue.done
  echo "--- [$n] exit=$rc" >> results/bench/queue.log
done
