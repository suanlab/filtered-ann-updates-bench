#!/usr/bin/env bash
# Gated sequential queue (successor of queue.sh):
#  - a quiet-machine gate: before each job, wait until the calibration probe reaches GATE QPS on two
#    consecutive checks (the shared machine is periodically saturated by unrelated workloads);
#  - contention check: if the post-job probe is below GATE, the job is recorded with exit code 99
#    ("contended") and appended again to the end of the queue file for a re-run.
# Jobs whose line starts with "python3 bench/stream_prep" or "python3 bench/prep" are not gated.
# Every queue line is executed with bash, so the queue file is restricted to its owner (chmod 600).
# Done tracking is by content, not line number: a job's key is the SHA-1 of its command line plus its
# occurrence number among identical lines (so a re-queued copy of a contended job is a new instance), and
# the done file records "<key> <exit code> <start> <end> <line number>". Editing or inserting other lines
# therefore never marks the wrong job done. (Done files written by the earlier line-number format are not read.)
# A probe that fails (no numeric output) is retried; PROBE_FAILS consecutive failures stop the queue.
# Disk gate: a job starts only when the file system holding the repository has at least MIN_FREE_GB free
# (default 20); otherwise the queue waits, so that a full disk pauses the queue instead of failing every job.
cd "$(dirname "$0")/.."; mkdir -p results/bench
QF=${QUEUE_FILE:-bench/queue.txt}; DF=${DONE_FILE:-results/bench/queue.done}; LF=${LOG_FILE:-results/bench/queue.log}
# MANIFEST (optional, follow-up 5 on): one JSON line per executed job with its key, line, command, exit code, times,
# the repository's commit and dirty state, and the gate probes, so that the done file's entries can be audited.
MF=${MANIFEST:-}
touch "$DF"; chmod 600 "$QF" "$DF"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
GATE=${GATE:-380}; MIN_FREE_GB=${MIN_FREE_GB:-20}; PROBE_FAILS=${PROBE_FAILS:-5}; PROBE_LOG=${PROBE_LOG:-results/bench/probe.log}; PROBE_CPU=${PROBE_CPU:-16}
probe() {   # prints the probe's QPS as an integer, or -1 if the probe failed
  local q
  q=$(taskset -c "$PROBE_CPU" python3 bench/probe.py "$1" | tee -a "$PROBE_LOG" |
      python3 -c "import json,sys;print(int(json.load(sys.stdin)['qps']))" 2>/dev/null)
  [[ $q =~ ^[0-9]+$ ]] && echo "$q" || echo -1
}
checked_probe() {   # probe with retries; exits the queue after PROBE_FAILS consecutive failures
  local q f=0
  while true; do
    q=$(probe "$1"); [[ $q -ge 0 ]] && { echo "$q"; return; }
    f=$((f+1)); echo "probe $1 failed ($f/$PROBE_FAILS) $(date +%T)" >> "$LF"
    [[ $f -ge $PROBE_FAILS ]] && { echo "QUEUE_ABORTED: calibration probe keeps failing $(date +%T)" >> "$LF"; exit 1; }
    sleep ${PROBE_RETRY_S:-30}
  done
}
while true; do
  n=0; job=""; key=""; declare -A seen=()
  while IFS= read -r line; do
    n=$((n+1)); [[ -z "$line" || "$line" == \#* ]] && continue
    h=$(printf '%s' "$line" | sha1sum | cut -c1-16); seen[$h]=$(( ${seen[$h]:-0} + 1 ))
    k="$h.${seen[$h]}"
    grep -q "^$k " "$DF" && continue
    job="$line"; key="$k"; break
  done < "$QF"
  unset seen
  [[ -z "$job" ]] && { echo "QUEUE_EMPTY $(date +%T)" >> "$LF"; break; }
  while true; do
    free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    [[ $free -ge $MIN_FREE_GB ]] && break
    echo "disk gate [$n] ${free}G free < ${MIN_FREE_GB}G, waiting $(date +%T)" >> "$LF"; sleep 300
  done
  gated=1; [[ "$job" == python3\ bench/stream_prep* || "$job" == python3\ bench/prep* ]] && gated=0
  if [[ $gated == 1 ]]; then
    ok=0
    while [[ $ok -lt 2 ]]; do
      q=$(checked_probe "gate:$n") || exit 1
      if [[ $q -ge $GATE ]]; then ok=$((ok+1)); else ok=0; echo "gate [$n] probe=$q < $GATE, waiting $(date +%T)" >> "$LF"; sleep 120; fi
    done
  fi
  start=$(date +%s)
  echo "=== [$n $key] $(date '+%F %T') $job" >> "$LF"
  bash -c "$job" >> "$LF" 2>&1; rc=$?
  post=$(checked_probe "post:$n") || exit 1
  if [[ -n "$MF" ]]; then
    JOB="$job" KEY="$key" N="$n" RC="$rc" T0="$start" T1="$(date +%s)" PRE="${q:-}" POST="$post" \
    GIT="$(git rev-parse HEAD 2>/dev/null)" DIRTY="$(git status --porcelain --untracked-files=no -- bench src scripts 2>/dev/null | wc -l)" \
    python3 -c 'import json,os;e=os.environ;print(json.dumps({"key":e["KEY"],"line":int(e["N"]),"cmd":e["JOB"],"rc":int(e["RC"]),"start":int(e["T0"]),"end":int(e["T1"]),"git":e["GIT"],"dirty_code_files":int(e["DIRTY"]),"probe_pre":e["PRE"],"probe_post":e["POST"]}))' >> "$MF"
  fi
  if [[ $gated == 1 && $post -lt $GATE && $rc == 0 ]]; then
    echo "$key 99 $start $(date +%s) $n" >> "$DF"
    echo "$job" >> "$QF"
    echo "--- [$n] contended (post probe $post); re-queued" >> "$LF"
  else
    echo "$key $rc $start $(date +%s) $n" >> "$DF"
    echo "--- [$n] exit=$rc" >> "$LF"
  fi
done
