#!/bin/bash
# Local overnight watcher (2026-10-19): polls the tagged instances' queue status
# every $EVERY s into logs/vast_overnight/watch.log. Exits 2 if any queue stopped
# on a failure, 0 when every queue finished, 4 when MILESTONE (a remote shell
# test run on $MILESTONE_TAG) succeeds, 3 after $MAX_S seconds.
cd "$(dirname "$0")/../.." || exit 1
TAGS="${TAGS:-cdA cdB cdC unc nom}"; EVERY="${EVERY:-600}"; MAX_S="${MAX_S:-36000}"
LOG=logs/vast_overnight/watch.log; t0=$(date +%s)
while true; do
  fin=0; n=0
  echo "=== $(date '+%F %T') ===" >> $LOG
  for t in $TAGS; do
    n=$((n+1))
    s="$(VAST_TAG=$t timeout 60 bash cluster/vast/rx.sh 'tail -2 logs/queue/status.txt 2>/dev/null | tr "\n" " "; f=$(ls -t logs/queue/*.log 2>/dev/null | head -1); grep "design it=\|^step=" $f 2>/dev/null | tail -1 | cut -c1-150' 2>&1)"
    echo "$t: $s" >> $LOG
    case "$s" in *QUEUE_STOPPED*) echo "STOPPED on $t" >> $LOG; exit 2 ;; esac
    case "$s" in *QUEUE_FINISHED*) fin=$((fin+1)) ;; esac
  done
  [ "$fin" = "$n" ] && { echo "ALL FINISHED" >> $LOG; exit 0; }
  if [ -n "${MILESTONE:-}" ] && VAST_TAG=$MILESTONE_TAG timeout 60 bash cluster/vast/rx.sh "$MILESTONE" >/dev/null 2>&1; then
    echo "MILESTONE reached: $MILESTONE" >> $LOG; exit 4
  fi
  [ $(( $(date +%s) - t0 )) -gt "$MAX_S" ] && { echo "TIMEOUT" >> $LOG; exit 3; }
  sleep "$EVERY"
done
