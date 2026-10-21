#!/bin/bash
# Runs ON a Vast instance: the jobs of a queue file, one after another (one GPU
# job at a time, the standing rule), each logged to logs/queue/<name>.log.
#
#   bash cluster/vast/remote_queue.sh cluster/vast/queues/<file>.tsv
#
# Queue file: one job per line, "<name><TAB><shell command>"; blank lines and
# lines starting with # are skipped. Progress goes to logs/queue/status.txt:
#   START <name> <utc>
#   DONE <name> <exit status> <utc>
#   QUEUE_FINISHED <utc>      or      QUEUE_STOPPED <name> <utc>
# A job that fails STOPS the queue (later jobs usually depend on it, e.g. a
# fixed-body run on the co-design run before it). Re-running the queue skips
# jobs already DONE with status 0, so it resumes after a fix.
set -uo pipefail
cd /root/MorphoSafety || exit 1
source /opt/conda_profile.sh && conda activate morpho
Q="${1:?usage: remote_queue.sh <queue file>}"
mkdir -p logs/queue
S=logs/queue/status.txt
while IFS=$'\t' read -r name cmd; do
  [ -z "${name// /}" ] && continue
  case "$name" in \#*) continue ;; esac
  if grep -q "^DONE $name 0 " "$S" 2>/dev/null; then continue; fi
  echo "START $name $(date -u +%FT%TZ)" >> "$S"
  bash -c "$cmd" > "logs/queue/$name.log" 2>&1
  st=$?
  echo "DONE $name $st $(date -u +%FT%TZ)" >> "$S"
  if [ "$st" != 0 ]; then
    echo "QUEUE_STOPPED $name $(date -u +%FT%TZ)" >> "$S"
    exit "$st"
  fi
done < "$Q"
echo "QUEUE_FINISHED $(date -u +%FT%TZ)" >> "$S"
