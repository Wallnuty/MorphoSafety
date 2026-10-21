#!/bin/bash
# Run one command on a tagged Vast instance:  VAST_TAG=cdA bash cluster/vast/rx.sh 'tail logs/queue/x.log'
# (runs in /root/MorphoSafety). Local helper; vast.sh's own subcommands remain the main interface.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
id="$(cat "cluster/vast/.instance${VAST_TAG:+.$VAST_TAG}")" || exit 1
url="$(~/miniconda3/bin/vastai ssh-url "$id" 2>/dev/null)"; url="${url#ssh://}"
user="${url%%@*}"; hp="${url#*@}"; host="${hp%%:*}"; port="${hp##*:}"
exec ssh -q -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "$user@$host" "cd /root/MorphoSafety && $*"
