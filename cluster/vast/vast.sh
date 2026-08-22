#!/bin/bash
# Vast.ai driver for MorphoSafety training runs. One entry point, subcommands.
#
#   bash cluster/vast/vast.sh search          # find offers
#   bash cluster/vast/vast.sh up <offer_id>   # rent it
#   bash cluster/vast/vast.sh wait            # block til ready; bails on a dead host
#   bash cluster/vast/vast.sh provision       # sync code + build the env (in tmux)
#   bash cluster/vast/vast.sh provision-log   # re-attach to a running provision
#   bash cluster/vast/vast.sh smoke           # measure this card: num_envs sweep
#   bash cluster/vast/vast.sh throughput      # full hyperparameter sweep (~90 min)
#   bash cluster/vast/vast.sh throughput-log  # follow the sweep
#   bash cluster/vast/vast.sh train <args>    # real run, detached in tmux
#   bash cluster/vast/vast.sh logs            # follow the remote log
#   bash cluster/vast/vast.sh pull            # bring checkpoints/logs home
#   bash cluster/vast/vast.sh down            # DESTROY (billing stops here)
#
# WHY RSYNC AND NOT `git clone`. This shell has no git credentials and the
# branch carries unpushed commits, so a clone on the remote would silently
# build PRE-CHANGE code -- exactly the failure recorded against the mscluster
# checkout in the plan file ("the cluster checkout is well behind"). rsync of
# the working tree is ~1.5 MB, takes a second, and by construction ships what
# is actually on this disk. It also means you never have to push to test.
#
# BILLING IS PER-HOUR AND STARTS AT `up`, NOT AT `train`. An idle rented
# instance costs the same as a busy one. `down` is the only thing that stops
# the meter -- `stop` still bills for storage. See README.md.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1
ROOT="$PWD"
STATE="$ROOT/cluster/vast/.instance"      # gitignored; holds the instance id
REMOTE_DIR="/root/MorphoSafety"
IMAGE="${IMAGE:-pytorch/pytorch}"
# 20, down from 40 (2026-08-22). Vast bills ALLOCATED disk, not used, so the
# only change that reduces the storage line is this number -- deleting files
# on the box saves nothing by itself.
#
# What actually has to fit, measured rather than guessed:
#   pytorch/pytorch:latest   ~8-9 GB on disk (3.66 GB compressed on Docker Hub)
#   the `morpho` conda env    ~6.1 GB, of which 4.4 GB is the nvidia-*-cu12
#                             wheels jax requires -- irreducible
#   repo + checkpoints        ~0.1 GB (a checkpoint is 6.4 MB; 10 of them)
#
# NOT 12. 12 is the right number only once IMAGE is a slim base; against
# pytorch/pytorch the image plus the env is ~15 GB and provisioning would die
# partway through the wheel install, which costs a whole rental cycle to
# discover. 20 also stays safe under either answer to a thing never actually
# measured here: whether Vast counts the docker image against this allocation.
# Run `df -h /` on the next rental and shrink it further if it does not.
DISK="${DISK:-20}"

# vastai lives in conda base, not in mjx-safety-gym -- deliberately, so a CLI
# dependency can never perturb the training environment.
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
VASTAI="$HOME/miniconda3/bin/vastai"
[ -x "$VASTAI" ] || VASTAI="$(command -v vastai)" || {
  echo "vastai not found. pip install vastai" >&2; exit 1; }

die () { echo "ERROR: $*" >&2; exit 1; }

# Mirrors vastai's OWN resolution order (vastai/_base.py::_resolve_api_key):
# VAST_API_KEY, then $XDG_CONFIG_HOME/vastai/vast_api_key (defaulting to
# ~/.config), then the legacy ~/.vast_api_key. Checking fewer places than the
# tool itself means rejecting a key that would have worked -- which is exactly
# what this guard did on first use: `vastai set api-key` writes the XDG path,
# and only the legacy path was checked. -s not -f, so an empty file fails here
# rather than as an auth error later.
need_key () {
  local xdg="${XDG_CONFIG_HOME:-$HOME/.config}/vastai/vast_api_key"
  [ -n "${VAST_API_KEY:-}" ] && return 0
  [ -s "$xdg" ] && return 0
  [ -s "$HOME/.vast_api_key" ] && return 0
  die \
"No Vast API key found. Looked in:
    \$VAST_API_KEY   (unset)
    $xdg
    $HOME/.vast_api_key
Generate one at https://cloud.vast.ai/manage-keys then run, yourself:
    ~/miniconda3/bin/vastai set api-key <key>
Do NOT paste the key into a chat -- it is a bearer credential."
}

instance_id () {
  [ -f "$STATE" ] || die "No instance recorded. Run: $0 up <offer_id>"
  cat "$STATE"
}

# vastai prints ssh://root@host:port -- split it for ssh/rsync, which want the
# pieces separately rather than a URL.
ssh_parts () {
  local url; url="$($VASTAI ssh-url "$(instance_id)" 2>/dev/null)" \
    || die "could not get ssh-url; is the instance running? ($0 status)"
  url="${url#ssh://}"
  SSH_USER="${url%%@*}"; local hp="${url#*@}"
  SSH_HOST="${hp%%:*}";  SSH_PORT="${hp##*:}"
  [ -n "$SSH_HOST" ] && [ -n "$SSH_PORT" ] || die "unparseable ssh-url: $url"
}

rsh () { echo "ssh -p $SSH_PORT -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts"; }
rexec () { ssh_parts; ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new \
             "$SSH_USER@$SSH_HOST" "$@"; }

case "${1:-}" in

search)
  need_key
  # reliability and cuda_max_good are the two filters that actually prevent a
  # wasted rental: jax[cuda12] needs a driver new enough for CUDA 12, and a
  # low-reliability host can vanish mid-run. verified=true keeps it to hosts
  # Vast has actually tested.
  # PIN THE CARD TO THE RTX 3090 (user's standing rule, 2026-08-17), because
  # it is what mscluster's `bigbatch` partition has. Every throughput number
  # measured here then transfers to the cluster directly instead of needing a
  # per-card re-measurement -- and the knee is genuinely card-dependent: the
  # laptop saturated at num_envs 512 while a 4070 Ti SUPER kept scaling to
  # 2048 (2026-08-16). A rental that does not match the target hardware
  # measures the wrong machine.
  #
  # SORT BY RAW PRICE, not dlperf_usd. Perf-per-dollar is the right sort when
  # the card is free to vary -- the cheapest offer is then usually an old slow
  # card and wall-clock is what costs money. Once the model is pinned, every
  # offer has near-identical compute, so cheapest IS best value.
  #
  # reliability>0.99, NOT >0.98. The first rental ever attempted here scored
  # 0.983 -- the LOWEST in its result set -- and its host could not pull a
  # docker image at all. Reliability is Vast's own measure of how often a
  # host's rentals actually work; the few cents saved by dropping the floor
  # are worth far less than one dead boot.
  GPU="${GPU:-RTX_3090}"
  echo "Cheapest $GPU, reliability>0.99, CUDA 12+, verified:"
  $VASTAI search offers \
    "num_gpus=1 gpu_name=$GPU cuda_max_good>=12.0 reliability>0.99
     verified=true disk_space>50 inet_down>100 rentable=true" \
    -o 'dph_total' --limit "${LIMIT:-15}"
  echo
  echo "Then: $0 up <ID>   (cheapest = top row)"
  echo "Override the card with: GPU=RTX_4090 $0 search"
  ;;

up)
  need_key
  OFFER="${2:-}"; [ -n "$OFFER" ] || die "usage: $0 up <offer_id>"
  [ -f "$STATE" ] && die "instance $(cat "$STATE") already recorded. '$0 down' first."
  # --direct gives a direct TCP connection rather than Vast's proxy: faster
  # rsync, and the proxy has been flaky for long-lived sessions.
  # --raw (JSON) rather than scraping the human table: if the id cannot be
  # parsed, the instance is ALREADY RENTED AND BILLING, and an unrecorded id
  # means `down` cannot find it. Print the whole response on any parse failure
  # so the id is at least recoverable by eye.
  out="$($VASTAI create instance "$OFFER" --image "$IMAGE" --disk "$DISK" \
          --ssh --direct --raw 2>&1)" || die "create failed: $out"
  id="$(printf '%s' "$out" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(1)
print(d.get("new_contract") or d.get("id") or "")
' 2>/dev/null)"
  if [ -z "$id" ]; then
    echo "$out" >&2
    die "RENTED BUT COULD NOT PARSE THE ID -- an instance may be billing now.
Find it with:  $VASTAI show instances
Then record it:  echo <id> > $STATE
Then stop it:  $0 down"
  fi
  echo "$id" > "$STATE"
  echo
  echo "instance $id recorded in $STATE"
  echo "BILLING HAS STARTED. Stop it with: $0 down"
  echo "Next: $0 wait     (blocks until ready; bails if the host cannot boot)"
  ;;

status)
  need_key
  $VASTAI show instances
  [ -f "$STATE" ] && echo && echo "tracked instance: $(cat "$STATE")"
  ;;

wait)
  # Poll until the box is actually usable, and BAIL LOUDLY on a stuck boot.
  # Observed 2026-08-16 on the first rental ever attempted: the host's network
  # intercepted registry-1.docker.io with a *.facebook.com certificate, so its
  # docker daemon could never pull the image. The instance sat in "loading"
  # forever, billing the whole time, and `show instances` reported it as
  # cur_state=running -- only status_msg revealed it. A boot that is still
  # failing after ~4 min is not going to recover; destroy and rent elsewhere.
  need_key
  id="$(instance_id)"
  for i in $(seq 1 24); do
    # id goes through the environment, not string interpolation: nesting the
    # shell variable inside a quoted python -c is how this broke the first time.
    st="$($VASTAI show instances --raw 2>/dev/null | INST_ID="$id" python3 -c '
import json, os, sys
want = os.environ["INST_ID"]
rows = [x for x in json.load(sys.stdin) if str(x["id"]) == want]
if not rows:
    print("GONE|")
    raise SystemExit
row = rows[0]
msg = (row.get("status_msg") or "").strip()[:120]
print("|".join([str(row.get("actual_status")),
                str(row.get("intended_status")), msg]))
' 2>/dev/null)"
    state="${st%%|*}"; rest="${st#*|}"
    intended="${rest%%|*}"; msg="${rest#*|}"
    echo "[$(date +%H:%M:%S)] $state ${msg:+| $msg}"
    case "$state" in
      running)
        # Vast reports `running` when the CONTAINER is up, but sshd inside it
        # starts accepting a few seconds later. Observed 2026-08-17: `wait`
        # printed READY and the provision immediately after it died with
        # "Connection closed" on both the rsync and the tmux launch, burning a
        # whole provision cycle of billed time on a box that was fine. Poll
        # the thing actually needed -- a working ssh -- not the thing vast
        # reports.
        ssh_parts
        for j in $(seq 1 24); do
          if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
               -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" true 2>/dev/null; then
            echo "READY (ssh up). Next: $0 provision"; exit 0
          fi
          [ "$j" = 1 ] && echo "         container up; waiting for sshd"
          sleep 5
        done
        die "container is running but sshd never accepted after ~2 min" ;;
      GONE)    die "instance $id no longer exists" ;;
    esac
    # `create instance` can leave the box with intended_status=stopped and NO
    # error message -- so it never boots, and every poll looks like a slow
    # image pull. Observed on the second rental (2026-08-16): it sat "loading"
    # for 9 minutes, billing, until an explicit `start` was issued, after which
    # it was ready in 2. Nothing distinguishes it from a slow pull except this
    # field, so start it rather than wait out a boot that was never scheduled.
    if [ "$intended" = "stopped" ]; then
      echo "         intended_status=stopped -- it was never told to boot; starting it"
      $VASTAI start instance "$id" >/dev/null 2>&1 || true
    fi
    case "$msg" in
      *"failed to verify certificate"*|*"Error response from daemon"*|*"manifest unknown"*|*"no space left"*)
        [ "$i" -ge 8 ] && die \
"Boot is stuck on an image-pull error and will not recover -- the HOST is
broken, not your config. Stop paying for it and take another offer:
    $0 down
    $0 search      # then: $0 up <new id>" ;;
    esac
    timeout 20 bash -c 'read -t 20 </dev/zero' 2>/dev/null || true
  done
  die "still not running after ~8 min; check '$0 status' and consider '$0 down'"
  ;;

ssh)
  ssh_parts
  echo "ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
  exec ssh -p "$SSH_PORT" -o StrictHostKeyChecking=accept-new "$SSH_USER@$SSH_HOST"
  ;;

sync)
  ssh_parts
  # Excludes matter: .git is 144 MB and checkpoints 347 MB, neither of which
  # the remote needs to train. The working tree itself is ~1.5 MB.
  rsync -az --delete --info=stats1 \
    --exclude '.git' --exclude 'checkpoints' --exclude 'logs' \
    --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
    --exclude 'cluster/vast/.instance' \
    -e "$(rsh)" "$ROOT/" "$SSH_USER@$SSH_HOST:$REMOTE_DIR/" \
    || die "rsync failed -- the remote does NOT have your code. Do not train."
  echo "code synced to $SSH_HOST:$REMOTE_DIR"
  ;;

provision)
  "$0" sync || exit 1
  ssh_parts
  # IN TMUX, not over the raw ssh channel. Installing jax[cuda12] pulls ~3 GB
  # of wheels and builds two git-pinned forks from source -- 10-20 min during
  # which any network blip, or an impatient local timeout, kills the client
  # while the remote pip keeps running orphaned. That happened on the first
  # provision here (local `timeout 900` fired at 15 min; remote pip was still
  # going, and the ssh output had shown nothing at all). tmux decouples the
  # two: the install owns its own session, and this just follows a log.
  echo "=== provisioning in tmux (~10-20 min; safe to Ctrl-C, it keeps going) ==="
  rexec "cd $REMOTE_DIR && rm -f /root/provision.done && \
         tmux kill-session -t provision 2>/dev/null; \
         tmux new-session -d -s provision \
           'bash cluster/vast/remote_provision.sh > /root/provision.log 2>&1; \
            echo \$? > /root/provision.done' && echo 'started'"
  # Follow until the sentinel appears. Re-attachable: run `$0 provision-log`
  # if this end drops.
  rexec "tail -n +1 -f /root/provision.log 2>/dev/null & \
         while [ ! -f /root/provision.done ]; do sleep 5; done; \
         sleep 2; kill %1 2>/dev/null; \
         echo; echo \"=== remote_provision.sh exit=\$(cat /root/provision.done) ===\""
  ;;

provision-log)
  # Re-attach to a provision already in flight, or read the finished one.
  rexec "tail -n 200 -f /root/provision.log"
  ;;

smoke)
  ssh_parts
  # Two stages, cheapest first, mirroring the mscluster discipline in the plan:
  # prove the GPU is real before spending any compute on it.
  rexec "bash $REMOTE_DIR/cluster/vast/remote_smoke.sh"
  ;;

throughput)
  ssh_parts
  # ~60-90 min, so tmux: an SSH drop mid-sweep would otherwise waste the whole
  # rental. Poll it with `$0 throughput-log`.
  rexec "cd $REMOTE_DIR && mkdir -p logs && \
         tmux kill-session -t tput 2>/dev/null; \
         tmux new-session -d -s tput \
         'bash cluster/vast/remote_throughput.sh 2>&1 | tee logs/throughput.log' && \
         echo 'launched in tmux session: tput'"
  echo "Follow it with: $0 throughput-log"
  ;;

throughput-log)
  ssh_parts
  rexec "tail -n ${LINES:-40} -f $REMOTE_DIR/logs/throughput.log"
  ;;

train)
  shift
  [ $# -gt 0 ] || die "usage: $0 train --robot ant --task minefield ..."
  ssh_parts
  # tmux, so the run survives the SSH session dropping -- which it will.
  rexec "cd $REMOTE_DIR && mkdir -p logs && \
         tmux kill-session -t train 2>/dev/null; \
         tmux new-session -d -s train \
         'source /opt/conda/etc/profile.d/conda.sh && conda activate morpho && \
          python -u -m mjx_safety_gym.algorithms.train_ppo $* \
          2>&1 | tee logs/vast_train.log' && \
         echo 'launched in tmux session: train'"
  echo "Follow it with: $0 logs"
  ;;

logs)
  ssh_parts
  rexec "tail -f $REMOTE_DIR/logs/vast_train.log"
  ;;

pull)
  ssh_parts
  mkdir -p "$ROOT/checkpoints/vast" "$ROOT/logs/vast"
  rsync -az --info=stats1 -e "$(rsh)" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/checkpoints/" "$ROOT/checkpoints/vast/" || true
  rsync -az --info=stats1 -e "$(rsh)" \
    "$SSH_USER@$SSH_HOST:$REMOTE_DIR/logs/" "$ROOT/logs/vast/" || true
  echo "pulled into checkpoints/vast/ and logs/vast/"
  ;;

down)
  need_key
  id="$(instance_id)"
  echo "This DESTROYS instance $id. Anything not pulled is lost."
  read -r -p "Type the instance id to confirm: " confirm
  [ "$confirm" = "$id" ] || die "mismatch, aborted"
  # -y because `vastai destroy` prompts on its own too. Without it the command
  # silently "Aborted." while the instance kept billing -- observed 2026-08-16.
  $VASTAI destroy instance "$id" -y && rm -f "$STATE"
  echo "destroyed $id; billing stopped"
  $VASTAI show instances
  ;;

*)
  sed -n '2,26p' "$0"
  ;;
esac
