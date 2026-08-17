# Vast.ai runbook

Rented GPUs for MorphoSafety training. Complements `cluster/*.sbatch` (Wits
mscluster) rather than replacing it: mscluster is free but queued and, per the
plan file, has burned an 8-hour job on a dead GPU; Vast is instant and paid.

## One-time setup

Already done on this machine:

- `vastai` 1.5.4 installed into **conda `base`** (deliberately not into
  `mjx-safety-gym` — a CLI dependency must never be able to perturb the
  training env).
- SSH keypair generated at `~/.ssh/id_ed25519` (no passphrase, so unattended
  `rsync`/`ssh` works).

**Two things only you can do:**

1. **API key.** Generate at <https://cloud.vast.ai/manage-keys>, then run it
   yourself — don't paste the key into a chat, it's a bearer credential for an
   account with a payment method attached:

   ```bash
   ~/miniconda3/bin/vastai set api-key <key>
   ```

   **Use the full path.** `vastai` lives in conda `base`, and activating
   `mjx-safety-gym` drops `miniconda3/bin` from PATH entirely (each env gets its
   own `bin` and its own `site-packages` — base is Python 3.14, the env is
   3.11). So a bare `vastai` is "command not found" from the training env, which
   is the env you are normally in. The full path works from anywhere.

   The key itself is stored in your home directory
   (`~/.config/vastai/vast_api_key`, or `~/.vast_api_key`), **not** per-env, so
   setting it once covers every environment.

   `vast.sh` is unaffected either way — it resolves the absolute path with a
   `command -v` fallback, so it runs correctly from any env.
2. **Credit.** The account currently shows **$0.00**. Nothing can be rented
   until it's funded.

Then register the public key so instances accept your SSH:

```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
vastai show ssh-keys
```

## Normal flow

```bash
bash cluster/vast/vast.sh search           # cheapest 24GB+ single-GPU offers
bash cluster/vast/vast.sh up <offer_id>    # rent it   <-- BILLING STARTS
bash cluster/vast/vast.sh status           # wait for "running"
bash cluster/vast/vast.sh provision        # sync code, build env, GPU preflight
bash cluster/vast/vast.sh smoke            # measure THIS card's throughput
bash cluster/vast/vast.sh train --robot ant --task minefield --penalizer none \
     --num_envs 1024 --num_timesteps 50000000 --num_evals 20 \
     --checkpoint_logdir /root/MorphoSafety/checkpoints/vast_run
bash cluster/vast/vast.sh logs             # follow it
bash cluster/vast/vast.sh pull             # checkpoints + logs back to laptop
bash cluster/vast/vast.sh down             # DESTROY   <-- BILLING STOPS
```

## Things that will bite you

**Billing starts at `up`, not at `train`.** An idle rented instance costs the
same as a busy one. `vastai stop instance` still bills for storage — only
`down` (destroy) ends it. Nothing here auto-destroys, on purpose: a crashed run
you can still SSH into is worth more than the dollar it costs to keep.

**Pull before you destroy.** `down` is irreversible and takes the disk with it.

**Code ships by rsync, not `git clone`.** This shell has no git credentials and
the branch has unpushed commits, so a remote clone would build *pre-change*
code — the exact failure already recorded against the mscluster checkout. rsync
is ~1.5 MB and always matches this disk. Corollary: **the remote is a mirror,
not a peer.** Edits made on the instance are destroyed by the next `sync`
(`--delete`), so edit locally and re-sync.

**`--num_envs` does not carry over from the laptop.** Two separate ceilings
constrain it and both move on rented hardware: VRAM at run time (6 GB laptop →
24 GB+) and **host RAM during XLA compilation**, which is what actually
OOM-killed `num_envs=2048` on the laptop's 7.6 GB. The laptop's minefield
throughput was still climbing at 512 envs (4184 sps) when it ran out of card,
so the knee was never found. `smoke` sweeps for it. Do not quote a laptop
number for a rented card.

**Nothing in this repo auto-detects hardware.** `--num_envs`, `--num_eval_envs`,
`--policy_hidden_layer_sizes`, `XLA_PYTHON_CLIENT_MEM_FRACTION` are all manual.
`validate()` in `train_ppo.py` enforces `batch_size * num_minibatches %
num_envs == 0`, so retuning is a joint choice, not one number to bump.

**The GPU preflight is not decoration.** It asserts `device.platform == "gpu"`,
not merely that `import jax` worked. A missing or mismatched CUDA plugin makes
jax fall back to CPU *silently* — that is how mscluster job 40227 ran 8 hours
and produced zero data points.

**The laptop's segfault class should not follow us here.** Exit 139 inside
`libcuda.so.1.1` was specific to the laptop's RTX 4050 / WSL2 passthrough; the
same code ran clean on mscluster's RTX 3090. If it recurs on Vast, that is new
information and worth recording in the plan file.

## Reference

- Offers are filtered on `reliability>0.98` and `cuda_max_good>=12.0`.
  The CUDA filter matters: `pyproject.toml`'s `[cuda]` extra pins
  `jax[cuda12]==0.10.2`, whose bundled `nvidia-*-cu12` wheels need a host
  driver new enough for CUDA 12. Nothing depends on the image's own toolkit.
- Default image is `pytorch/pytorch` (widely cached on Vast hosts, so it starts
  fast). Override with `IMAGE=... bash cluster/vast/vast.sh up <id>`.
- Instance id is tracked in `cluster/vast/.instance` (gitignored).
