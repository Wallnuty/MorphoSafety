#!/bin/bash
# Stage 1 of the mscluster rollout (see the project plan, "Running on the Wits
# mscluster"): confirm the bigbatch node's NVIDIA driver actually works with
# this project's pip-bundled CUDA 12.9 runtime BEFORE spending any real time
# budget on a training run. This is the one real unknown identified in the
# plan -- everything else (env vars, module loads) is a non-issue for this
# project, since jax-cuda12-plugin brings its own CUDA runtime and only needs
# a compatible host driver (see the [cuda] extra's comment in pyproject.toml).
#
# Prerequisite (do this once, manually, before submitting):
#   1. Push this branch (`test`) so the cluster can pull it.
#   2. On the cluster: install your own Miniconda into
#      /home-mscluster/<you>/miniconda3 (per the guide's own recommendation --
#      don't use the shared Anaconda), matching what's already done locally.
#   3. Clone the repo, then:
#        conda create -n mjx-safety-gym python=3.11 -y
#        conda activate mjx-safety-gym
#        pip install -e ".[cuda]"
#      (add ".[cuda,evolve]" instead if you also want pymoo for evolve.py)
#
# Submit from the repo root on the cluster:
#   sbatch cluster/gpu_smoke_test.sbatch
# Then watch progress with:
#   squeue --me
# and read the result in gpu_smoke_test_<jobid>.log once it finishes (should
# be seconds to a couple of minutes -- this does no training, just imports).
#
# NOTE on --gres=gpu:1 below: the guide's own minimal bigbatch example doesn't
# show a GPU resource flag, so whether it's required here is unverified --
# that's exactly what this job settles. If sbatch REJECTS the job outright
# (rather than the job running and failing), that's a sign this flag isn't
# valid on this partition -- remove it and resubmit.

#SBATCH --job-name=gpu-smoke-test
#SBATCH --output=gpu_smoke_test_%j.log
#SBATCH --partition=bigbatch
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

set -u

echo "=== node ==="
hostname
date

echo
echo "=== nvidia-smi (host driver + GPU visibility) ==="
nvidia-smi

echo
echo "=== conda env ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mjx-safety-gym
python --version
pip show jax jaxlib jax-cuda12-plugin 2>&1 | grep -E "^Name|^Version"

echo
echo "=== jax device check ==="
# pyproject.toml's own warning: "on a headless cluster node a missing plugin
# does NOT fail, it silently falls back to CPU." So this asserts on platform,
# not just on jax.devices() running without an exception -- a silent CPU
# fallback would otherwise look like a pass.
python -c "
import jax
devices = jax.devices()
print('JAX devices:', devices)
gpu_devices = [d for d in devices if d.platform == 'gpu']
assert gpu_devices, (
    'FAIL: jax.devices() returned no GPU -- silently fell back to CPU. '
    'Check jax-cuda12-plugin is installed (pip install -e \".[cuda]\") and '
    'that the host driver supports the pinned CUDA 12.9 pip runtime.'
)
print(f'PASS: {len(gpu_devices)} GPU device(s) found: {gpu_devices}')
"

echo
echo "=== minimal MJX step (confirms the CUDA path actually executes, not just initializes) ==="
python -c "
import mujoco
from mujoco import mjx
print('mujoco', mujoco.__version__)
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><geom type=\"plane\" size=\"1 1 0.1\"/>'
    '<body pos=\"0 0 1\"><joint type=\"free\"/><geom type=\"sphere\" size=\"0.1\"/></body>'
    '</worldbody></mujoco>'
)
mx = mjx.put_model(m)
d = mjx.make_data(mx)
d = mjx.step(mx, d)
d.qpos.block_until_ready()
print('PASS: mjx.step executed on device, qpos =', d.qpos)
"

echo
echo "=== done ==="
