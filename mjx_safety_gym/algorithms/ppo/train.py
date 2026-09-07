# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import pathlib
import time
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from absl import logging
from brax import envs
from brax.training import pmap, types
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint
from brax.training.types import Params, PRNGKey
from ml_collections import config_dict

from mjx_safety_gym.algorithms.penalizers import Penalizer
from mjx_safety_gym.algorithms.ppo import _PMAP_AXIS_NAME, Metrics, TrainingState
from mjx_safety_gym.algorithms.ppo import losses as ppo_losses
from mjx_safety_gym.algorithms.ppo import networks as ppo_networks
from mjx_safety_gym.algorithms.ppo import training_step as ppo_training_step
from mjx_safety_gym.algorithms.ppo.wrappers import TrackOnlineCosts
from mjx_safety_gym.algorithms.rl.evaluation import ConstraintsEvaluator
from mjx_safety_gym.algorithms.rl.utils import restore_state


def _load_checkpoint(path: str):
    """Read a checkpoint written by the save block below.

    Deliberately does NOT use `brax.training.checkpoint.load`: that helper
    builds restore_args by tree_map-ing over orbax metadata assuming every leaf
    is an array, and dies on our optimizer-state subtree with "different types
    at key path ... list vs RestoreArgs". brax's own checkpoints only hold
    (normalizer, network params); ours also carry penalizer and optimizer state.

    orbax returns each saved dataclass as a plain dict, whose pytree leaves are
    ordered ALPHABETICALLY. `restore_state` re-flattens positionally, so the
    typed containers must be rebuilt by keyword here or the leaves land in the
    wrong fields -- silently, and destructively: SafePPONetworkParams is
    (policy, value, cost_value) while the dict sorts to (cost_value, policy,
    value), which would load the cost critic's weights into the policy.
    """
    loaded = list(ocp.PyTreeCheckpointer().restore(str(path)))
    if isinstance(loaded[0], dict):
        loaded[0] = running_statistics.RunningStatisticsState(**loaded[0])
    if len(loaded) >= 2 and isinstance(loaded[1], dict):
        loaded[1] = ppo_losses.SafePPONetworkParams(**loaded[1])
    return loaded


def _passthrough_block_normalizer(tail: int, offset: int):
    """Normalize every observation dim EXCEPT a contiguous block near the end.

    The block is `tail` dims wide and sits `offset` dims from the right, so
    `(tail=7, offset=0)` is a plain suffix and `(tail=7, offset=1)` skips one
    trailing dim.

    Split out of `train` only so it can be exercised without building a
    training run; the reason it takes an offset at all is that THE GENES ARE
    NOT ALWAYS LAST. `Saute` appends its remaining-budget scalar OUTSIDE the
    base env (`wrappers.py`, a plain `hstack([obs, saute_state])`), so under
    `--penalizer saute` the layout is [..., genes(7), saute_state]. A bare
    suffix of NUM_GENES would then pass through [gene_6, saute_state] and
    normalize genes 0-5 -- six of seven genes whitened against a batch whose
    design distribution is moving, and the budget left raw. Exactly backwards.
    """
    tail, offset = int(tail), int(offset)
    lo = -(tail + offset)
    hi = -offset if offset else None

    def normalize(x, params):
        normed = running_statistics.normalize(x, params)
        parts = [normed[..., :lo], x[..., lo:hi]]
        if offset:
            parts.append(normed[..., hi:])
        return jnp.concatenate(parts, axis=-1)

    return normalize


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _device_put_replicated(tree, devices):
    """Replicate a pytree across devices, adding a leading device axis.

    jax.device_put_replicated (used by the original ss2r implementation) was
    removed in jax 0.10, so fall back to stacking one copy per device and
    sharding over the new leading axis, which is what pmap consumes.
    """
    if hasattr(jax, "device_put_replicated"):
        return jax.device_put_replicated(tree, devices)
    mesh = jax.sharding.Mesh(np.asarray(devices), ("_devices",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("_devices"))
    num_devices = len(devices)
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            jnp.stack([jnp.asarray(x)] * num_devices), sharding
        ),
        tree,
    )


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return leaf.astype(leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def train(
    environment: envs.Env,
    num_timesteps: int,
    episode_length: int,
    update_step_factory=ppo_training_step.update_fn,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    num_eval_episodes: int = 10,
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    safety_discounting: float = 0.9,
    seed: int = 0,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_evals: int = 1,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    cost_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    safety_gae_lambda: float = 0.95,
    deterministic_eval: bool = False,
    network_factory: types.NetworkFactory[
        ppo_networks.SafePPONetworks
    ] = ppo_networks.make_ppo_networks,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    normalize_advantage: bool = True,
    eval_env: Optional[envs.Env] = None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    restore_checkpoint_path: Optional[str] = None,
    safety_budget: float = float("inf"),
    penalizer: Penalizer | None = None,
    penalizer_params: Params | None = None,
    safe: bool = False,
    use_disagreement: bool = False,
    normalize_budget: bool = True,
    # Divide the budget by the episode length the batch ACTUALLY shows rather
    # than by the episode-length cap. Off by default so every run before
    # 2026-09-05 reproduces bit-identically. See the long comment at the
    # constraint in ppo/losses.py for why the fixed cap penalises a policy for
    # finishing early.
    adaptive_budget_horizon: bool = False,
    design_loop=None,
    design_updates_per_eval: int = 1,
    unnormalized_obs_tail: int = 0,
    unnormalized_obs_tail_offset: int = 0,
):
    assert batch_size * num_minibatches % num_envs == 0
    if not safe:
        penalizer = None
        penalizer_params = None
    original_safety_budget = safety_budget
    # Hoisted out of the branch below: the adaptive horizon clamps against this
    # cap whether or not the budget itself was normalised.
    num_decision_steps = episode_length // action_repeat
    if normalize_budget:
        # Rescale an episode-total cost budget onto the cost critic's scale.
        #
        # DIVERGENCE FROM ss2r: upstream divides by `episode_length`, but the
        # cost critic predicts a discounted return over *decision* steps, and
        # CostEpisodeWrapper sums cost across the action_repeat scan -- so each
        # decision step carries action_repeat steps' worth of cost. Dividing by
        # episode_length therefore makes the threshold action_repeat times too
        # strict (with action_repeat=4 the agent trained toward an effective
        # budget of 25/4, while eval still judged it against 25). Dividing by
        # the decision-step count makes `safety_budget` mean what it says.
        # Identical to upstream when action_repeat == 1.
        safety_budget = (safety_budget / num_decision_steps) / (
            1.0 - safety_discounting
        )
    xt = time.time()
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    # The number of environment steps executed for every training step.
    env_step_per_training_step = (
        batch_size * unroll_length * num_minibatches * action_repeat
    )
    num_evals_after_init = max(num_evals - 1, 1)
    # The number of training_step calls per training_epoch call.
    # equals to ceil(num_timesteps / (num_evals * env_step_per_training_step *
    #                                 num_resets_per_eval))
    # With design optimization on, ONE EPOCH IS ONE DESIGN ITERATION: the
    # design can only be resampled at a Python boundary (the models are
    # compiled host-side by MuJoCo), and `training_epoch` is the only such
    # boundary. Dividing the epoch length by design_updates_per_eval keeps the
    # total step budget identical while giving the distribution that many
    # REINFORCE updates per eval. Schaff resamples every PPO iteration; this is
    # the closest equivalent that does not pay a host round-trip per minibatch.
    design_iters_per_eval = design_updates_per_eval if design_loop is not None else 1
    num_training_steps_per_epoch = np.ceil(
        num_timesteps
        / (
            num_evals_after_init
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
            * design_iters_per_eval
        )
    ).astype(int)

    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    del key
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key, 2)
    del global_key
    assert num_envs % device_count == 0
    env = environment
    env = TrackOnlineCosts(env)
    reset_fn = jax.jit(jax.vmap(env.reset))
    # Design-aware reset. `install` assigns the incoming arrays onto the
    # MorphologyDesignWrapper DURING TRACING, so they are compiled in as real
    # arguments rather than baked-in constants -- the same trick the
    # randomization wrappers already use when they mutate `_mjx_model` inside a
    # vmapped function. Measured: swapping to an entirely different population
    # costs 0.04 s with the compile cache unchanged, against ~57 s if the graph
    # retraced. That is the single thing that makes per-iteration design
    # resampling affordable here.
    design_reset_fn = None
    if design_loop is not None:

        def _reset_with_design(rng, fields, genes):
            design_loop.install(fields, genes)
            return env.reset(rng)

        design_reset_fn = jax.jit(jax.vmap(_reset_with_design))
    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs,
        (local_devices_to_use, -1) + key_envs.shape[1:],
    )
    env_state = reset_fn(key_envs)
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    normalize = lambda x, y: x
    if normalize_observations:
        if unnormalized_obs_tail:
            # SCHAFF EXCLUDES DESIGN PARAMS FROM OBSERVATION NORMALISATION
            # (`model.py:RunningObsNorm`, which slices them off before
            # delegating). It matters more here than it looks: the genes are a
            # fixed encoding of the body, but running statistics would whiten
            # them against a batch whose design distribution is MOVING -- so
            # the same body would present differently to the policy as the
            # search narrows, which is a moving target on top of a moving
            # target. The tail is pasted back raw; its statistics are still
            # accumulated but never used, which is harmless. The block is not
            # always a bare suffix -- see the helper's docstring for why.
            normalize = _passthrough_block_normalizer(
                unnormalized_obs_tail, unnormalized_obs_tail_offset
            )

        else:
            normalize = running_statistics.normalize
    ppo_network = network_factory(
        obs_shape, env.action_size, preprocess_observations_fn=normalize
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network)
    policy_optimizer = optax.adam(learning_rate=learning_rate)
    value_optimizer = optax.adam(learning_rate=learning_rate)
    cost_value_optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
        cost_value_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )
    policy_loss, value_loss, cost_value_loss = ppo_losses.make_losses(
        ppo_network=ppo_network,
        entropy_cost=entropy_cost,
        discounting=discounting,
        safety_discounting=safety_discounting,
        reward_scaling=reward_scaling,
        cost_scaling=cost_scaling,
        gae_lambda=gae_lambda,
        safety_gae_lambda=safety_gae_lambda,
        clipping_epsilon=clipping_epsilon,
        normalize_advantage=normalize_advantage,
        safety_budget=safety_budget,
        use_disagreement=use_disagreement,
        adaptive_budget_horizon=adaptive_budget_horizon,
        budget_decision_steps=num_decision_steps,
    )
    training_step = update_step_factory(
        policy_loss,
        value_loss,
        cost_value_loss,
        policy_optimizer,
        value_optimizer,
        cost_value_optimizer,
        env,
        unroll_length,
        num_minibatches,
        make_policy,
        penalizer,
        num_updates_per_batch,
        batch_size,
        num_envs,
        env_step_per_training_step,
        safe,
        use_disagreement,
    )

    def training_epoch(
        training_state: TrainingState, state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: TrainingState, env_state: envs.State, key: PRNGKey
    ) -> Tuple[TrainingState, envs.State, Metrics]:
        nonlocal training_walltime  # type: ignore
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return (
            training_state,
            env_state,
            metrics,
        )  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = ppo_losses.SafePPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
        cost_value=ppo_network.cost_value_network.init(key_value),
    )  # type: ignore
    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), env_state.obs
    )
    policy_optimizer_state = policy_optimizer.init(init_params.policy)
    value_optimizer_state = value_optimizer.init(init_params.value)
    cost_value_optimizer_state = cost_value_optimizer.init(init_params.cost_value)
    training_state = TrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=(
            policy_optimizer_state,
            value_optimizer_state,
            cost_value_optimizer_state,
        ),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=running_statistics.init_state(obs_shape),
        env_steps=0,
        penalizer_params=penalizer_params,
    )  # type: ignore

    if restore_checkpoint_path is not None:
        loaded_params = _load_checkpoint(restore_checkpoint_path)
        restored_normalizer = restore_state(
            loaded_params[0], training_state.normalizer_params
        )
        restored_network_params = training_state.params
        restored_penalizer_params = training_state.penalizer_params
        restored_optimizer_state = training_state.optimizer_state

        # A FAILED RESTORE MUST NOT BE SILENT. Each of these blocks used to
        # swallow its exception and leave the FRESHLY INITIALISED value in
        # place, so a resume that could not read its checkpoint would train
        # from scratch while every log line looked like a normal continuation.
        # That matters most for the network params, and most of all under
        # scripts/train_chain.sh, which resumes automatically across crashes:
        # one unnoticed failure there silently discards every generation of
        # progress. The layout fallbacks are still tolerated -- checkpoints
        # written by older versions have a different element order -- but
        # anything that falls through is reported.
        if len(loaded_params) >= 2:
            try:
                restored_network_params = restore_state(
                    loaded_params[1], training_state.params
                )
            except Exception as exc:  # older layout: policy and value split
                if len(loaded_params) >= 3:
                    logging.warning(
                        "Checkpoint network params did not match the current "
                        "structure (%s); falling back to the split "
                        "policy/value layout.",
                        exc,
                    )
                    restored_network_params = training_state.params.replace(  # type: ignore
                        policy=restore_state(
                            loaded_params[1], training_state.params.policy
                        ),
                        value=restore_state(
                            loaded_params[2], training_state.params.value
                        ),
                    )
                else:
                    raise RuntimeError(
                        f"Could not restore network parameters from "
                        f"{restore_checkpoint_path}: {exc}. Refusing to "
                        "continue, because the alternative is training from "
                        "randomly initialised weights while appearing to "
                        "resume."
                    ) from exc
        if len(loaded_params) >= 3:
            try:
                restored_penalizer_params = restore_state(
                    loaded_params[2], training_state.penalizer_params
                )
            except Exception as exc:
                logging.warning(
                    "Could not restore penalizer params (%s); starting the "
                    "penalizer from its initial value. Expected when resuming "
                    "a run trained under a different --penalizer.",
                    exc,
                )
        if len(loaded_params) >= 4:
            try:
                restored_optimizer_state = restore_state(
                    loaded_params[3], training_state.optimizer_state
                )
            except Exception as exc:
                logging.warning(
                    "Could not restore optimizer state (%s); Adam moments "
                    "restart from zero. Training continues from the restored "
                    "weights, but expect a transient dip after the resume.",
                    exc,
                )

        training_state = training_state.replace(  # type: ignore
            normalizer_params=restored_normalizer,
            params=restored_network_params,
            penalizer_params=restored_penalizer_params,
            optimizer_state=restored_optimizer_state,
        )  # type: ignore

        # THE DESIGN DISTRIBUTION IS THE DELIVERABLE OF A CO-DESIGN RUN, so a
        # resume that restores the policy and not the distribution is worse
        # than a resume that fails: it looks like a continuation and is a
        # restart of the search. Loud on both paths -- there is no reading of
        # "resumed a co-design run without its design state" that is fine.
        if design_loop is not None:
            if design_loop.load(restore_checkpoint_path):
                # print, not logging.info: absl's default verbosity swallows
                # INFO in these runs (checked -- brax's own "saving checkpoint
                # to ..." never appears either), and a silent success here is
                # indistinguishable from the silent failure this whole block
                # exists to prevent. The repo prints its other startup facts
                # the same way.
                print(
                    f"restored design state from {restore_checkpoint_path}: "
                    f"{design_loop.gmm.components_left()} of "
                    f"{design_loop.gmm.n_components} components live, last "
                    f"chop at step {design_loop._last_chop}, design axis "
                    f"resumes at {design_loop._last_step} of tmax "
                    f"{design_loop.tmax}",
                    flush=True,
                )
            else:
                logging.warning(
                    "NO design_state.npz in %s -- the policy resumed but the "
                    "design distribution did NOT. It restarts at its init "
                    "(uniform means, std_init, all components live) while the "
                    "policy continues, so design/mode_* will jump and the "
                    "chopping schedule restarts. Checkpoints written before "
                    "design checkpointing existed look exactly like this.",
                    restore_checkpoint_path,
                )

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state = _device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )
    evaluator = ConstraintsEvaluator(
        eval_env,
        functools.partial(make_policy, deterministic=deterministic_eval),
        num_eval_envs=num_eval_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        key=eval_key,
        budget=original_safety_budget,
        num_episodes=num_eval_episodes,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1:
        metrics = evaluator.run_evaluation(
            _unpmap(
                (
                    training_state.normalizer_params,
                    training_state.params.policy,
                    training_state.params.value,
                )
            ),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    training_metrics: Metrics = {}
    training_walltime = 0.0
    current_step = 0
    _total_design_iters = (
        num_evals_after_init * max(num_resets_per_eval, 1) * design_iters_per_eval
    )
    for it in range(num_evals_after_init):
        logging.info("starting iteration %s %s", it, time.time() - xt)

        for _ in range(max(num_resets_per_eval, 1) * design_iters_per_eval):
            if design_loop is not None:
                # Sample a fresh population, compile it host-side, and RESET
                # onto it. The reset is not optional: BraxAutoResetWrapper
                # captures `first_state` at reset and replays it on every
                # subsequent `done`, so a new body would otherwise keep being
                # respawned into the previous body's initial pose.
                fields, genes = design_loop.sample(
                    local_devices_to_use, num_envs // process_count
                )
                key_env, design_key = jax.random.split(key_env)
                design_keys = jnp.reshape(
                    jax.random.split(design_key, num_envs // process_count),
                    (local_devices_to_use, -1, 2),
                )
                env_state = design_reset_fn(design_keys, fields, genes)

            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = training_epoch_with_timing(
                training_state, env_state, epoch_keys
            )
            current_step = int(_unpmap(training_state.env_steps))
            if design_loop is not None:
                training_metrics = dict(training_metrics)
                _dm = design_loop.finish_iteration(env_state, current_step)
                training_metrics.update(_dm)
                # ONE LINE PER DESIGN ITERATION.
                #
                # `progress_fn` fires only at evals, and `training_metrics` is
                # REBUILT every iteration -- so the log otherwise keeps one
                # design iteration in every `design_updates_per_eval` and
                # discards the rest. At the recommended sizing that is 1 in 8
                # or 1 in 16, and a chop that lands on any other iteration
                # never appears anywhere. For a search whose entire output is
                # the design trajectory, that is the wrong thing to drop.
                #
                # It also makes a long run observable between evals: at 300M
                # steps this prints roughly every 13 minutes, against ~1.7
                # hours between eval lines.
                _n_mode = sum(1 for k in _dm if k.startswith("design/mode_"))
                _mode = " ".join(
                    f"{_dm[f'design/mode_{i}']:+.2f}" for i in range(_n_mode)
                )
                # Anomalies are appended only when they occur, so their
                # presence in the line is itself the signal.
                _flags = "".join(
                    f" {k.split('/')[-1]}={_dm[k]:g}"
                    for k in (
                        "design/chopped",
                        "design/UNSCORED_DESIGNS",
                        "design/SKIPPED_UPDATE",
                    )
                    if k in _dm
                )
                print(
                    f"design it={len(design_loop.history):>4}/"
                    f"{_total_design_iters} step={current_step:>12,} "
                    f"score={_dm['design/score_mean']:+9.3f} "
                    f"comp={_dm['design/components']:.0f} "
                    f"ep/design={_dm['design/episodes_per_design']:.1f} "
                    f"arrive={_dm.get('design/arrival_rate', float('nan')):.2f}"
                    f" mode=[{_mode}]{_flags}",
                    flush=True,
                )
            key_env, tmp_key = jax.random.split(key_env)
            key_envs = jax.random.split(tmp_key, num_envs // process_count)
            key_envs = jnp.reshape(
                key_envs,
                (local_devices_to_use, -1) + key_envs.shape[1:],
            )
            # TODO: move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id == 0:
            # Run evals.
            metrics = evaluator.run_evaluation(
                _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params.policy,
                        training_state.params.value,
                    )
                ),
                training_metrics,
            )
            logging.info(metrics)
            progress_fn(current_step, metrics)
            params = _unpmap((training_state.normalizer_params, training_state.params))
            policy_params_fn(current_step, make_policy, params)
            if checkpoint_logdir:
                checkpoint_params = _unpmap(
                    (
                        training_state.normalizer_params,
                        training_state.params,
                        training_state.penalizer_params,
                        training_state.optimizer_state,
                    )
                )
                dummy_ckpt_config = config_dict.ConfigDict()
                checkpoint.save(
                    checkpoint_logdir,
                    current_step,
                    checkpoint_params,
                    dummy_ckpt_config,
                )
                if design_loop is not None:
                    # The same directory brax::checkpoint.save just made --
                    # `epath.Path(path) / f'{step:012d}'`. `DesignLoop.save`
                    # raises if it is not there rather than writing the state
                    # somewhere the resume path never looks, so a change to
                    # that layout upstream fails at the FIRST checkpoint
                    # instead of at the first resume.
                    design_loop.save(
                        pathlib.Path(checkpoint_logdir) / f"{current_step:012d}"
                    )

    total_steps = current_step
    assert total_steps >= num_timesteps

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap(
        (
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        )
    )
    logging.info("total steps: %s", total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
