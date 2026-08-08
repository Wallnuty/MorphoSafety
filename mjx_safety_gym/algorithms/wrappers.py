"""Cost-aware training wrappers, ported from ss2r/benchmark_suites/wrappers.py.

`CostEpisodeWrapper`: unlike brax's vanilla `EpisodeWrapper`, it threads the
`info["cost"]` key through episode aggregation instead of relying on
`state.metrics`, which matters because `cost` is only added as a
`state.metrics` key later, by the eval wrapper
(mjx_safety_gym.algorithms.rl.evaluation.ConstraintEvalWrapper).

`Saute`: state-augmentation safety, ported from
safe-learning/ss2r/benchmark_suites/wrappers.py:391. Unlike CRPO/Lagrangian
(mjx_safety_gym.algorithms.penalizers), Saute is not a `Penalizer` -- it needs
no cost critic at all. It appends a "safety budget remaining" scalar to the
observation, drains it by cost/budget each step, and once exhausted clamps
reward to -penalty and (probabilistically) ends the episode. Training then
proceeds with plain unconstrained PPO on the shaped reward.

`MorphologyDomainRandomizationWrapper`: introduces the num_envs batch
dimension for morphology randomization, vmapping a per-env (model, gene) pair
together -- see its own docstring for why genes have to be threaded in at
this level rather than appended by an outer wrapper afterward (a real bug,
caught via an actual crash, not a hypothetical).
"""

import jax
import jax.numpy as jp
from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training as brax_training


class CostEpisodeWrapper(brax_training.EpisodeWrapper):
    """Maintains episode step count and sets done at episode end."""

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            maybe_cost = nstate.info.get("cost", None)
            maybe_eval_reward = nstate.info.get("eval_reward", None)
            return nstate, (nstate.reward, maybe_cost, maybe_eval_reward)

        state, (rewards, maybe_costs, maybe_eval_rewards) = jax.lax.scan(
            f, state, (), self.action_repeat
        )
        state = state.replace(reward=jp.sum(rewards, axis=0))
        if maybe_costs is not None:
            state.info["cost"] = jp.sum(maybe_costs, axis=0)
        if maybe_eval_rewards is not None:
            state.info["eval_reward"] = jp.sum(maybe_eval_rewards, axis=0)
        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps
        return state.replace(done=done)


class Saute(Wrapper):
    """State-augmentation safety (https://arxiv.org/abs/2202.06558).

    Ported from safe-learning/ss2r/benchmark_suites/wrappers.py:391. Wraps
    *outside* the full vmap -> episode -> autoreset stack (i.e. around the
    result of `wrap_for_brax_training`, not inside it) -- see
    mjx_safety_gym.algorithms.train_ppo.

    `discounting` is accepted only to mirror upstream's constructor; upstream
    stores it and never reads it back, and neither does this port.

    NOTE on budget convention: divides raw `cost` by `budget` every step,
    undiscounted -- unlike this repo's CRPO/Lagrangian path
    (mjx_safety_gym.algorithms.ppo.train.train), which deliberately rescales
    `safety_budget` by decision-step count and `safety_discounting` (see that
    module for why -- upstream's raw-episode-budget convention was found to
    be action_repeat-sensitive there). Passing the same raw `--safety_budget`
    to a Saute run and a CRPO/Lagrangian run is NOT an apples-to-apples
    comparison.
    """

    def __init__(
        self,
        env: Env,
        discounting: float,
        budget: float,
        penalty: float,
        terminate: bool,
        termination_probability: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.budget = budget
        self.discounting = discounting
        self.terminate = terminate
        self.penalty = penalty
        self.termination_probability = termination_probability

    @property
    def observation_size(self):
        observation_size = self.env.observation_size
        if isinstance(observation_size, dict):
            return {k: v + 1 for k, v in observation_size.items()}
        return observation_size + 1

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["saute_state"] = jp.ones(())
        state.info["eval_reward"] = state.reward
        state.info["prng"] = jax.random.split(rng, 2)[0]
        if isinstance(state.obs, jax.Array):
            obs = jp.hstack([state.obs, state.info["saute_state"]])
        else:
            obs = {
                k: jp.hstack([v, state.info["saute_state"]])
                for k, v in state.obs.items()
            }
        state = state.replace(obs=obs)
        state.metrics["saute_unsafe"] = jp.zeros_like(state.reward)
        state.metrics["saute_reward"] = state.reward
        state.metrics["saute_terminate"] = jp.zeros_like(state.reward)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        saute_state = state.info["saute_state"]
        ones = jp.ones_like(saute_state)
        saute_state = jp.where(
            state.info.get("truncation", jp.zeros_like(state.done)),
            ones,
            saute_state,
        )
        nstate = self.env.step(state, action)
        cost = nstate.info.get("cost", jp.zeros_like(nstate.reward))
        # `disagreement` is an ss2r/SPiDR-specific info key this repo's envs
        # never set; kept for fidelity with upstream, harmless as a no-op.
        cost = cost + nstate.info.get("disagreement", 0.0)
        saute_state = saute_state - cost / self.budget
        saute_reward = jp.where(saute_state <= 0.0, -self.penalty, nstate.reward)
        terminate = jp.where(
            ((saute_state <= 0.0) & self.terminate) | nstate.done.astype(jp.bool_),
            True,
            False,
        )
        rng = state.info["prng"]
        if self.termination_probability >= 1.0:
            # Bernoulli(1.0) is always True, so `jp.where(terminate, True,
            # False) == terminate` exactly -- the split+sample below is
            # mathematically a no-op at the CLI default (1.0), just extra
            # per-step GPU RNG kernel launches for a result already known.
            # `self.termination_probability` is a plain python float fixed at
            # __init__, so this branch resolves at trace time (one compiled
            # graph per value), not a runtime cond.
            nstate.info["prng"] = rng
        else:
            rng, sample_rng = jax.random.split(rng)
            nstate.info["prng"] = rng
            terminate = jp.where(
                terminate,
                jax.random.bernoulli(sample_rng, self.termination_probability).astype(
                    jp.bool_
                ),
                jp.zeros_like(terminate),
            )
        saute_state = jp.where(terminate, ones, saute_state)
        nstate.info["saute_state"] = saute_state
        nstate.info["eval_reward"] = nstate.reward
        nstate.metrics["saute_reward"] = saute_reward
        nstate.metrics["saute_unsafe"] = (saute_state <= 0.0).astype(jp.float32)
        nstate.metrics["saute_terminate"] = terminate.astype(jp.float32)
        if isinstance(nstate.obs, jax.Array):
            obs = jp.hstack([nstate.obs, saute_state])
        else:
            obs = {k: jp.hstack([v, saute_state]) for k, v in nstate.obs.items()}
        return nstate.replace(
            obs=obs,
            done=terminate.astype(jp.float32),
            reward=saute_reward,
        )


class MorphologyDomainRandomizationWrapper(Wrapper):
    """Introduces the num_envs batch dimension for morphology randomization,
    vmapping a per-env (model, gene) pair together.

    Structurally like `mujoco_playground._src.wrapper.BraxDomainRandomizationVmapWrapper`
    (`env.unwrapped._mjx_model = mjx_model`, called from inside a `jax.vmap`),
    but ALSO threads a per-lane gene vector into `env.unwrapped._morphology_genes`
    at the same time. This has to be one wrapper, not two separately composed
    ones: genes cannot be recovered from a compiled `mjx.Model`, so a
    'BraxDomainRandomizationVmapWrapper + obs-appending wrapper applied
    afterward' split was tried first and doesn't work -- appending genes
    *after* this wrapper (i.e. outside it, alongside CostEpisodeWrapper)
    changes obs width only once the vmap has already run, which crashes
    CostEpisodeWrapper's own internal `action_repeat` scan: that scan's carry
    type is fixed by whatever `state` arrives from the previous full step
    (already gene-widened), but its body re-invokes the *inner*, un-widened
    env, so carry and body-output types mismatch on the very first step. Genes
    must be part of the observation from the FIRST `reset()` onward, which
    means threading them through GoToGoal.get_obs itself (see
    `morphology_conditioning` on GoToGoal), set here at the same vmap level as
    the model swap that makes `_mjx_model` per-lane.

    Must therefore wrap the RAW env, before `CostEpisodeWrapper`/
    `BraxAutoResetWrapper` -- i.e. it replaces `VmapWrapper`, at the same
    position, not something applied afterward.
    """

    def __init__(
        self,
        env: Env,
        mjx_model_v,
        in_axes,
        genes_v: jax.Array,
    ) -> None:
        super().__init__(env)
        self._mjx_model_v = mjx_model_v
        self._in_axes = in_axes
        self._genes_v = genes_v

    def _env_fn(self, mjx_model, genes) -> Env:
        env = self.env
        env.unwrapped._mjx_model = mjx_model
        env.unwrapped._morphology_genes = genes
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(mjx_model, genes, rng):
            return self._env_fn(mjx_model, genes).reset(rng)

        return jax.vmap(reset, in_axes=(self._in_axes, 0, 0))(
            self._mjx_model_v, self._genes_v, rng
        )

    def step(self, state: State, action: jax.Array) -> State:
        def step(mjx_model, genes, s, a):
            return self._env_fn(mjx_model, genes).step(s, a)

        return jax.vmap(step, in_axes=(self._in_axes, 0, 0, 0))(
            self._mjx_model_v, self._genes_v, state, action
        )
