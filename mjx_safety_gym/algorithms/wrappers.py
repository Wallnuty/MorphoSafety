"""Cost-aware training wrapper, ported from ss2r/benchmark_suites/wrappers.py.

Only `CostEpisodeWrapper` is ported: unlike brax's vanilla `EpisodeWrapper`, it
threads the `info["cost"]` key through episode aggregation instead of relying
on `state.metrics`, which matters because `cost` is only added as a
`state.metrics` key later, by the eval wrapper
(mjx_safety_gym.algorithms.rl.evaluation.ConstraintEvalWrapper).
"""

import jax
import jax.numpy as jp
from brax.envs.base import State
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
