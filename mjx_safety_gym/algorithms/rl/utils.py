"""Small shared helpers, ported from ss2r/rl/utils.py (trimmed to what
mjx_safety_gym.algorithms.ppo.train needs)."""

import jax


def restore_state(tree, target_example):
    state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(target_example), jax.tree_util.tree_leaves(tree)
    )
    return state
