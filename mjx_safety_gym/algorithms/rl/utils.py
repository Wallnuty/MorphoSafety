"""Small shared helpers, ported from ss2r/rl/utils.py (trimmed to what
mjx_safety_gym.algorithms.ppo.train needs)."""

import jax


def restore_state(tree, target_example):
    # brax >= 0.13 (since the Warp migration, 2026-10-04): the normaliser's count became a UInt64
    # (hi/lo uint32) and std_eps was added, so a checkpoint written by brax
    # 0.12 (a dict of count/mean/std/summed_variance) no longer lines up leaf
    # for leaf. Map it by field name instead; std_eps keeps its default 0.0.
    if hasattr(target_example, "std_eps"):
        import numpy as np
        from brax.training.types import UInt64

        get = tree.get if isinstance(tree, dict) else lambda k: getattr(tree, k)
        count = get("count")
        # A checkpoint written by brax 0.14 itself comes back from orbax with
        # the UInt64 as a plain {"hi", "lo"} dict: already the new format, so
        # it must not be read as an old scalar count (2026-10-15: resuming any
        # Warp-era run died here).
        if isinstance(count, dict) and set(count) == {"hi", "lo"}:
            return target_example.replace(
                mean=get("mean"),
                std=get("std"),
                summed_variance=get("summed_variance"),
                count=UInt64(hi=np.uint32(count["hi"]), lo=np.uint32(count["lo"])),
                **({"std_eps": get("std_eps")} if get("std_eps") is not None else {}),
            )
        if not isinstance(count, UInt64):
            n = int(np.asarray(get("count")).round())
            return target_example.replace(
                mean=get("mean"),
                std=get("std"),
                summed_variance=get("summed_variance"),
                count=UInt64(hi=np.uint32(n >> 32), lo=np.uint32(n & 0xFFFFFFFF)),
            )
    state = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(target_example), jax.tree_util.tree_leaves(tree)
    )
    return state
