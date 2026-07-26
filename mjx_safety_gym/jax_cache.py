"""Shared JAX persistent-compilation-cache setup.

XLA compile time dominates startup for MJX workloads (measured: 45s at 256
parallel envs, 264s at 1024 -- and it super-scales with env count while host
memory does not, so it is the compiler, not memory pressure). The persistent
cache turns that into a first-run-only cost.

Kept in one place so every entry point agrees on the directory: two entry
points pointing at different caches would each pay their own compiles.
"""

import os

import jax

# Default to a user-level cache rather than /tmp. On this WSL setup /tmp is
# ordinary disk under a non-systemd init, so it happens to survive reboots --
# but /usr/lib/tmpfiles.d/tmp.conf carries a "D /tmp ... 30d" rule that would
# start wiping it the moment systemd is enabled (increasingly the WSL default).
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/jax")


def configure(cache_dir: str | None = None) -> str:
    """Point JAX at a persistent on-disk compilation cache.

    Returns the directory in use. Override with the JAX_CACHE_DIR environment
    variable, or the `cache_dir` argument.
    """
    cache_dir = cache_dir or os.environ.get("JAX_CACHE_DIR") or DEFAULT_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Cache everything, not just entries above JAX's default size/time floors
    # (min_compile_time_secs defaults to 1.0s, which skips the many small
    # kernels that still add up across an MJX step).
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    return cache_dir
