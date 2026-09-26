"""Float32 matmul precision for every entry point (default since 2026-09-26).

On RTX 30/40-series GPUs (the 3090s and the laptop's 4050) JAX's default
float32 matmul rounds its inputs to TF32 -- about 3 significant digits.
Measured on the 4050: relative error 2.8e-4 at the default against 2e-7 at
'highest', which matches the CPU exactly. That covers the policy and value
networks and the larger products inside MJX's physics step; 3x3 products and
elementwise math were float32 either way.

'highest' is the default here because (1) it cost nothing in training
throughput (2784 vs 2787 steps/s, same run otherwise), (2) MuJoCo Playground
recommends it on these GPUs for RL training stability and reproducibility,
and (3) a morphology result should not rest on TF32 rounding in contact
dynamics. TF32-trained policies replay the same under float32 (256-episode
A/B, 2026-09-26), so older checkpoints stay usable.

Set on import of `mjx_safety_gym`, so every script agrees without having to
remember. Precedence: an explicit value (the --matmul_precision flag) >
$JAX_DEFAULT_MATMUL_PRECISION > 'highest'. 'default' restores JAX's own
behaviour, i.e. TF32 on these GPUs -- what every run before 2026-09-26 used.
"""

import os

import jax

ENV_VAR = "JAX_DEFAULT_MATMUL_PRECISION"
DEFAULT = "highest"
CHOICES = ("highest", "default")


def configure_matmul_precision(value: str | None = None) -> str:
    """Apply the matmul precision and return the one in force."""
    value = value or os.environ.get(ENV_VAR) or DEFAULT
    jax.config.update("jax_default_matmul_precision", value)
    return value
