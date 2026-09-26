__version__ = "0.1.6"

# Float32 matmuls in every entry point, not TF32 -- see numerics.py.
from mjx_safety_gym.numerics import configure_matmul_precision as _configure_matmul_precision

_configure_matmul_precision()
