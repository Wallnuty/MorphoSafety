cd ~/MorphoSafety && git pull
ls checkpoints/ant_minefield_none_wide_ctrl/          # confirm 000050380800 is the last dir

# A. shaping alone (obs 47)
SAFE_PENALIZER=none SAFE_SHAPING_W=0.3 sbatch cluster/ant_minefield_safe_b50.sbatch

# B. shaping + per-foot clearance obs (obs 63)
SAFE_PENALIZER=none SAFE_SHAPING_W=0.3 SAFE_FOOT_OBS=1 sbatch cluster/ant_minefield_safe_b50.sbatch

# C. foot obs alone
SAFE_PENALIZER=none SAFE_FOOT_OBS=1 sbatch cluster/ant_minefield_safe_b50.sbatch

# D. Lagrangian warm-started from the 50M control (obs 47 — the checkpoint's width)
SAFE_RESUME=$PWD/checkpoints/ant_minefield_none_wide_ctrl/000050380800 \
SAFE_NAME=ant_minefield_ppo_lagrangian_b15_warm \
SAFE_PENALIZER=ppo_lagrangian SAFE_BUDGET=15 SAFE_MULT_LR=3e-5 \
  sbatch cluster/ant_minefield_safe_b50.sbatch