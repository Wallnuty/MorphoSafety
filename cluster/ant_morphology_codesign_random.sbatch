#!/bin/bash
# CO-DESIGN ON RANDOM HAZARD FIELDS (2026-10-03): the full 2B evolution loop of
# ant_morphology_codesign.sbatch, identical in every setting except hazard
# PLACEMENT -- rejection-sampled at every reset instead of the fixed staggered
# lattice. Same 20 discs at radius 0.16, wholly inside the same hazard zone,
# same obs width (250), same budget 2. Against
# checkpoints/cluster/ant_codesign_lagrangian_b2_grid7_2000M it asks whether
# the evolved body (and the walk-over strategy) survive when the hazards stop
# arriving at a fixed pitch: on the lattice, safety can be a memorised stride;
# here it has to come from the lidar and the foot grid.
#
# Layouts are redrawn whenever a lane resets, and co-design resets every lane
# at each design iteration (~1.5 episodes per lane), so the run sees ~530k
# distinct fields. Eval draws its own, held out. Discs may overlap by up to
# 3 cm (keepout 0.9 r, as Safety Gym); 0.7 overlapping pairs per field on
# average.
#
# A thin wrapper: exports HAZARD_PLACEMENT and hands over to the main script,
# whose preflight also checks that two resets give different fields. Only THIS
# file's #SBATCH lines are read. Every other knob of the main script (BUDGET,
# HAZARD_SIZE, ...) still overrides at submit time.
#
# Run name: ant_codesign_lagrangian_b2_grid7_random_2000M.
#
# BEFORE SUBMITTING: this file, --hazard_placement (envs/run_forward.py,
# train_ppo.py) and the knob in ant_morphology_codesign.sbatch must be
# committed, pushed, and pulled:
#
#   git pull && sbatch cluster/ant_morphology_codesign_random.sbatch
#
# SIZING: same physics as the lattice run (42.9k sps, 13.9 h); --time 36h.
#
#SBATCH --job-name=ant-codesign-random
#SBATCH --output=ant_codesign_random_%j.log
#SBATCH --partition=bigbatch
# mscluster75 added 2026-10-03 (user's call).
#SBATCH --exclude=mscluster45,mscluster75,mscluster83
#SBATCH --time=36:00:00

export HAZARD_PLACEMENT=random

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
exec bash cluster/ant_morphology_codesign.sbatch
