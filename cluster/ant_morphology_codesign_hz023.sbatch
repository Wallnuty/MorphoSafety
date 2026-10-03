#!/bin/bash
# CO-DESIGN WITH BIGGER HAZARDS (2026-10-03): the full 2B evolution loop of
# ant_morphology_codesign.sbatch, identical in every setting except the hazard
# RADIUS, 0.23 instead of 0.16 (2.07x the disc area: 16.6% of the hazard zone
# instead of 8.0%, closest discs 0.31 m apart instead of 0.52). Same lattice
# centres, same obs width (250), same budget 2 -- so against
# checkpoints/cluster/ant_codesign_lagrangian_b2_grid7_2000M it isolates what
# the evolved body does when the hazards grow.
#
# A thin wrapper: exports HAZARD_SIZE and hands over to the main script, which
# carries the GPU/code preflights (they now also assert the built model's
# hazard radius), the training command and the checkpoint listing. Only THIS
# file's #SBATCH lines are read -- the wrapped script's are inert under
# `exec bash`. Every other knob of the main script (BUDGET, STEPS, ...) still
# overrides at submit time.
#
# Run name: ant_codesign_lagrangian_b2_grid7_hz0.23_2000M (the hazard tag keeps
# it from ever resuming into the 0.16 run).
#
# BEFORE SUBMITTING: this file and the HAZARD_SIZE knob in
# ant_morphology_codesign.sbatch must be committed, pushed, and pulled:
#
#   git pull && sbatch cluster/ant_morphology_codesign_hz023.sbatch
#
# SIZING: hazards have no collision geometry, so throughput matches the 0.16
# run (42.9k sps, 13.9 h on mscluster); --time 36h as there.
#
#SBATCH --job-name=ant-codesign-hz023
#SBATCH --output=ant_codesign_hz023_%j.log
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster45,mscluster83
#SBATCH --time=36:00:00

export HAZARD_SIZE=0.23

cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
exec bash cluster/ant_morphology_codesign.sbatch
