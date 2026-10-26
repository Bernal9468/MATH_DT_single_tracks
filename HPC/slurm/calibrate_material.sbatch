#!/bin/bash
#SBATCH --job-name=st_calmat
#SBATCH --partition=compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x-%j.out
# Joint calibration of heat source (eta, c0, n) and IN718 material (k, L, solidus, liquidus) on all six NIST quantities of
# the 7 cases (post/calibrate_material.py). This job only drives the iterations: each iteration submits the 7 cases as
# separate jobs (slurm/case.sbatch, one node each, with sensitivities) and waits for them (~1.5 h per iteration on 40 cores).
# 1 driver + 7 case jobs = 8 jobs (account limit 10). From the HPC folder:
#   sbatch slurm/calibrate_material.sbatch OUTROOT [--maxit 4]
cd "$SLURM_SUBMIT_DIR" || exit 1
OUT=${1:?usage: sbatch slurm/calibrate_material.sbatch OUTROOT [options]}; shift
LAUNCH=slurm python3 -u post/calibrate_material.py "$OUT" "$@"
