#!/bin/bash
#SBATCH --job-name=st_calib
#SBATCH --partition=compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/%x-%j.out
# Heat-source calibration (eta, c0, n) on all 7 cases. This job only drives the iterations: every iteration submits the
# 7 cases as separate jobs (slurm/case.sbatch, one node each) and waits for them, so an iteration takes as long as one
# case. 1 driver + 7 case jobs = 8 jobs (account limit 10). From the HPC folder:
#   sbatch slurm/calibrate.sbatch OUTROOT [--eta 0.748695 --c0 0.182604 --n 0.98581 --maxit 3]
# Starting values default to config/calibration.env. Log: OUTROOT/iterations.csv and logs/st_calib-<job>.out
cd "$SLURM_SUBMIT_DIR" || exit 1
source <(grep -E '^(ABSORP|C0|N_EXP)=' config/calibration.env | sed 's/#.*//')
OUT=${1:?usage: sbatch slurm/calibrate.sbatch OUTROOT [options]}; shift
LAUNCH=slurm python3 -u post/calibrate_multi.py "$OUT" --eta "${ABSORP}" --c0 "${C0}" --n "${N_EXP}" "$@"
