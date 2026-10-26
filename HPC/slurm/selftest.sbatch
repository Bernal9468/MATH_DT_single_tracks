#!/bin/bash
#SBATCH --job-name=st_selftest
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Self-test on a compute node (a few minutes): sbatch slurm/selftest.sbatch     (with CONTAINER=... for the Singularity image)
cd "$SLURM_SUBMIT_DIR" || exit 1
bash check/selftest.sh
