#!/bin/bash
#SBATCH --job-name=gm_p0c0
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 0 verification: gm_single on the baseline mesh (h = 20 um), case 0, vs arc/results/all/case0 (dorder 1 run, same nominal).
cd "$SLURM_SUBMIT_DIR" || exit 1
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:24}")
NT=24 taskset -c $S jobs/run_gm_case.sh 0 results/p0_case0
python3 post_compare.py ../../arc/results/all/case0 results/p0_case0
