#!/bin/bash
#SBATCH --job-name=gm_p3full
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 3 on the full 10 mm track (case 0): moving L2 (0.04) + L3 (0.02) boxes, vs the baseline arc/results/case0_full10
# (single level, pre-fix physics = no top-face loss, 490 min on 40 cores) -> run with -toploss 0 to compare like with like.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
X0=-5 X1=5 TOPLOSS=0 NT=16 taskset -c $S jobs/run_moving_case.sh 0 results/p3_full10 "-l1 $M/L1_h0.2_full.mesh -l2 $M/M2_full_h0.04.mesh -l3 $M/M3_full_h0.02.mesh -h2 0.04 -h3 0.02 -peakmesh $M/P_full_h0.02.mesh"
B=../../arc/results/case0_full10
python3 post_levels.py $B $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh results/p3_full10 $M/P_full_h0.02.mesh
