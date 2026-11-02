#!/bin/bash
#SBATCH --job-name=gm_mlshort
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Multi-level moving boxes + first-order Enzyme sensitivities + mechanical dumps, case 0 (2.1 mm track), toploss 0 vs the old baseline arc/results/all/case0
# (thermal dorder 1, 102 min on 40 cores).  L1 0.1 mm / L2 0.04 / L3 0.02, two-way injection, 14 threads.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:14}")
BM=$PWD/../../arc/mesh/track_h0.02_x-0.6_1.5.mesh
TOPLOSS=0 NT=14 taskset -c $S jobs/run_moving_case.sh 0 results/ml_short_c0 "-l1 $M/L1_h0.1_w.mesh -l2 $M/M2_h0.04.mesh -l3 $M/M3_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh -dumpdir $PWD/results/ml_short_c0/dumps -dumpmesh $BM"
B=../../arc/results/all/case0
python3 post_sens.py $B results/ml_short_c0 $BM $M/L3_h0.02.mesh
python3 post_levels.py $B $BM results/ml_short_c0 $M/L3_h0.02.mesh
