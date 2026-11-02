#!/bin/bash
#SBATCH --job-name=gm_mlstat
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Diagnostic: same as ml_short_sens but the L2/L3 boxes are STATIC (cover the whole track), to see whether the T0/Tw errors come from the fresh-slab seeds of moving boxes.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:14}")
BM=$PWD/../../arc/mesh/track_h0.02_x-0.6_1.5.mesh
TOPLOSS=0 NT=14 taskset -c $S jobs/run_moving_case.sh 0 results/ml_short_static "-l1 $M/L1_h0.1_w.mesh -l2 $M/L2_h0.04.mesh -l3 $M/L3_h0.02.mesh -h2 0.04 -h3 0.02 -mv2 0 -mv3 0 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh"
B=../../arc/results/all/case0
python3 post_sens.py $B results/ml_short_static $BM $M/L3_h0.02.mesh | head -22
