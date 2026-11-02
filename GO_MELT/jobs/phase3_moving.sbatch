#!/bin/bash
#SBATCH --job-name=gm_p3
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 3: moving L2/L3 boxes (laser along +x), case 0, vs the fixed baseline. Two variants side by side (8 cores each).
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
PK="-peakmesh $M/L3_h0.02.mesh"
NT=8 taskset -c $(sl 0 8) jobs/run_moving_case.sh 0 results/p3_mv_h04 "-l1 $M/L1_h0.2_w.mesh -l2 $M/M2_h0.04.mesh -l3 $M/M3_h0.02.mesh -h2 0.04 -h3 0.02 $PK" &
NT=8 taskset -c $(sl 8 8) jobs/run_moving_case.sh 0 results/p3_mv_h08 "-l1 $M/L1_h0.2_w.mesh -l2 $M/M2_h0.08.mesh -l3 $M/M3_h0.02.mesh -h2 0.08 -h3 0.02 $PK" &
wait
