#!/bin/bash
#SBATCH --job-name=gm_p1
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 1: static nested levels, one-way coupling, case 0 (top-face loss on). Three variants side by side (5+5+6 cores; compute3 nodes have 16 CPUs).
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
NT=5 taskset -c $(sl 0 5) jobs/run_levels_case.sh 0 results/p1_L1_L3     "-l1 $M/L1_h0.2.mesh -l3 $M/L3_h0.02.mesh" &
NT=5 taskset -c $(sl 5 5) jobs/run_levels_case.sh 0 results/p1_L1_L2a_L3 "-l1 $M/L1_h0.2.mesh -l2 $M/L2_h0.08.mesh -l3 $M/L3_h0.02.mesh" &
NT=6 taskset -c $(sl 10 6) jobs/run_levels_case.sh 0 results/p1_L1_L2b_L3 "-l1 $M/L1_h0.2.mesh -l2 $M/L2_h0.04.mesh -l3 $M/L3_h0.02.mesh" &
wait
