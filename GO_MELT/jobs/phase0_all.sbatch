#!/bin/bash
#SBATCH --job-name=gm_p0all
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 0: the other six cases, packed into ONE allocation (6 slots x 4 cores), each vs arc/results/all/case*
cd "$SLURM_SUBMIT_DIR" || exit 1
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); i=0
for c in 1.1 1.2 2.1 2.2 3.1 3.2; do
  S=$(IFS=,; echo "${CORES[*]:$((4*i)):4}"); NT=4 taskset -c $S jobs/run_gm_case.sh $c results/p0_case$c & i=$((i+1))
done; wait
for c in 1.1 1.2 2.1 2.2 3.1 3.2; do
  python3 ../HPC/post/analyze_case.py results/p0_case$c --mesh $PWD/../../arc/mesh/track_h0.02_x-0.6_1.5.mesh --case $c --json results/p0_case$c/result.json > results/p0_case$c/analysis.txt 2>&1
  echo "== case $c"; python3 post_compare.py ../../arc/results/all/case$c results/p0_case$c
done
