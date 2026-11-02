#!/bin/bash
#SBATCH --job-name=gm_p0fix
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 0 vs the FIXED upstream baseline (thermo repo origin/main, top face loses heat): baseline binary (ref/, container) 12 cores
# + gm_single (-toploss 1 -kq 3, the new defaults) 12 cores, case 0, h = 20 um, nominal only. One allocation, disjoint slots.
cd "$SLURM_SUBMIT_DIR" || exit 1
GM=$PWD; ARC=$GM/../../arc
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh))
S1=$(IFS=,; echo "${CORES[*]:0:12}"); S2=$(IFS=,; echo "${CORES[*]:12:12}")
mkdir -p results/p0fix_base
(cd $ARC && DORDER=0 BIN=$GM/results/p0fix_base/bin/thermal_in718_cpu CONTAINER=$ARC/ubuntu22.sif NT=12 OMP_PROC_BIND=false \
   taskset -c $S1 bash run/run_case.sh 0 $GM/results/p0fix_base > $GM/results/p0fix_base/wrapper.log 2>&1) &
NT=12 taskset -c $S2 jobs/run_gm_case.sh 0 results/p0fix_gm &
wait
python3 post_compare.py results/p0fix_base results/p0fix_gm
