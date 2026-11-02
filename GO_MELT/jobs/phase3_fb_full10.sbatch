#!/bin/bash
#SBATCH --job-name=gm_p3fb
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 3 + two-way feedback by injection (-fb 1), full 10 mm track, vs baseline case0_full10 (pre-fix, -toploss 0). Two variants (8 cores each):
# A: moving L2 0.04 + L3 0.02 + feedback;  B: same L3 with a finer/longer L1 band is not needed -> B = feedback with L2 0.08.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
gen() { [ -f $M/$1 ] || python3 mesh/gen_box.py $M/$1 "${@:2}"; }
gen M2_full_h0.08.mesh --x -7.6 -4.2 --y -0.6 0.6 --z -0.8 --h 0.08 > /dev/null
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 0 8) jobs/run_moving_case.sh 0 results/p3fb_full10_h04 "-l1 $M/L1_h0.2_full.mesh -l2 $M/M2_full_h0.04.mesh -l3 $M/M3_full_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh" &
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 8 8) jobs/run_moving_case.sh 0 results/p3fb_full10_h08 "-l1 $M/L1_h0.2_full.mesh -l2 $M/M2_full_h0.08.mesh -l3 $M/M3_full_h0.02.mesh -h2 0.08 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh" &
wait
B=../../arc/results/case0_full10
for d in p3fb_full10_h04 p3fb_full10_h08; do echo "=========== $d"; python3 post_levels.py $B $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh results/$d $M/P_full_h0.02.mesh; done
