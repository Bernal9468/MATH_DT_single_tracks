#!/bin/bash
#SBATCH --job-name=gm_p3pk
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Full 10 mm track, two-way injection; the peak record now includes the coarse levels outside the fine box (wake keeps heating after the pass).
# A: L1 0.2, L2 0.04, L3 0.02 (as before)   B: L1 0.1 near the track (finer wake carrier)
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
[ -f $M/L1_h0.1_full.mesh ] || python3 mesh/make_plate_mesh.py $M/L1_h0.1_full.mesh --h 0.1 --fine-x -8 8 --fine-y 0.8 --fine-z 0.8 --r 1.3 --hmax 1.0 > /dev/null
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 0 8) jobs/run_moving_case.sh 0 results/p3pk_L1_0.2 "-l1 $M/L1_h0.2_full.mesh -l2 $M/M2_full_h0.04.mesh -l3 $M/M3_full_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh" &
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 8 8) jobs/run_moving_case.sh 0 results/p3pk_L1_0.1 "-l1 $M/L1_h0.1_full.mesh -l2 $M/M2_full_h0.04.mesh -l3 $M/M3_full_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh" &
wait
B=../../arc/results/case0_full10
for d in p3pk_L1_0.2 p3pk_L1_0.1; do echo "=========== $d"; python3 post_levels.py $B $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh results/$d $M/P_full_h0.02.mesh; done
