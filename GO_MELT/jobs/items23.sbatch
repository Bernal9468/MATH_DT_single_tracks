#!/bin/bash
#SBATCH --job-name=gm_i23
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Item 3: 10 mm track nominal, multi-level final config (L1 0.1, L2 0.04, deeper L3 z=-0.4), toploss 0 vs arc/results/case0_full10 (8 cores, pinned).
# Item 2: hc check with the top face loss ON: multi-level (final config, -dorder 1) vs single-level gm_single (-dorder 1), both toploss 1, case 0 (2.1 mm), 8 cores, run in sequence.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
BM=$PWD/../../arc/mesh/track_h0.02_x-0.6_1.5.mesh
(
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 0 8) jobs/run_moving_case.sh 0 results/i3_full10_deep "-l1 $M/L1_h0.1_full.mesh -l2 $M/M2_full_h0.04.mesh -l3 $M/M3d_full_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh"
echo "=========== item 3: 10 mm"; grep -E "^done|^level" results/i3_full10_deep/run.log
python3 post_levels.py ../../arc/results/case0_full10 $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh results/i3_full10_deep $M/P_full_h0.02.mesh
) > logs/i3_part.log 2>&1 &
TOPLOSS=1 NT=8 taskset -c $(sl 8 8) jobs/run_moving_case.sh 0 results/i2_ml_tl1 "-l1 $M/L1_h0.1_w.mesh -l2 $M/M2_h0.04.mesh -l3 $M/M3d_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh"
NT=8 EXTRAARGS="-dorder 1 -toploss 1" taskset -c $(sl 8 8) jobs/run_gm_case.sh 0 results/i2_single_tl1 $BM
echo "=========== item 2: hc (toploss 1), ML vs single-level"; grep -E "^done|^sensitiv" results/i2_ml_tl1/run.log results/i2_single_tl1/run.log
TMIN=5e-4 python3 post_sens.py results/i2_single_tl1 results/i2_ml_tl1 $BM $M/L3_h0.02.mesh
wait; cat logs/i3_part.log
