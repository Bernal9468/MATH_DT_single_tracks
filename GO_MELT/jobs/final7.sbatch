#!/bin/bash
#SBATCH --job-name=gm_final7
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# FINAL: deeper L3 (z=-0.4) + baseline half-step (cases 1.1,3.1); all seven AMB2022-03 cases (2.1 mm tracks), multi-level moving boxes + first-order Enzyme sensitivities (no dumps), toploss 0, vs arc/results/all/case* (baseline AD, 40 cores).
# Run one after the other on the same 14 cores so every wall time is clean.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:14}")
BM=$PWD/../../arc/mesh/track_h0.02_x-0.6_1.5.mesh
for c in 0 1.1 1.2 2.1 2.2 3.1 3.2; do HA=-1; [ $c = 1.1 ] && HA=6; [ $c = 3.1 ] && HA=7
  EXTRAARGS="-halfat $HA" TOPLOSS=0 NT=14 taskset -c $S jobs/run_moving_case.sh $c results/fin_c$c "-l1 $M/L1_h0.1_w.mesh -l2 $M/M2_h0.04.mesh -l3 $M/M3d_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh"
  B=../../arc/results/all/case$c
  echo "=================== case $c"; grep -E "^done|^sensitiv" results/fin_c$c/run.log
  TMIN=5e-4 python3 post_sens.py $B results/fin_c$c $BM $M/L3_h0.02.mesh
  python3 post_levels.py $B $BM results/fin_c$c $M/L3_h0.02.mesh $c | grep -v "^steps\|time stamps\|^peakT:"
done
