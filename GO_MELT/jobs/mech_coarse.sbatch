#!/bin/bash
#SBATCH --job-name=gm_mech2
#SBATCH --partition=compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Mechanical stage, option 1: coarse thermal dump subset (steps 0,100,200,300,400,518), case 0, 12 threads per run.
# ref = baseline thermal dumps; ml = multi-level (final config: deeper L3) dumps generated here, concurrently with ref mech.
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; M=$GM/mesh; ROOT=$(cd $GM/../.. && pwd)
MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
echo "cores: ${CORES[*]}"
BASE=$ROOT/thermo/HPC/results/thermo/case0/fields; RD=$GM/results/coarse_ref_dumps; mkdir -p $RD
for s in 0 100 200 300 400 518; do ln -sf $BASE/f_$s.bin $RD/f_$s.bin; done
mech() { # name dumpdir cores
  mkdir -p results/coarse_mech_$1; t0=$(date +%s.%N)
  OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 MKL_THREADING_LAYER=GNU taskset -c $3 singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $2 -o $GM/results/coarse_mech_$1 -order 1 -bulk -abqanneal > results/coarse_mech_$1/log.txt 2>&1
  echo "coarse_mech_$1 exit=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) s" | tee -a results/coarse_mech_$1/log.txt; }
mech ref $RD $(sl 0 12) &
# thermal multi-level with dumps (final config), then its mechanical run
TOPLOSS=0 NT=12 taskset -c $(sl 12 12) jobs/run_moving_case.sh 0 results/coarse_ml "-l1 $M/L1_h0.1_w.mesh -l2 $M/M2_h0.04.mesh -l3 $M/M3d_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh -dumpdir $GM/results/coarse_ml/dumps -dumpmesh $MESH -dumpevery 100 -dumpevery2 100"
grep -E "^done|^sensitiv" results/coarse_ml/run.log; ls results/coarse_ml/dumps
mech ml $GM/results/coarse_ml/dumps $(sl 12 12)
wait
python3 post_mech.py results/coarse_mech_ref results/coarse_mech_ml
