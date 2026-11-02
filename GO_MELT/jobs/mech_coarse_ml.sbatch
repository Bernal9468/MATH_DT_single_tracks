#!/bin/bash
#SBATCH --job-name=gm_mml
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Mechanical stage, option 1 (multi-level half): thermal multi-level (final config, deeper L3) with coarse dumps, then mech, 16 threads.
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; M=$GM/mesh; ROOT=$(cd $GM/../.. && pwd); MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
t0=$(date +%s.%N)
TOPLOSS=0 NT=16 taskset -c $S jobs/run_moving_case.sh 0 results/coarse_ml "-l1 $M/L1_h0.1_w.mesh -l2 $M/M2_h0.04.mesh -l3 $M/M3d_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh -dumpdir $GM/results/coarse_ml/dumps -dumpmesh $MESH -dumpevery 100 -dumpevery2 100"
grep -E "^done|^sensitiv" results/coarse_ml/run.log; ls results/coarse_ml/dumps
t1=$(date +%s.%N); mkdir -p results/coarse_mech_ml
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 MKL_THREADING_LAYER=GNU taskset -c $S singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $GM/results/coarse_ml/dumps -o $GM/results/coarse_mech_ml -order 1 -bulk -abqanneal > results/coarse_mech_ml/log.txt 2>&1
echo "coarse_mech_ml exit=$? wall=$(echo "$(date +%s.%N) - $t1" | bc) s (thermal+dumps $(echo "$t1 - $t0" | bc) s)" | tee -a results/coarse_mech_ml/log.txt
