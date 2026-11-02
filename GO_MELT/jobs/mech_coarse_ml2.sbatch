#!/bin/bash
#SBATCH --job-name=gm_mml2
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Mechanical stage, option 1 (multi-level half): thermal multi-level (final config, deeper L3) with coarse dumps, mech only (thermal done), 16 threads.
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; M=$GM/mesh; ROOT=$(cd $GM/../.. && pwd); MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:15}")
t0=$(date +%s.%N)
# thermal part already done (results/coarse_ml, 565 s on 16 cores in job 855235); f_500 removed so both mech runs use the same 6 samples
t1=$(date +%s.%N); mkdir -p results/coarse_mech_ml
OMP_NUM_THREADS=15 MKL_NUM_THREADS=15 MKL_THREADING_LAYER=GNU taskset -c $S singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $GM/results/coarse_ml/dumps -o $GM/results/coarse_mech_ml -order 1 -bulk -abqanneal > results/coarse_mech_ml/log.txt 2>&1
echo "coarse_mech_ml exit=$? wall=$(echo "$(date +%s.%N) - $t1" | bc) s (thermal+dumps $(echo "$t1 - $t0" | bc) s)" | tee -a results/coarse_mech_ml/log.txt
