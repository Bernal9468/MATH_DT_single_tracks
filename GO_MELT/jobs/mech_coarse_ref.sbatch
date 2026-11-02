#!/bin/bash
#SBATCH --job-name=gm_mref
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Mechanical stage, option 1 (baseline half): baseline thermal dumps subset (steps 0,100,200,300,400,518), case 0, 16 threads.
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; ROOT=$(cd $GM/../.. && pwd); MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
BASE=$ROOT/thermo/HPC/results/thermo/case0/fields; RD=$GM/results/coarse_ref_dumps; mkdir -p $RD results/coarse_mech_ref
for s in 0 100 200 300 400 518; do ln -sf $BASE/f_$s.bin $RD/f_$s.bin; done
t0=$(date +%s.%N)
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 MKL_THREADING_LAYER=GNU taskset -c $S singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $RD -o $GM/results/coarse_mech_ref -order 1 -bulk -abqanneal > results/coarse_mech_ref/log.txt 2>&1
echo "coarse_mech_ref exit=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) s" | tee -a results/coarse_mech_ref/log.txt
