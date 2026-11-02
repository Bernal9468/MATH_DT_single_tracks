#!/bin/bash
#SBATCH --job-name=gm_mzt3A
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Mechanical stage on the zt3A thermal dumps (L1 ztop, L2 uniform 0.02 top, L3 deep).
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; ROOT=$(cd $GM/../.. && pwd); MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
t1=$(date +%s.%N); mkdir -p results/zt3A_mech
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 MKL_THREADING_LAYER=GNU taskset -c $S singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $GM/results/zt3A/dumps -o $GM/results/zt3A_mech -order 1 -bulk -abqanneal > results/zt3A_mech/log.txt 2>&1
echo "zt3A_mech exit=$? wall=$(echo "$(date +%s.%N) - $t1" | bc) s" | tee -a results/zt3A_mech/log.txt
echo "=========== mechanical check vs baseline"; python3 post_mech.py results/coarse_mech_ref results/zt3A_mech
