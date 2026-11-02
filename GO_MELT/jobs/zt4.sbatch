#!/bin/bash
#SBATCH --job-name=gm_zt4
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Case 0: thermal multi-level + sens + coarse dumps, thermal checks only (mechanical is a separate job once T0/Tw look right).
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; M=$GM/mesh; ROOT=$(cd $GM/../.. && pwd); MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
TOPLOSS=0 NT=16 taskset -c $S jobs/run_moving_case.sh 0 results/zt4 "-l1 $M/L1_h0.1_ztop.mesh -l2 $M/M2z1_h0.04.mesh -l3 $M/M3d_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh -dumpdir $GM/results/zt4/dumps -dumpmesh $MESH -dumpevery 100 -dumpevery2 100"
rm -f results/zt4/dumps/f_500.bin; grep -E "^done|^sensitiv|^level" results/zt4/run.log
B=../../arc/results/all/case0
echo "=========== thermal check (case 0)"; TMIN=5e-4 python3 post_sens.py $B results/zt4 $MESH $M/L3_h0.02.mesh; python3 post_levels.py $B $MESH results/zt4 $M/L3_h0.02.mesh 0 | grep -v "^steps\|time stamps\|^peakT:"

t1=$(date +%s.%N); mkdir -p results/zt4_mech
OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 MKL_THREADING_LAYER=GNU taskset -c $S singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $GM/results/zt4/dumps -o $GM/results/zt4_mech -order 1 -bulk -abqanneal > results/zt4_mech/log.txt 2>&1
echo "zt4_mech exit=$? wall=$(echo "$(date +%s.%N) - $t1" | bc) s" | tee -a results/zt4_mech/log.txt
echo "=========== mechanical check vs baseline"; python3 post_mech.py results/coarse_mech_ref results/zt4_mech
