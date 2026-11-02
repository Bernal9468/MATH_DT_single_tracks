#!/bin/bash
#SBATCH --job-name=gm_p0test
#SBATCH --partition=compute1,compute2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
# Phase 0 smoke test: new non-OTI single-level driver (gm_single) vs the baseline binary, same coarse mesh (h=0.04), case 0,
# short track. Both run side by side in disjoint core slots of ONE allocation (12 + 12).
cd "$SLURM_SUBMIT_DIR" || exit 1
GM=$PWD; ARC=$GM/../../arc
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh))
S1=$(IFS=,; echo "${CORES[*]:0:12}"); S2=$(IFS=,; echo "${CORES[*]:12:12}")
echo "node $(hostname) slots: $S1 | $S2"
MESH=$GM/mesh/test_h0.04.mesh; X0=-0.3; X1=0.6
TF=$(python3 -c "print(($X1-($X0))/960+4e-4)")
OUT_B=$GM/results/test_base; OUT_G=$GM/results/test_gm; mkdir -p $OUT_B $OUT_G
python3 inputs/make_laser.py $OUT_G/laser.inp --case 0 --x0 $X0 --x1 $X1 --tend $(python3 -c "print(1.05*$TF)")
(cd $ARC && DORDER=0 X0=$X0 X1=$X1 MESH=$MESH CONTAINER=$ARC/ubuntu22.sif NT=12 OMP_PROC_BIND=false \
   taskset -c $S1 bash run/run_case.sh 0 $OUT_B > $OUT_B/wrapper.log 2>&1) &
OMP_NUM_THREADS=12 OMP_PROC_BIND=close taskset -c $S2 $GM/bin/gm_single -m $MESH -laserfile $OUT_G/laser.inp -o $OUT_G \
   -fdt 5e-6 -tf $TF -absorp 0.748695 -gb 0.041029 -gc 0.182604 -gsub 12 -gcut 1.5 -nseg 20 > $OUT_G/run.log 2>&1 &
wait
echo finished $(date)
