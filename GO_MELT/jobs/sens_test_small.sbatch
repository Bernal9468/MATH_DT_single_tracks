#!/bin/bash
#SBATCH --job-name=gm_senstest
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
# First-order Enzyme sensitivities of gm_single vs the baseline AD (fixed upstream binary, dorder 1) on the small test mesh (h = 0.04, short track).
cd "$SLURM_SUBMIT_DIR" || exit 1
GM=$PWD; ARC=$GM/../../arc
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
MESH=$GM/mesh/test_h0.04.mesh; X0=-0.3; X1=0.6; mkdir -p results/sens_base results/sens_gm
mkdir -p results/sens_base/bin; cp results/p0fix_base/bin/thermal_in718_cpu results/sens_base/bin/; ln -sfn $ARC/lib results/sens_base/lib
(cd $ARC && DORDER=1 X0=$X0 X1=$X1 MESH=$MESH BIN=$GM/results/sens_base/bin/thermal_in718_cpu CONTAINER=$ARC/ubuntu22.sif NT=7 OMP_PROC_BIND=false \
   taskset -c $(sl 0 7) bash run/run_case.sh 0 $GM/results/sens_base > $GM/results/sens_base/wrapper.log 2>&1) &
X0=$X0 X1=$X1 NT=7 EXTRAARGS="-dorder 1" taskset -c $(sl 7 7) jobs/run_gm_case.sh 0 results/sens_gm $MESH &
wait
