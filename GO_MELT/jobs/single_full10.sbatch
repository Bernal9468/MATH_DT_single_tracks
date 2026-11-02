#!/bin/bash
#SBATCH --job-name=gm_s_full
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Control experiment: gm_single (phase-0 verified single-level solver) on the baseline's full 10 mm mesh, case 0, toploss 0 (pre-fix physics).
# If it reproduces arc/results/case0_full10, the 0.5% / +15 K peak-field bias of the multi-level run is a multi-level effect, not a baseline quirk.
cd "$SLURM_SUBMIT_DIR" || exit 1
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
sed -i 's|-fleut $FL_EUT -teut $T_EUT >|-fleut $FL_EUT -teut $T_EUT ${EXTRAARGS} >|' jobs/run_gm_case.sh
X0=-5 X1=5 NT=16 EXTRAARGS="-toploss 0" taskset -c $S jobs/run_gm_case.sh 0 results/single_full10 $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh
B=../../arc/results/case0_full10
python3 post_levels.py $B $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh results/single_full10 $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh
