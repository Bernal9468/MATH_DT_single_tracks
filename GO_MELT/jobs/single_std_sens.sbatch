#!/bin/bash
#SBATCH --job-name=gm_sstd
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Single-level gm_single, first-order sensitivities, standard 20 um mesh (2.1 mm track), case 0, toploss 0 vs arc/results/all/case0 (baseline AD, 102 min on 40 cores).
cd "$SLURM_SUBMIT_DIR" || exit 1
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:14}")
BM=$PWD/../../arc/mesh/track_h0.02_x-0.6_1.5.mesh
NT=14 EXTRAARGS="-dorder 1 -toploss 0" taskset -c $S jobs/run_gm_case.sh 0 results/single_std_c0 $BM
python3 post_sens.py ../../arc/results/all/case0 results/single_std_c0 $BM $BM
