#!/bin/bash
#SBATCH --job-name=gm_mech
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Mechanical stage (residual stress + 20 first-order derivatives) on (a) the baseline thermal dumps and (b) the multi-level dumps, same 20 um mesh, same solver
# (arc/bin/mech_main_aa_src in the ubuntu22 container).  Wall time of each is recorded.  Usage: sbatch jobs/mech_compare.sbatch REF_DUMPS NEW_DUMPS
cd "$SLURM_SUBMIT_DIR" || exit 1
REF=${1:-$PWD/../../thermo/HPC/results/thermo/case0/fields}; NEW=${2:-$PWD/results/ml_short_c0/dumps}
GM=$PWD; ARC=$GM/../../arc; MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh; ROOT=$(cd $GM/../.. && pwd)
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
run() { # name dumpdir cores
  mkdir -p results/mech_$1; t0=$(date +%s.%N)
  OMP_NUM_THREADS=7 MKL_NUM_THREADS=7 MKL_THREADING_LAYER=GNU taskset -c $3 singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $2 -o $GM/results/mech_$1 -order 1 -bulk -abqanneal > results/mech_$1/log.txt 2>&1
  echo "mech_$1 exit=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) s" | tee -a results/mech_$1/log.txt; }
run ref $REF $(sl 0 7) &
run ml  $NEW $(sl 7 7) &
wait
python3 post_mech.py results/mech_ref results/mech_ml
