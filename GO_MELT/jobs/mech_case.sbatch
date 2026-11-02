#!/bin/bash
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x-%j.out
# usage: sbatch --job-name=gm_mc_<case> --export=ALL,CASE=<case> jobs/mech_case.sbatch
# One case: multi-level thermal (final config) with coarse dumps -> mechanical (new) ; baseline mechanical from baseline dumps -> compare.
cd "$SLURM_SUBMIT_DIR" || exit 1; GM=$PWD; ARC=$GM/../../arc; M=$GM/mesh; ROOT=$(cd $GM/../.. && pwd); MESH=$ARC/mesh/track_h0.02_x-0.6_1.5.mesh; c=$CASE
HA=-1; [ $c = 1.1 ] && HA=6; [ $c = 3.1 ] && HA=7
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); S=$(IFS=,; echo "${CORES[*]:0:16}")
BASE=$ROOT/thermo/HPC/results/thermo/case$c/fields
LAST=$(ls $BASE | sed 's/f_//;s/.bin//' | sort -n | tail -1); STEPS="0 100 200 300 400"; [ $LAST -gt 500 ] && STEPS="$STEPS 500 600"; STEPS="$STEPS $LAST"
STEPS=$(for s in $STEPS; do [ $s -le $LAST ] && echo $s; done | sort -nu | tr '\n' ' '); echo "case $c steps: $STEPS"
RUN=results/mc_c$c; mkdir -p $RUN
EXTRAARGS="-halfat $HA" TOPLOSS=0 NT=16 taskset -c $S jobs/run_moving_case.sh $c $RUN "-l1 $M/L1_h0.1_ztop.mesh -l2 $M/M2z1_h0.04.mesh -l3 $M/M3d_h0.02.mesh -h2 0.04 -h3 0.02 -fb 1 -dorder 1 -peakmesh $M/L3_h0.02.mesh -dumpdir $GM/$RUN/dumps -dumpmesh $MESH -dumpevery 100 -dumpevery2 100"
grep -E "^done|^sensitiv" $RUN/run.log
echo "=========== thermal check"; TMIN=5e-4 python3 post_sens.py $ARC/results/all/case$c $RUN $MESH $M/L3_h0.02.mesh
ND=$GM/$RUN/dsub; RD=$GM/$RUN/refdump; mkdir -p $ND $RD
for s in $STEPS; do ln -sf $GM/$RUN/dumps/f_$s.bin $ND/f_$s.bin; ln -sf $BASE/f_$s.bin $RD/f_$s.bin; done
run_mech() { # dumpdir outdir
  mkdir -p $2; t1=$(date +%s.%N)
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 MKL_THREADING_LAYER=GNU taskset -c $S singularity exec --bind $ROOT $ARC/ubuntu22.sif $ARC/bin/mech_main_aa_src -m $MESH -tie ${MESH%.mesh}_tie.txt -dump $1 -o $2 -order 1 -bulk -abqanneal > $2/log.txt 2>&1
  echo "$2 exit=$? wall=$(echo "$(date +%s.%N) - $t1" | bc) s" | tee -a $2/log.txt; }
run_mech $ND $GM/$RUN/mech_new
run_mech $RD $GM/$RUN/mech_ref
echo "=========== mechanical check vs baseline (case $c)"; python3 post_mech.py $RUN/mech_ref $RUN/mech_new
