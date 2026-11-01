#!/bin/bash
# Mechanical stage (residual stress, IN718, first derivatives w.r.t. the 20 thermal parameters) on the thermal field dumps of
# a single-track run (results/thermo/case<c>/fields: T + 20 first derivatives every 20 steps).
# Solver: src/mechanical/mech_main_aa.cpp built with MATERIAL=IN718 (bin_mech/):
#   cpu -> mech_main_aa_src (all on the CPU), gpu -> mech_main_aa_gpu (element/point loops on the GPU, PARDISO on the CPU).
# -bulk: every element is solid from the start (bare plate; attribute 2 is the top element layer, not powder).
# usage: run/run_mech.sh cpu|gpu CASE OUTDIR [extra mech options, e.g. -nsamp 4]
#   MESH=<mesh> DUMP=<field dump dir> override the 20 um mesh and results/thermo/case<C>/fields
cd "$(dirname "$0")/.."
W=$1; C=$2; OUT=$3; shift 3
BIN=${BIN:-$PWD/bin_mech/mech_main_aa_$([ "$W" = gpu ] && echo gpu || echo src)}
NT=${NT:-12}; CORES=${CORES:-0-11}
MESH=${MESH:-$PWD/mesh/track_h0.02_x-0.6_1.5.mesh}; DUMP=${DUMP:-$PWD/results/thermo/case$C/fields}
mkdir -p "$OUT"
t0=$(date +%s.%N)
OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT taskset -c $CORES "$BIN" -m "$MESH" -tie "${MESH%.mesh}_tie.txt" \
    -dump "$DUMP" -o "$OUT" -order 1 -bulk -abqanneal "$@" > "$OUT/log.txt" 2>&1
echo "exit=$? wall=$(echo "$(date +%s.%N) - $t0" | bc) s" | tee -a "$OUT/log.txt"
