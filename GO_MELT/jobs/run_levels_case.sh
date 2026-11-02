#!/bin/bash
# usage: jobs/run_levels_case.sh CASE OUTDIR "L1 [L2] L3 mesh args (-l1 .. -l2 .. -l3 ..)"   env NT, DT, X0, X1, TAIL, TOPLOSS
GM=$(cd "$(dirname "$0")/.." && pwd); ARC=$GM/../../arc; CASE=$1; OUT=$2; MESHARGS=$3; mkdir -p $OUT
while IFS='=' read -r k v; do [[ -z "$k" || "$k" == \#* ]] && continue; v=${v%%#*}; v=${v//[[:space:]]/}; [ -z "${!k+x}" ] && export "$k=$v"; done < $ARC/config/calibration.env
read SPOT SPEED GCASE < <(python3 - $GM/inputs/cases.csv $CASE $C0 $N_EXP <<'PY'
import csv, math, sys
R = {r["case"]: r for r in csv.DictReader(open(sys.argv[1]))}
X = lambda r: float(r["power_W"]) / math.sqrt(float(r["speed_mm_s"]) * float(r["spot_D4sigma_um"]) ** 3)
r = R[sys.argv[2]]; print(r["spot_D4sigma_um"], r["speed_mm_s"], "%.6f" % (float(sys.argv[3]) * (X(r) / X(R["0"])) ** float(sys.argv[4])))
PY
)
GB=$(python3 -c "print(round(6**0.5 / 4 * $SPOT * 1e-3, 6))"); TF=$(python3 -c "print(($X1 - ($X0)) / $SPEED + $TAIL)")
python3 $GM/inputs/make_laser.py $OUT/laser.inp --case $CASE --x0 $X0 --x1 $X1 --tend $(python3 -c "print(1.05 * $TF)") > /dev/null
export OMP_NUM_THREADS=${NT:-8} OMP_PROC_BIND=close
echo "case=$CASE gb=$GB gc=$GCASE tf=$TF $MESHARGS" > $OUT/settings.txt
$GM/bin/gm_levels $MESHARGS -laserfile $OUT/laser.inp -o $OUT -fdt ${DT:-5e-6} -tf $TF -absorp $ABSORP -gb $GB -gc $GCASE -gsub 12 -gcut 1.5 -nseg 20 -fleut $FL_EUT -teut $T_EUT -toploss ${TOPLOSS:-1} > $OUT/run.log 2>&1
