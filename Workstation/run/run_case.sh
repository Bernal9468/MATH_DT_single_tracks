#!/bin/bash
# One AMB2022-03 single track (IN718 bare plate) with the calibrated model of config/calibration.env.
# Usage: run/run_case.sh CASE OUTDIR      (CASE = 0, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2; see inputs/cases.csv)
#
# Every setting of config/calibration.env can be overridden from the environment, plus:
#   GC        Goldak depth c [mm] for this run (default: C0 * (X/X_0)^N_EXP of the case)
#   MESH      mesh file (default: mesh/track_h<H_MESH>.mesh, generated on first use for the X0..X1 track)
#   DORDER    1 = temperatures + 20 first-order AD sensitivities (default), 0 = temperatures only
#   NT        threads (default: SLURM_CPUS_PER_TASK, else 12 (the physical cores of the Xeon W-2265))
#   EXTRA     further driver options
#   BIN       program (default: see PLATFORM below)
#   CORES CPU list for taskset (default 0-(NT-1)); SOLVER auto|pardiso|cudss (CPU/GPU solver switch); BIN=$ROOT/bin/thermal_in718_cpu: CPU only
# Outputs in OUTDIR: run.log, probe.csv (T at the surface centreline x = y = 0), sens.csv, peakT.bin, settings.txt, laser.inp
# Analysis: python3 post/analyze_case.py OUTDIR
PLATFORM=workstation
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# The program needs glibc >= 2.35 (built on Ubuntu 22.04). Where the system is older (cluster: glibc 2.28), it runs inside a
# plain Ubuntu 22.04 Singularity image, ubuntu22.sif in this folder (container/get_image.sh); bin/ and lib/ are bind-mounted,
# so the image never needs rebuilding. Found automatically; CONTAINER=<image.sif> selects another one.
[ "$PLATFORM" = hpc ] && [ -z "$CONTAINER" ] && [ -f "$ROOT/ubuntu22.sif" ] && CONTAINER=$ROOT/ubuntu22.sif
CASE=${1:?usage: run/run_case.sh CASE OUTDIR}; OUT=${2:?usage: run/run_case.sh CASE OUTDIR}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

# calibrated settings: the environment wins over the config file
while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    v=${v%%#*}; v=${v//[[:space:]]/}
    [ -z "${!k+x}" ] && export "$k=$v"
done < "$ROOT/config/calibration.env"
DORDER=${DORDER:-1}; GSUB=${GSUB:-12}; GCUT=${GCUT:-1.5}

read SPOT SPEED GCASE < <(python3 - "$ROOT/inputs/cases.csv" "$CASE" "$C0" "$N_EXP" <<'EOF'
import csv, math, sys
R = {r["case"]: r for r in csv.DictReader(open(sys.argv[1]))}
if sys.argv[2] not in R: sys.exit("unknown case " + sys.argv[2] + "; cases: " + " ".join(R))
X = lambda r: float(r["power_W"]) / math.sqrt(float(r["speed_mm_s"]) * float(r["spot_D4sigma_um"]) ** 3)
r = R[sys.argv[2]]
print(r["spot_D4sigma_um"], r["speed_mm_s"], "%.6f" % (float(sys.argv[3]) * (X(r) / X(R["0"])) ** float(sys.argv[4])))
EOF
) || exit 1
GC=${GC:-$GCASE}
GB=$(python3 -c "print(round(6**0.5 / 4 * $SPOT * 1e-3, 6))")          # a = b = sqrt(6) sigma, sigma = D4sigma / 4
TF=$(python3 -c "print(($X1 - ($X0)) / $SPEED + $TAIL)")                  # end of the track + cooling tail

if [ -z "$MESH" ]; then                                                    # fine box = track +- 0.1 mm, |y| <= 0.2, depth 0.3 mm
    MESH=$ROOT/mesh/track_h${H_MESH}_x${X0}_${X1}.mesh
    [ -f "$MESH" ] || python3 "$ROOT/mesh/make_plate_mesh.py" "$MESH" --h $H_MESH \
        --fine-x $(python3 -c "print($X0 - 0.1, $X1 + 0.1)") --fine-y 0.2 --fine-z 0.3 --r 1.3 --hmax 1.0 || exit 1
fi
MESH=$(cd "$(dirname "$MESH")" && pwd)/$(basename "$MESH"); TIE=${MESH%.mesh}_tie.txt
python3 "$ROOT/inputs/make_laser.py" "$OUT/laser.inp" --case "$CASE" --x0 $X0 --x1 $X1 --tend $(python3 -c "print(1.05 * $TF)") > /dev/null || exit 1

export MKL_THREADING_LAYER=GNU OMP_STACKSIZE=512M PARD11_TWOLEVEL=${PARD11_TWOLEVEL:-0} NEWTON_LS=auto
export AMG_MIN_DOFS=${AMG_MIN_DOFS:-200000}      # Newton solves CG + BoomerAMG from 200k unknowns (5-7x faster than PARDISO at 700k)
NT=${NT:-${SLURM_CPUS_PER_TASK:-12}}
export OMP_NUM_THREADS=$NT MKL_NUM_THREADS=$NT
if [ "$PLATFORM" = hpc ]; then
    BIN=${BIN:-$ROOT/bin/thermal_in718_cpu}
    export OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
    LAUNCH=()
    if [ -n "$CONTAINER" ]; then                                           # program from bin/, run inside the image
        LAUNCH=(singularity exec --bind "$ROOT","$OUT","$(dirname "$MESH")" "$CONTAINER")
    fi
else
    BIN=${BIN:-$ROOT/bin/thermal_in718_gpu}
    export SOLVER=${SOLVER:-auto}                                           # CPU/GPU solver switch (PARDISO or cuDSS)
    LAUNCH=(taskset -c ${CORES:-0-$((NT - 1))})
fi

cat > "$OUT/settings.txt" <<EOT
case=$CASE spot_D4sigma_um=$SPOT goldak_b=af=ar=$GB c=$GC absorptivity=$ABSORP dt=$DT x0=$X0 x1=$X1 t_final=$TF
mesh=$MESH dorder=$DORDER bin=$BIN gsub=$GSUB gcut=$GCUT fleut=$FL_EUT teut=$T_EUT threads=$NT platform=$PLATFORM extra=${EXTRA:-}
EOT
cd "$ROOT/inputs"
t0=$SECONDS
PEAK_T_FILE="$OUT/peakT.bin" "${LAUNCH[@]}" "$BIN" \
  -m "$MESH" -tie "$TIE" -laserfile "$OUT/laser.inp" -absorp $ABSORP -nseg 20 -nmax 200 -solver pardiso \
  -probex 0.0 -probey 0.0 -pf 20 -spread 2 -gaf $GB -gar $GB -gb $GB -gc $GC -gsub $GSUB -gcut $GCUT -gff 1.0 -gfr 1.0 \
  -fdt $DT -ts 0.0 -tf $TF -zshift 0.0 -tambc 23.5 -tsub 23.5 -dorder $DORDER -fleut $FL_EUT -teut $T_EUT \
  -pcsv "$OUT/probe.csv" -scsv "$OUT/sens.csv" ${EXTRA:-} > "$OUT/run.log" 2>&1
st=$?
echo "case $CASE: exit=$st wall=$((SECONDS - t0)) s  ($OUT)"
exit $st
