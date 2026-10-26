#!/bin/bash
# Self-test (2-5 min): case 0 on a short 0.6 mm track and a coarse 67k-node mesh, with the 20 first-order sensitivities,
# compared with the reference computed on the workstation (CPU build; check/reference/). PASS = every probe temperature and
# every sensitivity history agrees with the reference to 1e-8 relative (CPU and GPU builds agree to ~1e-15; different thread
# counts change the last digits only). Run from the folder that contains check/:  bash check/selftest.sh
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
[ -z "$CONTAINER" ] && [ -f "$ROOT/ubuntu22.sif" ] && export CONTAINER=$ROOT/ubuntu22.sif      # see run/run_case.sh
OUT=results/selftest; rm -rf "$OUT"; mkdir -p results
echo "== program: $( [ -n "$CONTAINER" ] && echo "container $CONTAINER" || ls bin/ | tr '\n' ' ')"
if [ -z "$CONTAINER" ]; then
    for b in bin/thermal_in718_*; do
        miss=$(ldd "$b" 2>&1 | grep -E "not found|GLIBC.*not found|version .* not found")
        [ -n "$miss" ] && { echo "FAIL: $b cannot be loaded here:"; echo "$miss"; echo "-> bash container/get_image.sh (README, 'Container'), then run this test again"; exit 1; }
    done
fi
python3 mesh/make_plate_mesh.py mesh/selftest.mesh --h 0.04 --fine-x -0.4 0.5 --fine-y 0.2 --fine-z 0.3 --r 1.3 --hmax 1.0 > /dev/null || exit 1
MESH=mesh/selftest.mesh X0=-0.3 X1=0.3 DT=1e-5 TAIL=2e-4 DORDER=1 ABSORP=0.748695 GC=0.182604 FL_EUT=0.23 T_EUT=1423.15 \
    bash run/run_case.sh 0 "$OUT" || { echo "FAIL: the run stopped, see $OUT/run.log"; tail -5 "$OUT/run.log"; exit 1; }
python3 - "$OUT" <<'EOF'
import csv, sys
worst, bad = 0.0, []
for f in ("probe.csv", "sens.csv"):
    A = list(csv.reader(open("check/reference/" + f))); B = list(csv.reader(open(sys.argv[1] + "/" + f)))
    if len(A) != len(B): bad.append(f"{f}: {len(B) - 1} rows, reference {len(A) - 1}"); continue
    for j in range(1, len(A[0])):
        sc = max(abs(float(r[j])) for r in A[1:])
        if sc < 1e-8: continue                         # columns that are zero for a bare plate (powder parameters)
        d = max(abs(float(a[j]) - float(b[j])) for a, b in zip(A[1:], B[1:])) / sc
        worst = max(worst, d)
        if d > 1e-8: bad.append(f"{f}: {A[0][j]} differs by {d:.2e} (relative)")
print(f"largest relative difference from the reference: {worst:.2e}")
print("SELFTEST " + ("PASS" if not bad else "FAIL\n  " + "\n  ".join(bad)))
sys.exit(1 if bad else 0)
EOF
