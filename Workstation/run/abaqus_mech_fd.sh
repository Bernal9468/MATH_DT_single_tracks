#!/bin/bash
# Abaqus finite differences of the stresses w.r.t. s = k, cp, rho, L: the mechanical replica (tools/abaqus_mech_deck.py) on
# each of the 9 thermal FD runs of run/abaqus_fd.sh (made with DECK_ARGS="--nt-every N": temperatures in the odb).
# usage: run/abaqus_mech_fd.sh THERMAL_FD_DIR OUTDIR MESH ELEMS_FILE [EVERY]   (JOBS, CPUS, JOBLIST from the environment)
# Paths may contain spaces ("Daniel Morales"): every path is quoted.
set -euo pipefail
cd "$(dirname "$0")/.."
TH=$(cd "$1" && pwd); OUT=$2; MESH=$3; ELEMS=$4; EVERY=${5:-10}; JOBS=${JOBS:-3}; CPUS=${CPUS:-4}
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); cp "$TH/abaqus_v6.env" "$OUT/"
NSTEP=$(grep -c "^\*STEP" "$TH/nom.inp"); JL=${JOBLIST:-"nom k_p k_m cp_p cp_m rho_p rho_m L_p L_m"}
for j in $JL; do
  ln -sf "$TH/$j.odb" "$OUT/th_$j.odb"
  python3 tools/abaqus_mech_deck.py "$MESH" th_$j "$OUT/m_$j.inp" --steps $NSTEP --every $EVERY --elems "$ELEMS" | cut -c1-120
done
EXTRACT=$PWD/tools/abaqus_mech_extract.py; export OUT CPUS EXTRACT
printf "%s\n" $JL | xargs -P "$JOBS" -I{} sh -c '
  set -eu
  cd "$OUT"
  test ! -e m_{}.lck
  stamp=$(mktemp .m_{}.started.XXXXXX)
  trap "rm -f \"$stamp\"" EXIT
  if abaqus job=m_{} interactive cpus="$CPUS" < /dev/null > m_{}.out 2>&1; then
    :
  else
    rc=$?
    # This Abaqus installation can abort during cleanup AFTER a completed solve.
    # Accept extraction only with a freshly written successful status file.
    if [ -f m_{}.sta ] && [ m_{}.sta -nt "$stamp" ] && grep -q "THE ANALYSIS HAS COMPLETED SUCCESSFULLY" m_{}.sta; then
      echo "WARNING: solver exit=$rc after successful analysis; inspect cleanup error" >> m_{}.out
    else
      exit "$rc"
    fi
  fi
  abaqus python "$EXTRACT" m_{} < /dev/null >> m_{}.out 2>&1
  echo m_{} $(tail -1 m_{}.out) $(date +%H:%M)
'
