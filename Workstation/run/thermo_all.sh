#!/bin/bash
# Nominal response + the full set of 20 first-order AD sensitivities for the 7 AMB2022-03 cases, with field dumps, then the
# comparison with the NIST thermography and the derivative report. Hybrid CPU+GPU: start one worker of each kind; they
# share the case list through lock directories.
#   bash run/thermo_all.sh cpu &     # CPU program, cores 12-17
#   bash run/thermo_all.sh gpu &     # GPU program, cores 0-11
# Needs data/nist_mds2-2716 (bash get_nist_thermography.sh). Output: results/thermo/case<c>/
cd "$(dirname "$0")/.."
W=${1:-cpu}
if [ "$W" = gpu ]; then export BIN=$PWD/bin/thermal_in718_gpu NT=${NT:-12} CORES=${CORES:-0-11}
else export BIN=$PWD/bin/thermal_in718_cpu NT=${NT:-6} CORES=${CORES:-12-17}; fi
export SENS_SOLVER=krylov DORDER=1 FIELD_DUMP_EVERY=20 FIELD_DUMP_EVERY2=20
mkdir -p results/thermo
for c in 0 1.1 1.2 2.1 2.2 3.1 3.2; do
  out=results/thermo/case$c
  mkdir results/thermo/.lock_$c 2>/dev/null || continue
  echo "[$W] case $c start $(date +%H:%M)"
  mkdir -p $out/fields
  FIELD_DUMP_DIR=$PWD/$out/fields bash run/run_case.sh $c $out &&
    python3 post/analyze_case.py $out --json $out/result.json > $out/analysis.txt &&
    python3 tools/compare_thermography.py $out &&
    python3 tools/sensitivity_report.py $out
  echo "[$W] case $c done $(date +%H:%M)"
done
