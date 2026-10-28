#!/bin/bash
# Dense temperature-only runs (DORDER=0, field dump every 2 steps = 10 us, ~1.3 GB per case) for the simulation videos
# (tools/make_video_3d.py, tools/make_video_plate.py), all 7 cases, then the 3D video of each case.
# Worker: bash run/video_fields.sh gpu|cpu  - cases are claimed with lock folders, so both workers can run at the same time
# (GPU worker on cores 0-11, CPU worker on cores 12-17; adjust NT/CORES to the machine).
cd "$(dirname "$0")/.."
W=${1:-gpu}
if [ "$W" = gpu ]; then export BIN=$PWD/bin/thermal_in718_gpu NT=12 CORES=0-11
else export BIN=$PWD/bin/thermal_in718_cpu NT=6 CORES=12-17; fi
export DORDER=0 FIELD_DUMP_EVERY=2 FIELD_DUMP_EVERY2=2
mkdir -p results/video
for c in 0 1.1 1.2 2.1 2.2 3.1 3.2; do
  out=results/video/case$c
  mkdir results/video/.lock_$c 2>/dev/null || continue          # another worker has (or had) this case
  mkdir -p $out/fields
  echo "[$W] case $c start $(date +%H:%M)"
  FIELD_DUMP_DIR=$PWD/$out/fields bash run/run_case.sh $c $out &&
    python3 tools/make_video_3d.py $out > $out/video.log 2>&1
  echo "[$W] case $c done $(date +%H:%M)"
done
