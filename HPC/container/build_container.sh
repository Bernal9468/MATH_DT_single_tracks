#!/bin/bash
# Build single_tracks.sif (Singularity image of the CPU program) on the cluster login node. Run from the HPC folder:
#   bash container/build_container.sh
# 1st choice: singularity build --fakeroot (no root needed if the site enabled fakeroot for your user);
# 2nd choice: rootless Podman image -> docker archive -> singularity build (no special rights needed).
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
IMG=single_tracks.sif
if singularity build --fakeroot "$IMG" container/single_tracks.def; then
    echo "built $IMG with singularity --fakeroot"
elif command -v podman > /dev/null; then
    echo "singularity --fakeroot not available -> podman"
    podman build -t single_tracks:latest -f container/Containerfile . &&
    podman save --format docker-archive -o single_tracks.tar single_tracks:latest &&
    singularity build "$IMG" docker-archive://single_tracks.tar && rm -f single_tracks.tar || exit 1
    echo "built $IMG via podman"
else
    echo "neither 'singularity build --fakeroot' nor podman works here; ask the HPC support for fakeroot, or build"
    echo "the image on a machine with root (sudo singularity build $IMG container/single_tracks.def) and copy it"; exit 1
fi
singularity test "$IMG" && echo "now: CONTAINER=\$PWD/$IMG bash check/selftest.sh"
