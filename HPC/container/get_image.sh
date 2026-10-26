#!/bin/bash
# Plain Ubuntu 22.04 image for clusters whose glibc is older than 2.35 (the program's build system). No root, fakeroot or
# podman needed: ~30 MB, a few seconds. Run once from the HPC folder:  bash container/get_image.sh
# Afterwards run/run_case.sh and check/selftest.sh use ./ubuntu22.sif automatically (the program and its libraries are
# bind-mounted from bin/ and lib/, so updates of the program never require a new image).
# Tested on the cluster: singularity-ce 4.5.1 (EL8, glibc 2.28); 'ignoring ENOTSUP on setxattr' warnings are harmless.
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT" || exit 1
export SINGULARITY_CACHEDIR=${SINGULARITY_CACHEDIR:-$ROOT/.sing_cache}
export SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-${TMPDIR:-/tmp}}
singularity pull --force ubuntu22.sif docker://ubuntu:22.04 || exit 1
singularity exec ubuntu22.sif ldd --version | head -1
echo "ubuntu22.sif ready; now: bash check/selftest.sh"
