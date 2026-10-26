#!/bin/bash
# Download the NIST AMB2022-03 in-situ thermography (dataset mds2-2716, https://doi.org/10.18434/mds2-2716) used by
# tools/compare_thermography.py: 550 MB HDF5 + its SHA-256 + README, into data/nist_mds2-2716/. Run from this folder.
# Python packages for the comparison: h5py, scipy, matplotlib (e.g. python3 -m pip install --user h5py scipy matplotlib).
cd "$(dirname "$0")" || exit 1
D=data/nist_mds2-2716; mkdir -p "$D"; cd "$D" || exit 1
B=https://data.nist.gov/od/ds/ark:/88434/mds2-2716/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5
curl -fL -o AMB2022-03-718-AMMT-StaringCamera_Signal.h5.sha256 "$B.sha256" &&
curl -fL -o AMB2022-03-718-AMMT-StaringCamera_Signal.h5 "$B" &&
curl -fsL -o 2716_README.txt https://data.nist.gov/od/ds/mds2-2716/2716_README.txt || exit 1
echo "$(cat AMB2022-03-718-AMMT-StaringCamera_Signal.h5.sha256 | awk '{print $1}')  AMB2022-03-718-AMMT-StaringCamera_Signal.h5" | sha256sum -c
