#!/bin/bash
#SBATCH --job-name=st_thermo
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
# Nominal thermal response + the full set of 20 first derivatives of all 7 cases, compared with the NIST thermography.
# Per case: run with sensitivities (SENS_SOLVER=krylov: Jacobi-BiCGSTAB instead of a PARDISO factorisation per step) and
# field dumps every 20 steps, then post/analyze_case.py, tools/compare_thermography.py (model vs NIST figure) and
# tools/sensitivity_report.py (derivatives.csv + figure). Needs data/nist_mds2-2716 (bash get_nist_thermography.sh) and
# python3 with h5py, scipy, matplotlib. From the HPC folder:  sbatch slurm/thermo_all.sbatch [OUTROOT]  (default results/thermo)
CASES=(0 1.1 1.2 2.1 2.2 3.1 3.2)
cd "$SLURM_SUBMIT_DIR" || exit 1
C=${CASES[$SLURM_ARRAY_TASK_ID]}; OUT=${1:-results/thermo}/case$C; mkdir -p "$OUT/fields"
echo "case $C on $(hostname), $SLURM_CPUS_PER_TASK cores, started $(date)"
SENS_SOLVER=${SENS_SOLVER:-krylov} DORDER=1 FIELD_DUMP_DIR=$PWD/$OUT/fields FIELD_DUMP_EVERY=20 FIELD_DUMP_EVERY2=20 \
    bash run/run_case.sh "$C" "$OUT" || exit 1
python3 post/analyze_case.py "$OUT" --json "$OUT/result.json" > "$OUT/analysis.txt"
python3 tools/compare_thermography.py "$OUT"
python3 tools/sensitivity_report.py "$OUT"
echo "finished $(date)"
