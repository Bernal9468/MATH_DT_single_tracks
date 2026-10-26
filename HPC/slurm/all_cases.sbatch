#!/bin/bash
#SBATCH --job-name=st_all
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x-%A_%a.out
# All 7 AMB2022-03 cases at once, one node each (job array; the account allows 10 jobs at a time). From the HPC folder:
#   sbatch slurm/all_cases.sbatch [OUTROOT]        (default results/all) -> OUTROOT/case<c>/, then
#   python3 post/summarize.py OUTROOT/summary.csv 0=OUTROOT/case0 1.1=OUTROOT/case1.1 ...   (see README)
CASES=(0 1.1 1.2 2.1 2.2 3.1 3.2)
cd "$SLURM_SUBMIT_DIR" || exit 1
C=${CASES[$SLURM_ARRAY_TASK_ID]}; OUT=${1:-results/all}/case$C
echo "case $C on $(hostname), $SLURM_CPUS_PER_TASK cores, started $(date)"
bash run/run_case.sh "$C" "$OUT"
st=$?
[ $st -eq 0 ] && python3 post/analyze_case.py "$OUT" --json "$OUT/result.json"
echo "finished $(date), exit $st"
exit $st
