#!/bin/bash
#SBATCH --job-name=st_case
#SBATCH --partition=compute1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
# One single track on one whole node (40 cores; the program is multithreaded, not MPI). Submit from the HPC folder:
#   sbatch slurm/case.sbatch CASE OUTDIR                      e.g. sbatch slurm/case.sbatch 1.1 results/case1.1
# The settings of config/calibration.env can be overridden with variables in the submitting shell (sbatch exports the
# environment by default), e.g.  DORDER=0 sbatch slurm/case.sbatch 0 results/case0_T
# Singularity instead of the native program:  CONTAINER=$PWD/single_tracks.sif sbatch slurm/case.sbatch 0 results/case0
# Long runs (full 10 mm track with sensitivities): --partition=compute2 --time=10-00:00:00 on the sbatch command line.
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "node $(hostname), $SLURM_CPUS_PER_TASK cores, started $(date)"
bash run/run_case.sh "$1" "$2"
st=$?
[ $st -eq 0 ] && python3 post/analyze_case.py "$2" --json "$2/result.json"
echo "finished $(date), exit $st"
exit $st
