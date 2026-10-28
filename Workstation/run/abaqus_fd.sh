#!/bin/bash
# Abaqus finite differences of dT/ds for s = k, cp, rho, L (central, relative step H = 1 %) on the Abaqus replica of an
# MFEM run (tools/abaqus_deck.py; the MFEM run needs SRC_DUMP_DIR, binary bin_src). 9 Abaqus jobs, JOBS at a time.
# usage: run/abaqus_fd.sh MFEM_RUNDIR OUTDIR [H]
cd "$(dirname "$0")/.."
RUN=$1; OUT=$2; H=${3:-0.01}; JOBS=${JOBS:-3}; CPUS=${CPUS:-6}
mkdir -p $OUT
printf "mp_mpi_implementation=IMPI\nmp_mpirun_path={IMPI: '/opt/intel/oneapi/mpi/2021.6.0/bin/mpirun'}\nmp_mode=MPI\n" > $OUT/abaqus_v6.env
P=$(python3 -c "print(1 + $H)"); M=$(python3 -c "print(1 - $H)")
jobs_=("nom:" "k_p:--sk $P" "k_m:--sk $M" "cp_p:--scp $P" "cp_m:--scp $M" "rho_p:--srho $P" "rho_m:--srho $M" "L_p:--sl $P" "L_m:--sl $M")
for j in "${jobs_[@]}"; do python3 tools/abaqus_deck.py $RUN $OUT/${j%%:*}.inp ${j#*:} | cut -c1-120; done
printf "%s\n" "${jobs_[@]%%:*}" | xargs -P $JOBS -I{} sh -c "cd $OUT && rm -f {}.lck && abaqus job={} interactive double=both cpus=$CPUS > {}.out 2>&1; echo {} \$(tail -1 {}.out) \$(date +%H:%M)"
