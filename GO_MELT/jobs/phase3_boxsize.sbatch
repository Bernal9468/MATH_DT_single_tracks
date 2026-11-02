#!/bin/bash
#SBATCH --job-name=gm_p3pk
#SBATCH --partition=compute2,compute3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Full 10 mm track, two-way injection, LARGER L3/L2 cross-sections (the peak-field error sits on the L3 boundary). Peak record stays on the baseline's y +-0.2, z -0.3 grid.
cd "$SLURM_SUBMIT_DIR" || exit 1; M=$PWD/mesh
CORES=($(bash /work/gyj329/MFEM/implementations/bare_plate/bare_plate_enzyme_mechanical/jobs/alloc_cores.sh)); sl() { (IFS=,; echo "${CORES[*]:$1:$2}"); }
gen() { [ -f $M/$1 ] || python3 mesh/gen_box.py $M/$1 "${@:2}" > /dev/null; }
gen M3_full_y3z4.mesh --x -6.8 -4.7 --y -0.3 0.3 --z -0.4 --h 0.02;  gen M2_full_y8z10.mesh --x -7.6 -4.2 --y -0.8 0.8 --z -1.0 --h 0.04
gen M3_full_y4z5.mesh --x -6.8 -4.7 --y -0.4 0.4 --z -0.5 --h 0.02;  gen M2_full_y10z12.mesh --x -7.6 -4.2 --y -1.0 1.0 --z -1.2 --h 0.04
python3 mesh/make_plate_mesh.py $M/L1_h0.2_full_w.mesh --h 0.2 --fine-x -8 8 --fine-y 1.2 --fine-z 1.4 --r 1.3 --hmax 1.0 > /dev/null
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 0 8) jobs/run_moving_case.sh 0 results/p3box_y3z4 "-l1 $M/L1_h0.2_full_w.mesh -l2 $M/M2_full_y8z10.mesh -l3 $M/M3_full_y3z4.mesh -h2 0.04 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh" &
X0=-5 X1=5 TOPLOSS=0 NT=8 taskset -c $(sl 8 8) jobs/run_moving_case.sh 0 results/p3box_y4z5 "-l1 $M/L1_h0.2_full_w.mesh -l2 $M/M2_full_y10z12.mesh -l3 $M/M3_full_y4z5.mesh -h2 0.04 -h3 0.02 -fb 1 -peakmesh $M/P_full_h0.02.mesh" &
wait
B=../../arc/results/case0_full10
for d in p3box_y3z4 p3box_y4z5; do echo "=========== $d"; python3 post_levels.py $B $PWD/../../arc/mesh/track_h0.02_x-5_5.mesh results/$d $M/P_full_h0.02.mesh; done
