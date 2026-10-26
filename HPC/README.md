# AMB2022-03 single tracks — HPC version (CPU only, SLURM)

Calibrated thermal model of the NIST AM Bench AMB2022-03 single laser tracks on a bare IN718 plate (phase 1, melt-pool
scale), with first-order automatic-differentiation sensitivities (Enzyme) of the temperature field with respect to
20 parameters. This folder is the **CPU-only** build for the cluster; it needs no GPU and no CUDA. The workstation version
(`../Workstation`) runs the same model with GPU kernels; both give the same results to ~1e-14 (checked).

## 1. First steps on the cluster

```bash
# copy the folder (from the workstation)
rsync -a HPC/ <user>@<cluster>:~/single_tracks/          # ~350 MB
cd ~/single_tracks
bash check/selftest.sh                                    # on the login node: a 2-5 min test, must print SELFTEST PASS
#   or on a compute node:  sbatch slurm/selftest.sbatch    (output in logs/)
```

The program was built on Ubuntu 22.04 and needs **glibc ≥ 2.35**. On our cluster (EL8, glibc 2.28) it runs inside a plain
Ubuntu 22.04 image: `bash container/get_image.sh` once (no root needed, ~30 MB), then everything below works unchanged
(section 5). Verified on the cluster: self-test PASS (max. difference 8e-16), all 7 cases reproduce the workstation results.

Requirements on the cluster: `python3` (standard library only; 3.12 on the compute nodes), `bash`, `singularity` (for the
image). No modules need to be loaded. This cluster requires `--ntasks` on every `sbatch` (all `slurm/*.sbatch` set it).

## 2. Run one case

```bash
sbatch slurm/case.sbatch 0 results/case0                  # one whole node (40 cores); ~85-90 min with sensitivities
python3 post/analyze_case.py results/case0                # model vs NIST: depth, width, TAM, cooling rates + sensitivities
```

Cases: `0, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2` (`inputs/cases.csv`: power, speed, spot and the NIST measurements).
Every setting in `config/calibration.env` can be overridden from the submitting shell (sbatch passes the environment):

| variable | default | meaning |
|---|---|---|
| `DORDER` | 1 | 1 = temperatures + 20 sensitivities, 0 = temperatures only (much cheaper) |
| `X0`, `X1` | -0.6, 1.5 | laser track [mm]; the probe and melt-pool section are at x = 0 |
| `H_MESH`, `DT` | 0.02, 5e-6 | element size [mm] in the refined box, time step [s] (converged values) |
| `ABSORP`, `C0`, `N_EXP` | calibrated | heat source (see 4) |
| `FL_EUT`, `T_EUT` | 0.23, 1423.15 | two-stage liquid fraction (see 4) |
| `GC` | from C0, N_EXP | Goldak depth [mm] for this run only |
| `MESH` | generated | own mesh file; by default `mesh/track_h<H>_x<X0>_<X1>.mesh` is generated on first use |
| `TAIL` | 4e-4 | cooling time after the laser stops [s] |
| `NT` | cores of the node | threads |
| `CONTAINER` | — | Singularity image to run the program in (section 5) |

Example: `DORDER=0 DT=1e-5 sbatch slurm/case.sbatch 2.2 results/case2.2_T`.

## 3. All cases, calibration, monitoring

```bash
sbatch slurm/all_cases.sbatch results/all                 # 7 jobs in parallel (job array), one node each
python3 post/summarize.py results/all/summary.csv 0=results/all/case0 1.1=results/all/case1.1 1.2=results/all/case1.2 \
        2.1=results/all/case2.1 2.2=results/all/case2.2 3.1=results/all/case3.1 3.2=results/all/case3.2

sbatch slurm/calibrate.sbatch results/calib --maxit 3     # recalibrate eta, c0, n: each iteration = 7 parallel jobs
sbatch slurm/calibrate_material.sbatch results/calmat     # source + material (k, L, solidus, liquidus) on all 6 quantities
python3 tools/jobs.py -w 60                               # status: SLURM jobs, progress % and ETA of every run
```

The account allows 10 jobs at a time: the calibration uses 1 driver + 7 case jobs. Default partition `compute1`
(3-day limit); for long runs add `--partition=compute2 --time=10-00:00:00` to `sbatch`.

**Full 10 mm NIST track**: `X0=-5 X1=5 sbatch --partition=compute2 --time=10-00:00:00 slurm/case.sbatch 0 results/case0_full`
(~1.3M nodes; temperatures only est. 15-20 h, with sensitivities considerably longer — start with `DORDER=0`).

## 3b. Nominal response vs NIST thermography + full set of first derivatives

```bash
bash get_nist_thermography.sh                             # NIST mds2-2716 thermography (550 MB, checksum verified)
python3 -m pip install --user h5py scipy matplotlib       # once, for the comparison scripts
sbatch slurm/thermo_all.sbatch results/thermo             # 7 cases: run with 20 sensitivities + field dumps, then figures
```

Per case (`results/thermo/case<c>/`): `thermography_case<c>.png` - NIST camera frame vs model surface field (same colour
scale, both referenced to the front edge of the saturated spot), isotherms, centreline profile, midpoint history with the
NIST cooling-rate windows; `derivatives.csv` + `derivatives_case<c>.png` - the 20 first derivatives of depth, width, TAM
and the three cooling rates (also as elasticities) and dT/ds(t) at the midpoint for every parameter.
The camera saturates above ~1520-1660 C and records nothing below ~940 C; NIST's emissivity per track is reproduced from
their procedure (solidification plateau at 1298 C; tools/nist_thermo.py).

**Sensitivity solver:** `SENS_SOLVER=krylov` (used by `thermo_all.sbatch`) solves the sensitivity system with
Jacobi-preconditioned BiCGSTAB (~5 iterations per right-hand side) instead of a PARDISO factorisation every step: 4.7x faster
on the test mesh, results within 1.5e-11 of PARDISO; a solve that does not converge falls back to PARDISO automatically.
Default remains PARDISO.

## 4. Model and calibration (summary)

* IN718 properties from open measured data (`src/thermal/material_in718.hpp`); solidus 1260 °C, liquidus 1336 °C.
* Plate 25.4 × 25.4 × 3.17 mm, bottom at 23.5 °C (baseplate contact), convection + radiation on top; graded hex mesh.
* Heat source: Goldak double ellipsoid, width/length from the measured spot (a = b = √6 σ), depth per case
  **c = C0 (X/X₀)^n**, X = P/√(v·D4σ³) (normalised-enthalpy scaling). Calibrated on depth and width of all 7 cases with the
  AD Jacobian (Levenberg–Marquardt): **η = 0.749, C0 = 183 µm, n = 0.986**, RMS error 8.2 %.
* Two-stage latent heat: **23 %** of the latent heat released between **1150 °C** and the solidus (terminal Nb/Laves
  solidification), calibrated on the solid cooling rate of case 0.
* Converged discretisation (mesh study 40 → 13.3 µm, dt 10 → 5 µs): **h = 20 µm, dt = 5 µs**.
* Known limitation: the transition cooling rate (1336 → 1260 °C) is ~25 % slower than measured at every resolution.

### Validation (all 7 cases, h = 20 µm, dt = 5 µs, final calibration) — model error vs NIST [%]

| case | P [W] | v [mm/s] | spot [µm] | depth | width | TAM | liquid CR | transition CR | solid CR |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 285 | 960 | 67 | −0.1 | −0.9 | +4.9 | +14.1 | −25.9 | +3.2 |
| 1.1 | 285 | 960 | 49 | −20.9 | −3.7 | −31.3 | +56.1 | +26.1 | −2.5 |
| 1.2 | 285 | 960 | 82 | +9.9 | +8.8 | +36.3 | +23.6 | −25.9 | +40.0 |
| 2.1 | 285 | 1200 | 67 | +10.6 | +5.8 | +12.2 | +20.6 | −38.1 | +11.2 |
| 2.2 | 285 | 800 | 67 | −9.8 | −4.3 | −2.6 | +19.9 | −18.2 | −1.4 |
| 3.1 | 325 | 960 | 67 | −4.9 | +5.0 | +2.6 | +10.4 | −18.9 | −11.2 |
| 3.2 | 245 | 960 | 67 | +3.2 | −0.6 | +9.8 | +26.1 | −27.1 | +10.7 |
| **RMS** | | | | **10.5** | **4.9** | **19.2** | **28.1** | **26.5** | **16.8** |

Width is predicted within 9 % in every case; depth within 11 % except the smallest spot (1.1, keyhole, −21 %).
Power and speed changes (2.x, 3.x) are captured well for depth, width and TAM. The spot-size cases (1.1, 1.2) carry the
largest TAM errors. Systematic biases: the liquid cooling rate is ~24 % too fast and the transition rate ~18 % too slow on
average (latent-heat model), the solid rate scatters around the measurement (mean +7 %).

Note: since 2026-10-26 `post/analyze_case.py` computes the cooling rates as the least-squares slope over the continuous
window between the level crossings (smooth in the parameters, needed for their AD sensitivities); the table above used the
sample-based fit, which differs by ≤ 2 %.

Details, figures and the convergence study: the PDF report of the project (`report/` on the workstation).

## 5. Container (clusters with glibc < 2.35)

```bash
bash container/get_image.sh                               # singularity pull docker://ubuntu:22.04 -> ubuntu22.sif
bash check/selftest.sh                                    # uses ./ubuntu22.sif automatically: SELFTEST PASS
```

`run/run_case.sh` and `check/selftest.sh` pick up `ubuntu22.sif` from this folder automatically and run the program from
`bin/` inside it (`bin/`, `lib/`, the mesh and output folders are bind-mounted), so a new program version never needs a new
image. `CONTAINER=<image.sif>` selects another image. Keep the folder path free of commas (bind list).
`container/single_tracks.def`/`build_container.sh` build a self-contained image instead; they need `--fakeroot` or podman,
which this cluster does not allow (no subuid, NFS).

## 6. Contents

| path | content |
|---|---|
| `bin/thermal_in718_cpu`, `lib/` | program (MFEM 4.9.1, MKL PARDISO, hypre BoomerAMG, Enzyme AD) and its libraries |
| `config/calibration.env` | calibrated parameters and default discretisation |
| `run/run_case.sh` | one case (any machine); `slurm/*.sbatch` — job scripts |
| `post/` | `analyze_case.py` (NIST quantities + AD sensitivities of all of them), `summarize.py`, `calibrate_multi.py`, `calibrate_material.py`, `calibrate_solid.py` |
| `inputs/`, `mesh/` | cases and NIST data, laser path generator, mesh generator |
| `tools/jobs.py` | job status |
| `check/` | self-test and its reference (computed on the workstation) |
| `container/` | `get_image.sh` (plain Ubuntu 22.04 image, used on this cluster); Singularity / Podman recipes |
| `src/` | source code and Makefile of the program (for reference / rebuilding) |

Solver notes: the Newton systems use CG + BoomerAMG above 200k unknowns (`AMG_MIN_DOFS`), PARDISO below; the sensitivity
system (unsymmetric) uses PARDISO. A Newton failure is retried with dt/2 (up to 6 times, `NEWTON_CUTS`).
