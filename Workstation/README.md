# AMB2022-03 single tracks — Workstation version (CPU + GPU)

Calibrated thermal model of the NIST AM Bench AMB2022-03 single laser tracks on a bare IN718 plate (phase 1, melt-pool
scale), with first-order automatic-differentiation sensitivities (Enzyme) with respect to 20 parameters. This folder is the
**CPU + GPU** build for this workstation (Xeon W-2265, 12 cores; Quadro RTX 4000, 8 GB). The cluster version (`../HPC`)
runs the same model on CPUs only; both give the same results to ~1e-14 (checked).

## 1. Run

```bash
bash check/selftest.sh                                    # 2-5 min, must print SELFTEST PASS
bash run/run_case.sh 0 results/case0                      # one case (~20-35 min at the converged resolution)
python3 post/analyze_case.py results/case0                # model vs NIST: depth, width, TAM, cooling rates + sensitivities
python3 tools/jobs.py -w                                  # status of the running jobs (refresh every 5 min)
```

Cases: `0, 1.1, 1.2, 2.1, 2.2, 3.1, 3.2` (`inputs/cases.csv`). Settings come from `config/calibration.env` and can be
overridden in the environment, e.g. `DORDER=0 DT=1e-5 bash run/run_case.sh 2.2 results/case2.2_T`:

| variable | default | meaning |
|---|---|---|
| `DORDER` | 1 | 1 = temperatures + 20 sensitivities, 0 = temperatures only |
| `X0`, `X1` | -0.6, 1.5 | laser track [mm]; probe and melt-pool section at x = 0 |
| `H_MESH`, `DT` | 0.02, 5e-6 | element size [mm] in the refined box, time step [s] (converged values) |
| `ABSORP`, `C0`, `N_EXP`, `GC` | calibrated | heat source; `GC` = Goldak depth [mm] for this run only |
| `FL_EUT`, `T_EUT` | 0.23, 1423.15 | two-stage liquid fraction |
| `NT`, `CORES` | 12, 0-11 | threads and the cores they are pinned to (the 12 physical cores) |
| `SOLVER` | auto | CPU/GPU solver switch for the direct solves: auto, pardiso, cudss |
| `BIN` | `bin/thermal_in718_gpu` | `bin/thermal_in718_cpu` = the CPU-only program (same as the cluster's) |

All cases one after the other: `for c in 0 1.1 1.2 2.1 2.2 3.1 3.2; do bash run/run_case.sh $c results/all/case$c; done`,
then `python3 post/summarize.py results/all/summary.csv 0=results/all/case0 1.1=results/all/case1.1 ...`.
Calibration: `python3 post/calibrate_multi.py results/calib --eta 0.748695 --c0 0.182604 --n 0.98581 --maxit 3`.

## 2. What runs on the GPU

Goldak source and its AD derivatives, Newton assembly (conductivity refill, boundary terms, capacitance) and the Enzyme
sensitivity blocks run on the GPU; the linear solvers run on the CPU:

* Newton systems: CG + BoomerAMG above 200k unknowns (5-7× faster than PARDISO at 700k), PARDISO below.
* Sensitivity system (unsymmetric): PARDISO. The solver switch measures cuDSS against PARDISO at run time; on this card
  PARDISO always wins (FP64 rate of the RTX 4000 is 1/32 of FP32), and cuDSS runs out of the 8 GB above ~300k unknowns
  (the switch then releases the GPU memory and continues with PARDISO).
* GPU memory guard: if the Newton-assembly arrays do not fit (≳1M elements), that part falls back to the CPU.
* Newton failures are retried with dt/2 (up to 6 times, `NEWTON_CUTS`).

## 3. Model and calibration (summary)

* IN718 properties from open measured data; solidus 1260 °C, liquidus 1336 °C; plate on a 23.5 °C baseplate.
* Goldak source, depth per case **c = C0 (X/X₀)^n**, X = P/√(v·D4σ³): **η = 0.749, C0 = 183 µm, n = 0.986**
  (calibrated on depth and width of all 7 cases, RMS error 8.2 %).
* Two-stage latent heat: **23 %** released between **1150 °C** and the solidus (calibrated on the solid cooling rate).
* Converged discretisation: **h = 20 µm, dt = 5 µs**. Known limitation: transition cooling rate ~25 % too slow.

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

Full history (calibration runs, convergence study, figures, PDF report): the parent folder `single_tracks/`
(`results/`, `report/`).

## 4. Contents

| path | content |
|---|---|
| `bin/thermal_in718_gpu` | CPU + GPU program (Quadro RTX 4000, sm_75); `bin/thermal_in718_cpu` = CPU only |
| `lib/` | bundled libraries (MKL, GCC runtime; same files as `HPC/lib`); CUDA 12 / cuDSS from `MFEM/toolchain` (program RPATH) |
| `config/calibration.env` | calibrated parameters and default discretisation |
| `run/run_case.sh` | one case; `post/` analysis and calibration; `tools/jobs.py` status |
| `inputs/`, `mesh/` | cases and NIST data, laser path and mesh generators |
| `check/` | self-test and its reference |
| `src/` | source code and Makefile (`make gpu_enzyme cpu MATERIAL=IN718`; `GPU_ARCHS="70 75 80"` for other GPUs) |
