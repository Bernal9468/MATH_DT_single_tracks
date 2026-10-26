#!/usr/bin/env python3
"""Process-dependent Goldak depth, calibrated on all AMB2022-03 cases at once.

Source depth per case from the normalised-enthalpy scaling (King et al. 2014, Hann et al. 2011: dH/h_s ~ eta P / sqrt(v sigma^3)):
    c_i = c0 * X_i^n,   X_i = (P_i / sqrt(v_i sigma_i^3)) / (P_0 / sqrt(v_0 sigma_0^3))     (X = 1 for the baseline case 0)
Unknowns theta = (eta, c0, n); residuals r = (depth_i - depth_NIST_i) / depth_NIST_i and the same for the width, i = all cases.
Jacobian from the AD sensitivities of every run (post/analyze_case.py, relative scales s):
    dD_i/ds_eta = d_depth[Eta],  dD_i/ds_c0 = d_depth[Rz],  dD_i/dn = d_depth[Rz] * ln X_i      (c_i relative change = ds_c0 + ln X_i dn)
Levenberg-Marquardt step (lambda adapted: halved after a decrease of the cost; after an increase the step is rejected and
retried with lambda = max(10 lambda, 0.1), large enough to shorten it),
steps capped at |ds_eta|, |ds_c0| <= 0.3 and |dn| <= 1. Stops when the cost decreases by < 2 % or after --maxit iterations.
usage: calibrate_multi.py OUTROOT [--eta 0.750774 --c0 0.184612 --n 0 --maxit 4]   (MESH, DT, X0, X1, TAIL ... passed through)
Runs in OUTROOT/it<k>/case<c>/ (an existing run there, e.g. a symlink, is reused); log OUTROOT/iterations.csv.
LAUNCH=slurm (HPC): the cases of an iteration run in parallel, one SLURM job each (slurm/case.sbatch).
"""
import argparse, csv, json, math, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P = argparse.ArgumentParser(); P.add_argument("out")
P.add_argument("--eta", type=float, default=0.750774); P.add_argument("--c0", type=float, default=0.184612)
P.add_argument("--n", type=float, default=0.0); P.add_argument("--maxit", type=int, default=4)
P.add_argument("--cases", default="0,1.1,1.2,2.1,2.2,3.1,3.2"); P.add_argument("--lam", type=float, default=1e-3)
a = P.parse_args(); os.makedirs(a.out, exist_ok=True)
meas = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}
cases = a.cases.split(",")
xh = lambda r: float(r["power_W"]) / math.sqrt(float(r["speed_mm_s"]) * float(r["spot_D4sigma_um"]) ** 3)
X = {c: xh(meas[c]) / xh(meas["0"]) for c in cases}


def solve(A, b):   # small dense Gauss elimination with partial pivoting
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k])); M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]; M[i] = [M[i][j] - f * M[k][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1): x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def launch(c, run, env):
    """one case: here (run/run_case.sh), or with LAUNCH=slurm as its own job (slurm/case.sbatch, sbatch --wait)"""
    if os.environ.get("LAUNCH") == "slurm":
        cmd = ["sbatch", "--wait", "--parsable", f"--job-name=st_{c}", "--export=ALL", os.path.join(ROOT, "slurm", "case.sbatch"), c, run]
        r = subprocess.run(cmd, env=env, cwd=ROOT, capture_output=True, text=True)
        return r.returncode, f"case {c}: slurm job {r.stdout.strip()} exit={r.returncode} ({os.path.relpath(run, ROOT)})"
    r = subprocess.run(["bash", os.path.join(ROOT, "run", "run_case.sh"), c, run], env=env, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def evaluate(it, eta, c0, n):
    """run (or reuse) all cases; return residuals, Jacobian rows, per-case results"""
    res, J, rows = [], [], {}
    todo = []
    for c in cases:
        run = os.path.join(a.out, f"it{it}", f"case{c}"); ci = c0 * X[c] ** n
        if not os.path.exists(os.path.join(run, "peakT.bin")):
            todo.append((c, run, dict(os.environ, ABSORP=f"{eta:.6f}", GC=f"{ci:.6f}")))
    # LAUNCH=slurm: all cases of the iteration at once (one job each); otherwise one after the other
    with ThreadPoolExecutor(max_workers=len(todo) if os.environ.get("LAUNCH") == "slurm" and todo else 1) as ex:
        for (c, run, _), (rc, msg) in zip(todo, ex.map(lambda t: launch(*t), todo)):
            print("  " + msg); sys.stdout.flush()
            if rc != 0: sys.exit(f"run failed: {run}")
    for c in cases:
        run = os.path.join(a.out, f"it{it}", f"case{c}"); ci = c0 * X[c] ** n
        js = os.path.join(run, "result.json")
        if not os.path.exists(js):
            subprocess.run([sys.executable, os.path.join(HERE, "analyze_case.py"), run, "--json", js], check=True, capture_output=True)
        m = json.load(open(js))["melt_pool_mid_1298C"]; lx = math.log(X[c])
        Dm, Wm = float(meas[c]["depth_um"]), float(meas[c]["width_um"])
        res += [(m["depth_um"] - Dm) / Dm, (m["width_um"] - Wm) / Wm]
        for q, qm in (("d_depth_um", Dm), ("d_width_um", Wm)):
            J.append([m[q]["Eta"] / qm, m[q]["Rz"] / qm, m[q]["Rz"] * lx / qm])
        rows[c] = (ci, m["depth_um"], m["width_um"])
    return res, J, rows


log = os.path.join(a.out, "iterations.csv")
if not os.path.exists(log):
    open(log, "w").write("it,eta,c0_mm,n,cost," + ",".join(f"c_{c},D_{c},W_{c}" for c in cases) + "\n")
eta, c0, n, lam = a.eta, a.c0, a.n, a.lam
cost_prev = None; it = 0
while it < a.maxit:
    print(f"it {it}: eta {eta:.4f}  c0 {c0 * 1e3:.1f} um  n {n:.3f}  (c_i = " + ", ".join(f"{c} {c0 * X[c] ** n * 1e3:.0f}" for c in cases) + " um)")
    sys.stdout.flush()
    r, J, rows = evaluate(it, eta, c0, n)
    cost = sum(x * x for x in r)
    rms = math.sqrt(cost / len(r)) * 100
    open(log, "a").write(f"{it},{eta:.6f},{c0:.6f},{n:.5f},{cost:.5f}," + ",".join(f"{rows[c][0]:.6f},{rows[c][1]:.2f},{rows[c][2]:.2f}" for c in cases) + "\n")
    print(f"  cost {cost:.4f} (rms error {rms:.1f} %); depth err " + " ".join(f"{c}:{r[2*k]*100:+.0f}" for k, c in enumerate(cases))
          + " | width err " + " ".join(f"{c}:{r[2*k+1]*100:+.0f}" for k, c in enumerate(cases))); sys.stdout.flush()
    if cost_prev is not None:
        if cost > cost_prev:                       # rejected: back to the previous point with a larger damping
            print("  cost increased -> step rejected"); eta, c0, n, r, J = best; lam = max(lam * 10, 0.1); cost = cost_prev
        else:
            if (cost_prev - cost) / cost_prev < 0.02: print("  converged (cost decrease < 2 %)"); break
            lam *= 0.5
    best = (eta, c0, n, r, J); cost_prev = cost
    JTJ = [[sum(J[k][i] * J[k][j] for k in range(len(J))) for j in range(3)] for i in range(3)]
    JTr = [-sum(J[k][i] * r[k] for k in range(len(J))) for i in range(3)]
    for i in range(3): JTJ[i][i] *= 1 + lam
    ds_eta, ds_c0, dn = solve(JTJ, JTr)
    f = max(abs(ds_eta) / 0.3, abs(ds_c0) / 0.3, abs(dn) / 1.0, 1.0); ds_eta, ds_c0, dn = ds_eta / f, ds_c0 / f, dn / f
    pred = sum((r[k] + J[k][0] * ds_eta + J[k][1] * ds_c0 + J[k][2] * dn) ** 2 for k in range(len(r)))
    print(f"  step: ds_eta {ds_eta:+.3f} ds_c0 {ds_c0:+.3f} dn {dn:+.3f} (linear prediction of the cost {pred:.4f})")
    eta, c0, n = eta * (1 + ds_eta), c0 * (1 + ds_c0), n + dn
    it += 1
