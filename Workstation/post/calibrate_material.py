#!/usr/bin/env python3
"""Joint calibration of heat source and IN718 material on all NIST quantities of all AMB2022-03 cases.

Unknowns theta (multiplicative ones as relative steps, the rest absolute):
    eta      laser absorptivity              (AD seed Eta)
    c0, n    Goldak depth c_i = c0 (X_i/X_0)^n (AD seed Rz; dq/dn = dq/ds_c * ln(X_i/X_0))
    k        scale of the conductivity curve  (AD seed k)          prior 1 +- 15 %
    L        scale of the latent heat         (AD seed Lfus)       prior 1 +- 15 %
    Ts, Tl   solidus / liquidus [K]           (AD seeds Ts, Tliq; relative -> per K) prior 1533.15 / 1609.15 +- 15 K
Residuals per case: depth, width (1298 C pool at x = 0), TTAM, TLCR, TTCR, TSCR, relative to NIST and divided by an
assumed measurement uncertainty (3, 3, 5, 10, 10, 10 %), plus the prior terms. Jacobian from the AD sensitivities of every
run (post/analyze_case.py: melt-pool and probe sensitivities; TTCR/TSCR derivatives are approximate because the 1260 C
window edge sits on the solidus kink - the Levenberg-Marquardt loop checks every step with real runs and rejects steps
that do not lower the cost). The two-stage latent-heat fraction (FL_EUT, T_EUT) stays fixed.
Runs with the Workstation or HPC run/run_case.sh (RUN_CASE, default ../Workstation/run/run_case.sh); LAUNCH=slurm as in
calibrate_multi.py. usage: calibrate_material.py OUTROOT [--maxit 4]  (DT, X0, X1, TAIL, MESH ... passed through)
Log: OUTROOT/iterations.csv (parameters, cost, per-quantity RMS errors); runs in OUTROOT/it<k>/case<c>/.
"""
import argparse, csv, json, math, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P = argparse.ArgumentParser(); P.add_argument("out"); P.add_argument("--maxit", type=int, default=4)
P.add_argument("--cases", default="0,1.1,1.2,2.1,2.2,3.1,3.2"); P.add_argument("--lam", type=float, default=1e-2)
P.add_argument("--eta", type=float, default=0.748695); P.add_argument("--c0", type=float, default=0.182604)
P.add_argument("--n", type=float, default=0.98581); P.add_argument("--k", type=float, default=1.0)
P.add_argument("--L", type=float, default=1.0); P.add_argument("--Ts", type=float, default=1533.15)
P.add_argument("--Tl", type=float, default=1609.15)
a = P.parse_args(); os.makedirs(a.out, exist_ok=True)
# working folder: its Workstation/run/run_case.sh (its own run/run_case.sh is the older interface); inside HPC/ or
# Workstation/: run/run_case.sh of that folder
RUN_CASE = os.environ.get("RUN_CASE") or next(p for p in (os.path.join(ROOT, "Workstation", "run", "run_case.sh"),
                                                      os.path.join(ROOT, "run", "run_case.sh")) if os.path.exists(p))
meas = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}
cases = a.cases.split(",")
xh = lambda r: float(r["power_W"]) / math.sqrt(float(r["speed_mm_s"]) * float(r["spot_D4sigma_um"]) ** 3)
LX = {c: math.log(xh(meas[c]) / xh(meas["0"])) for c in cases}
Q = [("depth_um", "depth_um", 0.03), ("width_um", "width_um", 0.03), ("TTAM_s", "TTAM_s", 0.05),
     ("TLCR_C_s", "TLCR_C_s", 0.10), ("TTCR_C_s", "TTCR_C_s", 0.10), ("TSCR_C_s", "TSCR_C_s", 0.10)]
NAMES = ["eta", "c0", "n", "k", "L", "Ts", "Tl"]
PRIOR = {"k": (1.0, 0.15), "L": (1.0, 0.15), "Ts": (1533.15, 15.0), "Tl": (1609.15, 15.0)}     # (centre, sigma)
CAP = {"eta": 0.2, "c0": 0.2, "n": 0.5, "k": 0.2, "L": 0.2, "Ts": 20.0, "Tl": 20.0}           # step caps (rel / abs)
REL = {"eta", "c0", "k", "L"}                                                                  # multiplicative unknowns


def solve(A, b):
    n = len(b); M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(M[i][k])); M[k], M[p] = M[p], M[k]
        for i in range(k + 1, n):
            f = M[i][k] / M[k][k]; M[i] = [M[i][j] - f * M[k][j] for j in range(n + 1)]
    x = [0.0] * n
    for i in range(n - 1, -1, -1): x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def env_for(th, c):
    ex = f"-kscale {th['k']:.6f} -lfusscale {th['L']:.6f} -tsolidus {th['Ts']:.4f} -tliquidus {th['Tl']:.4f}"
    return dict(os.environ, ABSORP=f"{th['eta']:.6f}", C0=f"{th['c0']:.6f}", N_EXP=f"{th['n']:.6f}", DORDER="1",
                EXTRA=(os.environ.get("EXTRA", "") + " " + ex).strip())


def launch(c, run, env):
    if os.environ.get("LAUNCH") == "slurm":
        cmd = ["sbatch", "--wait", "--parsable", f"--job-name=st_{c}", "--export=ALL", os.path.join(os.path.dirname(os.path.dirname(RUN_CASE)), "slurm", "case.sbatch"), c, run]
        r = subprocess.run(cmd, env=env, cwd=os.path.dirname(os.path.dirname(RUN_CASE)), capture_output=True, text=True)
        return r.returncode, f"case {c}: slurm job {r.stdout.strip()} exit={r.returncode}"
    r = subprocess.run(["bash", RUN_CASE, c, run], env=env, capture_output=True, text=True)
    return r.returncode, r.stdout.strip()


def evaluate(it, th):
    """all cases at theta: weighted residuals, Jacobian rows (d residual / d step), per-quantity relative errors"""
    todo = []
    for c in cases:
        run = os.path.join(a.out, f"it{it}", f"case{c}")
        if not os.path.exists(os.path.join(run, "peakT.bin")): todo.append((c, run, env_for(th, c)))
    with ThreadPoolExecutor(max_workers=len(todo) if os.environ.get("LAUNCH") == "slurm" and todo else 1) as ex:
        for (c, run, _), (rc, msg) in zip(todo, ex.map(lambda t: launch(*t), todo)):
            print("  " + msg); sys.stdout.flush()
            if rc != 0: sys.exit(f"run failed: {run}")
    r, J, errs = [], [], {q: [] for q, _, _ in Q}
    for c in cases:
        run = os.path.join(a.out, f"it{it}", f"case{c}"); js = os.path.join(run, "result.json")
        if not os.path.exists(js):
            subprocess.run([sys.executable, os.path.join(HERE, "analyze_case.py"), run, "--json", js], check=True, capture_output=True)
        R = json.load(open(js)); m = R["melt_pool_mid_1298C"]; pr = R["probe"]; ps = R.get("probe_sens", {})
        for q, mq, u in Q:
            if q in ("depth_um", "width_um"): v, d = m[q], m["d_" + q]
            else: v, d = pr.get(q, float("nan")), ps.get(q)
            if v != v or d is None: continue                       # quantity not available in this run
            M = float(meas[c][mq]); w = 1.0 / (M * u)
            r.append((v - M) * w); errs[q].append((v - M) / M)
            J.append([w * d["Eta"], w * d["Rz"], w * d["Rz"] * LX[c], w * d["k"], w * d["Lfus"],
                      w * d["Ts"] / th["Ts"], w * d["Tliq"] / th["Tl"]])
    for nm, (c0, sg) in PRIOR.items():                             # prior terms (theta - centre) / sigma
        j = NAMES.index(nm); row = [0.0] * len(NAMES)
        if nm in REL: r.append((th[nm] - c0) / sg); row[j] = th[nm] / sg            # d/d(relative step)
        else: r.append((th[nm] - c0) / sg); row[j] = 1.0 / sg
        J.append(row)
    return r, J, errs


log = os.path.join(a.out, "iterations.csv")
if not os.path.exists(log):
    open(log, "w").write("it," + ",".join(NAMES) + ",cost," + ",".join(f"rms_{q}" for q, _, _ in Q) + "\n")
th = {"eta": a.eta, "c0": a.c0, "n": a.n, "k": a.k, "L": a.L, "Ts": a.Ts, "Tl": a.Tl}
lam, best, cost_prev = a.lam, None, None
for it in range(a.maxit):
    print(f"it {it}: " + "  ".join(f"{k} {v:.5g}" for k, v in th.items())); sys.stdout.flush()
    r, J, errs = evaluate(it, th)
    cost = sum(x * x for x in r)
    rms = {q: math.sqrt(sum(e * e for e in v) / len(v)) * 100 if v else float("nan") for q, v in errs.items()}
    open(log, "a").write(f"{it}," + ",".join(f"{th[k]:.6g}" for k in NAMES) + f",{cost:.4f}," + ",".join(f"{rms[q]:.2f}" for q, _, _ in Q) + "\n")
    print(f"  cost {cost:.3f}; RMS error % " + "  ".join(f"{q.split('_')[0]} {rms[q]:.1f}" for q, _, _ in Q)); sys.stdout.flush()
    if cost_prev is not None:
        if cost > cost_prev:
            print("  cost increased -> step rejected"); th, r, J = dict(best[0]), best[1], best[2]; cost = cost_prev
            lam = max(lam * 10, 0.1)
        else:
            if (cost_prev - cost) / cost_prev < 0.02: print("  converged (cost decrease < 2 %)"); break
            lam *= 0.5
    best = (dict(th), r, J); cost_prev = cost
    nP = len(NAMES)
    JTJ = [[sum(J[k][i] * J[k][j] for k in range(len(J))) for j in range(nP)] for i in range(nP)]
    JTr = [-sum(J[k][i] * r[k] for k in range(len(J))) for i in range(nP)]
    for i in range(nP): JTJ[i][i] *= 1 + lam
    d = solve(JTJ, JTr)
    f = max(max(abs(d[i]) / CAP[NAMES[i]] for i in range(nP)), 1.0); d = [x / f for x in d]
    print("  step: " + "  ".join(f"d{NAMES[i]} {d[i]:+.4g}" for i in range(nP))); sys.stdout.flush()
    for i, nm in enumerate(NAMES):
        th[nm] = th[nm] * (1 + d[i]) if nm in REL else th[nm] + d[i]
