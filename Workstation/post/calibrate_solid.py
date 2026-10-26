#!/usr/bin/env python3
"""Calibrate the two-stage liquid fraction (driver options -fleut phi -teut T_e) on the NIST transition and solid cooling
rates of one case (default 0): a fraction phi of the latent heat is released between T_e and the solidus (terminal Nb/Laves
solidification of IN718), 1 - phi between solidus and liquidus.

phi and T_e are plain run-time constants of the program (no AD seed), so the 2x2 Jacobian of the residuals
r = ((TTCR - TTCR_NIST)/TTCR_NIST, (TSCR - TSCR_NIST)/TSCR_NIST) is taken by one-sided finite differences
(phi + dphi, T_e - dTe): 3 runs per Newton iteration. Bounds: 0 <= phi <= 0.8, 1323 K <= T_e <= T_solidus - 20 K.
The heat source (ABSORP, GC) and discretisation come from the environment as in run/run_case.sh; BIN must be a program
built with the two-stage liquid fraction.
--phi-only: T_e fixed, phi alone fitted to the solid cooling rate by secant steps (the transition rate barely responds to
phi or T_e on case 0 at h = 40 um, so the 2x2 system is close to singular): 1 run per iteration after the first.
usage: calibrate_solid.py OUTROOT [--case 0 --phi 0.25 --te 1423.15 --maxit 4 --tol 0.05 --phi-only]
Log: OUTROOT/iterations.csv; runs in OUTROOT/it<k>_{base,phi,te}/.
"""
import argparse, csv, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
T_SOL = 1533.15
P = argparse.ArgumentParser(); P.add_argument("out"); P.add_argument("--case", default="0")
P.add_argument("--phi", type=float, default=0.25); P.add_argument("--te", type=float, default=1423.15)
P.add_argument("--dphi", type=float, default=0.05); P.add_argument("--dte", type=float, default=25.0)
P.add_argument("--maxit", type=int, default=4); P.add_argument("--tol", type=float, default=0.05)
P.add_argument("--phi-only", action="store_true")
a = P.parse_args(); os.makedirs(a.out, exist_ok=True)
meas = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}[a.case]
M = (float(meas["TTCR_C_s"]), float(meas["TSCR_C_s"]))


def run(tag, phi, te):
    d = os.path.join(a.out, tag)
    if not os.path.exists(os.path.join(d, "result.json")):
        env = dict(os.environ, FL_EUT=f"{phi:.6f}", T_EUT=f"{te:.4f}")
        r = subprocess.run(["bash", os.path.join(ROOT, "run", "run_case.sh"), a.case, d], env=env, capture_output=True, text=True)
        print("  " + r.stdout.strip()); sys.stdout.flush()
        if r.returncode != 0: sys.exit(f"run failed: {d}")
        subprocess.run([sys.executable, os.path.join(HERE, "analyze_case.py"), d, "--json", os.path.join(d, "result.json")],
                       check=True, capture_output=True)
    res = json.load(open(os.path.join(d, "result.json"))); p = res["probe"]; m = res["melt_pool_mid_1298C"]
    return [(p["TTCR_C_s"] - M[0]) / M[0], (p["TSCR_C_s"] - M[1]) / M[1]], p, m


log = os.path.join(a.out, "iterations.csv")
if not os.path.exists(log): open(log, "w").write("it,phi,Te_K,TTCR,TSCR,err_TTCR_pct,err_TSCR_pct,TAM_s,TLCR,depth_um,width_um\n")
phi, te = a.phi, a.te
if a.phi_only:
    pts = []                                       # (phi, solid-rate residual)
    for it in range(a.maxit):
        r, p, m = run(f"it{it}_base", phi, te); pts.append((phi, r[1]))
        open(log, "a").write(f"{it},{phi:.5f},{te:.2f},{p['TTCR_C_s']:.4g},{p['TSCR_C_s']:.4g},{r[0]*100:.2f},{r[1]*100:.2f},"
                             f"{p['TTAM_s']:.4g},{p['TLCR_C_s']:.4g},{m['depth_um']:.2f},{m['width_um']:.2f}\n")
        print(f"it {it}: phi {phi:.4f} Te {te - 273.15:.1f} C -> TTCR err {r[0]*100:+.1f} %, TSCR err {r[1]*100:+.1f} %"); sys.stdout.flush()
        if abs(r[1]) < a.tol: print("converged"); break
        if len(pts) == 1: phi = phi + (a.dphi if r[1] > 0 else -a.dphi)      # more phi -> slower solid cooling
        else:
            (p0, r0), (p1, r1) = pts[-2], pts[-1]
            phi = p1 - r1 * (p1 - p0) / (r1 - r0)
        phi = min(max(phi, 0.0), 0.8)
    sys.exit(0)
for it in range(a.maxit):
    r, p, m = run(f"it{it}_base", phi, te)
    open(log, "a").write(f"{it},{phi:.5f},{te:.2f},{p['TTCR_C_s']:.4g},{p['TSCR_C_s']:.4g},{r[0]*100:.2f},{r[1]*100:.2f},"
                         f"{p['TTAM_s']:.4g},{p['TLCR_C_s']:.4g},{m['depth_um']:.2f},{m['width_um']:.2f}\n")
    print(f"it {it}: phi {phi:.3f} Te {te - 273.15:.1f} C -> TTCR err {r[0]*100:+.1f} %, TSCR err {r[1]*100:+.1f} %"); sys.stdout.flush()
    if max(abs(x) for x in r) < a.tol: print("converged"); break
    rp, _, _ = run(f"it{it}_phi", phi + a.dphi, te)
    rt, _, _ = run(f"it{it}_te", phi, te - a.dte)
    J = [[(rp[i] - r[i]) / a.dphi, (rt[i] - r[i]) / (-a.dte)] for i in range(2)]
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]
    d_phi = (-r[0] * J[1][1] + J[0][1] * r[1]) / det; d_te = (-J[0][0] * r[1] + J[1][0] * r[0]) / det
    f = max(abs(d_phi) / 0.2, abs(d_te) / 60.0, 1.0); d_phi /= f; d_te /= f   # trust region: 0.2 in phi, 60 K in T_e
    print(f"  J = {J}, step dphi {d_phi:+.3f} dTe {d_te:+.1f} K"); sys.stdout.flush()
    phi = min(max(phi + d_phi, 0.0), 0.8); te = min(max(te + d_te, 1323.15), T_SOL - 20.0)
