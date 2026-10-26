#!/usr/bin/env python3
"""Model vs NIST table for all AMB2022-03 cases from analysed runs (result.json of post/analyze_case.py).
usage: summarize.py OUT.csv CASE=RUNDIR [CASE=RUNDIR ...]      (the first case is marked as the calibration case)
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
meas = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}
Q = [("depth_um", "depth_um"), ("width_um", "width_um"), ("TTAM_s", "TTAM_s"), ("TLCR_C_s", "TLCR_C_s"), ("TTCR_C_s", "TTCR_C_s"), ("TSCR_C_s", "TSCR_C_s")]
w = csv.writer(open(sys.argv[1], "w"))
w.writerow(["case", "power_W", "speed_mm_s", "spot_um", "role"] + [f"{q}_{s}" for q, _ in Q for s in ("model", "nist", "err_pct")]
           + ["dDepth_dsRz", "dDepth_dsEta", "dWidth_dsRz", "dWidth_dsEta"])
print(f"{'case':5s} {'role':9s}" + "".join(f"{q:>10s}" for q, _ in Q) + "   (model error vs NIST, %)")
for k, arg in enumerate(sys.argv[2:]):
    c, d = arg.split("=", 1); r = json.load(open(os.path.join(d, "result.json"))); m = r["melt_pool_mid_1298C"]; p = r["probe"]; M = meas[c]
    val = {"depth_um": m["depth_um"], "width_um": m["width_um"], **{q: p.get(q, float("nan")) for q in ("TTAM_s", "TLCR_C_s", "TTCR_C_s", "TSCR_C_s")}}
    row, errs = [c, M["power_W"], M["speed_mm_s"], M["spot_D4sigma_um"], "calibration" if k == 0 else "prediction"], []
    for q, mq in Q:
        e = (val[q] - float(M[mq])) / float(M[mq]) * 100; errs.append(e); row += [f"{val[q]:.4g}", M[mq], f"{e:.1f}"]
    row += [f"{m['d_depth_um']['Rz']:.1f}", f"{m['d_depth_um']['Eta']:.1f}", f"{m['d_width_um']['Rz']:.1f}", f"{m['d_width_um']['Eta']:.1f}"]
    w.writerow(row); print(f"{c:5s} {row[4][:9]:9s}" + "".join(f"{e:+10.1f}" for e in errs))
