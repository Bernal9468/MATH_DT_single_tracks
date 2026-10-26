#!/usr/bin/env python3
"""Full set of first-order derivatives of one run (DORDER=1): the 20 AD sensitivities of the NIST quantities and of the
temperature history at the track midpoint.

Inputs (RUNDIR): result.json of post/analyze_case.py (melt-pool depth/width derivatives, probe_sens of TTAM and the cooling
rates), probe.csv and sens.csv (T and dT/ds_p at x = y = 0 every step).
Outputs:
  RUNDIR/derivatives.csv   quantity, parameter, value, d quantity / d s_p (s_p = relative change of p), elasticity
                           (% change of the quantity per 1 % change of p; not defined for the shift parameters X0, TL)
  RUNDIR/derivatives_case<c>.png  (a) elasticities of depth, width, TTAM, TLCR, TTCR, TSCR, ranked;
                                  (b) dT/ds_p(t) at the midpoint for all 20 parameters with the NIST cooling-rate windows
Parameters (driver order): k cp rho Lfus Ts Tliq hc T0 P Eta Rx Ry Rz X0 Vx TL Tw hp kp Rhop - relative scales of the
conductivity, heat capacity, density, latent heat, solidus, liquidus, convection coefficient, ambient temperature, power,
absorptivity, Goldak semi-axes (length a, width b, depth c), start position X0 (per mm), speed, laser time shift TL (per s),
substrate temperature, and the powder parameters hp, kp, Rhop (no powder on a bare plate: zero).
usage: sensitivity_report.py RUNDIR
"""
import csv, json, math, os, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "post")); import analyze_case as A

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "legend.frameon": False})
LABEL = {"k": "conductivity k", "cp": "heat capacity cp", "rho": "density ρ", "Lfus": "latent heat L", "Ts": "solidus Ts",
         "Tliq": "liquidus Tl", "hc": "convection h", "T0": "ambient T0", "P": "power P", "Eta": "absorptivity η",
         "Rx": "Goldak length a", "Ry": "Goldak width b", "Rz": "Goldak depth c", "X0": "start x0 [per mm]", "Vx": "speed v",
         "TL": "time shift [per s]", "Tw": "substrate T", "hp": "powder h", "kp": "powder k", "Rhop": "powder ρ"}
SHIFT = {"X0", "TL"}                                  # absolute shifts: no elasticity
Q = [("depth_um", "depth"), ("width_um", "width"), ("TTAM_s", "TAM"), ("TLCR_C_s", "liquid CR"), ("TTCR_C_s", "transition CR"),
     ("TSCR_C_s", "solid CR")]

run = sys.argv[1]
R = json.load(open(os.path.join(run, "result.json"))); case = R["case"]
m = R["melt_pool_mid_1298C"]; pr = R["probe"]; ps = R.get("probe_sens", {})
if not m.get("d_depth_um"): sys.exit("no sensitivities in this run (DORDER=0)")
rows, el = [], {}
for q, lbl in Q:
    if q in ("depth_um", "width_um"): v, d = m[q], m["d_" + q]
    else: v, d = pr.get(q, float("nan")), ps.get(q, {})
    el[lbl] = {}
    for p in A.PARAMS:
        dv = d.get(p, float("nan"))
        e = dv / v if (p not in SHIFT and v == v and v != 0) else float("nan")      # (dq/q) / (dp/p)
        rows.append([lbl, p, f"{v:.6g}", f"{dv:.6g}", "" if e != e else f"{e:.4g}"])
        if e == e: el[lbl][p] = e
with open(os.path.join(run, "derivatives.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "parameter", "value", "d_quantity_d_s_p", "elasticity"]); w.writerows(rows)

# ---- figure
t, T = A.read_probe(os.path.join(run, "probe.csv")); S = A.read_sens(os.path.join(run, "sens.csv"))
fig = plt.figure(figsize=(7.4, 10.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.55)
# (a) elasticity heat table: quantities x parameters (bare-plate parameters only), ranked by the largest |elasticity|
pars = [p for p in A.PARAMS if p not in SHIFT and any(abs(el[l].get(p, 0.0)) > 1e-6 for _, l in Q)]
pars.sort(key=lambda p: -max(abs(el[l].get(p, 0.0)) for _, l in Q))
ax = fig.add_subplot(gs[0])
M = [[el[l].get(p, float("nan")) for p in pars] for _, l in Q]
vmax = max(abs(x) for r in M for x in r if x == x)
im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_yticks(range(len(Q))); ax.set_yticklabels([l for _, l in Q])
ax.set_xticks(range(len(pars))); ax.set_xticklabels([LABEL[p] for p in pars], rotation=40, ha="right")
for i, r in enumerate(M):
    for j, x in enumerate(r):
        if x == x: ax.text(j, i, f"{x:+.2f}", ha="center", va="center", fontsize=6.5, color=INK if abs(x) < 0.6 * vmax else SURF)
[sp.set_visible(False) for sp in ax.spines.values()]; ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("elasticity: % change per 1 % change", fontsize=7)
ax.set_title(f"(a) case {case}: first-order sensitivities of the NIST quantities (AD), as elasticities", fontsize=8.5, loc="left", color=INK)
# (b) dT/ds_p(t) at the midpoint: small multiples, one per parameter
sub = gs[1].subgridspec(5, 4, hspace=0.75, wspace=0.35)
tm = [x * 1e3 for x in t]
cool = [i for i in range(len(T)) if T[i] < A.T_1400 and t[i] > t[T.index(max(T))]]
for n, p in enumerate(A.PARAMS):
    a2 = fig.add_subplot(sub[n // 4, n % 4])
    for hi, lo, c in ((A.T_1400, A.T_LIQ, PAL[1]), (A.T_LIQ, A.T_SOL, PAL[3]), (A.T_SOL, A.T_1150, PAL[2])):
        ts = [t[i] * 1e3 for i in cool if lo <= T[i] <= hi]
        if ts: a2.axvspan(min(ts), max(ts), color=c, alpha=0.15, lw=0)
    s = S.get(p); smax = max(max(abs(x) for x in v) for v in S.values())
    if s and max(abs(x) for x in s) > 1e-9 * smax:          # below: zero up to round-off
        a2.plot(tm, s, color=PAL[0], lw=1.2)
        a2.axhline(0, color=GRID, lw=0.6)
    else:
        a2.text(0.5, 0.5, "0 (no powder)" if p in ("hp", "kp", "Rhop") else "≈ 0", transform=a2.transAxes, ha="center", va="center", color=INK2, fontsize=7)
        a2.set_xlim(tm[0], tm[-1]); a2.set_yticks([])
    a2.set_title(LABEL[p], fontsize=7, color=INK, pad=2); a2.tick_params(labelsize=6)
    if n // 4 == 4: a2.set_xlabel("t [ms]", fontsize=6.5)
fig.text(0.02, 0.50, "(b) dT/ds_p at the track midpoint [K per unit relative change] vs time; shaded: liquid / transition / solid "
         "cooling-rate windows", fontsize=8.5, color=INK)
fig.suptitle(f"AMB2022-03 case {case}: full set of first derivatives (20 parameters)", fontsize=10, x=0.02, ha="left", color=INK)
out = os.path.join(run, f"derivatives_case{case}.png"); fig.savefig(out, dpi=170, bbox_inches="tight")
print("written", out, "and", os.path.join(run, "derivatives.csv"))
