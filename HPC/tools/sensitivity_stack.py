#!/usr/bin/env python3
"""Cumulative ranked sensitivities of the temperature at the track midpoint: at each time, the share of every parameter in
sum_p |dT/ds_p|, stacked (largest time-averaged share at the bottom), with the nominal T(t) on a second axis.

s_p is the relative change of parameter p. The two shift parameters are put on the same footing: X0 [per mm] is scaled by
the distance from the start to the midpoint, TL [per s] by the laser arrival time at the midpoint (both then mean "a 1 x
relative shift", and they are physically the same effect). Parameters that are zero on a bare plate (convection, powder)
are left out. P and η enter only as their product, so their bands are equal.
usage: sensitivity_stack.py RUNDIR [--out FIG.png]
"""
import argparse, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "post")); import analyze_case as A

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
LABEL = {"k": "Thermal conductivity", "cp": "Specific heat", "rho": "Density", "Lfus": "Latent heat", "Ts": "Solidus temperature",
         "Tliq": "Liquidus temperature", "T0": "Ambient temperature", "P": "Laser power", "Eta": "Absorptivity",
         "Rx": "Beam radius x", "Ry": "Beam radius y", "Rz": "Beam depth z", "X0": "Laser start position", "Vx": "Scan speed",
         "TL": "Laser start time", "Tw": "Substrate temperature"}
COL = {"Eta": "#f0a35e", "P": "#a5532a", "Tliq": "#e39b00", "Ts": "#ff3b3b", "k": "#2d6e9e", "TL": "#6a3d9a", "Rz": "#4cae4c",
       "cp": "#2a8a2a", "X0": "#8c8c8c", "Vx": "#bcbd22", "Tw": "#17a2d6", "rho": "#c81e2e", "Ry": "#3f7fc0",
       "Rx": "#d62728", "Lfus": "#22c22a", "T0": "#e040e0"}

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out"); a = ap.parse_args()
st = open(os.path.join(a.run, "settings.txt")).read()
case = re.search(r"case=(\S+)", st).group(1); x0 = float(re.search(r"x0=(\S+)", st).group(1))
lp = [l.split(",") for l in open(os.path.join(a.run, "laser.inp")) if l.strip() and not l.startswith("#")]
v = abs(float(lp[1][1]) - float(lp[0][1])) / (float(lp[1][0]) - float(lp[0][0]))      # [mm/s]
t, T = A.read_probe(os.path.join(a.run, "probe.csv")); S = A.read_sens(os.path.join(a.run, "sens.csv"))
t = np.array(t); T = np.array(T)
scale = {"X0": abs(x0), "TL": abs(x0) / v}                    # midpoint at x = 0: distance [mm] and arrival time [s]
pars = [p for p in A.PARAMS if p in LABEL and np.max(np.abs(S[p])) > 1e-9 * np.max(np.abs(S["P"]))]
M = np.array([np.abs(np.array(S[p])) * scale.get(p, 1.0) for p in pars])
tot = M.sum(0); share = np.divide(M, tot, out=np.zeros_like(M), where=tot > 0)
order = np.argsort(-share.mean(1)); pars = [pars[i] for i in order]; share = share[order]

fig, ax = plt.subplots(figsize=(8.6, 5.6))
tm = t * 1e3
ax.stackplot(tm, share * 100, colors=[COL[p] for p in pars], linewidth=0)
cum = np.vstack([np.zeros(len(t)), np.cumsum(share, 0)]) * 100
small, placed = [], []                                         # labels: thickest free spot away from the edges and the others
inner = (tm > tm[0] + 0.1 * (tm[-1] - tm[0])) & (tm < tm[-1] - 0.1 * (tm[-1] - tm[0]))
for i in np.argsort(-(share * inner).max(1)):
    p = pars[i]; raw = cum[i + 1] - cum[i]; hw = max(1, int(0.12 / np.median(np.diff(tm))))       # band thickness over the label width
    th = np.array([raw[max(0, j - hw):j + hw + 1].min() for j in range(len(raw))]) * inner
    for j in np.argsort(-th):
        if th[j] < 6: small.append(p); break
        y = 0.5 * (cum[i, j] + cum[i + 1, j])
        if all(abs(tm[j] - u) > 0.32 * (tm[-1] - tm[0]) / 2.6 or abs(y - w) > 5 for u, w in placed):
            placed.append((tm[j], y))
            ax.text(tm[j], y, LABEL[p], ha="center", va="center", fontsize=7, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)); break
ax.set_xlim(tm[0], tm[-1]); ax.set_ylim(0, 100); ax.set_xlabel("t [ms]")
ax.set_ylabel("Cumulative ranked sensitivities [%]")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter())
a2 = ax.twinx(); a2.plot(tm, T, color=INK, lw=2.4); a2.set_ylabel("T [K]")
a2.set_ylim(300, max(2400, 1.05 * T.max()))
for s in ax.spines.values(): s.set_linestyle("--"); s.set_color(INK2)
ton = (0 - x0) / v * 1e3
a2.axhline(A.T_LIQ, color=INK2, lw=0.6, ls=":"); a2.axhline(A.T_SOL, color=INK2, lw=0.6, ls=":")
if small:
    hs = [matplotlib.patches.Patch(color=COL[p], label=LABEL[p]) for p in small]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=min(4, len(hs)), frameon=False, fontsize=7,
              title="not labelled (thin bands)", title_fontsize=7)
R = json.load(open(os.path.join(a.run, "result.json"))) if os.path.exists(os.path.join(a.run, "result.json")) else {}
ax.set_title(f"Thermal field, track midpoint (x = y = z = 0), AMB2022-03 case {case}: share of each parameter in Σ|dT/ds_p| "
             f"(Enzyme AD)\nlaser over the midpoint at t = {ton:.2f} ms; dotted: liquidus / solidus; peak T {T.max():.0f} K",
             fontsize=8.5, loc="left", color=INK)
out = a.out or os.path.join(a.run, f"sensitivity_stack_case{case}.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out)
