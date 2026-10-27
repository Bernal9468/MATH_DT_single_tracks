#!/usr/bin/env python3
"""Model temperature at the track midpoint against the NIST AMB2022-03 temperature data.

Per case (RUNDIR/nist_temperature_case<c>.png): T(t) at the midpoint, model (probe x = y = z = 0) against the 3 thermography
repeats (tools/nist_thermo.py, NIST emissivity procedure), both timed from the saturation front edge passing the point (as
tools/compare_thermography.py); shaded: the NIST cooling-rate windows; table: the NIST-reported TTAM, TLCR, TTCR, TSCR
(inputs/cases.csv, challenge results) against the model.
All cases (DIR/nist_temperature_all.png, --all DIR): the four reported quantities, model vs NIST, per case, with the error.
The camera reads only between its floor (~940 C) and saturation (T_sat ~1520-1660 C, per repeat).
usage: nist_temperature.py RUNDIR [--h5 FILE] [--out FIG.png]     |     nist_temperature.py --all DIR
"""
import argparse, csv, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "post"))
import analyze_case as A

INK, INK2, GRID, SURF, MODEL = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "legend.frameon": False, "font.weight": "bold", "axes.labelweight": "bold",
                     "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
K0 = A.K0
CASES = ["0", "1.1", "1.2", "2.1", "2.2", "3.1", "3.2"]
Q = [("TTAM_s", "Time above melting TAM [ms]", 1e3), ("TLCR_C_s", "Liquid cooling rate [10⁵ K/s]", 1e-5),
     ("TTCR_C_s", "Transition cooling rate [10⁵ K/s]", 1e-5), ("TSCR_C_s", "Solid cooling rate [10⁵ K/s]", 1e-5)]
BANDS = ((1400, 1336, "#eb6834", "liquid CR"), (1336, 1260, "#eda100", "transition CR"),
         (1260, 1150, "#1baf7a", "solid CR"))
NIST = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}

ap = argparse.ArgumentParser(); ap.add_argument("run", nargs="?"); ap.add_argument("--all")
ap.add_argument("--h5", default=os.path.join(ROOT, "data", "nist_mds2-2716", "AMB2022-03-718-AMMT-StaringCamera_Signal.h5"))
ap.add_argument("--out"); ap.add_argument("--full", action="store_true", help="whole history (heating, peak, cooling) instead of the camera range")
a = ap.parse_args()


def model_values(run):
    R = json.load(open(os.path.join(run, "result.json"))); return R["probe"]


if a.all:                                    # ---------------- all cases: reported quantities
    done = [c for c in CASES if os.path.exists(os.path.join(a.all, f"case{c}", "result.json"))]
    fig, axs = plt.subplots(2, 2, figsize=(9.0, 6.6)); fig.subplots_adjust(hspace=0.38, wspace=0.25)
    x = np.arange(len(done))
    for ax, (q, lab, sc) in zip(axs.flat, Q):
        nv = np.array([float(NIST[c][q]) for c in done]) * sc
        mv = np.array([model_values(os.path.join(a.all, f"case{c}"))[q] for c in done]) * sc
        ax.vlines(x, np.minimum(nv, mv), np.maximum(nv, mv), color=GRID, lw=3, zorder=1)
        ax.scatter(x, nv, s=70, facecolor=SURF, edgecolor=INK, lw=1.8, zorder=3, label="NIST (reported)")
        ax.scatter(x, mv, s=55, color=MODEL, edgecolor=SURF, lw=1.5, zorder=4, label="model")
        top = max(nv.max(), mv.max())
        for xi, n_, m_ in zip(x, nv, mv):
            ax.text(xi, max(n_, m_) + 0.06 * top, f"{100 * (m_ - n_) / n_:+.0f}%", ha="center", va="bottom", fontsize=8,
                    color=INK2, fontweight="bold")
        ax.set_ylim(0, 1.25 * top); ax.set_xticks(x); ax.set_xticklabels(done); ax.set_xlabel("Case"); ax.set_ylabel(lab)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axs[0, 0].legend(loc="lower left", fontsize=8.5)
    out = a.out or os.path.join(a.all, "nist_temperature_all.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out); sys.exit()

# ---------------- one case: midpoint history
import nist_thermo as NT
run = a.run
case = re.search(r"case=(\S+)", open(os.path.join(run, "settings.txt")).read()).group(1)
reps = []
for r in (1, 2, 3):
    tr = NT.Track(a.h5, case, r); tr.laser_path(); tr.fit_emissivity(); reps.append(tr)
T_sat = float(np.mean([tr.t_sat() for tr in reps]))
t_p, T_p = A.read_probe(os.path.join(run, "probe.csv")); t_p = np.array(t_p); T_p = np.array(T_p) - K0
up = np.where((T_p[:-1] < T_sat) & (T_p[1:] >= T_sat))[0][0]
t_front = t_p[up] + (t_p[up + 1] - t_p[up]) * (T_sat - T_p[up]) / (T_p[up + 1] - T_p[up])

fig = plt.figure(figsize=(11.6, 5.2)); gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.1], wspace=0.05)
ax = fig.add_subplot(gs[0]); axt = fig.add_subplot(gs[1]); axt.axis("off")
for hi, lo, col, lab in BANDS: ax.axhspan(lo + K0, hi + K0, color=col, alpha=0.16, lw=0, label=lab)
for n, (tr, ls) in enumerate(zip(reps, ("--", "-.", ":"))):
    th, Th, sat = tr.history(int(round(tr.i0 + tr.u_px * np.median(tr.frames))))
    ax.plot(th * 1e3, Th + K0, color=INK2, lw=1.6, ls=ls, marker="o", ms=2.5, label=f"NIST repeat {n + 1} (ε {tr.eps:.2f})")
m_eps = re.search(r"emissivity=(\S+)", open(os.path.join(run, "run.log")).read())      # surface emissivity of the model
ax.plot((t_p - t_front) * 1e3, T_p + K0, color=MODEL, lw=2.6, label=f"model (ε {float(m_eps.group(1)):.2f})" if m_eps else "model")
ax.axhline(T_sat + K0, color=INK2, lw=0.8, ls=(0, (1, 2)))
if a.full:                                                       # whole run; the camera reads only the grey band
    xl = ((t_p[0] - t_front) * 1e3, (t_p[-1] - t_front) * 1e3)
    ax.axhspan(940 + K0, T_sat + K0, color=GRID, alpha=0.45, lw=0, zorder=0, label=f"camera range ({940 + K0:.0f} K to saturation)")
    ax.axhline(23.5 + K0, color=INK2, lw=0.6, ls=":"); ax.set_ylim(K0, 1.05 * (T_p.max() + K0))
else:
    xl = (-0.1, 2.6); ax.set_ylim(900 + K0, T_sat + K0 + 60)
ax.set_xlim(*xl)
ax.text(xl[1] - 0.02 * (xl[1] - xl[0]), T_sat + K0 + 8, f"camera saturation ≈ {T_sat + K0:.0f} K", ha="right", va="bottom", fontsize=8,
        color=INK2)
ax.set_xlabel("Time [ms]"); ax.set_ylabel("T [K]")
axt.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=8.5, bbox_to_anchor=(0.0, 1.0))
P = model_values(run); rows = []
for q, lab, sc in Q:
    n_, m_ = float(NIST[case][q]) * sc, P[q] * sc
    name, unit = lab.split(" [")
    rows.append([name.replace(" cooling rate", " CR").replace("Time above melting ", "") + "\n[" + unit,
                 f"{n_:.2f}", f"{m_:.2f}", f"{100 * (m_ - n_) / n_:+.1f} %"])
tb = axt.table(cellText=rows, colLabels=["Zone", "NIST", "model", "error"], cellLoc="center", colLoc="center",
              bbox=[0.04, 0.0, 0.96, 0.5])
tb.auto_set_font_size(False); tb.set_fontsize(8.5)
nr, nc = len(rows), 4
for (i, j), cell in tb.get_celld().items():         # outer frame only, no internal borders
    cell.visible_edges = ("T" if i == 0 else "") + ("B" if i == nr else "") + ("L" if j == 0 else "") + ("R" if j == nc - 1 else "")
    cell.set_edgecolor(INK2); cell.set_linewidth(1.0); cell.set_facecolor("none"); cell.get_text().set_fontweight("bold")
    cell.set_width((0.38, 0.19, 0.19, 0.24)[j])
fig.text(0.07, 0.0, "NIST: National Institute of Standards and Technology (AM Bench 2022, AMB2022-03).   ε: emissivity.\n"
         "TAM: time above melting, the time spent above 1571 K (midpoint of solidus 1533 K and liquidus 1609 K).\n"
         "CR: cooling rate;  liquid CR: 1673 → 1609 K,  transition CR: 1609 → 1533 K,  solid CR: 1533 → 1423 K.\n"
         "Time is counted from the camera-saturation front edge passing the track midpoint.",
         fontsize=8.5, color=INK, va="top", fontweight="normal", linespacing=1.5)
out = a.out or os.path.join(run, f"nist_temperature{'_full' if a.full else ''}_case{case}.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out)
