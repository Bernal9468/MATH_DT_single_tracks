#!/usr/bin/env python3
"""All 7 AMB2022-03 cases together: nominal response vs NIST and the full set of first derivatives (runs of
run/thermo_fields.sh, each post-processed by post/analyze_case.py and tools/sensitivity_report.py).

Outputs in DIR (default results/thermo):
  summary.csv          model, NIST and error (%) of depth, width, TAM, LCR, TCR, SCR per case (post/summarize.py)
  derivatives_all.csv  the derivatives.csv of every case, with a case column
  summary_thermo.png   (a) model error vs NIST, cases x quantities, with the RMS per quantity;
                       (b) elasticities of the six quantities, cases x parameters (one small table per quantity)
usage: thermo_summary.py [DIR]
"""
import csv, math, os, subprocess, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "thermo")
CASES = ["0", "1.1", "1.2", "2.1", "2.2", "3.1", "3.2"]
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
done = [c for c in CASES if os.path.exists(os.path.join(D, f"case{c}", "derivatives.csv"))]
if not done: sys.exit("no finished cases in " + D)

subprocess.run([sys.executable, os.path.join(ROOT, "post", "summarize.py"), os.path.join(D, "summary.csv")]
               + [f"{c}={os.path.join(D, 'case' + c)}" for c in done], check=True)
S = {r["case"]: r for r in csv.DictReader(open(os.path.join(D, "summary.csv")))}
Q = [("depth_um", "depth"), ("width_um", "width"), ("TTAM_s", "TAM"), ("TLCR_C_s", "liquid CR"), ("TTCR_C_s", "transition CR"),
     ("TSCR_C_s", "solid CR")]

E, rows = {}, []                         # E[case][quantity label][parameter] = elasticity
for c in done:
    E[c] = {}
    for r in csv.DictReader(open(os.path.join(D, f"case{c}", "derivatives.csv"))):
        rows.append([c] + [r[k] for k in ("quantity", "parameter", "value", "d_quantity_d_s_p", "elasticity")])
        if r["elasticity"]: E[c].setdefault(r["quantity"], {})[r["parameter"]] = float(r["elasticity"])
with open(os.path.join(D, "derivatives_all.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "quantity", "parameter", "value", "d_quantity_d_s_p", "elasticity"]); w.writerows(rows)

def label(c):
    r = S[c]; return f"{c}  ({r['power_W']} W, {r['speed_mm_s']} mm/s, {r['spot_um']} µm)"

fig = plt.figure(figsize=(8.2, 11.0))
gs = fig.add_gridspec(2, 1, height_ratios=[0.75, 2.0], hspace=0.42)
# (a) errors
ax = fig.add_subplot(gs[0])
M = [[float(S[c][f"{q}_err_pct"]) for q, _ in Q] for c in done]
rms = [math.sqrt(sum(M[i][j] ** 2 for i in range(len(done))) / len(done)) for j in range(len(Q))]
im = ax.imshow(M, cmap="RdBu_r", vmin=-60, vmax=60, aspect="auto")
ax.set_xticks(range(len(Q))); ax.set_xticklabels([f"{l}\nRMS {e:.1f} %" for (_, l), e in zip(Q, rms)])
ax.set_yticks(range(len(done))); ax.set_yticklabels([label(c) for c in done])
for i, r in enumerate(M):
    for j, x in enumerate(r): ax.text(j, i, f"{x:+.1f}", ha="center", va="center", fontsize=7.5, color=INK if abs(x) < 35 else SURF)
[sp.set_visible(False) for sp in ax.spines.values()]; ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("model − NIST [%]", fontsize=7)
ax.set_title("(a) nominal response vs NIST (case 0: calibration; others: prediction)", loc="left", fontsize=9, color=INK)
# (b) elasticities: one table per quantity, parameters ranked by the largest |elasticity| over all cases and quantities
pars = sorted({p for c in done for l in E[c] for p, v in E[c][l].items() if abs(v) > 0.05},
              key=lambda p: -max(abs(E[c].get(l, {}).get(p, 0)) for c in done for _, l in Q))
NAME = {"k": "k", "cp": "cp", "rho": "ρ", "Lfus": "L", "Ts": "Ts", "Tliq": "Tl", "hc": "h", "T0": "T0", "P": "P", "Eta": "η",
        "Rx": "a", "Ry": "b", "Rz": "c", "Vx": "v", "Tw": "Tsub"}
sub = gs[1].subgridspec(3, 2, hspace=0.45, wspace=0.12)
for n, (_, l) in enumerate(Q):
    a2 = fig.add_subplot(sub[n // 2, n % 2])
    if n == 0: ytop = a2.get_position().y1
    X = [[E[c].get(l, {}).get(p, float("nan")) for p in pars] for c in done]
    vm = max(abs(x) for r in X for x in r if x == x)
    a2.imshow(X, cmap="RdBu_r", vmin=-vm, vmax=vm, aspect="auto")
    a2.set_xticks(range(len(pars))); a2.set_xticklabels([NAME.get(p, p) for p in pars], fontsize=7)
    a2.set_yticks(range(len(done))); a2.set_yticklabels(done if n % 2 == 0 else [], fontsize=7)
    for i, r in enumerate(X):
        for j, x in enumerate(r):
            if x == x: a2.text(j, i, f"{x:.1f}" if abs(x) >= 0.95 else f"{x:.2f}".replace("0.", "."), ha="center", va="center",
                               fontsize=5.2, color=INK if abs(x) < 0.6 * vm else SURF)
    [sp.set_visible(False) for sp in a2.spines.values()]; a2.tick_params(length=0)
    a2.set_title(f"{l}   (colour scale ±{vm:.1f})", fontsize=8, loc="left", color=INK)
fig.text(0.02, ytop + 0.03, "(b) elasticities (% change of the quantity per 1 % change of the parameter), AD, rows = cases; "
         "a, b, c: Goldak semi-axes;\n     X0, TL (shifts) and the powder parameters (zero on a bare plate) are in derivatives_all.csv",
         fontsize=8.5, color=INK, va="bottom")
fig.suptitle(f"AMB2022-03, IN718 bare plate: {len(done)} cases, nominal response and the full set of first derivatives",
             fontsize=10, x=0.02, ha="left", color=INK)
out = os.path.join(D, "summary_thermo.png"); fig.savefig(out, dpi=170, bbox_inches="tight")
print("written", out, os.path.join(D, "summary.csv"), os.path.join(D, "derivatives_all.csv"))
