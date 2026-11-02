"""CPU-overhead figure (extra cost of each derivative order / SINGLE-LEVEL nominal run, common reference for both), single-level vs multi-level.  Single-level split recovered from the reference figures
(overhead 0.998x / 1.78x with thermal 82 % / 53 %, totals 12.3 h / 23.3 h, thermal 80 % / 67 %).  Multi-level: thermal parts divided by the measured thermal speed-up (6145 s / 927 s), mechanical unchanged.  usage: fig_overhead_ref.py OUT"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]; ST = 6145.0 / 927.0
tot1, tot2 = 12.3, 23.3; th1_tot, th2_tot = 0.80 * tot1, 0.67 * tot2
o1, o2, f1, f2 = 0.998, 1.78, 0.82, 0.53          # overheads and thermal share of each extra
N = ((tot2 - tot1) / o2 + tot1 / (1 + o1)) / 2    # nominal run [h], two consistent estimates averaged
e1 = (f1 * o1 * N, (1 - f1) * o1 * N); e2 = (f2 * o2 * N, (1 - f2) * o2 * N)   # extra [h] (thermal, mechanical)
th0 = th1_tot - e1[0]; me0 = tot1 * 0.20 - e1[1]                                 # nominal split [h]
M0 = th0 / ST + me0; E1 = (e1[0] / ST, e1[1]); E2 = (e2[0] / ST, e2[1])
S = [(e1[0] / N, e1[1] / N), (e2[0] / N, e2[1] / N)]; M = [(E1[0] / N, E1[1] / N), (E2[0] / N, E2[1] / N)]   # same reference for both: the single-level nominal run
plt.rcParams.update({"font.size": 20, "axes.linewidth": 2}); ORANGE, BLUE = "#f28e2b", "#2b7bba"; ymin = 1e-2
def stacked(ax, x, th, me, col, lt, lm):
    tot = th + me; b = np.exp(np.log(ymin) + (np.log(tot) - np.log(ymin)) * th / tot)
    ax.bar(x, b - ymin, bottom=ymin, width=0.36, color=col, edgecolor="black", linewidth=3, label=lt)
    ax.bar(x, tot - b, bottom=b, width=0.36, color=col, edgecolor="black", linewidth=3, hatch="//", alpha=0.55, label=lm)
    ax.text(x, tot, f"{tot:.3g}$\\times$", ha="center", va="bottom", fontsize=15)
    ax.text(x, ymin * 1.25, "thermal %.0f %%\nmechanical %.0f %%" % (100 * th / tot, 100 * me / tot), ha="center", va="bottom", fontsize=12, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
f, ax = plt.subplots(figsize=(13, 7.2))
for g in range(2):
    stacked(ax, g - 0.2, *S[g], ORANGE, "Single-level, thermal" if g == 0 else None, "Single-level, mechanical" if g == 0 else None)
    stacked(ax, g + 0.2, *M[g], BLUE, "Multi-level, thermal" if g == 0 else None, "Multi-level, mechanical" if g == 0 else None)
ax.set_yscale("log"); ax.set_ylim(ymin, 60); ax.set_xticks([0, 1]); ax.set_xticklabels(["First-order (extra)", "Second-order (extra)"]); ax.set_ylabel("CPU overhead"); ax.tick_params(width=2, length=6)
ax.legend(loc="upper left", frameon=False, fontsize=15); f.tight_layout(); f.savefig(out + ".png", dpi=170)
print("nominal single %.2f h (th %.2f me %.2f)  multi %.2f h" % (N, th0, me0, M0)); print("single", [round(sum(s), 3) for s in S], "multi", [round(sum(m), 3) for m in M])
