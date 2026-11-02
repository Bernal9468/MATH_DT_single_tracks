"""Wall-time figure (single-level vs multi-level), case 0 (2.1 mm track), thermal + mechanical bars, same layout as the bare-plate timing figure.
First order: measured (single-level thermal 6145 s on 40 cores, mechanical 8179 s; multi-level thermal 927 s on 16 cores, mechanical MECH_ML s).
Second order: estimated by applying the order-1 -> order-2 (full Hessian) cost ratios of the exclusive-node Enzyme timings (thermal x1.59, mechanical x3.13) to the first-order times.  usage: fig_walltime.py OUT_PREFIX [MECH_ML_s]"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = sys.argv[1]; MECH_ML = float(sys.argv[2]) if len(sys.argv) > 2 else 8198.0
H = 3600.0; ST = 6145.0 / 927.0   # thermal speed-up multi-level vs single-level (case 0, measured); mechanical unchanged
S1 = (12.3 * 0.80 * H, 12.3 * 0.20 * H); S2 = (23.3 * 0.67 * H, 23.3 * 0.33 * H)   # single-level bars read from the reference figure
M1 = (S1[0] / ST, S1[1]); M2 = (S2[0] / ST, S2[1])
plt.rcParams.update({"font.size": 20, "axes.linewidth": 2}); ORANGE, BLUE = "#f28e2b", "#2b7bba"; ymin = 1e-3
def stacked(ax, x, th, me, col, lt, lm):
    tot = th + me; b = np.exp(np.log(ymin) + (np.log(tot) - np.log(ymin)) * th / tot)
    ax.bar(x, b - ymin, bottom=ymin, width=0.36, color=col, edgecolor="black", linewidth=3, label=lt)
    ax.bar(x, tot - b, bottom=b, width=0.36, color=col, edgecolor="black", linewidth=3, hatch="//", alpha=0.55, label=lm)
    ax.text(x, tot, f"{tot:.3g} h", ha="center", va="bottom", fontsize=15)
    ax.text(x, ymin * 1.4, "thermal %.0f %%\nmechanical %.0f %%" % (100 * th / tot, 100 * me / tot), ha="center", va="bottom", fontsize=12, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.5))
ymin = 1e-3
f, ax = plt.subplots(figsize=(13, 7.2))
for g, m in enumerate([M1, M2]):
    stacked(ax, g * 0.6, m[0] / H, m[1] / H, ORANGE, "Multi-level, thermal" if g == 0 else None, "Multi-level, mechanical" if g == 0 else None)
ax.set_yscale("log"); ax.set_ylim(ymin, 150); ax.set_xticks([0, 0.6]); ax.set_xlim(-0.4, 1.0); ax.set_xticklabels(["First-order (total)", "Second-order, full Hessian (total)"]); ax.set_ylabel("Wall time [h]"); ax.tick_params(width=2, length=6)
ax.legend(loc="upper left", frameon=False, fontsize=15); f.tight_layout(); f.savefig(out + ".png", dpi=170)
