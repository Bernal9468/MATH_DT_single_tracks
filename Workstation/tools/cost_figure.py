#!/usr/bin/env python3
"""Wall time of the 7-case campaign (nominal T + all first derivatives, results/thermo/case*): CPU only vs CPU + GPU.

Measured per-case wall times from run.log; the binary in settings.txt tells which worker ran the case (GPU: *gpu*).
CPU only: the 7 cases one after the other on the CPU; the cases that ran on the GPU are estimated from the mean CPU
time per step of the cases that ran on the CPU (hatched). CPU + GPU: the two workers as they ran, in parallel.
usage: cost_figure.py [RESULTS_THERMO] [OUT.png]
"""
import glob, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = sys.argv[1] if len(sys.argv) > 1 else "results/thermo"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "wall_time_cpu_vs_gpu.png")
C = []
for d in sorted(glob.glob(os.path.join(root, "case*"))):
    log = open(os.path.join(d, "run.log")).read(); st = open(os.path.join(d, "settings.txt")).read()
    C.append(dict(case=os.path.basename(d)[4:], gpu="gpu" in os.path.basename(re.search(r"bin=(.+?) gsub=", st).group(1)),
                  wall=float(re.search(r"Wall time *: *([0-9.]+)", log).group(1)),
                  steps=int(re.search(r"Accepted steps *: *(\d+)", log).group(1))))
cpu = [c for c in C if not c["gpu"]]; gpu = [c for c in C if c["gpu"]]
s_cpu = sum(c["wall"] for c in cpu) / sum(c["steps"] for c in cpu)                # CPU s/step
s_gpu = sum(c["wall"] for c in gpu) / sum(c["steps"] for c in gpu)
for c in C: c["cpu_t"] = c["wall"] if not c["gpu"] else s_cpu * c["steps"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.2})
COL = {"cpu": "#3b6fb6", "gpu": "#e07b39"}
fig, ax = plt.subplots(figsize=(14, 4.6))
lanes = [("CPU only", [(c, c["cpu_t"], "cpu", c["gpu"]) for c in C]),
         ("CPU + GPU:\nCPU worker", [(c, c["wall"], "cpu", False) for c in cpu]),
         ("CPU + GPU:\nGPU worker", [(c, c["wall"], "gpu", False) for c in gpu])]
Y = [2.6, 1.2, 0.2]; tot = []
for y, (name, segs) in zip(Y, lanes):
    x = 0.0
    for c, t, k, est in segs:
        h = t / 3600
        ax.barh(y, h, left=x, height=0.72, color=COL[k], ec="white", lw=2, hatch="//" if est else None,
                alpha=0.55 if est else 1.0)
        if h > 0.75: ax.text(x + h / 2, y, f"{c['case']}\n{t / 60:.0f} min", ha="center", va="center", fontsize=10.5,
                             color="white" if not est else "black")
        else: ax.text(x + h / 2, y, f"{c['case']}\n{t / 60:.0f}′", ha="center", va="center", fontsize=10, color="white")
        x += h
    tot.append(x)
wall_cg = max(tot[1], tot[2])
ax.text(tot[0] + 0.08, Y[0], f"{tot[0]:.1f} h", va="center", fontsize=14, fontweight="bold")
ax.text(wall_cg + 0.08, 0.7, f"{wall_cg:.1f} h wall\n({tot[0] / wall_cg:.2f}× faster)", va="center", fontsize=14,
        fontweight="bold")
ax.plot([wall_cg] * 2, [-0.18, 1.65], color="k", lw=1.2, ls=(0, (3, 2)))
ax.set_yticks(Y); ax.set_yticklabels([l for l, _ in lanes]); ax.set_ylim(-0.75, 3.15)
ax.set_xlim(0, tot[0] * 1.13); ax.set_xlabel("wall time [h]")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.8); ax.set_axisbelow(True)
ax.legend(handles=[Patch(color=COL["cpu"], label=f"CPU, measured ({s_cpu:.1f} s/step)"),
                   Patch(fc=COL["cpu"], alpha=0.55, hatch="//", ec="white", label="CPU, estimated from the CPU s/step"),
                   Patch(color=COL["gpu"], label=f"GPU, measured ({s_gpu:.1f} s/step)")],
          loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False, fontsize=12)
fig.text(0.01, -0.06, f"7 cases, 20 µm mesh (323,610 nodes), 5 µs steps, nominal T + all first derivatives (Enzyme forward "
         f"mode). Xeon W-2265 (CPU worker: 6 threads) + Quadro RTX 4000. CPU + GPU: both workers at the same time.",
         fontsize=10.5, color="#444444")
# best split of the same 7 cases over the two workers (GPU time of a CPU case from the GPU s/step)
gt = [c["wall"] if c["gpu"] else s_gpu * c["steps"] for c in C]
best = min((max(sum(C[i]["cpu_t"] for i in range(len(C)) if not m >> i & 1), sum(gt[i] for i in range(len(C)) if m >> i & 1)), m)
           for m in range(1 << len(C)))
ax.text(0.0, Y[2] - 0.62, f"(′ = min)   GPU idle for {(wall_cg - tot[2]):.1f} h; a balanced split (GPU: "
        + ", ".join(C[i]["case"] for i in range(len(C)) if best[1] >> i & 1) + f") would take ≈ {best[0] / 3600:.1f} h",
        va="center", ha="left", fontsize=11, color="#444444")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out, f"CPU only {tot[0]:.2f} h, CPU+GPU {wall_cg:.2f} h")
