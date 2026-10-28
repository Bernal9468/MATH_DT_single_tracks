#!/usr/bin/env python3
"""Wall time of the nominal T + all first derivatives runs, CPU only vs CPU + GPU, split by solver phase.

(a) the same case run with both programs: results/thermo/case0 (CPU worker) and results/video/case0 (GPU program,
    dense field dumps), (b) the 7-case campaign results/thermo/case*: CPU only (the 7 cases one after the other on the
    CPU; the cases that ran on the GPU are estimated from the per-step CPU phase times of the cases that ran on the CPU)
    and CPU + GPU (the two workers as they ran, in parallel).
Phases from the per-step timers of run.log ([timers ms], [timers2 ms], [timers3 ms]); 'other' = wall - the timed phases.
usage: cost_figure.py [RESULTS_DIR] [OUT.png]   (default results, OUT = results/thermo/wall_time_cpu_vs_gpu.png)
"""
import glob, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

res = sys.argv[1] if len(sys.argv) > 1 else "results"
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(res, "thermo", "wall_time_cpu_vs_gpu.png")
# phase: (label, colour, timer keys)
PH = [("tangent assembly", "#2a77d4", ["kasm"]),
      ("residual + Newton linear solve", "#ee6a3a", ["bdry", "cap", "tie", "pard"]),
      ("laser source (nominal)", "#d6589b", ["src"]),
      ("sensitivities (20 derivatives: AD terms + linear solves)", "#1db07c", ["a", "b", "c", "d", "e", "f", "fresh", "solve"]),
      ("other (I/O, probes, set-up)", "#b4b4b4", [])]


def phases(d):
    L = open(os.path.join(d, "run.log")).read(); S = {}
    pats = [(r"\[timers ms\] kasm=(\d+)\(x\d+\) bdry=(\d+) cap=(\d+) tie=(\d+) pardiso=(\d+)", "kasm bdry cap tie pard"),
            (r"\[timers2 ms\] .*? sens: a=(\d+) b=(\d+) c=(\d+) d=(\d+) e=(\d+) f=(\d+) prep=\d+ fresh=(\d+) solve=(\d+)",
             "a b c d e f fresh solve"), (r"\[timers3 ms\] src_nominal=([\d.]+)", "src")]
    for p, keys in pats:
        for m in re.finditer(p, L):
            for k, v in zip(keys.split(), m.groups()): S[k] = S.get(k, 0.0) + float(v) / 1e3
    st = open(os.path.join(d, "settings.txt")).read()
    v = np.array([sum(S.get(k, 0.0) for k in keys) for _, _, keys in PH])
    wall = float(re.search(r"Wall time *: *([0-9.]+)", L).group(1)); v[-1] = wall - v[:-1].sum()
    return dict(case=os.path.basename(d)[4:], ph=v, wall=wall, steps=int(re.search(r"Accepted steps *: *(\d+)", L).group(1)),
                gpu="gpu" in os.path.basename(re.search(r"bin=(.+?) gsub=", st).group(1)))


C = [phases(d) for d in sorted(glob.glob(os.path.join(res, "thermo", "case*")))]
cpu = [c for c in C if not c["gpu"]]; gpu = [c for c in C if c["gpu"]]
per_step = sum(c["ph"] for c in cpu) / sum(c["steps"] for c in cpu)              # CPU phase times per step [s]
cpu_only = sum(c["ph"] if not c["gpu"] else per_step * c["steps"] for c in C)
cpu_w, gpu_w = sum(c["ph"] for c in cpu), sum(c["ph"] for c in gpu)
A_cpu = next(c for c in C if c["case"] == "0"); A_gpu = phases(os.path.join(res, "video", "case0"))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.2})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.22})


def stack(ax, x, v, scale, w=0.62):
    b = 0.0; tot = v.sum() * scale
    for (name, col, _), s in zip(PH, v * scale):
        ax.bar(x, s, w, bottom=b, color=col, ec="white", lw=1.5)
        if s > 0.045 * ax.get_ylim()[1]:
            ax.text(x, b + s / 2, f"{s:.0f}" if s >= 5 else f"{s:.1f}", ha="center", va="center", fontsize=10.5,
                    color="white" if col != "#b4b4b4" else "#333333")
        b += s
    return tot


# (a) case 0 on each program, minutes
ax1.set_ylim(0, A_cpu["wall"] / 60 * 1.15)
for x, c in [(0, A_cpu), (1, A_gpu)]:
    t = stack(ax1, x, c["ph"], 1 / 60)
    ax1.text(x, t + 1.2, f"{t:.0f} min", ha="center", fontsize=13, fontweight="bold")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["CPU only\n(CPU program)", "CPU + GPU\n(GPU program)"])
ax1.text(1, A_gpu["wall"] / 60 + 12, f"same case, same mesh,\nsame {A_cpu['steps']} time steps,\nsame 20 derivatives;\nonly the program differs",
         ha="center", va="bottom", fontsize=11, color="#444444")
ax1.set_ylabel("wall time [min]"); ax1.set_title(f"(a) one case (case 0) run once with each program,\n     both measured: {A_cpu['wall'] / A_gpu['wall']:.1f}× faster with the GPU", loc="left", fontsize=13)

# (b) campaign, hours
ax2.set_ylim(0, cpu_only.sum() / 3600 * 1.15)
t0 = stack(ax2, 0, cpu_only, 1 / 3600); t1 = stack(ax2, 1.4, cpu_w, 1 / 3600); t2 = stack(ax2, 2.6, gpu_w, 1 / 3600)
ax2.text(0, t0 + 0.2, f"{t0:.1f} h", ha="center", fontsize=13, fontweight="bold")
ax2.text(2.6, t2 + 0.2, f"{t2:.1f} h", ha="center", fontsize=12)
wall = max(t1, t2)
ax2.plot([1.0, 3.0], [wall, wall], color="k", lw=1.3, ls=(0, (4, 2)))
ax2.text(3.05, wall, f"{wall:.1f} h wall\n({t0 / wall:.2f}× faster)", va="center", fontsize=13, fontweight="bold")
ax2.set_xticks([0, 1.4, 2.6])
ax2.set_xticklabels(["CPU only\n(7 cases)", "CPU worker\n(" + ", ".join(c["case"] for c in cpu) + ")",
                     "GPU worker\n(" + ", ".join(c["case"] for c in gpu) + ")"])
ax2.text(2.0, -0.155, "└──── CPU + GPU, in parallel ────┘", transform=ax2.get_xaxis_transform(), ha="center", fontsize=13)
ax2.set_xlim(-0.6, 4.0); ax2.set_ylabel("wall time [h]")
ax2.set_title("(b) 7-case campaign\n     (nominal T + 20 first derivatives each)", loc="left", fontsize=13)
for a in (ax1, ax2):
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.grid(axis="y", color="#e2e2e2", lw=0.8); a.set_axisbelow(True)
fig.legend(handles=[Patch(color=c, label=n) for n, c, _ in PH], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
           frameon=False, fontsize=12)
note = (
    "CPU program (bin/thermal_in718_cpu) = CPU only: every phase runs on the CPU cores.\n"
    "GPU program (bin/thermal_in718_gpu) = CPU + GPU: tangent assembly, laser source and the Enzyme derivative kernels run on the Quadro RTX 4000 GPU; "
    "the linear solves\n(Newton: CG + BoomerAMG; sensitivities: Jacobi-BiCGSTAB) stay on the CPU.  Same model, same equations and same results; "
    "only where the work is done differs.\n"
    "(a) case 0 run once with each program, both measured (the GPU run is the video re-run; its field dumps every 2 steps are inside 'other').\n"
    "(b) the 7 cases were run by two workers at the same time, each taking the next free case from the list: CPU worker = CPU program, 6 threads "
    "(logical CPUs 12-17);\nGPU worker = GPU program, 12 threads (logical CPUs 0-11) + the GPU.  CPU only = all 7 cases one after the other with the CPU "
    f"program; {', '.join(c['case'] for c in gpu)} never ran on the CPU,\nso their times are estimated from the per-step CPU phase times of the other cases.  "
    "Xeon W-2265, 12 cores x 2 threads: logical CPUs 12-17 are the second threads of cores 0-5,\nwhich the GPU worker also used, so the CPU times are "
    "pessimistic.  20 µm mesh (323,610 nodes), 5 µs steps; phase times from the per-step timers in run.log.")
fig.text(0.01, -0.075, note, ha="left", va="top", fontsize=10, color="#444444", linespacing=1.45)
fig.savefig(out, dpi=170, bbox_inches="tight")
print("written", out, f"| case 0: CPU {A_cpu['wall'] / 60:.1f} min, GPU {A_gpu['wall'] / 60:.1f} min | campaign: CPU only "
      f"{t0:.2f} h, CPU worker {t1:.2f} h, GPU worker {t2:.2f} h")
for n, a, b, c, d in zip([p[0] for p in PH], A_cpu["ph"], A_gpu["ph"], cpu_w, gpu_w):
    print(f"  {n:34s} case0 CPU {a / 60:6.1f} min  GPU {b / 60:6.1f} min | CPU worker {c / 3600:5.2f} h  GPU worker {d / 3600:5.2f} h")
