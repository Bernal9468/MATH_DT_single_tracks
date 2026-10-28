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
      ("residual + Newton solve", "#ee6a3a", ["bdry", "cap", "tie", "pard", "src"]),
      ("sensitivities", "#1db07c", ["a", "b", "c", "d", "e", "f", "fresh", "solve"]),
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
# GPU only: the 7 cases one after the other with the GPU program; measured = the GPU worker's cases + results/gpu_only
G = {c["case"]: c for c in gpu}
for d in glob.glob(os.path.join(res, "gpu_only", "case*")):
    try: G.setdefault(os.path.basename(d)[4:], phases(d))
    except (AttributeError, FileNotFoundError): pass                        # run not finished
g_step = sum(c["ph"] for c in G.values()) / sum(c["steps"] for c in G.values())
g_est = [c["case"] for c in C if c["case"] not in G]
gpu_only = sum(G[c["case"]]["ph"] if c["case"] in G else g_step * c["steps"] for c in C)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.2})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1, 1.7], "wspace": 0.22})


def stack(ax, x, v, scale, w=0.62, b=0.0):
    tot = v.sum() * scale
    for (name, col, _), s in zip(PH, v * scale):
        ax.bar(x, s, w, bottom=b, color=col, ec="white", lw=1.5)
        if s > 0.045 * ax.get_ylim()[1]:
            ax.text(x, b + s / 2, f"{s:.0f}" if s >= 5 and scale > 1e-3 else f"{s:.1f}", ha="center", va="center", fontsize=10.5,
                    color="white" if col != "#b4b4b4" else "#333333")
        b += s
    return tot


# (a) case 0 on each program, minutes
ax1.set_ylim(0, A_cpu["wall"] / 60 * 1.15)
for x, c in [(0, A_cpu), (1, A_gpu)]:
    t = stack(ax1, x, c["ph"], 1 / 60)
    ax1.text(x, t + 1.2, f"{t:.0f} min" + (f"\n({A_cpu['wall'] / c['wall']:.1f}× faster)" if x else ""), ha="center",
             va="bottom", fontsize=13, fontweight="bold")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["CPU only\n(CPU program)", "CPU + GPU\n(GPU program)"])
ax1.set_ylabel("wall time [min]"); ax1.set_title("(a) One single track run", loc="left", fontsize=13)

# (b) campaign, hours
ax2.set_ylim(0, cpu_only.sum() / 3600 * 1.15)
t0 = stack(ax2, 0, cpu_only, 1 / 3600)
t1 = stack(ax2, 1.3, cpu_w, 1 / 3600); t2 = stack(ax2, 2.3, gpu_w, 1 / 3600); t12 = t1 + t2
t3 = stack(ax2, 3.7, gpu_only, 1 / 3600)
ax2.text(0, t0 + 0.2, f"{t0:.1f} h", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax2.text(1.3, t1 + 0.2, f"{t1:.1f} h", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax2.text(2.3, t2 + 0.2, f"{t2:.1f} h", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax2.text(1.8, t1 + 1.1, f"CPU + GPU: {t1:.1f} + {t2:.1f} = {t12:.1f} h\n({t0 / t12:.1f}× faster)", ha="center", va="bottom",
         fontsize=13, fontweight="bold")
ax2.text(3.7, t3 + 0.2, f"{t3:.1f} h\n({t0 / t3:.1f}× faster)", ha="center", va="bottom", fontsize=13, fontweight="bold")
ax2.set_xticks([0, 1.3, 2.3, 3.7])
ax2.set_xticklabels(["CPU only\n(7 cases)", "CPU worker\n(" + ", ".join(c["case"] for c in cpu) + ")",
                     "GPU worker\n(" + ", ".join(c["case"] for c in gpu) + ")",
                     "GPU only\n(7 cases)"])
ax2.text(1.8, -0.155, "└─── CPU + GPU (7 cases) ───┘", transform=ax2.get_xaxis_transform(), ha="center", fontsize=13)
ax2.set_xlim(-0.6, 4.3); ax2.set_ylabel("total run time, sum of the cases [h]")
ax2.set_title("(b) Total single tracks run", loc="left", fontsize=13)
for a in (ax1, ax2):
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.legend(handles=[Patch(color=c, label=n) for n, c, _ in PH], loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2,
           frameon=False, fontsize=12)
note = (
    "CPU program (bin/thermal_in718_cpu) = CPU only: every phase runs on the CPU cores.\n"
    "GPU program (bin/thermal_in718_gpu) = CPU + GPU: tangent assembly, laser heat input and the Enzyme derivative kernels run on the Quadro RTX 4000 GPU; "
    "the linear solves\n(Newton: CG + BoomerAMG; sensitivities: Jacobi-BiCGSTAB) stay on the CPU.  Same model, same equations and same results; "
    "only where the work is done differs.\n"
    "(a) case 0 run once with each program, both measured (the GPU run is the video re-run; its field dumps every 2 steps are inside 'other').\n"
    "(b) bars = sum of the run times of the 7 cases. CPU + GPU: two workers, each taking the next free case from the list: CPU worker = CPU program, 6 threads "
    "(logical CPUs 12-17);\nGPU worker = GPU program, 12 threads (logical CPUs 0-11) + the GPU; they ran at the same time, so the elapsed clock time was "
    f"{max(t1, t2):.1f} h ({t0 / max(t1, t2):.2f}× less than CPU only).\nCPU only = all 7 cases one after the other with the CPU "
    f"program; {', '.join(c['case'] for c in gpu)} never ran on the CPU,\nso their times are estimated from the per-step CPU phase times of the other cases.  "
    "Xeon W-2265, 12 cores x 2 threads: logical CPUs 12-17 are the second threads of cores 0-5,\nwhich the GPU worker also used, so the CPU times are "
    "pessimistic.  GPU only = all 7 cases one after the other with the GPU program, 12 threads + the GPU "
    f"({'estimated for ' + ', '.join(g_est) + ' from the per-step GPU phase times' if g_est else 'each case measured; ' + ', '.join(sorted(set(G) - set(c['case'] for c in gpu))) + ' re-run alone on the GPU program'}).\n"
    "Thermal analysis only: nominal T + 20 first derivatives per track (no mechanics).  20 µm mesh (323,610 nodes), 5 µs steps; phase times from the per-step timers in run.log.")
fig.text(0.01, -0.075, note, ha="left", va="top", fontsize=10, color="#444444", linespacing=1.45)
fig.savefig(out, dpi=170, bbox_inches="tight")
print("written", out, f"| case 0: CPU {A_cpu['wall'] / 60:.1f} min, GPU {A_gpu['wall'] / 60:.1f} min | campaign: CPU only "
      f"{t0:.2f} h, CPU worker {t1:.2f} h, GPU worker {t2:.2f} h")
for n, a, b, c, d in zip([p[0] for p in PH], A_cpu["ph"], A_gpu["ph"], cpu_w, gpu_w):
    print(f"  {n:34s} case0 CPU {a / 60:6.1f} min  GPU {b / 60:6.1f} min | CPU worker {c / 3600:5.2f} h  GPU worker {d / 3600:5.2f} h")
