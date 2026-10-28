#!/usr/bin/env python3
"""Wall time of the mechanical stage of one single track (case 0), CPU program vs GPU program, split by solver phase.

Runs: run/run_mech.sh cpu|gpu (results/mech/case0_h40_{cpu,gpu}); phases from the 'timers [s]' line of log.txt
(mech_main_aa.cpp), the wall time from the 'exit=... wall=...' line; 'other' = wall - the timed phases.
usage: cost_figure_mech.py [CPU_RUN GPU_RUN OUT.png]
"""
import os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

cpu_d, gpu_d = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("results/mech/case0_h40_cpu", "results/mech/case0_h40_gpu")
out = sys.argv[3] if len(sys.argv) > 3 else "results/mech/wall_time_mech_cpu_vs_gpu.png"
# phase: (label, colour, timer keys) -- colours as in cost_figure.py
PH = [("tangent + residual assembly", "#2a77d4", ["newton_asm"]),
      ("Newton solve", "#ee6a3a", ["newton_pardiso"]),
      ("sensitivities", "#1db07c", ["jets", "d_rhs", "d_pardiso", "d_update"]),
      ("other (I/O, set-up)", "#b4b4b4", [])]


def phases(d):
    L = open(os.path.join(d, "log.txt")).read()
    S = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", re.search(r"timers \[s\]:(.*)", L).group(1))}
    v = np.array([sum(S.get(k, 0.0) for k in keys) for _, _, keys in PH])
    wall = float(re.search(r"exit=0 wall=([\d.]+)", L).group(1)); v[-1] = wall - v[:-1].sum()
    mesh = re.search(r"mesh: (\d+) elements (\d+) nodes", L).groups(); dofs = re.search(r"dofs (\d+)", L).group(1)
    return dict(ph=v, wall=wall, mesh=mesh, dofs=int(dofs), nsamp=int(re.search(r"(\d+) thermal samples", L).group(1)),
                newton=int(re.search(r"newton iterations (\d+)", L).group(1)))


C, G = phases(cpu_d), phases(gpu_d)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.2})
fig, ax = plt.subplots(figsize=(7.5, 6.4))
ax.set_ylim(0, C["wall"] / 60 * 1.18)
for x, r in [(0, C), (1, G)]:
    b = 0.0
    for (name, col, _), s in zip(PH, r["ph"] / 60):
        ax.bar(x, s, 0.62, bottom=b, color=col, ec="white", lw=1.5)
        if s > 0.045 * ax.get_ylim()[1]:
            ax.text(x, b + s / 2, f"{s:.0f}" if s >= 5 else f"{s:.1f}", ha="center", va="center", fontsize=10.5,
                    color="white" if col != "#b4b4b4" else "#333333")
        b += s
    sp = C["wall"] / r["wall"]
    ax.text(x, b + 1.0, f"{b:.0f} min" + ((f"\n({sp:.1f}× faster)" if sp >= 1.05 else f"\n({sp:.2f}×: no speed-up)") if x else ""), ha="center",
            va="bottom", fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["CPU only\n(CPU program)", "CPU + GPU\n(GPU program)"]); ax.set_xlim(-0.6, 1.6)
ax.set_ylabel("wall time [min]"); ax.set_title("One single track run: mechanical stage", loc="left", fontsize=13)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.legend(handles=[Patch(color=c, label=n) for n, c, _ in PH], loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2,
           frameon=False, fontsize=12)
note = (
    f"Case 0, residual stress (IN718, Hill thermo-plasticity, small strain hex8 B-bar, bottom face fixed) + its 20 first derivatives\n"
    f"w.r.t. the thermal parameters; {C['mesh'][0]} elements, {C['mesh'][1]} nodes ({C['dofs']:,} unknowns, 40 µm mesh), "
    f"{C['nsamp']} thermal dumps (every 10 thermal steps).\n"
    "CPU program (mech_main_aa_src): every phase on the CPU, 12 threads.  GPU program (mech_main_aa_gpu): element assembly,\n"
    "return map, derivative right-hand sides and updates on the Quadro RTX 4000; the linear solves (PARDISO) stay on the CPU.\n"
    "Material: Special Metals INCONEL 718 datasheet SMC-045 (E, ν, CTE vs T; annealed yield 433 MPa, linear hardening).")
fig.text(0.01, -0.02, note, ha="left", va="top", fontsize=10, color="#444444", linespacing=1.45)
fig.savefig(out, dpi=170, bbox_inches="tight")
print("written", out, f"| CPU {C['wall'] / 60:.1f} min ({C['newton']} Newton), GPU {G['wall'] / 60:.1f} min ({G['newton']} Newton)")
for (n, _, _), a, b in zip(PH, C["ph"], G["ph"]): print(f"  {n:28s} CPU {a / 60:6.1f} min  GPU {b / 60:6.1f} min")
