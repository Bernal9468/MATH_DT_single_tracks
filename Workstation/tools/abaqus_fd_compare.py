#!/usr/bin/env python3
"""MFEM Enzyme AD derivatives against Abaqus central finite differences (run/abaqus_fd.sh) for s = k, cp, rho, L.

dT/ds_p = (T(1 + H) - T(1 - H)) / (2 H) from the Abaqus replica (tools/abaqus_deck.py), s_p = relative change of parameter p.
Figures (in FDDIR):
  abaqus_fd_line_t<ms>.png : the 4 derivatives along the track centreline (y = z = 0) at time t (MFEM field dump)
  abaqus_fd_probe.png      : the 4 derivative histories at the MFEM probe node (nearest x = y = 0 on the top surface,
                             run.log 'Probe row1_top dof'), one of the 8 printed PROBE nodes in Abaqus
NRMSE = RMS(AD - FD) / (max FD - min FD); also prints the nominal T difference MFEM vs Abaqus.
usage: abaqus_fd_compare.py MFEM_RUNDIR FDDIR [--t 1.5] [--h 0.01]
"""
import argparse, csv, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import field_io as F

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("fd")
ap.add_argument("--t", type=float, nargs="+", default=[1.5], help="times of the line plots [ms]")
ap.add_argument("--h", type=float, default=0.01)
a = ap.parse_args()
PAR = [("k", "dT_dk_row1", r"$\partial T/\partial k$"), ("cp", "dT_dcp_row1", r"$\partial T/\partial c_p$"),
       ("rho", "dT_drho_row1", r"$\partial T/\partial \rho$"), ("L", "dT_dLfus_row1", r"$\partial T/\partial L$")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.6})


def read_dat(fn):
    """Abaqus NODE PRINT: {time: {node: NT11}}"""
    out, t = {}, None
    for l in open(fn):
        m = re.match(r" STEP TIME COMPLETED\s+\S+\s+TOTAL TIME COMPLETED\s+(\S+)", l)
        if m: t = round(float(m.group(1).rstrip(",")) * 1e9) / 1e9; out.setdefault(t, {}); continue
        m = re.match(r"\s+(\d+)\s+([-0-9.]+E[-+]\d+)\s*$", l)
        if m and t is not None: out[t][int(m.group(1))] = float(m.group(2))
    return out


st = open(os.path.join(a.run, "settings.txt")).read()
mesh = re.search(r"mesh=(.+?\.mesh)", st).group(1)
L = open(mesh).read().split("\n"); i = L.index("vertices"); nv = int(L[i + 1])
V = np.array([list(map(float, l.split())) for l in L[i + 3:i + 3 + nv]])
inp = open(os.path.join(a.fd, "nom.inp")).read()
probe = [int(x) for x in re.search(r"\*NSET, NSET=PROBE\n([^*]*)", inp).group(1).replace(",", " ").split()]
line = [int(x) for x in re.search(r"\*NSET, NSET=LINE\n([^*]*)", inp).group(1).replace(",", " ").split()]
pdof = int(re.search(r"Probe row1_top dof=(\d+)", open(os.path.join(a.run, "run.log")).read()).group(1))
if pdof + 1 not in probe: sys.exit(f"MFEM probe node {pdof + 1} is not in the Abaqus PROBE set")
wts, probe = [1.0], [pdof + 1]                                     # MFEM probe = the mesh node nearest x = y = 0 on top

D = {j: read_dat(os.path.join(a.fd, j + ".dat")) for j in ["nom"] + [p + s for p, _, _ in PAR for s in ("_p", "_m")]}
times = sorted(set.intersection(*[set(d) for d in D.values()]))
print(len(times), "common Abaqus times, last", times[-1])
prb = {j: np.array([sum(w * D[j][t][n] for w, n in zip(wts, probe)) for t in times]) for j in D}

# MFEM probe histories
rows = list(csv.DictReader(open(os.path.join(a.run, "sens.csv")))); tm = np.array([float(r["time_native"]) for r in rows])
pr = list(csv.DictReader(open(os.path.join(a.run, "probe.csv"))))
Tm = np.interp(times, [float(r["time_native"]) for r in pr], [float(r["layer1_top"]) for r in pr])
print(f"nominal probe T: max |MFEM - Abaqus| = {np.max(np.abs(Tm - prb['nom'])):.3f} K  (peak {prb['nom'].max():.1f} K)")


def nrmse(ad, fd): return 100 * np.sqrt(np.mean((ad - fd) ** 2)) / (fd.max() - fd.min())


def panel(xs, ad, fd, ax, lab, xl):
    ax.plot(xs, ad, color="red", lw=3.0, label="Enzyme AD (MFEM)")
    ax.plot(xs, fd, color="black", lw=2.0, ls=(0, (4, 3)), label="Abaqus FD")
    ax.set_ylabel(lab + "  [K]"); ax.text(0.97, 0.06, f"NRMSE = {nrmse(ad, fd):.2f} %", transform=ax.transAxes, ha="right",
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax.set_xlabel(xl)


fig, axs = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True); res = {}
for ax, (p, col, lab) in zip(axs.flat, PAR):
    ad = np.interp(times, tm, [float(r[col]) for r in rows]); fd = (prb[p + "_p"] - prb[p + "_m"]) / (2 * a.h)
    panel(np.array(times) * 1e3, ad, fd, ax, lab, "t [ms]" if ax in axs[1] else ""); res["probe " + p] = nrmse(ad, fd)
axs[0, 0].legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(1.05, 1.22), ncol=2, frameon=False)
fig.savefig(os.path.join(a.fd, "abaqus_fd_probe.png"), dpi=170, bbox_inches="tight"); print("written abaqus_fd_probe.png")

dumps = {round(F.header(f)["t"] * 1e9) / 1e9: f for f in F.dump_files(os.path.join(a.run, "fields"))}
xs_ = V[np.array(line) - 1, 0]; o = np.argsort(xs_); xs_ = xs_[o]; ln = np.array(line)[o]
for tms in a.t:
    t = min(times, key=lambda x: abs(x - tms * 1e-3))
    if t not in dumps or any(n not in D["nom"][t] for n in ln): print("no MFEM dump / Abaqus line output at", t); continue
    G, _ = F.read(dumps[t], [0, 1, 2, 3, 4], ln - 1)
    print(f"t = {t * 1e3:.3f} ms: nominal line T max |MFEM - Abaqus| = {np.max(np.abs(G[0] - np.array([D['nom'][t][n] for n in ln]))):.3f} K")
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.2), sharex=True)
    for q, (ax, (p, col, lab)) in enumerate(zip(axs.flat, PAR)):
        fd = (np.array([D[p + "_p"][t][n] for n in ln]) - np.array([D[p + "_m"][t][n] for n in ln])) / (2 * a.h)
        panel(xs_, G[q + 1], fd, ax, lab, "x [mm]" if ax in axs[1] else ""); res[f"line {t * 1e3:.2f} ms {p}"] = nrmse(G[q + 1], fd)
    axs[0, 0].legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(1.05, 1.22), ncol=2, frameon=False)
    out = os.path.join(a.fd, f"abaqus_fd_line_t{t * 1e3:.2f}.png"); fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out)
for k, v in res.items(): print(f"  NRMSE {k:22s} {v:6.2f} %")
