#!/usr/bin/env python3
"""Nominal temperature field and its first derivatives in space, one snapshot of a run with field dumps (DORDER=1).

Snapshot: the dump with the laser closest to the thermography snapshot of tools/compare_thermography.py (x = x0 + 2.6 mm,
or 0.05 mm before the end of a short segment). Panels:
  (a) nominal T on the top surface z = 0;  (b) nominal T on the longitudinal section y = 0 (melt pool depth);
  (c) dT/ds_p on the section y = 0 for the 12 non-zero material / process / source parameters, in K per 1 % change of the
      parameter (s_p = relative change), each with its own symmetric colour scale; contours: nominal solidus and liquidus.
P and η enter only as their product, so their fields are identical (one panel, "P or η").
usage: fields_figure.py RUNDIR [--fields DIR] [--t SECONDS] [--out FIG.png]
"""
import argparse, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "post"))
import analyze_case as A
import field_io as F

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
K0 = A.K0; TS, TL = A.T_SOL - K0, A.T_LIQ - K0
SHOW = [("Eta", "power P or absorptivity η"), ("Vx", "speed v"), ("k", "conductivity k"), ("cp", "heat capacity cp"),
        ("rho", "density ρ"), ("Lfus", "latent heat L"), ("Ts", "solidus Ts"), ("Tliq", "liquidus Tl"),
        ("Rz", "Goldak depth c"), ("Ry", "Goldak width b"), ("Rx", "Goldak length a"), ("Tw", "substrate T")]

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--fields"); ap.add_argument("--t", type=float)
ap.add_argument("--out")
a = ap.parse_args()
st = open(os.path.join(a.run, "settings.txt")).read()
case = re.search(r"case=(\S+)", st).group(1); x0 = float(re.search(r"x0=(\S+)", st).group(1)); x1 = float(re.search(r"x1=(\S+)", st).group(1))
mesh = re.search(r"mesh=(.+?\.mesh)", st).group(1)
lp = [l.split(",") for l in open(os.path.join(a.run, "laser.inp")) if l.strip() and not l.startswith("#")]
v = abs(float(lp[1][1]) - float(lp[0][1])) / (float(lp[1][0]) - float(lp[0][0]))      # [mm/s] from the first leg
files = F.dump_files(a.fields or os.path.join(a.run, "fields"))
if not files: sys.exit("no field dumps")
times = np.array([F.header(f)["t"] for f in files])
if a.t is None: k = int(np.argmin(np.abs(x0 + v * times - min(max(0.0, x0 + 2.6), x1 - 0.05))))
else: k = int(np.argmin(np.abs(times - a.t)))
h = F.header(files[k])
if h["n1"] < 20: sys.exit("this run has no first-order fields (DORDER=0)")
V = A.read_mesh_vertices(mesh)
top, sec = F.Plane(V, 2, 0.0), F.Plane(V, 1, 0.0)
idx = [1 + A.PARAMS.index(p) for p, _ in SHOW]
Vt, t = F.read(files[k], [0], top.nodes); Vs, _ = F.read(files[k], [0] + idx, sec.nodes)
xl = x0 + v * t
GT = top.grid(Vt[0]) - K0; GS = sec.grid(Vs[0]) - K0
xw = (xl - 1.9, xl + 0.35); yw, zw = 0.32, 0.30

fig = plt.figure(figsize=(8.2, 11.4))
gs = fig.add_gridspec(3, 1, height_ratios=[0.55, 0.42, 3.0], hspace=0.42)
cmap = plt.get_cmap("inferno"); norm = matplotlib.colors.Normalize(25, 2500)

def frame(ax, ylab="z [mm]"):
    ax.set_xlim(*xw); ax.set_xlabel("x [mm]"); ax.set_ylabel(ylab)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = fig.add_subplot(gs[0])
iu = (top.u > xw[0] - 0.1) & (top.u < xw[1] + 0.1); iw = np.abs(top.w) < yw + 0.1
m = ax.pcolormesh(top.u[iu], top.w[iw], GT[np.ix_(iu, iw)].T, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
ax.contour(top.u[iu], top.w[iw], GT[np.ix_(iu, iw)].T, levels=[TS, TL], colors=["#9fd0ff", "white"], linewidths=0.9)
ax.axvline(xl, color="#9fd0ff", lw=0.6, ls=":"); frame(ax, ylab="y [mm]"); ax.set_ylim(-yw, yw)
ax.set_title(f"(a) nominal T, top surface z = 0 (t = {t * 1e3:.2f} ms, laser at x = {xl:.2f} mm, dotted); contours: solidus "
             f"{TS:.0f} °C (blue), liquidus {TL:.0f} °C (white)", fontsize=8, loc="left", color=INK)
ax2 = fig.add_subplot(gs[1])
iu = (sec.u > xw[0] - 0.1) & (sec.u < xw[1] + 0.1); iw = sec.w > -zw - 0.05
ax2.pcolormesh(sec.u[iu], sec.w[iw], GS[np.ix_(iu, iw)].T, cmap=cmap, norm=norm, shading="gouraud", rasterized=True)
ax2.contour(sec.u[iu], sec.w[iw], GS[np.ix_(iu, iw)].T, levels=[TS, TL], colors=["#9fd0ff", "white"], linewidths=0.9)
ax2.axvline(xl, color="#9fd0ff", lw=0.6, ls=":"); frame(ax2); ax2.set_ylim(-zw, 0)
ax2.set_title("(b) nominal T, longitudinal section y = 0 (track centreline)", fontsize=8, loc="left", color=INK)
cb = fig.colorbar(m, ax=[ax, ax2], fraction=0.025, pad=0.01, extend="max"); cb.set_label("T [°C]")

sub = gs[2].subgridspec(6, 2, hspace=0.75, wspace=0.18)
for n, (p, lbl) in enumerate(SHOW):
    a2 = fig.add_subplot(sub[n // 2, n % 2])
    D = sec.grid(Vs[1 + n])[np.ix_(iu, iw)] / 100.0                    # K per 1 % change
    dmax = np.nanmax(np.abs(D)); vm = np.nanpercentile(np.abs(D), 99.0) or dmax or 1.0     # robust scale; front spikes saturate
    q = a2.pcolormesh(sec.u[iu], sec.w[iw], D.T, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="gouraud", rasterized=True)
    c_ext = "both" if dmax > vm * 1.001 else "neither"
    a2.contour(sec.u[iu], sec.w[iw], GS[np.ix_(iu, iw)].T, levels=[TS, TL], colors=[INK2, INK], linewidths=0.6)
    a2.set_xlim(*xw); a2.set_ylim(-zw, 0); a2.tick_params(labelsize=6.5)
    a2.set_xlabel("x [mm]" if n >= 10 else "", fontsize=7); a2.set_ylabel("z [mm]" if n % 2 == 0 else "", fontsize=7)
    a2.set_title(f"{lbl}   (scale ±{vm:.1f} K, max {dmax:.0f} K)", fontsize=7.5, loc="left", color=INK)
    c = fig.colorbar(q, ax=a2, fraction=0.03, pad=0.01, extend=c_ext); c.ax.tick_params(labelsize=6)
ax3 = fig.axes[3]
fig.text(0.02, ax3.get_position().y1 + 0.022, "(c) first derivatives dT/ds_p on the section y = 0, in K per 1 % increase of the "
         "parameter (red: hotter; colour scale\n     at the 99th percentile of |dT/ds_p|, the laser-front spikes saturate); contours: nominal solidus (grey), liquidus (black)", fontsize=8.5, color=INK)
fig.suptitle(f"AMB2022-03 case {case}: nominal temperature field and its first derivatives (Enzyme AD)", fontsize=10,
             x=0.02, ha="left", color=INK)
out = a.out or os.path.join(a.run, f"fields_case{case}.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out)
