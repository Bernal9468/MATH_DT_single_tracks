#!/usr/bin/env python3
"""Geometry / mesh / boundary-condition figure of one single-track run (the Abaqus FD check model by default).

(a) the whole plate in 3D with its mesh on the visible faces, the fine box around the track and the boundary conditions,
(b) top view of the track region: mesh, laser path, FD centreline (Abaqus NSET LINE), probe node,
(c) longitudinal section y = 0 of the same region.
usage: geometry_figure.py MFEM_RUNDIR [OUT.png]   (default OUT: MFEM_RUNDIR/../geometry.png)
"""
import os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

run = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(run, "..", "geometry.png")
st = open(os.path.join(run, "settings.txt")).read(); log = open(os.path.join(run, "run.log")).read()
mesh = re.search(r"mesh=(.+?\.mesh)", st).group(1)
L = open(mesh).read().split("\n"); i = L.index("vertices"); nv = int(L[i + 1])
V = np.array([list(map(float, l.split())) for l in L[i + 3:i + 3 + nv]]); ne = int(L[L.index("elements") + 1])
X, Y, Z = (np.unique(np.round(V[:, d], 6)) for d in range(3))
h_conv = float(re.search(r"Effective boundary loss scale=\S+ h=(\S+)", log).group(1))
eps = float(re.search(r"emissivity=(\S+)", log).group(1))
T_sub = float(re.search(r"Dirichlet BC nodes: \d+ \(T=(\S+?)K\)", log).group(1))
pdof = int(re.search(r"Probe row1_top dof=(\d+)", log).group(1)); PR = V[pdof]
las = np.loadtxt(os.path.join(run, "laser.inp"), delimiter=",")              # t, x, y, z, P[mW]
xs, xe, t_on, P = las[0, 1], las[1, 1], las[1, 0], las[0, 4] / 1e3
D4S = float(re.search(r"spot_D4sigma_um=(\S+)", st).group(1)); dt = float(re.search(r"dt=(\S+)", st).group(1))
t_end = float(re.search(r"t_final=(\S+)", st).group(1))
d = np.diff(X); hf = d.min()
fine = lambda A: A[np.r_[np.diff(A) < 1.5 * hf, False] | np.r_[False, np.diff(A) < 1.5 * hf]]
FX, FY, FZ = fine(X), fine(Y), fine(Z)
LX = (-0.75, 1.65)                                                             # abaqus_deck.py NSET LINE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 1.4})
C_MESH, C_FINE, C_LAS, C_PRB = "#9a9a9a", "#1f5fbf", "#d62728", "#e6a400"

fig = plt.figure(figsize=(16, 8.4))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[1, 0.8], wspace=0.22, hspace=0.42)

# ---- (a) 3D plate
ax = fig.add_subplot(gs[:, 0], projection="3d"); ZS = 2.0                     # z exaggerated x2 for legibility
x0, x1, y0, y1, z0, z1 = X[0], X[-1], Y[0], Y[-1], Z[0], Z[-1]
kw = dict(color=C_MESH, lw=0.35)
for x in X: ax.plot([x, x], [y0, y1], [z1 * ZS] * 2, **kw); ax.plot([x, x], [y0, y0], [z0 * ZS, z1 * ZS], **kw)
for y in Y: ax.plot([x0, x1], [y, y], [z1 * ZS] * 2, **kw); ax.plot([x1, x1], [y, y], [z0 * ZS, z1 * ZS], **kw)
for z in Z: ax.plot([x0, x1], [y0, y0], [z * ZS] * 2, **kw); ax.plot([x1, x1], [y0, y1], [z * ZS] * 2, **kw)
for e in [((x0, x1), (y0, y0), (z0, z0)), ((x1, x1), (y0, y1), (z0, z0)), ((x0, x1), (y1, y1), (z1, z1)),
          ((x0, x0), (y0, y1), (z1, z1)), ((x0, x0), (y0, y0), (z0, z1)), ((x1, x1), (y1, y1), (z0, z1)),
          ((x0, x1), (y0, y0), (z1, z1)), ((x1, x1), (y0, y1), (z1, z1))]:
    ax.plot(*e[:2], np.array(e[2]) * ZS, color="k", lw=1.3)
fx0, fx1, fy0, fy1 = FX[0], FX[-1], FY[0], FY[-1]
ax.plot([fx0, fx1, fx1, fx0, fx0], [fy0, fy0, fy1, fy1, fy0], [z1 * ZS + 0.01] * 5, color=C_FINE, lw=2.2)
ax.plot([xs, xe], [0, 0], [z1 * ZS + 0.02] * 2, color=C_LAS, lw=3.0)
ax.text(4.0, -7.5, 0.3, "fine box +\nlaser track", color=C_FINE, fontsize=11, ha="center")
fig.text(0.07, -0.02, f"Boundary conditions:  bottom z = {z0:g} mm: T = {T_sub} K (Dirichlet);  all other faces: "
         f"convection h = {h_conv * 1e3:g} W/m²K + radiation ε = {eps}, T_amb = {T_sub} K", fontsize=11.5)
ax.set_box_aspect((x1 - x0, y1 - y0, (z1 - z0) * ZS)); ax.view_init(elev=28, azim=-58)
ax.set_xlabel("x [mm]", labelpad=6); ax.set_ylabel("y [mm]", labelpad=6); ax.set_zlabel("")
ax.set_zticks([z0 * ZS, z1 * ZS]); ax.set_zticklabels([f"  {z0:g}", "  0"])
for a_ in (ax.xaxis, ax.yaxis, ax.zaxis): a_.pane.set_alpha(0.0)
ax.grid(False)
fig.text(0.07, 0.93, f"(a) IN718 plate {x1 - x0:g} × {y1 - y0:g} × {z1 - z0:g} mm (z drawn ×2),  {nv:,} nodes,  "
         f"{ne:,} hex elements", fontsize=13)

# ---- (b) top view, (c) section y = 0
XL, YL, ZL = (-1.4, 2.4), (-0.9, 0.9), (-0.9, 0.04)
for k, (lbl, V2, lim2, F2, yl) in enumerate([("(b) top view z = 0", Y, YL, FY, "y [mm]"),
                                               ("(c) section y = 0", Z, ZL, FZ, "z [mm]")]):
    a = fig.add_subplot(gs[k, 1])
    for x in X[(X >= XL[0] - 1) & (X <= XL[1] + 1)]: a.plot([x, x], [V2[0], V2[-1]], color=C_MESH, lw=0.5)
    for v in V2: a.plot([X[0], X[-1]], [v, v], color=C_MESH, lw=0.5)
    a.add_patch(Rectangle((fx0, F2[0]), fx1 - fx0, F2[-1] - F2[0], fill=False, ec=C_FINE, lw=2.0,
                          label=f"fine box, h = {hf * 1e3:.0f} µm"))
    zl = 0 if k == 0 else Z[-1]
    a.plot(LX, [zl, zl], color="k", lw=2.4, ls=(0, (4, 2)), label="FD centreline (y = z = 0)", zorder=4)
    a.annotate("", xy=(xe, zl), xytext=(xs, zl), zorder=5,
               arrowprops=dict(arrowstyle="-|>", color=C_LAS, lw=2.6, mutation_scale=18))
    a.plot([], [], color=C_LAS, lw=2.6, label=f"laser path {xs:g} → {xe:g} mm")
    a.plot(PR[0], PR[1] if k == 0 else PR[2], marker="*", ms=17, mfc=C_PRB, mec="k", ls="none", zorder=6,
           label=f"probe node ({PR[0]:.3f}, {PR[1]:g}, {PR[2]:g})")
    if k == 0:
        a.add_patch(plt.Circle((xs, 0), D4S / 2e3, fill=False, ec=C_LAS, lw=1.5, zorder=5))
        a.text(xs - 0.1, 0.1, f"spot D4σ = {D4S:g} µm", color=C_LAS, ha="left", va="bottom", fontsize=11)
    a.set_xlim(XL); a.set_ylim(lim2 if k == 0 else ZL); a.set_aspect("equal")
    a.set_xlabel("x [mm]"); a.set_ylabel(yl); a.set_title(lbl, loc="left", fontsize=13)
    if k == 0: a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=11)
fig.text(0.07, -0.06, f"Case 0: P = {P:g} W, v = {(xe - xs) / t_on:.0f} mm/s, laser on 0 – {t_on * 1e3:.4g} ms, "
         f"cooling to {t_end * 1e3:.4g} ms;  dt = {dt * 1e6:g} µs ({int(t_end / dt + 1e-9)} steps).  "
         "MFEM: trilinear H1 hexes; Abaqus replica: DC3D8 on the same nodes.", ha="left", fontsize=11.5)
fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out)
