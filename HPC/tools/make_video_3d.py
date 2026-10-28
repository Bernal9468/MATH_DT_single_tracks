#!/usr/bin/env python3
"""Simulation video in the style of an FE post-processor: banded temperature contour [K], 3D view of the block around the
track (cut on the track centreline y = 0, so the front face shows the melt pool depth), top view, laser marker, clock.

Input: a run with dense field dumps (FIELD_DUMP_DIR, FIELD_DUMP_EVERY=2), e.g. results/video/case0.
usage: make_video_3d.py RUNDIR [--fields DIR] [--fps 30] [--every N] [--out VIDEO.mp4] [--frame N --png FIG.png]
"""
import argparse, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
import matplotlib.patheffects
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "post"))
import analyze_case as A
import field_io as F
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass                                                             # system ffmpeg

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white"})
# 12 bands, 300-2400 K, blue -> red (FE post-processor spectrum); above 2400 K: the top band
BANDS = ["#0012ff", "#0060ff", "#00aaff", "#00dcf0", "#00d28c", "#00b843", "#66cc00", "#c8e000", "#ffd700", "#ffa500",
         "#ff5a00", "#ff1e00"]
LEV = np.linspace(300.0, 2400.0, len(BANDS) + 1)
CMAP = matplotlib.colors.ListedColormap(BANDS); CMAP.set_over(BANDS[-1]); CMAP.set_under(BANDS[0])
NORM = matplotlib.colors.BoundaryNorm(LEV, CMAP.N)

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--fields"); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--every", type=int, default=1); ap.add_argument("--out"); ap.add_argument("--frame", type=int)
ap.add_argument("--png"); ap.add_argument("--box", type=float, nargs=6, default=(-0.7, 1.6, 0.0, 0.6, -0.45, 0.0),
                                         help="xmin xmax ymin ymax zmin zmax of the 3D block [mm]")
a = ap.parse_args()
st = open(os.path.join(a.run, "settings.txt")).read()
case = re.search(r"case=(\S+)", st).group(1); x0 = float(re.search(r"x0=(\S+)", st).group(1)); x1 = float(re.search(r"x1=(\S+)", st).group(1))
mesh = re.search(r"mesh=(.+?\.mesh)", st).group(1)
lp = [l.split(",") for l in open(os.path.join(a.run, "laser.inp")) if l.strip() and not l.startswith("#")]
v = abs(float(lp[1][1]) - float(lp[0][1])) / (float(lp[1][0]) - float(lp[0][0]))      # [mm/s]
t_off = (x1 - x0) / v
files = F.dump_files(a.fields or os.path.join(a.run, "fields"))[::a.every]
V = np.array(A.read_mesh_vertices(mesh))
bx0, bx1, by0, by1, bz0, bz1 = a.box


def face(axis, value, ulim, wlim):
    """nodes of the plane axis = value inside the block, as a tensor grid: (plane, iu mask, iw mask)"""
    P = F.Plane(V, axis, value)
    iu = (P.u >= ulim[0] - 1e-9) & (P.u <= ulim[1] + 1e-9); iw = (P.w >= wlim[0] - 1e-9) & (P.w <= wlim[1] + 1e-9)
    return P, iu, iw


xs = np.unique(np.round(V[:, 0], 9)); xr = xs[np.argmin(np.abs(xs - bx1))]                 # mesh plane nearest the block end
top, ti, tj = face(2, 0.0, (bx0, xr), (by0, by1))              # z = 0 : (x, y)
fro, fi, fj = face(1, 0.0, (bx0, xr), (bz0, bz1))              # y = 0 : (x, z)
rig, ri, rj = face(0, xr, (by0, by1), (bz0, bz1))              # x = xr: (y, z)
tv, vi, vj = face(2, 0.0, (bx0, xr), (-by1, by1))              # top view, both sides of the track
bz0 = fro.w[fj].min(); by1 = top.w[tj].max()                    # block snapped to the mesh nodes
STEP = 0.01                                                    # resampling of the faces for drawing [mm]


def regular(P, iu, iw):
    u, w = P.u[iu], P.w[iw]
    return u, w, np.clip(np.arange(u[0], u[-1] + STEP, STEP), u[0], u[-1]), np.clip(np.arange(w[0], w[-1] + STEP, STEP), w[0], w[-1])


RG = [regular(top, ti, tj), regular(fro, fi, fj), regular(rig, ri, rj)]


def resample(G, rg):
    u, w, U, W = rg
    f = RegularGridInterpolator((u, w), G); UU, WW = np.meshgrid(U, W, indexing="ij")
    return UU, WW, f(np.stack([UU.ravel(), WW.ravel()], -1)).reshape(UU.shape)


nodes = np.concatenate([top.nodes, fro.nodes, rig.nodes, tv.nodes]); cut = np.cumsum([len(top.nodes), len(fro.nodes), len(rig.nodes)])


def fields(fn):
    T, t = F.read(fn, [0], nodes); T = T[0]
    Tt, Tf, Tr, Tv = np.split(T, cut)
    return t, top.grid(Tt)[np.ix_(ti, tj)], fro.grid(Tf)[np.ix_(fi, fj)], rig.grid(Tr)[np.ix_(ri, rj)], tv.grid(Tv)[np.ix_(vi, vj)]


def tmax(fn):
    h = F.header(fn); M = np.memmap(fn, dtype=np.float64, mode="r", offset=28, shape=(1 + h["n1"] + h["n2"] + h["nm"], h["ndof"]))
    return float(M[0].max())


fig = plt.figure(figsize=(12.8, 6.0), dpi=150)
axc = fig.add_axes([0.012, 0.16, 0.022, 0.66])
cb = matplotlib.colorbar.ColorbarBase(axc, cmap=CMAP, norm=NORM, boundaries=LEV, ticks=LEV, spacing="uniform")
cb.ax.set_yticklabels([f"{x:+.3e}" for x in LEV]); cb.ax.tick_params(labelsize=9, length=0); cb.outline.set_edgecolor("#555555")
for k in range(1, len(LEV) - 1): cb.ax.axhline(LEV[k], color="#333333", lw=0.6)
fig.text(0.008, 0.86, r"$T$ [K]", fontsize=17, fontweight="bold", va="bottom")
ax3 = fig.add_axes([0.16, 0.0, 0.47, 1.0], projection="3d")
axt = fig.add_axes([0.635, 0.22, 0.35, 0.42])
import csv  # noqa: E402
CR = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}[case]
fig.text(0.16, 0.93, f"AMB2022-03  case {case}:   P = {float(CR['power_W']):.0f} W,   v = {float(CR['speed_mm_s']):.0f} mm/s,   "
         f"spot D4σ = {float(CR['spot_D4sigma_um']):.0f} µm", fontsize=16, fontweight="bold")
h_t = fig.text(0.635, 0.86, "", fontsize=19, fontweight="bold")
h_m = fig.text(0.635, 0.80, "", fontsize=13)
mark3 = fig.text(0, 0, "▼", fontsize=26, ha="center", va="bottom", visible=False)
h_l = fig.text(0.635, 0.745, "", fontsize=13, fontweight="bold")


def edges3d(ax):
    c = dict(color="black", lw=1.4, zorder=10)
    for (xa, ya, za), (xb, yb, zb) in [((bx0, by0, bz1), (xr, by0, bz1)), ((xr, by0, bz1), (xr, by1, bz1)), ((xr, by1, bz1), (bx0, by1, bz1)),
                                     ((bx0, by1, bz1), (bx0, by0, bz1)), ((bx0, by0, bz0), (xr, by0, bz0)), ((xr, by0, bz0), (xr, by1, bz0)),
                                     ((bx0, by0, bz0), (bx0, by0, bz1)), ((xr, by0, bz0), (xr, by0, bz1)), ((xr, by1, bz0), (xr, by1, bz1))]:
        ax.plot([xa, xb], [ya, yb], [za, zb], **c)


def triad(ax, o, L):
    for d, lab in ((np.array([L, 0, 0]), "x"), (np.array([0, L, 0]), "y"), (np.array([0, 0, L]), "z")):
        e = o + d; ax.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color="black", lw=2)
        ax.scatter([e[0]], [e[1]], [e[2]], marker="o", s=14, color="black", depthshade=False)
        ax.text(*(o + 1.3 * d), lab, fontsize=17, fontstyle="italic")


def draw(k):
    fn = files[k]; t, Gt, Gf, Gr, Gv = fields(fn); xl = x0 + v * t; on = t <= t_off + 1e-12
    ax3.cla(); ax3.set_axis_off()
    kw = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
    X, Y, G = resample(Gt, RG[0]); ax3.plot_surface(X, Y, np.zeros_like(X), facecolors=CMAP(NORM(G)), **kw)
    X, Z, G = resample(Gf, RG[1]); ax3.plot_surface(X, np.zeros_like(X) + by0, Z, facecolors=CMAP(NORM(G)), **kw)
    Y, Z, G = resample(Gr, RG[2]); ax3.plot_surface(np.zeros_like(Y) + xr, Y, Z, facecolors=CMAP(NORM(G)), **kw)
    edges3d(ax3)
    span = xr - bx0; cx = 0.5 * (bx0 + xr)
    ax3.set_xlim(cx - span / 2, cx + span / 2); ax3.set_ylim(-span / 2 + 0.5 * by1, span / 2 + 0.5 * by1)
    ax3.set_zlim(-span / 2 + 0.5 * bz0, span / 2 + 0.5 * bz0)
    ax3.view_init(elev=28, azim=-58); ax3.dist = 6.0
    triad(ax3, np.array([bx0 + 0.25, by0 - 0.25, bz0 - 0.45]), 0.3)
    if on and bx0 <= xl <= xr:                                  # laser marker on top of everything: in projected 2D coordinates
        px, py, _ = proj3d.proj_transform(xl, by0, 0.0, ax3.get_proj())
        fx, fy = fig.transFigure.inverted().transform(ax3.transData.transform((px, py)))
        mark3.set_position((fx, fy + 0.004)); mark3.set_visible(True)
    else:
        mark3.set_visible(False)
    axt.cla()
    axt.contourf(tv.u[vi], tv.w[vj], Gv.T, levels=np.concatenate([[0], LEV[1:-1], [1e5]]), colors=BANDS)
    axt.set_aspect("equal")
    axt.plot([bx0, xr, xr, bx0, bx0], [by0, by0, by1, by1, by0], color="black", lw=1.2, ls="--")       # the 3D block
    if on and bx0 <= xl <= xr: axt.scatter([xl], [0.06], marker="v", s=180, color="black", zorder=5)
    axt.set_xticks([]); axt.set_yticks([])
    for s in axt.spines.values(): s.set_linewidth(1.4); s.set_color("black")
    ox, oy, L = bx0 + 0.1, -by1 + 0.08, 0.2
    axt.annotate("", xy=(ox + L, oy), xytext=(ox, oy), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    axt.annotate("", xy=(ox, oy + L), xytext=(ox, oy), arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
    for txt, xy in (("x", (ox + L + 0.04, oy)), ("y", (ox - 0.02, oy + L + 0.03))):
        axt.text(*xy, txt, fontsize=16, fontstyle="italic", color="white", va="center",
                 path_effects=[matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])
    h_t.set_text(rf"$t$ =  {t * 1e3:.4f} ms")
    h_m.set_text(rf"$T_{{max}}$ =  {tmax(fn):6.0f} K")
    h_l.set_text("Laser ON" if on else "Laser OFF"); h_l.set_color("#e00000" if on else "#666666")


if a.frame is not None:
    draw(a.frame); fig.savefig(a.png or "frame.png", dpi=150); print("written", a.png or "frame.png"); sys.exit()
out = a.out or os.path.join(a.run, f"video3d_case{case}.mp4")
w = anim.FFMpegWriter(fps=a.fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with w.saving(fig, out, dpi=150):
    for k in range(len(files)):
        draw(k); w.grab_frame()
        if k % 25 == 0: print(f"frame {k}/{len(files)}", flush=True)
print("written", out)
