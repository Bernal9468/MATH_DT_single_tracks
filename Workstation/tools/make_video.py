#!/usr/bin/env python3
"""Simulation video of one case (MP4): nominal temperature field, its first derivatives and the NIST thermography, frame by
frame, from a run with dense field dumps (DORDER=1, FIELD_DUMP_EVERY=2 FIELD_DUMP_EVERY2=2).

Left:  (a) NIST camera, repeat 1, at the same time; (b) model top surface seen as the camera sees it (same scale, saturated
       above T_sat, nothing below 940 C); (c) model longitudinal section y = 0 with solidus / liquidus; (d) temperature
       history at the track midpoint, model and the 3 NIST repeats, with the current time.
Right: dT/ds_p on the section y = 0 for P (= η), k, liquidus Tl and the Goldak depth c, in K per 1 % increase; fixed colour
       scale per parameter over the whole video (99th percentile of |dT/ds_p|).
Time alignment: NIST frame k shows the model time t = t_front + (k - k_mid) / 30 kHz, where t_front (model) and k_mid (NIST)
are when the saturated-spot front edge passes the track midpoint x = 0, as in tools/compare_thermography.py.
The model segment starts cold at x0 (2.1 mm segment); the NIST track started 4.4 mm earlier, so its trail is longer at the
start, and NIST's laser keeps going after the model's segment ends.
usage: make_video.py RUNDIR [--fields DIR] [--fps 30] [--out FILE.mp4] [--every N]
"""
import argparse, copy, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "post"))
import analyze_case as A
import field_io as F
import nist_thermo as NT
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass                                                             # system ffmpeg

INK, INK2, SURF, S1 = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "legend.frameon": False})
K0 = A.K0; TS, TL = A.T_SOL - K0, A.T_LIQ - K0
DER = [("Eta", "dT/d(power P or absorptivity η)"), ("k", "dT/d(conductivity k)"), ("Tliq", "dT/d(liquidus Tl)"),
       ("Rz", "dT/d(Goldak depth c)")]

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--fields"); ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--every", type=int, default=1, help="use every N-th dump"); ap.add_argument("--out")
ap.add_argument("--h5", default=os.path.join(ROOT, "data", "nist_mds2-2716", "AMB2022-03-718-AMMT-StaringCamera_Signal.h5"))
a = ap.parse_args()
st = open(os.path.join(a.run, "settings.txt")).read()
case = re.search(r"case=(\S+)", st).group(1); x0 = float(re.search(r"x0=(\S+)", st).group(1)); x1 = float(re.search(r"x1=(\S+)", st).group(1))
mesh = re.search(r"mesh=(.+?\.mesh)", st).group(1)
files = F.dump_files(a.fields or os.path.join(a.run, "fields"))[::a.every]
times = np.array([F.header(f)["t"] for f in files])
print(f"{len(files)} frames, t = {times[0] * 1e3:.3f} .. {times[-1] * 1e3:.3f} ms")

# ---- NIST
reps = []
for r in (1, 2, 3):
    tr = NT.Track(a.h5, case, r); tr.laser_path(); tr.fit_emissivity(); reps.append(tr)
T_sat = float(np.mean([tr.t_sat() for tr in reps])); v = reps[0].v
tr = reps[0]; k_mid = float(np.median(tr.frames))
t_p, T_p = A.read_probe(os.path.join(a.run, "probe.csv")); t_p = np.array(t_p); T_p = np.array(T_p) - K0
up = np.where((T_p[:-1] < T_sat) & (T_p[1:] >= T_sat))[0][0]
t_front = t_p[up] + (t_p[up + 1] - t_p[up]) * (T_sat - T_p[up]) / (T_p[up + 1] - T_p[up])
hist = []
for trr in reps:
    th, Th, _ = trr.history(int(round(trr.i0 + trr.u_px * np.median(trr.frames)))); hist.append((th + t_front, Th))

# ---- model planes, window
xw, yw, zw = (x0 - 0.35, x1 + 0.3), 0.32, 0.30
V = A.read_mesh_vertices(mesh)
top, sec = F.Plane(V, 2, 0.0), F.Plane(V, 1, 0.0)
ut, wt = (top.u > xw[0] - 0.05) & (top.u < xw[1] + 0.05), np.abs(top.w) < yw + 0.05
us, ws = (sec.u > xw[0] - 0.05) & (sec.u < xw[1] + 0.05), sec.w > -zw - 0.05
XX, YY = np.linspace(*xw, 700), np.linspace(-yw, yw, 160)
PXY = np.array(np.meshgrid(XX, YY, indexing="ij")).reshape(2, -1).T
idx = [0] + [1 + A.PARAMS.index(p) for p, _ in DER]

def model(fn):
    Vt, t = F.read(fn, [0], top.nodes); Vs, _ = F.read(fn, idx, sec.nodes)
    GT = top.grid(Vt[0])[np.ix_(ut, wt)] - K0
    GS = [sec.grid(Vs[i])[np.ix_(us, ws)] for i in range(len(idx))]
    return t, GT, GS[0] - K0, [g / 100.0 for g in GS[1:]]

scale = np.zeros(len(DER))
for fn in files[::max(1, len(files) // 25)]:
    _, _, _, D = model(fn)
    scale = np.maximum(scale, [np.nanpercentile(np.abs(d), 99.0) for d in D])

# ---- figure
fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.0], hspace=0.62, wspace=0.2, left=0.05, right=0.97, top=0.9, bottom=0.07)
cam = copy.copy(plt.get_cmap("inferno")); cam.set_bad("#d9d8d3"); cam.set_over("#fff7c2")
ncam = matplotlib.colors.Normalize(940, T_sat); nfull = matplotlib.colors.Normalize(25, 2500)
axs = [fig.add_subplot(gs[i, 0]) for i in range(4)] + [fig.add_subplot(gs[i, 1]) for i in range(4)]
title = fig.suptitle("", x=0.05, ha="left", fontsize=11, color=INK)
cbs = []

def draw(n):
    t, GT, GS, D = model(files[n])
    for ax in axs: ax.clear()
    k = int(round(k_mid + (t - t_front) * NT.FPS))
    # (a) NIST
    ax = axs[0]
    if 0 <= k < tr.S.shape[0]:
        xi, y, T = tr.frame_temperature(k); xfront = v * (k - k_mid) / NT.FPS
        Tm = np.where(tr.S[k] >= 4095, 1e4, T)
        ax.imshow(Tm.T, origin="lower", extent=[xi[0] + xfront, xi[-1] + xfront, y[0], y[-1]], cmap=cam, norm=ncam, aspect="auto",
                  interpolation="nearest")
        ax.set_title(f"(a) NIST thermography, case {case} repeat 1, frame {k} (30 kfps)", fontsize=8, loc="left", color=INK)
    else:
        ax.set_title("(a) NIST thermography: outside the recording", fontsize=8, loc="left", color=INK)
    ax.set_xlim(*xw); ax.set_ylim(-yw, yw); ax.set_ylabel("y [mm]")
    # (b) model, camera view
    ax = axs[1]
    G = RegularGridInterpolator((top.u[ut], top.w[wt]), GT, bounds_error=False, fill_value=np.nan)(PXY).reshape(len(XX), len(YY))
    Gc = np.where(G >= T_sat, 1e4, np.where(G < 940, np.nan, G))
    ax.imshow(Gc.T, origin="lower", extent=[xw[0], xw[1], -yw, yw], cmap=cam, norm=ncam, aspect="auto", interpolation="bilinear")
    ax.set_xlim(*xw); ax.set_ylim(-yw, yw); ax.set_ylabel("y [mm]")
    ax.set_title(f"(b) model top surface as the camera sees it (940 °C … T_sat {T_sat:.0f} °C)", fontsize=8, loc="left", color=INK)
    # (c) section
    ax = axs[2]
    ax.pcolormesh(sec.u[us], sec.w[ws], GS.T, cmap="inferno", norm=nfull, shading="gouraud", rasterized=True)
    if np.nanmax(GS) > TS: ax.contour(sec.u[us], sec.w[ws], GS.T, levels=[TS, TL], colors=["#9fd0ff", "white"], linewidths=0.8)
    ax.set_xlim(*xw); ax.set_ylim(-zw, 0); ax.set_ylabel("z [mm]"); ax.set_xlabel("x [mm]")
    ax.set_title("(c) model section y = 0 (25 … 2500 °C); solidus (blue), liquidus (white)", fontsize=8, loc="left", color=INK)
    # (d) history
    ax = axs[3]
    for (th, Th), lab in zip(hist, ("NIST repeats", None, None)): ax.plot(th * 1e3, Th, color=INK2, lw=0.9, ls="--", label=lab)
    ax.plot(t_p * 1e3, T_p, color=S1, lw=1.8, label="model")
    ax.axvline(t * 1e3, color="#eb6834", lw=1.2)
    for l in (TS, TL): ax.axhline(l, color="#e4e3df", lw=0.8)
    ax.set_xlim(times[0] * 1e3, times[-1] * 1e3); ax.set_ylim(900, T_sat + 50)
    ax.set_xlabel("t [ms]"); ax.set_ylabel("T [°C]"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(d) temperature at the track midpoint x = 0 (surface)", fontsize=8, loc="left", color=INK)
    # derivatives
    for j, (p, lbl) in enumerate(DER):
        ax = axs[4 + j]
        ax.pcolormesh(sec.u[us], sec.w[ws], D[j].T, cmap="RdBu_r", vmin=-scale[j], vmax=scale[j], shading="gouraud", rasterized=True)
        if np.nanmax(GS) > TS: ax.contour(sec.u[us], sec.w[ws], GS.T, levels=[TS, TL], colors=[INK2, INK], linewidths=0.6)
        ax.set_xlim(*xw); ax.set_ylim(-zw, 0); ax.set_ylabel("z [mm]")
        if j == 3: ax.set_xlabel("x [mm]")
        ax.set_title(f"({'efgh'[j]}) {lbl}, section y = 0  [K per 1 %, ±{scale[j]:.0f}]", fontsize=8, loc="left", color=INK)
    if not cbs:                                                      # colour bars once (fixed scales)
        sm = lambda c, nrm: matplotlib.cm.ScalarMappable(norm=nrm, cmap=c)
        cbs.append(fig.colorbar(sm(cam, ncam), ax=axs[0:2], fraction=0.015, pad=0.01, extend="max"))
        cbs.append(fig.colorbar(sm("inferno", nfull), ax=axs[2], fraction=0.015, pad=0.01, extend="max"))
        for j in range(4):
            cbs.append(fig.colorbar(sm("RdBu_r", matplotlib.colors.Normalize(-scale[j], scale[j])), ax=axs[4 + j], fraction=0.015, pad=0.01))
        for c in cbs: c.ax.tick_params(labelsize=6.5)
    laser = "laser ON" if t <= (x1 - x0) / v + 1e-9 else "laser OFF (cool-down)"
    title.set_text(f"AMB2022-03 case {case} ({tr.P:.0f} W, {v:.0f} mm/s, D4σ {tr.spot:.0f} µm): nominal field, first derivatives "
                   f"(Enzyme AD) and NIST thermography     t = {t * 1e3:.3f} ms, {laser}")
    return []

out = a.out or os.path.join(a.run, f"video_case{case}.mp4")
w = anim.FFMpegWriter(fps=a.fps, bitrate=6000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with w.saving(fig, out, dpi=150):
    for n in range(len(files)):
        draw(n); w.grab_frame()
        if n % 25 == 0: print(f"  frame {n}/{len(files)}", flush=True)
print("written", out)
