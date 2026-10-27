#!/usr/bin/env python3
"""Model surface temperature field against the NIST AMB2022-03 thermography (mds2-2716), one figure per case.

Model: a nominal run of run/run_case.sh with field dumps (FIELD_DUMP_DIR=<run>/fields FIELD_DUMP_EVERY=2), its mesh, its
probe.csv. Measurement: the 3 repeats of the case (tools/nist_thermo.py: NIST emissivity procedure, signal -> temperature).
Both are put in the frame of the front edge of the saturated spot: the point on the track centreline, ahead of the laser,
where the temperature passes the camera saturation temperature T_sat of the repeat (~1520-1660 C); for the model the same
T_sat (mean of the repeats) is used. Only the band between the camera floor (~940 C) and T_sat is compared.
Panels: (a) measured field (repeat 1), (b) model field, same colour scale; (c) isotherms 1150 / 1260 / 1336 / 1400 C,
measured (3 repeats) vs model; (d) centreline profile T(xi); (e) temperature history at the track midpoint vs time after
the front edge passed.
usage: compare_thermography.py RUNDIR [--fields DIR] [--h5 FILE] [--out FIG.png]
"""
import argparse, copy, glob, os, re, struct, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "post"))
import nist_thermo as NT
import analyze_case as A

S1, S2, S3, INK, INK2, GRID, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "savefig.facecolor": SURF, "legend.frameon": False})
LEVELS = [1150, 1260, 1336, 1400]
K0 = 273.15

ap = argparse.ArgumentParser(); ap.add_argument("run")
ap.add_argument("--h5", default=os.path.join(ROOT, "data", "nist_mds2-2716", "AMB2022-03-718-AMMT-StaringCamera_Signal.h5"))
ap.add_argument("--out"); ap.add_argument("--fields", help="field-dump folder (default RUNDIR/fields)"); ap.add_argument("--xi", type=float, nargs=2, default=(-2.6, 0.3)); ap.add_argument("--ylim", type=float, default=0.45)
a = ap.parse_args()
st = open(os.path.join(a.run, "settings.txt")).read()
case = re.search(r"case=(\S+)", st).group(1); x0 = float(re.search(r"x0=(\S+)", st).group(1)); x1 = float(re.search(r"x1=(\S+)", st).group(1))
mesh = re.search(r"mesh=(.+?\.mesh)", st).group(1)

# ---------------- measurement
reps = []
for r in (1, 2, 3):
    tr = NT.Track(a.h5, case, r); tr.laser_path(); tr.fit_emissivity(); reps.append(tr)
T_sat = float(np.mean([tr.t_sat() for tr in reps]))
v = reps[0].v                                                       # commanded speed [mm/s]

def front_edge(xi, T, ts):
    """xi of the saturation front edge on a centreline profile T(xi) (first crossing of ts from ahead of the laser)"""
    for k in range(len(xi) - 1, 0, -1):
        if T[k] < ts <= T[k - 1] and np.isfinite(T[k]):
            return xi[k] + (xi[k - 1] - xi[k]) * (ts - T[k]) / (T[k - 1] - T[k])
    return np.nan

meas_profiles = []                                                  # (xi, T) centreline, mean of steady frames, per repeat
for tr in reps:
    kmid = int(np.median(tr.frames)); ks = range(kmid - 30, kmid + 31, 3)
    grid = np.linspace(a.xi[0], a.xi[1], 600); acc = []
    for k in ks:
        xi, y, T = tr.frame_temperature(k); j = int(round(tr.j0))
        c = np.nanmean(T[:, j - 1:j + 2], axis=1)
        sat = tr.S[k][:, j] >= 4095
        c = np.where(sat, np.inf, c)                                # saturated -> above T_sat
        acc.append(np.interp(grid, xi, np.where(np.isfinite(c), c, np.where(np.isinf(c), 1e4, np.nan)), left=np.nan, right=np.nan))
    prof = np.nanmedian(np.array(acc), axis=0); prof[prof > 5e3] = np.nan
    meas_profiles.append((grid, prof))

# ---------------- model
V = np.array(A.read_mesh_vertices(mesh)); top = np.where(np.abs(V[:, 2]) < 1e-9)[0]
xs, gi = np.unique(np.round(V[top, 0], 9), return_inverse=True); ys, gj = np.unique(np.round(V[top, 1], 9), return_inverse=True)

def read_field(fn):
    with open(fn, "rb") as f:
        ndof, step, n1, n2, nm = struct.unpack("5i", f.read(20)); t = struct.unpack("d", f.read(8))[0]
        T = np.frombuffer(f.read(8 * ndof), dtype=np.float64)
    G = np.full((len(xs), len(ys)), np.nan); G[gi, gj] = T[top] - K0
    return t, G

files = sorted(glob.glob(os.path.join(a.fields or os.path.join(a.run, "fields"), "f_*.bin")), key=lambda p: int(re.search(r"f_(\d+)", p).group(1)))
if not files: sys.exit(f"no field dumps in {a.run}/fields (run with FIELD_DUMP_DIR=<run>/fields FIELD_DUMP_EVERY=2)")
times = []
for fn in files:
    with open(fn, "rb") as f: f.read(20); times.append(struct.unpack("d", f.read(8))[0])
times = np.array(times); x_las = x0 + v * times
# steady snapshot: laser at the track midpoint x = 0 (as NIST's mid-track frame) when at least 2.6 mm of trail lie behind it,
# else as far along as possible while still scanning (2.1 mm segment runs); any dump interval works (one snapshot is used)
x_target = min(max(0.0, x0 + 2.6), x1 - 0.05)
k_snap = int(np.argmin(np.abs(x_las - x_target)))
t_snap, Gs = read_field(files[k_snap])
jc = int(np.argmin(np.abs(ys)))
xm = xs - (x0 + v * t_snap)                                          # relative to the laser centre
xi_front_model = front_edge(xm, Gs[:, jc], T_sat)
xm_rel = xm - xi_front_model                                         # relative to the model's saturation front edge
model_prof = (xm_rel, Gs[:, jc])

# ---------------- histories at the midpoint
t_p, T_p = A.read_probe(os.path.join(a.run, "probe.csv")); T_p = np.array(T_p) - K0; t_p = np.array(t_p)
up = np.where((T_p[:-1] < T_sat) & (T_p[1:] >= T_sat))[0]
t_front = t_p[up[0]] + (t_p[up[0] + 1] - t_p[up[0]]) * (T_sat - T_p[up[0]]) / (T_p[up[0] + 1] - T_p[up[0]]) if len(up) else np.nan
meas_hist = []
for tr in reps:
    i_mid = int(round(tr.i0 + tr.u_px * np.median(tr.frames)))
    th, Th, sat = tr.history(i_mid)
    meas_hist.append((th, Th))

# ---------------- figure
fig = plt.figure(figsize=(7.4, 8.6))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1.15, 1.15], hspace=0.55, wspace=0.28)
norm = matplotlib.colors.Normalize(940, T_sat); cmap = copy.copy(plt.get_cmap("inferno")); cmap.set_bad("#d9d8d3"); cmap.set_over("#fff7c2")
ext = [a.xi[0], a.xi[1], -a.ylim, a.ylim]
# (a) measured field, repeat 1, steady frame
tr = reps[0]; kmid = int(np.median(tr.frames)); xi, y, T = tr.frame_temperature(kmid)
prof_k = np.nanmean(T[:, int(round(tr.j0)) - 1:int(round(tr.j0)) + 2], axis=1)
sat_c = tr.S[kmid][:, int(round(tr.j0))] >= 4095
front_k = xi[np.where(sat_c)[0].max()] if sat_c.any() else 0.0
Tm = np.where(tr.S[kmid] >= 4095, 1e4, T)
ax = fig.add_subplot(gs[0, :])
ax.imshow(Tm.T, origin="lower", extent=[xi[0] - front_k, xi[-1] - front_k, y[0], y[-1]], cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_ylabel("y [mm]")
ax.set_title(f"(a) NIST thermography, case {case} repeat 1 (frame {kmid}, eps {tr.eps:.3f}); light = saturated (> {tr.t_sat():.0f} °C), grey = below camera floor", fontsize=8, loc="left", color=INK)
# (b) model field
ax2 = fig.add_subplot(gs[1, :])
XX = np.linspace(*a.xi, 600); YY = np.linspace(-a.ylim, a.ylim, 150)
f = RegularGridInterpolator((xm_rel, ys), Gs, bounds_error=False, fill_value=np.nan)
P = np.array(np.meshgrid(XX, YY, indexing="ij")).reshape(2, -1).T; Gm = f(P).reshape(len(XX), len(YY))
Gm_plot = np.where(Gm >= T_sat, 1e4, np.where(Gm < 940, np.nan, Gm))
im = ax2.imshow(Gm_plot.T, origin="lower", extent=ext, cmap=cmap, norm=norm, aspect="auto", interpolation="bilinear")
ax2.set_ylabel("y [mm]"); ax2.set_xlabel("ξ = distance from the saturation front edge [mm] (scan direction →)")
ax2.set_title(f"(b) model, nominal (t = {t_snap * 1e3:.2f} ms, laser at x = {x0 + v * t_snap:.2f} mm); same scale and masks", fontsize=8, loc="left", color=INK)
cb = fig.colorbar(im, ax=[ax, ax2], pad=0.01, fraction=0.025, extend="max"); cb.set_label("T [°C]")
# (c) isotherms
ax3 = fig.add_subplot(gs[2, 0])
cols = {1150: S3, 1260: S1, 1336: S2, 1400: "#e87ba4"}
for tr_i, trr in enumerate(reps):
    k = int(np.median(trr.frames)); xi_, y_, T_ = trr.frame_temperature(k)
    sc = trr.S[k][:, int(round(trr.j0))] >= 4095; fk = xi_[np.where(sc)[0].max()] if sc.any() else 0.0
    ax3.contour(xi_ - fk, y_, np.nan_to_num(T_, nan=0.0).T, levels=LEVELS, colors=[cols[l] for l in LEVELS], linewidths=0.9, linestyles="--", alpha=0.8)
ax3.contour(XX, YY, np.nan_to_num(Gm, nan=0.0).T, levels=LEVELS, colors=[cols[l] for l in LEVELS], linewidths=1.8)
ax3.set_xlim(a.xi[0], 0.1); ax3.set_ylim(-0.3, 0.3); ax3.set_xlabel("ξ [mm]"); ax3.set_ylabel("y [mm]")
ax3.set_title("(c) isotherms: model (solid) vs NIST (dashed, 3 repeats)", fontsize=8, loc="left", color=INK)
from matplotlib.lines import Line2D
ax3.legend(handles=[Line2D([], [], color=cols[l], lw=1.6, label=f"{l} °C") for l in LEVELS], fontsize=7, loc="lower left", ncol=2)
# (d) centreline profile
ax4 = fig.add_subplot(gs[2, 1])
for (g, p_), lab in zip(meas_profiles, ("NIST repeats", None, None)):
    ax4.plot(g, p_, color=INK2, lw=1.0, ls="--", label=lab)
ax4.plot(model_prof[0], model_prof[1], color=S1, lw=2, label="model")
for l in LEVELS: ax4.axhline(l, color=GRID, lw=0.8)
ax4.set_xlim(a.xi[0], 0.1); ax4.set_ylim(900, T_sat + 40); ax4.set_xlabel("ξ [mm]"); ax4.set_ylabel("T [°C] (centreline)")
ax4.set_title("(d) centreline profile behind the laser", fontsize=8, loc="left", color=INK); ax4.legend(fontsize=7, loc="upper left")
# (e) midpoint history
ax5 = fig.add_subplot(gs[3, :])
bands = ((1400, 1336, S2, "liquid CR"), (1336, 1260, "#eda100", "transition CR"), (1260, 1150, S3, "solid CR"))
for hi, lo, c, lab in bands: ax5.axhspan(lo, hi, color=c, alpha=0.13, lw=0, label=lab)
for (th, Th), lab in zip(meas_hist, ("NIST repeats", None, None)):
    ax5.plot(th * 1e3, Th, color=INK2, lw=1.0, ls="--", label=lab)
ax5.plot((t_p - t_front) * 1e3, T_p, color=S1, lw=2, label="model (probe x = 0)")
ax5.set_xlim(-0.1, 2.6); ax5.set_ylim(900, T_sat + 40); ax5.set_xlabel("time after the saturation front edge passed [ms]")
ax5.set_ylabel("T [°C]"); ax5.legend(fontsize=7, loc="upper right", ncol=2)
ax5.set_title("(e) temperature history at the track midpoint", fontsize=8, loc="left", color=INK)
fig.suptitle(f"AMB2022-03 case {case} (P {reps[0].P:.0f} W, v {v:.0f} mm/s, D4σ {reps[0].spot:.0f} µm): model vs NIST thermography",
             fontsize=10, x=0.02, ha="left", color=INK)
out = a.out or os.path.join(a.run, f"thermography_case{case}.png")
fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out, f"| T_sat {T_sat:.0f} C, eps", [round(t.eps, 3) for t in reps])
