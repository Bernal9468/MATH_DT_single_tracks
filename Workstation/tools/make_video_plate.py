#!/usr/bin/env python3
"""Plate-layout video: the 21 AMB2022-03 single tracks (7 cases x 3 repeats) at their places on the IN718 plate, scanned one
after the other in the line-number order, banded temperature [K] as tools/make_video_3d.py.

Layout from the challenge description, Fig. 3 (measured on the figure, the 10 mm track length as scale): two groups of
parallel tracks along +X, pitch ~0.51 mm, 10 mm long, ~2.2 mm apart in x; right group Line 0_1 ... 2.1_2 (11 tracks), left
group Line 2.1_3 ... 3.2_3 (10 tracks), first line of the left group ~0.19 mm below the first line of the right group.
Each track shows its own simulation (results/video/case<c>, dense dumps): the simulated window x0..x1 around the track
midpoint (model x = 0 at the track midpoint, where the camera measured); the 3 repeats of a case share the simulation (the
model is deterministic). The plate is back at ambient before each track (NIST: < 0.1 C rise per line scan).
--cases: one track per case (0, 1.1, ... 3.2) instead of the 21 lines: 7 tracks on the right-group geometry (same pitch, length
and x), a simplified layout (NIST placed the repeats of a case side by side), labelled by case.
Panels: plate overview (top), the active track: top view and longitudinal section on the centreline (bottom).
usage: make_video_plate.py [DIR] [--cases] [--every 2] [--fps 30] [--out VIDEO.mp4] [--frame N --png FIG.png]
"""
import argparse, csv, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.patheffects

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "post"))
import analyze_case as A
import field_io as F
try:
    import imageio_ffmpeg; matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white"})
BANDS = ["#0012ff", "#0060ff", "#00aaff", "#00dcf0", "#00d28c", "#00b843", "#66cc00", "#c8e000", "#ffd700", "#ffa500",
         "#ff5a00", "#ff1e00"]                                   # same bands as make_video_3d.py
LEV = np.linspace(300.0, 2400.0, len(BANDS) + 1)
CLEV = np.concatenate([[0], LEV[1:-1], [1e5]])
CMAP = matplotlib.colors.ListedColormap(BANDS); NORM = matplotlib.colors.BoundaryNorm(LEV, CMAP.N)
D4S = "D4σ: laser spot diameter = 4σ, σ the standard deviation of the Gaussian intensity profile I(r) = I₀ exp(−r²/2σ²); equal to the 1/e² diameter (e.g. 67 µm: σ = 16.75 µm, 1/e² radius 33.5 µm)"

# plate layout [mm], y up (Fig. 3): line name -> (x of the track start, y)
L_TRACK, PITCH, X_RIGHT, DY_LEFT = 10.0, 0.51, 12.2, -0.19
RIGHT = ["0_1", "0_2", "0_3", "1.1_1", "1.1_2", "1.1_3", "1.2_1", "1.2_2", "1.2_3", "2.1_1", "2.1_2"]
LEFT = ["2.1_3", "2.2_1", "2.2_2", "2.2_3", "3.1_1", "3.1_2", "3.1_3", "3.2_1", "3.2_2", "3.2_3"]
POS = {n: (X_RIGHT, -k * PITCH) for k, n in enumerate(RIGHT)}
POS.update({n: (0.0, DY_LEFT - k * PITCH) for k, n in enumerate(LEFT)})
ORDER = sorted(POS, key=lambda n: (float(n.split("_")[0]), int(n.split("_")[1])))      # scan order = line numbering
HALF = PITCH / 2                                                # field drawn up to half the pitch on each side of a track

ap = argparse.ArgumentParser(); ap.add_argument("dir", nargs="?", default=os.path.join(ROOT, "results", "video"))
ap.add_argument("--every", type=int, default=2); ap.add_argument("--fps", type=int, default=30); ap.add_argument("--out")
ap.add_argument("--frame", type=int); ap.add_argument("--png"); ap.add_argument("--cases", action="store_true")
a = ap.parse_args()
LABEL = {n: n for n in POS}
if a.cases:                                                     # one track per case, right-group geometry
    ORDER = [f"{c}_1" for c in ("0", "1.1", "1.2", "2.1", "2.2", "3.1", "3.2")]
    POS = {n: (X_RIGHT, -k * PITCH) for k, n in enumerate(ORDER)}; LABEL = {n: "case " + n.split("_")[0] for n in ORDER}
CASES = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}


class Run:
    """one case: field dumps sampled on the top surface (z = 0) and on the centreline section (y = 0)"""
    def __init__(self, run):
        st = open(os.path.join(run, "settings.txt")).read()
        self.x0, self.x1 = (float(re.search(k + r"=(\S+)", st).group(1)) for k in ("x0", "x1"))
        lp = [l.split(",") for l in open(os.path.join(run, "laser.inp")) if l.strip() and not l.startswith("#")]
        self.v = abs(float(lp[1][1]) - float(lp[0][1])) / (float(lp[1][0]) - float(lp[0][0]))
        self.t_off = (self.x1 - self.x0) / self.v
        V = np.array(A.read_mesh_vertices(re.search(r"mesh=(.+?\.mesh)", st).group(1)))
        self.top = F.Plane(V, 2, 0.0); self.sec = F.Plane(V, 1, 0.0)
        self.ti = (self.top.u >= self.x0 - 0.1) & (self.top.u <= self.x1 + 0.1); self.tj = np.abs(self.top.w) <= 0.6 + 1e-9
        self.si = self.ti.copy() if np.array_equal(self.sec.u, self.top.u) else (self.sec.u >= self.x0 - 0.1) & (self.sec.u <= self.x1 + 0.1)
        self.sj = self.sec.w >= -0.45 - 1e-9
        self.files = F.dump_files(os.path.join(run, "fields"))[::a.every]
        self.nodes = np.concatenate([self.top.nodes, self.sec.nodes]); self.cut = len(self.top.nodes)

    def fields(self, fn):
        T, t = F.read(fn, [0], self.nodes); T = T[0]
        return t, self.top.grid(T[:self.cut])[np.ix_(self.ti, self.tj)], self.sec.grid(T[self.cut:])[np.ix_(self.si, self.sj)]


runs = {}
for n in ORDER:
    c = n.split("_")[0]
    if c not in runs and os.path.isdir(os.path.join(a.dir, f"case{c}", "fields")): runs[c] = Run(os.path.join(a.dir, f"case{c}"))
seq = [n for n in ORDER if n.split("_")[0] in runs]
frames = [(n, k) for n in seq for k in range(len(runs[n.split("_")[0]].files))]
print(len(seq), "tracks,", len(frames), "frames; missing cases:", sorted({n.split("_")[0] for n in ORDER} - set(runs)))

fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
axp = fig.add_axes([0.11, 0.52, 0.87, 0.34])                    # plate overview
axz = fig.add_axes([0.13, 0.13, 0.38, 0.31])                    # active track, top view
axs = fig.add_axes([0.57, 0.13, 0.41, 0.31])                    # active track, centreline section
axc = fig.add_axes([0.012, 0.13, 0.016, 0.73])
cb = matplotlib.colorbar.ColorbarBase(axc, cmap=CMAP, norm=NORM, boundaries=LEV, ticks=LEV, spacing="uniform")
cb.ax.set_yticklabels([f"{x:.0f}" for x in LEV]); cb.ax.tick_params(labelsize=8, length=0); cb.outline.set_edgecolor("#555555")
for k in range(1, len(LEV) - 1): cb.ax.axhline(LEV[k], color="#333333", lw=0.6)
fig.text(0.008, 0.875, r"$T$ [K]", fontsize=13, fontweight="bold")
h_c = fig.text(0.07, 0.945, "", fontsize=15, fontweight="bold")
h_t = fig.text(0.07, 0.905, "", fontsize=13, fontweight="bold")
h_l = fig.text(0.30, 0.905, "", fontsize=13, fontweight="bold")
h_m = fig.text(0.45, 0.905, "", fontsize=13)
fig.text(0.98, 0.008, ("one track per case on the AMB2022-03 track geometry (pitch, length: NIST challenge description, Fig. 3); "
         "simulated window around each track midpoint;\nplate back at ambient between tracks\n" + D4S if a.cases else
         "AMB2022-03 plate layout (NIST challenge description, Fig. 3); simulated window around each track midpoint;\n"
         "repeats share the simulation; tracks scanned in line order, plate back at ambient between tracks\n" + D4S),
         fontsize=8, color="#555555", ha="right", va="bottom")
ys = [p[1] for p in POS.values()]
PX = (min(p[0] for p in POS.values()) - 1.0, max(p[0] for p in POS.values()) + L_TRACK + 1.0); PY = (min(ys) - 0.45, max(ys) + 0.45)


def style(ax, xl, yl):
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_aspect("equal"); ax.tick_params(labelsize=8)
    for s in ax.spines.values(): s.set_linewidth(1.2); s.set_color("black")


def draw(f):
    n, k = frames[f]; c = n.split("_")[0]; R = runs[c]; fn = R.files[k]; t, Gt, Gs = R.fields(fn)
    xs, ys_ = POS[n]; xm = xs + L_TRACK / 2                     # model x = 0 at the track midpoint
    xl = R.x0 + R.v * t; on = t <= R.t_off + 1e-12
    u, w = R.top.u[R.ti], R.top.w[R.tj]; band = np.abs(w) <= HALF
    # plate overview
    axp.cla(); axp.set_facecolor(BANDS[0]); style(axp, PX, PY)
    done = seq[:seq.index(n)]
    for m, (x_, y_) in POS.items():
        if m == n: continue
        axp.plot([x_, x_ + L_TRACK], [y_, y_], color="#9a9a9a" if m in done else "white", lw=1.6 if m in done else 0.7,
                 ls="-" if m in done else (0, (3, 3)), alpha=1.0 if m in done else 0.6)
    axp.plot([xs, xs + L_TRACK], [ys_, ys_], color="white", lw=0.7, ls=(0, (3, 3)))
    axp.contourf(xm + u, ys_ + w[band], Gt[:, band].T, levels=CLEV, colors=BANDS)
    axp.plot([xm + R.x0, xm + R.x1, xm + R.x1, xm + R.x0, xm + R.x0], [ys_ - HALF, ys_ - HALF, ys_ + HALF, ys_ + HALF, ys_ - HALF],
             color="black", lw=0.8)
    if on: axp.scatter([xm + xl], [ys_ + 0.33], marker="v", s=70, color="black", zorder=6)
    for m, (x_, y_) in POS.items():
        side = (x_ - 0.1, "right") if x_ == 0.0 else (x_ + L_TRACK + 0.1, "left")
        axp.text(side[0], y_, LABEL[m], fontsize=7 if m == n else 6.5, va="center", ha=side[1], color="#ffd700" if m == n else "white",
                 fontweight="bold" if m == n else "normal", clip_on=False)
    axp.set_xlabel("x [mm]", fontsize=9); axp.set_ylabel("y [mm]", fontsize=9)
    # active track, top view and section
    axz.cla(); axz.contourf(u, w, Gt.T, levels=CLEV, colors=BANDS); style(axz, (u[0], u[-1]), (w[0], w[-1]))
    if on: axz.scatter([xl], [0.0], marker="v", s=140, color="black", zorder=5)
    axz.set_title(f"{'Line ' if not a.cases else ''}{LABEL[n]}: top surface", fontsize=10, fontweight="bold"); axz.set_xlabel("x from track midpoint [mm]", fontsize=9)
    axz.set_ylabel("y [mm]", fontsize=9)
    us, ws = R.sec.u[R.si], R.sec.w[R.sj]
    axs.cla(); axs.contourf(us, ws, Gs.T, levels=CLEV, colors=BANDS); style(axs, (us[0], us[-1]), (ws[0], ws[-1]))
    axs.contour(us, ws, Gs.T, levels=[1609.0], colors="black", linewidths=0.9)      # liquidus: the melt pool
    if on: axs.scatter([xl], [0.03], marker="v", s=140, color="black", zorder=5, clip_on=False)
    axs.set_title("centreline section (black: liquidus 1609 K)", fontsize=10, fontweight="bold", pad=16)
    axs.set_xlabel("x from track midpoint [mm]", fontsize=9); axs.set_ylabel("z [mm]", fontsize=9)
    C = CASES[c]
    pv = f"case {c}:  P = {float(C['power_W']):.0f} W,  v = {float(C['speed_mm_s']):.0f} mm/s,  spot D4σ = {float(C['spot_D4sigma_um']):.0f} µm"
    h_c.set_text(f"Track {seq.index(n) + 1}/{len(seq)}:  " + (pv if a.cases else f"Line {n}  ({pv})"))
    h_t.set_text(rf"$t$ = {t * 1e3:.4f} ms"); h_l.set_text("Laser ON" if on else "Laser OFF"); h_l.set_color("#e00000" if on else "#666666")
    h_m.set_text(rf"$T_{{max}}$ (top surface) = {Gt.max():.0f} K")


if a.frame is not None:
    draw(a.frame); fig.savefig(a.png or "frame.png", dpi=150); print("written", a.png or "frame.png"); sys.exit()
out = a.out or os.path.join(a.dir, "video_plate_7cases.mp4" if a.cases else "video_plate_all_tracks.mp4")
wr = anim.FFMpegWriter(fps=a.fps, bitrate=8000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
with wr.saving(fig, out, dpi=150):
    for f in range(len(frames)):
        draw(f); wr.grab_frame()
        if f % 100 == 0: print(f"frame {f}/{len(frames)}", flush=True)
print("written", out)
