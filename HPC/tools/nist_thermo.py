"""NIST AMB2022-03 in-situ thermography (dataset mds2-2716, file AMB2022-03-718-AMMT-StaringCamera_Signal.h5): read one
single track, convert the raw camera signal to temperature and locate the laser in every frame.

Camera: 30 kfps, 640 x 304 px, 20.4 x 20.8 um/px (challenge doc. sec. 2.4), signal = 12-bit digital levels; values below
100 DL are stored as 0 (no data, < ~940 C) and 4095 = saturated (melt pool). The scan runs along the first image axis (i).
Calibration (file attributes /Calibration/ThermalCal, corrected 9/12/24):
    T [C] = 14388 / (a ln(c eps / S + 1)) - b / a,    a = 0.9655, b = 197.2, c = 4.392e7
with an emissivity eps per track. NIST chose eps so that the solidification plateau of the apparent temperature sits at the
midpoint of solidus and liquidus, 1298 C (results doc. sec. 1, Fig. 1); fit_emissivity() repeats that procedure.
Groups: Line_<X>_<Y>_<R> = case X.Y, repeat R (case 0: Line_0_<R>).
"""
import math
import h5py
import numpy as np

PX_I_MM, PX_J_MM = 0.0204, 0.0208          # nominal pixel size along the scan (i) and across it (j) [mm]
# Along the scan the image moves 4.4 % less than the commanded speed predicts with 20.4 um/px in every case (919 vs 960,
# 1149 vs 1200, 765 vs 800 mm/s); the commanded speed is good to +-2.5 % (dataset README), so Track.px_i calibrates the
# pixel size along the scan from the commanded speed (~21.3 um/px).
FPS = 30000.0
T_PLATEAU = 1298.0                          # [C] midpoint of solidus 1260 and liquidus 1336


def group(case, rep):
    return f"Line_0_{rep}" if case == "0" else f"Line_{case.replace('.', '_')}_{rep}"


class Track:
    def __init__(self, h5file, case, rep):
        f = h5py.File(h5file, "r")
        g = f["ThermalData/" + group(case, rep)]
        self.S = g["Signal"][...].astype(np.float64)               # [frame, i, j]
        cal = f["Calibration/ThermalCal"].attrs
        self.a, self.b, self.c = float(cal["Coeff_a"][0]), float(cal["Coeff_b"][0]), float(cal["Coeff_c"][0])
        self.P, self.v, self.spot = (float(g.attrs[k][0]) for k in ("laser_power", "scan_speed", "spot_size"))
        self.case, self.rep = case, rep
        self.eps = None

    # ---- calibration
    def temp(self, S, eps):
        """signal -> temperature [C]; nan where there is no data (0) or the camera is saturated (4095)"""
        S = np.asarray(S, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            T = 14388.0 / (self.a * np.log(self.c * eps / S + 1.0)) - self.b / self.a
        return np.where((S > 0) & (S < 4095), T, np.nan)

    def signal_at(self, T, eps):
        return self.c * eps / (math.exp(14388.0 / (self.a * T + self.b)) - 1.0)

    def t_sat(self, eps=None):
        return float(self.temp(4094.0, eps or self.eps))

    # ---- laser position: hottest (saturated) region centroid per frame
    def laser_path(self):
        """frame -> (i, j) of the laser (centroid of the saturated pixels, else of the maximum); fitted line i(t) = i0 + u t
        over the scanning frames. Returns frames, i_fit, j_mean, speed in px/frame."""
        fr, ii, jj = [], [], []
        for k in range(self.S.shape[0]):
            F = self.S[k]
            if F.max() < 4095: continue
            idx = np.argwhere(F >= 4095)
            fr.append(k); ii.append(idx[:, 0].max()); jj.append(idx[:, 1].mean())   # front edge of the saturated spot
        fr, ii, jj = np.array(fr), np.array(ii, float), np.array(jj)
        u, i0 = np.polyfit(fr, ii, 1)
        self.frames, self.u_px, self.i0, self.j0 = fr, u, i0, float(np.median(jj))
        self.px_i = self.v / (u * FPS)                                     # [mm/px] from the commanded speed
        return fr, i0 + u * fr, self.j0, u

    # ---- emissivity as NIST: solidification plateau of the apparent temperature at 1298 C
    def fit_emissivity(self, n_px=30):
        """centreline pixels in the middle of the track: on each pixel's cooling branch (after saturation ends) take the
        signal where |dS/dt| is smallest (plateau), median over n_px pixels; eps such that T(S_plateau) = 1298 C."""
        if not hasattr(self, "u_px"): self.laser_path()
        j = int(round(self.j0)); i_lo, i_hi = int(self.i0 + self.u_px * self.frames[0]), int(self.i0 + self.u_px * self.frames[-1])
        im = (i_lo + i_hi) // 2
        plateau = []
        for i in range(im - n_px // 2, im + n_px // 2):
            s = self.S[:, i, j - 1:j + 2].mean(axis=1)                     # 3-pixel average across the track
            sat = np.where(s >= 4095 - 1e-9)[0]
            if len(sat) == 0: continue
            cool = s[sat[-1] + 1:]; cool = cool[cool > 0]
            if len(cool) < 8: continue
            ds = np.abs(np.convolve(np.diff(cool), np.ones(3) / 3, mode="valid"))
            k = int(np.argmin(ds[:len(ds) * 2 // 3])) + 1                     # plateau before the signal dies out
            plateau.append(cool[k])
        S_pl = float(np.median(plateau))
        self.eps = S_pl * (math.exp(14388.0 / (self.a * T_PLATEAU + self.b)) - 1.0) / self.c
        self.S_plateau = S_pl
        return self.eps

    # ---- data in the laser frame
    def frame_temperature(self, k):
        """temperature field [C] of frame k on (xi, y) [mm] relative to the front edge of the saturated spot (xi along the
        scan, negative behind it)"""
        T = self.temp(self.S[k], self.eps)
        xi = (np.arange(T.shape[0]) - (self.i0 + self.u_px * k)) * self.px_i
        y = (np.arange(T.shape[1]) - self.j0) * PX_J_MM
        return xi, y, T

    def history(self, i, j=None):
        """temperature history [C] of pixel (i, j) (3-pixel average across the track); time [s] from the saturated-spot
        front edge passing the pixel"""
        j = int(round(self.j0)) if j is None else j
        s = self.S[:, i, j - 1:j + 2]
        s = np.where((s > 0) & (s < 4095), s, np.nan); s = np.nanmean(s, axis=1) if s.ndim > 1 else s
        sat = (self.S[:, i, j] >= 4095)
        t = (np.arange(self.S.shape[0]) - (i - self.i0) / self.u_px) / FPS
        return t, self.temp(np.nan_to_num(s, nan=0.0), self.eps), sat
