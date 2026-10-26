#!/usr/bin/env python3
"""AMB2022-03 single-track quantities from one run (run/run_case.sh), compared with the NIST measurements.

  probe.csv  -> TTAM, TLCR, TTCR, TSCR at the top-surface centreline x = y = 0 (NIST definitions, challenge doc. sec. 3):
                TAM = time above 1298 C; liquid CR = linear fit 1400 C -> liquidus 1336 C; transition CR = liquidus ->
                solidus 1260 C; solid CR = solidus -> 1150 C (all on the cooling branch after the last melting).
  peakT.bin  -> melt-pool depth and width in the cross-section x = 0 (the mesh plane nearest 0): region where the peak
                temperature reached the threshold (default 1298 C; solidus and liquidus also reported), edges found by
                linear interpolation along the grid lines; depth = deepest point below z = 0, width = widest extent in y.
                Sensitivities d(depth)/ds_p and d(width)/ds_p: implicit differentiation of the isotherm Tpeak = Tm at the
                extreme point, d(position)/ds = -(dTpeak/ds) / (dTpeak/dn) (dTpeak/ds = dT/ds at the time of the peak).
usage: analyze_case.py RUNDIR [--mesh MESH] [--case 0] [--json out.json]
"""
import argparse, array, csv, json, math, os, re

PARAMS = ["k", "cp", "rho", "Lfus", "Ts", "Tliq", "hc", "T0", "P", "Eta", "Rx", "Ry", "Rz", "X0", "Vx", "TL", "Tw", "hp", "kp", "Rhop"]
K0 = 273.15
T_SOL, T_LIQ, T_MID, T_1400, T_1150 = 1260 + K0, 1336 + K0, 1298 + K0, 1400 + K0, 1150 + K0
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)


def read_probe(fn):
    rows = [l.split(",") for l in open(fn).read().split("\n") if l.strip()]
    data = [(float(r[0]), float(r[1])) for r in rows[1:]]
    return [d[0] for d in data], [d[1] for d in data]


def crossings(t, T, level):
    """times where T crosses level (linear interpolation), with direction +1 up / -1 down"""
    out = []
    for i in range(len(T) - 1):
        a, b = T[i] - level, T[i + 1] - level
        if a == 0: continue
        if a * b <= 0 and b != a:
            out.append((t[i] + (t[i + 1] - t[i]) * a / (a - b), 1 if b > a else -1))
    return out


def fit_slope(t, T, hi, lo, t_after):
    """linear-fit slope of T(t) on the cooling branch after t_after where lo <= T <= hi (needs >= 2 points; else secant)"""
    pts = [(ti, Ti) for ti, Ti in zip(t, T) if ti >= t_after and lo <= Ti <= hi]
    c_hi = [c for c, d in crossings(t, T, hi) if d < 0 and c >= t_after]; c_lo = [c for c, d in crossings(t, T, lo) if d < 0 and c >= t_after]
    if len(pts) >= 3:
        n = len(pts); mt = sum(p[0] for p in pts) / n; mT = sum(p[1] for p in pts) / n
        s = sum((p[0] - mt) * (p[1] - mT) for p in pts) / sum((p[0] - mt) ** 2 for p in pts)
        return -s, n
    if c_hi and c_lo and c_lo[0] > c_hi[0]: return (hi - lo) / (c_lo[0] - c_hi[0]), 0   # secant between the crossings
    return float("nan"), len(pts)


def probe_quantities(t, T):
    up = [c for c, d in crossings(t, T, T_MID) if d > 0]; dn = [c for c, d in crossings(t, T, T_MID) if d < 0]
    if not up or not dn: return {"TTAM_s": float("nan"), "peak_K": max(T)}
    t_up = up[0]; t_dn = [c for c in dn if c > t_up][-1]
    tpk = t[T.index(max(T))]
    lcr, n_l = fit_slope(t, T, T_1400, T_LIQ, tpk); tcr, n_t = fit_slope(t, T, T_LIQ, T_SOL, tpk); scr, n_s = fit_slope(t, T, T_SOL, T_1150, tpk)
    return {"TTAM_s": t_dn - t_up, "TLCR_C_s": lcr, "TTCR_C_s": tcr, "TSCR_C_s": scr, "peak_K": max(T),
            "fit_points": {"liquid": n_l, "transition": n_t, "solid": n_s}, "t_cool_below_1150C": ([c for c, d in crossings(t, T, T_1150) if d < 0] or [float('nan')])[-1]}


def read_mesh_vertices(fn):
    L = open(fn).read().split("\n"); j = L.index("vertices"); nv = int(L[j + 1])
    return [tuple(map(float, L[j + 3 + k].split())) for k in range(nv)]


def melt_pool(V, Tpk, dTpk, x_sec, level):
    """depth/width of {Tpeak >= level} in the plane x = x_sec (grid plane), and their derivatives"""
    plane = [i for i, v in enumerate(V) if abs(v[0] - x_sec) < 1e-9]
    ys = sorted({round(V[i][1], 9) for i in plane}); zs = sorted({round(V[i][2], 9) for i in plane})
    idx = {(round(V[i][1], 9), round(V[i][2], 9)): i for i in plane}
    npar = len(dTpk)
    def edge(i_in, i_out, coord_in, coord_out):   # interpolated isotherm position between an inside and an outside node
        Ta, Tb = Tpk[i_in], Tpk[i_out]; f = (Ta - level) / (Ta - Tb); pos = coord_in + f * (coord_out - coord_in)
        dTdn = (Tb - Ta) / (coord_out - coord_in)                              # along the search direction
        dT_s = [(1 - f) * dTpk[p][i_in] + f * dTpk[p][i_out] for p in range(npar)]
        return pos, [-dT_s[p] / dTdn for p in range(npar)]                       # d(pos)/ds
    best_depth = (0.0, [0.0] * npar)                                            # depth = -z of the deepest isotherm crossing
    for y in ys:
        col = [idx[(y, z)] for z in zs]                                           # z ascending (bottom -> top)
        for k in range(len(col) - 1, 0, -1):                                      # from the top down: first inside -> outside
            if Tpk[col[k]] >= level and Tpk[col[k - 1]] < level:
                z, dz = edge(col[k], col[k - 1], zs[k], zs[k - 1])
                if -z > best_depth[0]: best_depth = (-z, [-d for d in dz])
                break
    ymin, ymax = (0.0, [0.0] * npar), (0.0, [0.0] * npar)
    for z in zs:
        row = [idx[(y, z)] for y in ys]
        for k in range(len(row) - 1):
            if Tpk[row[k + 1]] >= level and Tpk[row[k]] < level:                  # left edge (outside at k, inside at k+1)
                y, dy = edge(row[k + 1], row[k], ys[k + 1], ys[k])
                if y < ymin[0]: ymin = (y, dy)
            if Tpk[row[k]] >= level and Tpk[row[k + 1]] < level:                  # right edge
                y, dy = edge(row[k], row[k + 1], ys[k], ys[k + 1])
                if y > ymax[0]: ymax = (y, dy)
    width = ymax[0] - ymin[0]; dwidth = [ymax[1][p] - ymin[1][p] for p in range(npar)]
    return best_depth[0], best_depth[1], width, dwidth


def main():
    P = argparse.ArgumentParser(); P.add_argument("run"); P.add_argument("--mesh"); P.add_argument("--case"); P.add_argument("--json")
    a = P.parse_args()
    st = open(os.path.join(a.run, "settings.txt")).read()
    case = a.case or re.search(r"case=(\S+)", st).group(1)
    mesh = a.mesh or re.search(r"mesh=(\S+(?: \S+)*?\.mesh)", st).group(1)
    meas = {r["case"]: r for r in csv.DictReader(open(os.path.join(ROOT, "inputs", "cases.csv")))}[case]
    t, T = read_probe(os.path.join(a.run, "probe.csv"))
    res = {"case": case, "settings": st.strip(), "probe": probe_quantities(t, T)}
    with open(os.path.join(a.run, "peakT.bin"), "rb") as f:
        hd = array.array("i"); hd.fromfile(f, 2); n, n1 = hd
        Tpk = array.array("d"); Tpk.fromfile(f, n); d = array.array("d"); d.fromfile(f, n * n1)
    dTpk = [d[p * n:(p + 1) * n] for p in range(n1)]
    V = read_mesh_vertices(mesh)
    xs = sorted({round(v[0], 9) for v in V}); x_sec = min(xs, key=lambda x: abs(x))
    res["section_x_mm"] = x_sec
    for name, lev in (("mid_1298C", T_MID), ("solidus", T_SOL), ("liquidus", T_LIQ)):
        dep, ddep, wid, dwid = melt_pool(V, Tpk, dTpk, x_sec, lev)
        res[f"melt_pool_{name}"] = {"depth_um": dep * 1e3, "width_um": wid * 1e3,
                                    "d_depth_um": {PARAMS[p]: ddep[p] * 1e3 for p in range(n1)},
                                    "d_width_um": {PARAMS[p]: dwid[p] * 1e3 for p in range(n1)}}
    m = res["melt_pool_mid_1298C"]; pr = res["probe"]
    def row(lbl, model, meas_v, unit, fmt="{:.3g}"):
        mv = float(meas_v); dev = (model - mv) / mv * 100 if model == model else float("nan")
        print(f"  {lbl:28s} model {fmt.format(model):>10s}   NIST {fmt.format(mv):>10s} {unit:6s}  ({dev:+.1f} %)")
    print(f"case {case}: {st.splitlines()[0]}")
    row("depth (1298 C isotherm)", m["depth_um"], meas["depth_um"], "um", "{:.1f}")
    row("width (1298 C isotherm)", m["width_um"], meas["width_um"], "um", "{:.1f}")
    row("TTAM", pr.get("TTAM_s", float("nan")), meas["TTAM_s"], "s")
    for q, k in (("TLCR", "TLCR_C_s"), ("TTCR", "TTCR_C_s"), ("TSCR", "TSCR_C_s")):
        row(q, pr.get(k, float("nan")), meas[k], "C/s")
    print(f"  peak T at the probe {pr['peak_K']:.0f} K; fit points (liquid/transition/solid) {pr.get('fit_points')}")
    print(f"  solidus / liquidus isotherms: depth {res['melt_pool_solidus']['depth_um']:.1f} / {res['melt_pool_liquidus']['depth_um']:.1f} um, "
          f"width {res['melt_pool_solidus']['width_um']:.1f} / {res['melt_pool_liquidus']['width_um']:.1f} um")
    top = lambda dd: ", ".join(f"{k} {v:+.1f}" for k, v in sorted(dd.items(), key=lambda kv: -abs(kv[1]))[:6])
    print(f"  d(depth)/ds [um per unit relative change]: {top(m['d_depth_um'])}")
    print(f"  d(width)/ds [um per unit relative change]: {top(m['d_width_um'])}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
