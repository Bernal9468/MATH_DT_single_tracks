#!/usr/bin/env python3
"""AMB2022-03 single-track quantities from one run (run/run_case.sh), compared with the NIST measurements.

  probe.csv  -> TTAM, TLCR, TTCR, TSCR at the top-surface centreline x = y = 0 (NIST definitions, challenge doc. sec. 3)
                and, with sens.csv (dT/ds histories at the probe), their exact derivatives (probe_sensitivities):
                TAM = time above 1298 C; liquid CR = linear fit 1400 C -> liquidus 1336 C; transition CR = liquidus ->
                solidus 1260 C; solid CR = solidus -> 1150 C (all on the cooling branch after the last melting).
  peakT.bin  -> melt-pool depth and width in the cross-section x = 0 (the mesh plane nearest 0): region where the peak
                temperature reached the threshold (default 1298 C; solidus and liquidus also reported), edges found by
                linear interpolation along the grid lines; depth = deepest point below z = 0, width = widest extent in y.
                Sensitivities d(depth)/ds_p and d(width)/ds_p: implicit differentiation of the isotherm Tpeak = Tm at the
                extreme point, d(position)/ds = -(dTpeak/ds) / (dTpeak/dn) (dTpeak/ds = dT/ds at the time of the peak).
usage: analyze_case.py RUNDIR [--mesh MESH] [--case 0] [--json out.json]
"""
import argparse, array, csv, json, math, os, re, sys

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
    # cooling rates: continuous-window least-squares slope (window_rate; smooth in the parameters, <= 2 % from the
    # sample-based fit_slope, which is kept for the fit point counts)
    n_l = fit_slope(t, T, T_1400, T_LIQ, tpk)[1]; n_t = fit_slope(t, T, T_LIQ, T_SOL, tpk)[1]; n_s = fit_slope(t, T, T_SOL, T_1150, tpk)[1]
    lcr, tcr, scr = (window_rate(t, T, hi, lo, tpk) for hi, lo in ((T_1400, T_LIQ), (T_LIQ, T_SOL), (T_SOL, T_1150)))
    return {"TTAM_s": t_dn - t_up, "TLCR_C_s": lcr, "TTCR_C_s": tcr, "TSCR_C_s": scr, "peak_K": max(T),
            "fit_points": {"liquid": n_l, "transition": n_t, "solid": n_s}, "t_cool_below_1150C": ([c for c, d in crossings(t, T, T_1150) if d < 0] or [float('nan')])[-1]}


def window_rate(t, T, hi, lo, t_after):
    """cooling rate over the continuous window: least-squares line through the piecewise-linear T(t) between the first
    downward crossing of hi after t_after and the next downward crossing of lo, integrated exactly; returns -slope (C/s),
    nan if the window is not complete. Smooth in T (no sample enters or leaves the window), so its derivative is exact."""
    t1 = t2 = None
    for i in range(len(T) - 1):
        if t[i + 1] < t_after: continue
        for lev in ((hi,) if t1 is None else (lo,)):
            a, b = T[i] - lev, T[i + 1] - lev
            if a > 0 >= b:
                tc = t[i] + (t[i + 1] - t[i]) * a / (a - b)
                if t1 is None and tc >= t_after: t1 = tc
                elif t1 is not None and tc > t1: t2 = tc
        if t2 is not None: break
    if t1 is None or t2 is None: return float("nan")
    I0 = I1 = I2 = J0 = J1 = 0.0
    for i in range(len(T) - 1):
        ta, tb = max(t[i], t1), min(t[i + 1], t2)
        if tb <= ta: continue
        lin = lambda x: T[i] + (T[i + 1] - T[i]) * (x - t[i]) / (t[i + 1] - t[i])
        Ta, Tb = lin(ta), lin(tb); h = tb - ta
        I0 += h; I1 += (tb ** 2 - ta ** 2) / 2; I2 += (tb ** 3 - ta ** 3) / 3
        J0 += h * (Ta + Tb) / 2
        J1 += h * (ta * (2 * Ta + Tb) + tb * (Ta + 2 * Tb)) / 6                    # exact integral of t T(t) on the segment
    return -(I0 * J1 - I1 * J0) / (I0 * I2 - I1 ** 2)


def read_sens(fn):
    """probe sensitivity histories dT/ds_p (sens.csv, columns dT_d<p>_row1) -> {p: [values]}; {} without sensitivities"""
    if not os.path.exists(fn): return {}
    rows = [l.split(",") for l in open(fn).read().split("\n") if l.strip()]
    out = {}
    for j, h in enumerate(rows[0]):
        m = re.match(r"dT_d(.+)_row1$", h)
        if m and m.group(1) in PARAMS: out[m.group(1)] = [float(r[j]) for r in rows[1:]]
    return out


def probe_sensitivities(t, T, S):
    """exact derivatives of TTAM and the three cooling rates of probe_quantities with respect to every parameter in S
    (dT/ds histories at the probe), for the same crossings and fit points:
      crossing time between samples i, i+1 (a = T_i - L, b = T_i+1 - L): t_c = t_i + dt a/(a - b),
      d t_c/ds = dt (a db - b da)/(a - b)^2;  TTAM = t_down - t_up;
      cooling rates (window_rate, continuous window between the level crossings): derivative along T + eps dT/ds.
    Returns ({quantity: value}, {quantity: {p: d quantity / d s_p}}); the values must equal probe_quantities()."""
    def cross(level, direction, after=-1.0, first=True):
        hits = []
        for i in range(len(T) - 1):
            a, b = T[i] - level, T[i + 1] - level
            if a == 0 or a * b > 0 or b == a: continue
            if (1 if b > a else -1) != direction: continue
            tc = t[i] + (t[i + 1] - t[i]) * a / (a - b)
            if tc >= after: hits.append((tc, i, a, b))
        if not hits: return None
        tc, i, a, b = hits[0] if first else hits[-1]
        dt = t[i + 1] - t[i]
        return tc, {p: dt * (a * s[i + 1] - b * s[i]) / (a - b) ** 2 for p, s in S.items()}
    val, der = {}, {}
    up = cross(T_MID, +1)
    if up is None: return val, der
    dn = cross(T_MID, -1, after=up[0], first=False)
    if dn is None: return val, der
    val["TTAM_s"] = dn[0] - up[0]; der["TTAM_s"] = {p: dn[1][p] - up[1][p] for p in S}
    # cooling rates: directional derivative of window_rate along the linearised history T + eps dT/ds (central
    # difference in eps on the post-processing only; the temperature derivative itself is the AD one)
    tpk = t[T.index(max(T))]
    for q, hi, lo in (("TLCR_C_s", T_1400, T_LIQ), ("TTCR_C_s", T_LIQ, T_SOL), ("TSCR_C_s", T_SOL, T_1150)):
        v0 = window_rate(t, T, hi, lo, tpk)
        if v0 != v0: continue
        val[q] = v0; der[q] = {}
        for p, sp in S.items():
            m = max(abs(x) for x in sp)
            if m == 0: der[q][p] = 0.0; continue
            eps = 1e-3 / m                                  # temperature perturbation of at most 1e-3 K
            vp = window_rate(t, [a + eps * b for a, b in zip(T, sp)], hi, lo, tpk)
            vm = window_rate(t, [a - eps * b for a, b in zip(T, sp)], hi, lo, tpk)
            der[q][p] = (vp - vm) / (2 * eps)
    return val, der


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
    S = read_sens(os.path.join(a.run, "sens.csv"))
    if S:
        if len(next(iter(S.values()))) != len(t): sys.exit("sens.csv and probe.csv have different numbers of rows")
        pv, pd = probe_sensitivities(t, T, S)
        for q, v in pv.items():          # same crossings and fit points as probe_quantities
            assert abs(v - res["probe"][q]) <= 1e-9 * abs(v), (q, v, res["probe"][q])
        res["probe_sens"] = pd           # d quantity / d s_p, s_p = relative change of parameter p (TL, X0: see PARAMS)
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
    for q, lbl in (("TTAM_s", "d(TTAM)/ds [s]"), ("TLCR_C_s", "d(TLCR)/ds [C/s]"), ("TTCR_C_s", "d(TTCR)/ds [C/s]"), ("TSCR_C_s", "d(TSCR)/ds [C/s]")):
        if q in res.get("probe_sens", {}):
            top_p = [(k, v) for k, v in sorted(res["probe_sens"][q].items(), key=lambda kv: -abs(kv[1])) if k not in ("TL", "X0", "Vx")]
            print(f"  {lbl:24s} " + ", ".join(f"{k} {v:+.3g}" for k, v in top_p[:6]))
    if a.json: json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
