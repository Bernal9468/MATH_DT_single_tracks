#!/usr/bin/env python3
"""First-order derivative comparison: probe sensitivities over time (sens.csv) and peak-field derivatives (peakT.bin, 20 arrays).
usage: post_sens.py BASE_DIR NEW_DIR [BASE_MESH NEW_MESH]      (meshes only needed for the node-matched peak-derivative comparison)"""
import sys, csv, math, array, os
TMIN = float(os.environ.get('TMIN', '0'))   # compare only t >= TMIN (the probe is read from the coarse level before the fine box reaches it)
PN = ["k","cp","rho","Lfus","Ts","Tliq","hc","T0","P","Eta","Rx","Ry","Rz","X0","Vx","TL","Tw","hp","kp","Rhop"]
def rd(d):
    R = list(csv.reader(open(d + "/sens.csv"))); h = R[0]; cols = {n: i for i, n in enumerate(h)}
    t = [float(r[0]) for r in R[1:]]; return t, {p: [float(r[cols["dT_d%s_row1" % p]]) for r in R[1:]] for p in PN if "dT_d%s_row1" % p in cols}
tb, Sb = rd(sys.argv[1]); tn, Sn = rd(sys.argv[2])
import bisect
def interp(t, y, tq):
    out = []
    for q in tq:
        i = bisect.bisect_left(t, q)
        if i == 0: out.append(y[0])
        elif i >= len(t): out.append(y[-1])
        else: f = (q - t[i-1]) / (t[i] - t[i-1]); out.append(y[i-1] + f * (y[i] - y[i-1]))
    return out
print("%-6s %12s %12s %9s %9s" % ("param", "max|base|", "max|new|", "NRMSE %", "max err %"))
for p in PN:
    if p not in Sb or p not in Sn: continue
    b = Sb[p]; n = Sn[p] if len(tn) != len(tb) or max(abs(x - y) for x, y in zip(tb, tn)) > 1e-12 else Sn[p]
    if len(tn) == len(tb) and max(abs(x - y) for x, y in zip(tb, tn)) > 1e-12: n = interp(tn, Sn[p], tb)
    m = max(abs(x) for x in b)
    if m < 1e-30: print("%-6s %12.3e %12.3e   (baseline zero)" % (p, m, max(abs(x) for x in Sn[p]))); continue
    sel = [i for i, t in enumerate(tb) if t >= TMIN]; b = [b[i] for i in sel]; n = [n[i] for i in sel]; m = max(abs(x) for x in b)
    e = [x - y for x, y in zip(b, n)]
    print("%-6s %12.4e %12.4e %9.4f %9.4f" % (p, m, max(abs(x) for x in n), 100 * math.sqrt(sum(x * x for x in e) / len(e)) / m, 100 * max(abs(x) for x in e) / m))
if len(sys.argv) > 4:
    def verts(fn):
        L = open(fn).read().split("\n"); j = L.index("vertices"); n = int(L[j + 1]); return [tuple(map(float, L[j + 3 + k].split())) for k in range(n)]
    def pk(d):
        with open(d + "/peakT.bin", "rb") as f:
            h = array.array("i"); h.fromfile(f, 2); a = array.array("d"); a.fromfile(f, h[0]); D = []
            for _ in range(h[1]): x = array.array("d"); x.fromfile(f, h[0]); D.append(list(x))
            return list(a), D
    Vb, Vn = verts(sys.argv[3]), verts(sys.argv[4]); Pb, Db = pk(sys.argv[1]); Pn, Dn = pk(sys.argv[2])
    key = lambda v: tuple(round(c, 6) for c in v); idx = {key(v): i for i, v in enumerate(Vb)}; pairs = [(idx[key(v)], i) for i, v in enumerate(Vn) if key(v) in idx]
    print("\npeak-field derivatives dTpeak/ds at %d matching nodes (value at each node's own peak time)" % len(pairs))
    for pi, p in enumerate(PN):
        if pi >= len(Db) or pi >= len(Dn): break
        m = max(abs(Db[pi][a]) for a, b in pairs)
        if m < 1e-30: continue
        e = [Db[pi][a] - Dn[pi][b] for a, b in pairs]
        print("%-6s max|base| %10.3e   NRMSE %8.4f %%   max err %8.4f %%" % (p, m, 100 * math.sqrt(sum(x * x for x in e) / len(e)) / m, 100 * max(abs(x) for x in e) / m))
