#!/usr/bin/env python3
"""compare two probe.csv (+ optional peakT.bin) : python3 post_compare.py BASE_DIR NEW_DIR"""
import sys, csv, array, math
def rd(d):
    R = list(csv.reader(open(d + "/probe.csv")))[1:]
    return [float(r[0]) for r in R], [float(r[1]) for r in R], [float(r[2]) for r in R]
def pk(d):
    with open(d + "/peakT.bin", "rb") as f:
        h = array.array("i"); h.fromfile(f, 2); a = array.array("d"); a.fromfile(f, h[0]); return list(a)
tb, pb, mb = rd(sys.argv[1]); tn, pn, mn = rd(sys.argv[2])
import bisect
def interp(t, y, tq):   # linear interpolation of the new run onto the baseline time stamps (a Newton cut shifts the baseline's grid by dt/2)
    out = []
    for q in tq:
        i = bisect.bisect_left(t, q)
        if i == 0: out.append(y[0])
        elif i >= len(t): out.append(y[-1])
        else: f = (q - t[i-1]) / (t[i] - t[i-1]); out.append(y[i-1] + f * (y[i] - y[i-1]))
    return out
if max(abs(a - b) for a, b in zip(tb, tn)) > 1e-12 and len(tb) == len(tn): print("time stamps differ (max %.2e s): new run interpolated onto the baseline stamps" % max(abs(a - b) for a, b in zip(tb, tn)))
pn, mn = interp(tn, pn, tb), interp(tn, mn, tb); tn = tb; n = min(len(tb), len(tn))
def stat(a, b, name):
    rng = max(a) - min(a); e = [x - y for x, y in zip(a, b)]
    print(f"{name}: peak base {max(a):.2f} new {max(b):.2f} ({100*(max(b)-max(a))/max(a):+.3f}%)  NRMSE {100*math.sqrt(sum(x*x for x in e)/len(e))/rng:.4f}%  maxabs {max(abs(x) for x in e):.2f} K")
print("steps base", len(tb), "new", len(tn), " t_end", tb[-1], tn[-1])
stat(pb[:n], pn[:n], "probe(top,x=y=0)"); stat(mb[:n], mn[:n], "T_max(t)")
try:
    a, b = pk(sys.argv[1]), pk(sys.argv[2])
    if len(a) != len(b): raise Exception('different meshes: use post_levels.py for the node-matched peak-field comparison')
    rng = max(a) - min(a); e = [x - y for x, y in zip(a, b)]
    print(f"peakT field: NRMSE {100*math.sqrt(sum(x*x for x in e)/len(e))/rng:.4f}%  max {max(abs(x) for x in e):.2f} K  (max T {max(a):.1f} vs {max(b):.1f})")
except Exception as ex: print("peakT:", ex)
