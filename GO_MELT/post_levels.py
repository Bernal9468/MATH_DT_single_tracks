#!/usr/bin/env python3
"""compare a multi-level run (peakT on its finest-level mesh) with a single-level baseline: probe, peak field at matching nodes, depth/width
usage: post_levels.py BASE_DIR BASE_MESH NEW_DIR NEW_MESH [case]"""
import sys, array, math, subprocess, json, os
bd, bm, nd, nm = sys.argv[1:5]; case = sys.argv[5] if len(sys.argv) > 5 else "0"
def verts(fn):
    L = open(fn).read().split("\n"); j = L.index("vertices"); n = int(L[j + 1]); return [tuple(map(float, L[j + 3 + k].split())) for k in range(n)]
def pk(d):
    with open(d + "/peakT.bin", "rb") as f:
        h = array.array("i"); h.fromfile(f, 2); a = array.array("d"); a.fromfile(f, h[0]); return list(a)
Vb, Vn = verts(bm), verts(nm); Pb, Pn = pk(bd), pk(nd)
key = lambda v: tuple(round(c, 6) for c in v); idx = {key(v): i for i, v in enumerate(Vb)}
pairs = [(idx[key(v)], i) for i, v in enumerate(Vn) if key(v) in idx]
e = [Pb[a] - Pn[b] for a, b in pairs]; rng = max(Pb) - min(Pb)
print(f"peak field on {len(pairs)}/{len(Vn)} matching nodes: NRMSE {100*math.sqrt(sum(x*x for x in e)/len(e))/rng:.4f}%  max {max(abs(x) for x in e):.2f} K  bias {sum(e)/len(e):+.3f} K")
subprocess.run(["python3", "post_compare.py", bd, nd])
for d, m in ((bd, bm), (nd, nm)):
    subprocess.run(["python3", "../HPC/post/analyze_case.py", d, "--mesh", m, "--case", case, "--json", d + "/result.json"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    B, N = json.load(open(bd + "/result.json")), json.load(open(nd + "/result.json"))
    for k in ["TTAM_s", "TLCR_C_s", "TTCR_C_s", "TSCR_C_s"]: print(f"{k}: {B['probe'][k]:.5g} {N['probe'][k]:.5g} {100*(N['probe'][k]/B['probe'][k]-1):+.3f}%")
    for k in ["depth_um", "width_um"]: print(f"{k}: {B['melt_pool_mid_1298C'][k]:.3f} {N['melt_pool_mid_1298C'][k]:.3f}")
except Exception as ex: print("analysis:", ex)
