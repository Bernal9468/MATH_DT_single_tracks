#!/usr/bin/env python3
"""Uniform tensor-product hex box (MFEM mesh v1.0) for a GO_MELT level.  The top face is z = ztop (default 0, laser side).
usage: gen_box.py OUT.mesh --x X0 X1 --y Y0 Y1 --z Z0 [ZTOP] --h H     (h = element size in all directions; ranges snapped to whole elements)
Boundary attributes: 1 = z min, 2 = z max, 3 = sides.  Element attribute 1 (top layer 2, like make_plate_mesh.py)."""
import argparse
P = argparse.ArgumentParser(); P.add_argument("out"); P.add_argument("--x", type=float, nargs=2, required=True); P.add_argument("--y", type=float, nargs=2, required=True)
P.add_argument("--z", type=float, nargs="+", required=True); P.add_argument("--h", type=float, required=True); P.add_argument("--ztop", type=float, default=0.0, help="if >0: top element thickness, growing by --rz per layer up to h (anisotropic top refinement)"); P.add_argument("--rz", type=float, default=1.25)
a = P.parse_args()
z0, ztop = a.z[0], (a.z[1] if len(a.z) > 1 else 0.0)
def ax(lo, hi, h):
    n = max(1, round((hi - lo) / h)); return [lo + (hi - lo) * i / n for i in range(n + 1)]
xs, ys, zs = ax(*a.x, a.h), ax(*a.y, a.h), ax(z0, ztop, a.h)
if a.ztop > 0:   # top-down march from ztop, growth rz, capped at h; last layer stretched to land on z0
    pts, z, size = [ztop], ztop, a.ztop
    while True:
        z -= size
        if z < z0 + 0.5 * size: pts.append(z0); break
        pts.append(z); size = min(size * a.rz, a.h)
    zs = pts[::-1]
nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1; NX, NY = nx + 1, ny + 1; vid = lambda i, j, k: i + NX * (j + NY * k)
with open(a.out, "w") as f:
    f.write(f"MFEM mesh v1.0\n\n# GO_MELT level box (gen_box.py)\n\ndimension\n3\n\nelements\n{nx*ny*nz}\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v = [vid(i, j, k), vid(i+1, j, k), vid(i+1, j+1, k), vid(i, j+1, k), vid(i, j, k+1), vid(i+1, j, k+1), vid(i+1, j+1, k+1), vid(i, j+1, k+1)]
                f.write(("2" if k == nz - 1 else "1") + " 5 " + " ".join(map(str, v)) + "\n")
    q = []
    for j in range(ny):
        for i in range(nx):
            q.append((1, [vid(i, j, 0), vid(i, j+1, 0), vid(i+1, j+1, 0), vid(i+1, j, 0)])); q.append((2, [vid(i, j, nz), vid(i+1, j, nz), vid(i+1, j+1, nz), vid(i, j+1, nz)]))
    for k in range(nz):
        for i in range(nx):
            q.append((3, [vid(i, 0, k), vid(i+1, 0, k), vid(i+1, 0, k+1), vid(i, 0, k+1)])); q.append((3, [vid(i, ny, k), vid(i, ny, k+1), vid(i+1, ny, k+1), vid(i+1, ny, k)]))
        for j in range(ny):
            q.append((3, [vid(0, j, k), vid(0, j, k+1), vid(0, j+1, k+1), vid(0, j+1, k)])); q.append((3, [vid(nx, j, k), vid(nx, j+1, k), vid(nx, j+1, k+1), vid(nx, j, k+1)]))
    f.write(f"\nboundary\n{len(q)}\n")
    for att, v in q: f.write(f"{att} 3 " + " ".join(map(str, v)) + "\n")
    f.write(f"\nvertices\n{(nx+1)*(ny+1)*(nz+1)}\n3\n")
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1): f.write(f"{xs[i]:.9f} {ys[j]:.9f} {zs[k]:.9f}\n")
print(f"{a.out}: {nx} x {ny} x {nz} = {nx*ny*nz} elements, h = {a.h} (actual {xs[1]-xs[0]:.5f}, {ys[1]-ys[0]:.5f}, {zs[1]-zs[0]:.5f})")
