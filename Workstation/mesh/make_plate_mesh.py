#!/usr/bin/env python3
"""Structured, graded hex mesh of the AMB2022-03 bare IN718 plate (MFEM mesh v1.0) for the single-track thermal model.

Plate 25.4 x 25.4 x 3.17 mm, top surface at z = 0 (laser side), bottom at z = -3.17 (contact with the SS304 baseplate,
Dirichlet in the driver). Tensor-product mesh: uniform spacing h in the fine box around the track, geometric growth
(ratio r, capped at hmax) from the fine box to the plate faces. One body, same material: attribute 1, except the top
element layer = attribute 2, which the thermal driver treats as 'row 1' (active from t = 0, no powder without
-powderstart, conforming, no tie) -- it only gives the driver its row-1 probe on the top surface.
Boundary faces: 1 = bottom (z min), 2 = top (z = 0), 3 = sides.  Also writes an empty tie file (no tied interfaces).

usage: make_plate_mesh.py OUT.mesh --h 0.02 --fine-x -5.2 5.2 --fine-y 0.2 --fine-z 0.3 [--r 1.25 --hmax 0.8]
"""
import argparse
import math
import os

P = argparse.ArgumentParser()
P.add_argument("out")
P.add_argument("--L", type=float, default=25.4, help="plate side [mm]")
P.add_argument("--T", type=float, default=3.17, help="plate thickness [mm]")
P.add_argument("--h", type=float, default=0.02, help="element size in the fine box [mm]")
P.add_argument("--fine-x", type=float, nargs=2, default=(-5.2, 5.2), help="fine box x range [mm]")
P.add_argument("--fine-y", type=float, default=0.2, help="fine box half-width in y [mm] (centred on y = 0)")
P.add_argument("--fine-z", type=float, default=0.3, help="fine box depth below the top [mm]")
P.add_argument("--r", type=float, default=1.25, help="growth ratio outside the fine box")
P.add_argument("--hmax", type=float, default=0.8, help="largest element size [mm]")
a = P.parse_args()


def graded(lo, hi, a_, b_, h, r, hmax):
    """1-D nodes on [lo, hi]: uniform h on [a_, b_] (snapped to whole elements), geometric growth outside."""
    n = max(1, round((b_ - a_) / h)); hh = (b_ - a_) / n
    core = [a_ + i * hh for i in range(n + 1)]
    def grow(start, end, sign):
        pts, x, s = [], start, hh
        while True:
            s = min(s * r, hmax)
            if sign * (end - (x + sign * s)) < 0.5 * s:   # last element: stretch / shrink to land on the face
                pts.append(end); break
            x += sign * s; pts.append(x)
        return pts
    left = grow(a_, lo, -1)[::-1] if a_ > lo + 1e-12 else []
    right = grow(b_, hi, +1) if b_ < hi - 1e-12 else []
    return left + core + right


half = 0.5 * a.L
xs = graded(-half, half, a.fine_x[0], a.fine_x[1], a.h, a.r, a.hmax)
ys = graded(-half, half, -a.fine_y, a.fine_y, a.h, a.r, a.hmax)
zs = graded(-a.T, 0.0, -a.fine_z, 0.0, a.h, a.r, a.hmax)
nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
NX, NY = nx + 1, ny + 1
vid = lambda i, j, k: i + NX * (j + NY * k)
ne, nv = nx * ny * nz, (nx + 1) * (ny + 1) * (nz + 1)

with open(a.out, "w") as f:
    f.write("MFEM mesh v1.0\n\n# AMB2022-03 bare IN718 plate, graded tensor-product hexes (make_plate_mesh.py)\n\n")
    f.write(f"dimension\n3\n\nelements\n{ne}\n")
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):   # MFEM hex vertex order: (0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),(1,0,1),(1,1,1),(0,1,1)
                v = [vid(i, j, k), vid(i+1, j, k), vid(i+1, j+1, k), vid(i, j+1, k),
                     vid(i, j, k+1), vid(i+1, j, k+1), vid(i+1, j+1, k+1), vid(i, j+1, k+1)]
                f.write(("2" if k == nz - 1 else "1") + " 5 " + " ".join(map(str, v)) + "\n")
    quads = []
    for j in range(ny):
        for i in range(nx):
            quads.append((1, [vid(i, j, 0), vid(i, j+1, 0), vid(i+1, j+1, 0), vid(i+1, j, 0)]))           # bottom (outward -z)
            quads.append((2, [vid(i, j, nz), vid(i+1, j, nz), vid(i+1, j+1, nz), vid(i, j+1, nz)]))       # top (+z)
    for k in range(nz):
        for i in range(nx):
            quads.append((3, [vid(i, 0, k), vid(i+1, 0, k), vid(i+1, 0, k+1), vid(i, 0, k+1)]))           # y min
            quads.append((3, [vid(i, ny, k), vid(i, ny, k+1), vid(i+1, ny, k+1), vid(i+1, ny, k)]))       # y max
        for j in range(ny):
            quads.append((3, [vid(0, j, k), vid(0, j, k+1), vid(0, j+1, k+1), vid(0, j+1, k)]))           # x min
            quads.append((3, [vid(nx, j, k), vid(nx, j+1, k), vid(nx, j+1, k+1), vid(nx, j, k+1)]))       # x max
    f.write(f"\nboundary\n{len(quads)}\n")
    for att, q in quads: f.write(f"{att} 3 " + " ".join(map(str, q)) + "\n")
    f.write(f"\nvertices\n{nv}\n3\n")
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1): f.write(f"{xs[i]:.9f} {ys[j]:.9f} {zs[k]:.9f}\n")

tie = os.path.splitext(a.out)[0] + "_tie.txt"
with open(tie, "w") as f: f.write("0\n")
dx = lambda c: (min(c[i+1] - c[i] for i in range(len(c)-1)), max(c[i+1] - c[i] for i in range(len(c)-1)))
print(f"{a.out}: {nx} x {ny} x {nz} = {ne} elements, {nv} nodes; fine h = {a.h} mm on x {a.fine_x}, |y| <= {a.fine_y}, "
      f"z >= -{a.fine_z}; spacing x {dx(xs)[0]:.4f}-{dx(xs)[1]:.3f}, y {dx(ys)[0]:.4f}-{dx(ys)[1]:.3f}, "
      f"z {dx(zs)[0]:.4f}-{dx(zs)[1]:.3f} mm; tie file {tie}")
