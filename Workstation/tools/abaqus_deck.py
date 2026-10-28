#!/usr/bin/env python3
"""Abaqus/Standard replica of one MFEM single-track run, for finite-difference checks of the AD derivatives.

Same mesh (DC3D8, node n = MFEM vertex n-1), same IN718 tables (bare_plate_single_layer/src/thermal/material_in718.hpp),
same boundary conditions (film h + radiation eps on every boundary face, bottom face held at T_sub), same time grid (fixed
increments dt), and the SAME laser load: the nodal source vector MFEM assembled in every step (SRC_DUMP_DIR -> fsrc.bin),
applied as *CFLUX in one Abaqus step per MFEM step (AMPLITUDE=STEP: the load is constant over the increment).
Latent heat: MFEM's two-stage smoothstep liquid fraction (fraction FL_EUT between T_eut and the solidus, the rest between
solidus and liquidus) as NSUB linear *LATENT HEAT rows per stage (Abaqus releases each row linearly; checked 2026-10-27).
Capacity: Abaqus uses cp(T_end) * dT per increment, as MFEM (checked on one element).

usage: abaqus_deck.py MFEM_RUNDIR OUT.inp [--sk 1] [--scp 1] [--srho 1] [--sl 1] [--nsub 40]
  --sk/--scp/--srho/--sl: multipliers of k(T), cp(T), rho, latent heat (FD perturbations)
"""
import argparse, os, re, struct, sys
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out")
for p in ("sk", "scp", "srho", "sl"): ap.add_argument("--" + p, type=float, default=1.0)
ap.add_argument("--nsub", type=int, default=40); ap.add_argument("--line-every", type=int, default=10)
a = ap.parse_args()

# ---- run settings (settings.txt, run.log)
st = open(os.path.join(a.run, "settings.txt")).read(); log = open(os.path.join(a.run, "run.log")).read()
g = lambda k, s=st: re.search(k + r"=(\S+)", s).group(1)
mesh_file = re.search(r"mesh=(.+?\.mesh)", st).group(1)
dt, fleut, teut = float(g("dt")), float(g("fleut")), float(g("teut"))
h_conv = float(re.search(r"Effective boundary loss scale=\S+ h=(\S+)", log).group(1))
eps = float(re.search(r"emissivity=(\S+)", log).group(1))
T_sub = float(re.search(r"Dirichlet BC nodes: \d+ \(T=(\S+?)K\)", log).group(1))
T_amb = T_sub                                                    # -tambc = -tsub = 23.5 C (C2K = 273.0)

# ---- IN718 (material_in718.hpp)
hpp = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "thermal", "material_in718.hpp")).read()
arr = lambda n: np.array([float(x) for x in re.search(r"#define " + n + r" \{([^}]*)\}", hpp).group(1).replace("\\", " ").split(",")])
num = lambda n: float(re.search(r"#define " + n + r" (\S+)", hpp).group(1))
kT, kV, cT, cV = arr("IN718_K_TA"), arr("IN718_K_VA"), arr("IN718_CP_TA"), arr("IN718_CP_VA")
RHO, LF, TS, TL = num("IN718_RHO"), num("IN718_LFUS"), num("IN718_T_SOLIDUS"), num("IN718_T_LIQUIDUS")

# ---- mesh
L = open(mesh_file).read().split("\n"); i = L.index("elements"); ne = int(L[i + 1])
E = np.array([list(map(int, l.split()[2:])) for l in L[i + 2:i + 2 + ne]])
i = L.index("vertices"); nv = int(L[i + 1])
V = np.array([list(map(float, l.split())) for l in L[i + 3:i + 3 + nv]])
# element faces (Abaqus C3D8 numbering) on the boundary of the box
FACES = {"S1": (0, 1, 2, 3), "S2": (4, 5, 6, 7), "S3": (0, 1, 5, 4), "S4": (1, 2, 6, 5), "S5": (2, 3, 7, 6), "S6": (3, 0, 4, 7)}
lo, hi = V.min(0), V.max(0); tol = 1e-9
surf = {}
for f, idx in FACES.items():
    P = V[E[:, list(idx)]]                                      # (ne, 4, 3)
    on = np.zeros(ne, bool)
    for d in range(3):
        on |= np.all(np.abs(P[:, :, d] - lo[d]) < tol, 1) | np.all(np.abs(P[:, :, d] - hi[d]) < tol, 1)
    surf[f] = np.where(on)[0] + 1
bottom = np.where(np.abs(V[:, 2] - lo[2]) < tol)[0] + 1
PX = np.zeros(3)                                                # MFEM probe point x = y = z = 0: the 8 nodes of its element
cand = np.where(np.all((V[E].min(1) <= PX + tol) & (V[E].max(1) >= PX - tol), 1))[0]
pel = int(cand[0]); probe = E[pel] + 1
line = np.where((np.abs(V[:, 1]) < tol) & (np.abs(V[:, 2]) < tol) & (V[:, 0] > -0.75) & (V[:, 0] < 1.65))[0] + 1

# ---- MFEM nodal source, one record per step attempt (the last record of each t_end wins: Newton step cuts)
raw = open(os.path.join(a.run, "fsrc.bin"), "rb").read(); off = 0; rec = {}
while off < len(raw):
    te, nz = struct.unpack_from("<dq", raw, off); off += 16
    d = np.frombuffer(raw, dtype=[("i", "<i8"), ("v", "<f8")], count=nz, offset=off); off += 16 * nz
    if abs(te / dt - round(te / dt)) < 1e-6: rec[round(te / dt)] = d   # the final partial step (t_final off the grid) is dropped
steps = sorted(rec); nstep = steps[-1]
if steps != list(range(1, nstep + 1)):
    sys.exit("source records are not on the fixed time grid (step cuts?)")
amp = {}
for k in steps:
    for i_, v_ in zip(rec[k]["i"], rec[k]["v"]): amp.setdefault(int(i_) + 1, {})[k] = float(v_)

# ---- latent heat rows: smoothstep stages as NSUB linear pieces each
def stage_rows(T0, T1, frac):
    x = np.linspace(0.0, 1.0, a.nsub + 1); s = x ** 3 * (10 - 15 * x + 6 * x * x); T = T0 + (T1 - T0) * x
    return [(frac * LF * a.sl * (s[j + 1] - s[j]), T[j] + 1e-6, T[j + 1]) for j in range(a.nsub)]   # Abaqus: ranges must not touch
rows = (stage_rows(teut, TS, fleut) if fleut > 0 else []) + stage_rows(TS, TL, 1.0 - fleut)

w = open(a.out, "w")
def block(vals, n=8):
    vals = list(vals)
    for j in range(0, len(vals), n): w.write(", ".join(str(v) for v in vals[j:j + n]) + "\n")
w.write(f"*HEADING\nAbaqus replica of {os.path.abspath(a.run)}  sk={a.sk} scp={a.scp} srho={a.srho} sl={a.sl}\n")
w.write("*NODE, NSET=ALL\n")
for n, (x, y, z) in enumerate(V, 1): w.write(f"{n}, {x:.12g}, {y:.12g}, {z:.12g}\n")
w.write("*ELEMENT, TYPE=DC3D8, ELSET=PLATE\n")
for e, c in enumerate(E, 1): w.write(f"{e}, " + ", ".join(str(v + 1) for v in c) + "\n")
for f, els in surf.items():
    w.write(f"*ELSET, ELSET=B_{f}\n"); block(els, 16)
w.write("*SURFACE, NAME=OUTER, TYPE=ELEMENT\n" + "".join(f"B_{f}, {f}\n" for f in surf))
w.write("*NSET, NSET=BOTTOM\n"); block(bottom, 16)
w.write("*NSET, NSET=PROBE\n"); block(probe, 16); w.write("*NSET, NSET=LINE\n"); block(line, 16)
w.write("*SOLID SECTION, ELSET=PLATE, MATERIAL=IN718\n*MATERIAL, NAME=IN718\n*CONDUCTIVITY\n")
for T, v in zip(kT, kV): w.write(f"{v * a.sk:.12g}, {T}\n")
w.write("*SPECIFIC HEAT\n")
for T, v in zip(cT, cV): w.write(f"{v * a.scp:.12g}, {T}\n")
w.write(f"*DENSITY\n{RHO * a.srho:.12g}\n*LATENT HEAT\n")
for Lr, T0, T1 in rows: w.write(f"{Lr:.12g}, {T0:.10f}, {T1:.10f}\n")
w.write("*PHYSICAL CONSTANTS, ABSOLUTE ZERO=0., STEFAN BOLTZMANN=5.670373e-11\n")
w.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALL, {T_amb}\n")
# One Abaqus step per MFEM time step: Abaqus applies the AVERAGE of an amplitude's two end values over an increment
# (checked 2026-10-27), so a tabular amplitude cannot hold MFEM's per-step load; AMPLITUDE=STEP + CFLUX, OP=NEW can.
for k in steps:
    w.write(f"*STEP, INC=100, UNSYMM=YES, AMPLITUDE=STEP\n*HEAT TRANSFER, END=PERIOD\n{dt}, {dt}\n")   # one fixed increment
    w.write("*CONTROLS, PARAMETERS=FIELD, FIELD=TEMPERATURE\n1e-6, 1e-5\n")          # tight: FD differences of ~1 K
    w.write("*CONTROLS, PARAMETERS=TIME INCREMENTATION\n40, 60, , 80, , , , 20\n")   # Abaqus Newton converges linearly here
    w.write("*CONTROLS, PARAMETERS=LINE SEARCH\n10, 4., 0.1, 0.25, 0.1\n")          # without it Newton cycles at the melt
    if k == 1:
        w.write(f"*BOUNDARY\nBOTTOM, 11, 11, {T_sub}\n")
        w.write(f"*SFILM\nOUTER, F, {T_amb}, {h_conv}\n*SRADIATE\nOUTER, R, {T_amb}, {eps}\n")
        w.write("*OUTPUT, FIELD, FREQUENCY=0\n*NODE OUTPUT\nNT\n")
    w.write("*CFLUX, OP=NEW\n")
    loads = [(int(i_) + 1, float(v_)) for i_, v_ in zip(rec[k]["i"], rec[k]["v"])] or [(1, 0.0)]
    for n, v in loads: w.write(f"{n}, 11, {v:.15g}\n")
    w.write("*NODE PRINT, NSET=PROBE, FREQUENCY=1, SUMMARY=NO, TOTALS=NO\nNT\n")
    w.write(f"*NODE PRINT, NSET=LINE, FREQUENCY={1 if k % a.line_every == 0 else 0}, SUMMARY=NO, TOTALS=NO\nNT\n*END STEP\n")
w.close()
tot = [rec[k]["v"].sum() for k in steps]
print(f"written {a.out}: {nv} nodes, {ne} elements, {len(amp)} loaded nodes, {nstep} increments of {dt}, "
      f"{len(rows)} latent rows; probe element {pel + 1} nodes {list(probe)}, {len(line)} line nodes; "
      f"h={h_conv} eps={eps} T_sub={T_sub}; source total min/max {min(tot):.4g}/{max(tot):.4g} mW")
