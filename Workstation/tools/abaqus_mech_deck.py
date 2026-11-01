#!/usr/bin/env python3
"""Abaqus/Standard replica of the mechanical stage (mech_main_aa.cpp, MATERIAL_IN718, options -order 1 -bulk -abqanneal)
on an Abaqus thermal replica run (tools/abaqus_deck.py --nt-every N): for finite-difference checks of the AD stress
derivatives.

Same mesh (C3D8: selectively reduced integration = mean-dilatation B-bar, as the hex8 B-bar of the solver; node n = MFEM
vertex n-1), same IN718 tables (src/mechanical/hill_phi.hpp, MATERIAL_IN718 branch: E(T), nu(T),
mean CTE from T_zero, initial temperature T_ref, linear isotropic hardening sigma_y0 -> sigma_y1 at eps_p1, isotropic Hill
= Mises), *ANNEAL TEMPERATURE = T_SOLIDUS (the solver's -abqanneal), bottom face fixed, small strain. One static step per
temperature sample of the thermal odb (every N thermal steps + the last), temperatures read with *TEMPERATURE, FILE=;
C3D8 evaluates the material temperature at the element mean (confirmed with TEMP/THE IP output); do not use -iptemp in the MFEM replica.
the solver also takes one quasi-static step per thermal dump.
Output: stresses at the integration points of the elements in ELEMS (0-based MFEM element ids, e.g.
results/mech/line_elems_h40.txt) every step (*EL PRINT); the comparison averages the 8 points (= the solver's element mean).
usage: abaqus_mech_deck.py MESH THERMAL_JOB OUT.inp --steps 517 --every 10 --elems FILE
"""
import argparse, os, re
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("mesh"); ap.add_argument("thermal_job"); ap.add_argument("out")
ap.add_argument("--steps", type=int, default=517); ap.add_argument("--every", type=int, default=10)
ap.add_argument("--elems", required=True)
ap.add_argument("--rtol", type=float, default=1e-8, help="mechanical force/correction convergence tolerance for FD")
ap.add_argument("--first-substeps", type=int, default=1, help="fixed subdivisions of the first temperature interval (match MFEM splits)")
a = ap.parse_args()
if not 0 < a.rtol < 1 or a.first_substeps < 1: ap.error("invalid convergence tolerance or first-step subdivision count")

# ---- IN718 deck of the solver
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "mechanical")
hpp = open(os.path.join(SRC, "hill_phi.hpp")).read(); cpp = open(os.path.join(SRC, "mech_main_aa.cpp")).read()
in718 = hpp[hpp.index("#ifdef MATERIAL_IN718"):]                     # first IN718 branch onwards
vals = lambda name, txt: [float(x) for x in re.search(r"#define " + name + r" \{([^}]*)\}", txt).group(1).split(",")]
Tt, Ta = vals("HILL_VALS_Tt", hpp), vals("HILL_VALS_Ta", hpp)
Et, nut, alphat = vals("HILL_VALS_Et", in718), vals("HILL_VALS_nut", in718), vals("HILL_VALS_alphat", in718)
cst = lambda name: float(re.search(name + r" = ([\d.eE+-]+)", in718).group(1))
T_ref, T_zero, sy0, sy1, ep1 = cst("T_ref"), cst("T_zero"), cst("sigma_y0"), cst("sigma_y1"), cst("eps_p1")
T_sol = float(re.search(r"#ifdef MATERIAL_IN718\s*\nstatic constexpr double T_SOLIDUS = ([\d.]+)", cpp).group(1))
assert re.search(r"R11 = 1.0, R22 = 1.0, R33 = 1.0, R12 = 1.0, R13 = 1.0, R23 = 1.0", in718), "IN718 deck must be isotropic"

# ---- mesh
L = open(a.mesh).read().split("\n"); i = L.index("elements"); ne = int(L[i + 1])
E = np.array([list(map(int, l.split()[2:])) for l in L[i + 2:i + 2 + ne]])
i = L.index("vertices"); nv = int(L[i + 1]); V = np.array([list(map(float, l.split())) for l in L[i + 3:i + 3 + nv]])
bottom = np.where(np.abs(V[:, 2] - V[:, 2].min()) < 1e-9)[0] + 1
elems = np.array([int(x) for x in open(a.elems).read().replace(",", " ").split()]) + 1
samples = list(range(a.every, a.steps + 1, a.every)) + ([a.steps] if a.steps % a.every else [])

w = open(a.out, "w")
def block(vs, n=16):
    vs = list(vs)
    for j in range(0, len(vs), n): w.write(", ".join(str(v) for v in vs[j:j + n]) + "\n")
w.write(f"*HEADING\nAbaqus mechanical replica (mech_main_aa IN718) on the temperatures of {a.thermal_job}.odb\n*NODE, NSET=ALL\n")
for n, (x, y, z) in enumerate(V, 1): w.write(f"{n}, {x:.12g}, {y:.12g}, {z:.12g}\n")
w.write("*ELEMENT, TYPE=C3D8, ELSET=PLATE\n")
for e, c in enumerate(E, 1): w.write(f"{e}, " + ", ".join(str(v + 1) for v in c) + "\n")
w.write("*NSET, NSET=BOTTOM\n"); block(bottom); w.write("*ELSET, ELSET=LINE\n"); block(elems)
w.write("*SOLID SECTION, ELSET=PLATE, MATERIAL=IN718\n*MATERIAL, NAME=IN718\n*ELASTIC\n")
for T, Ev, nu in zip(Tt, Et, nut): w.write(f"{Ev:.12g}, {nu}, {T}\n")
w.write(f"*EXPANSION, ZERO={T_zero}\n")
for T, al in zip(Ta, alphat): w.write(f"{al:.12g}, {T}\n")
# Abaqus requires zero hardening at and above the annealing temperature: perfectly plastic at sigma_y0 from T_sol on (the
# solver resets the plastic strain and hardening of elements at or above T_sol every step, so their yield stays sigma_y0)
w.write(f"*PLASTIC\n{sy0}, 0., {T_sol - 1.0}\n{sy1}, {ep1}, {T_sol - 1.0}\n{sy0}, 0., {T_sol}\n{sy0}, {ep1}, {T_sol}\n"
        f"*ANNEAL TEMPERATURE\n{T_sol}\n")
w.write(f"*INITIAL CONDITIONS, TYPE=TEMPERATURE\nALL, {T_ref}\n*BOUNDARY\nBOTTOM, ENCASTRE\n")
for k in samples:
    increment = 1.0 / a.first_substeps if k == samples[0] else 1.0
    w.write(f"*STEP, INC=1000\n*STATIC\n{increment:.12g}, 1., 1e-6, {increment:.12g}\n")
    w.write(f"*CONTROLS, PARAMETERS=FIELD, FIELD=DISPLACEMENT\n{a.rtol:.12g}, {a.rtol:.12g}\n")                 # temperatures ramp linearly over the step
    w.write(f"*TEMPERATURE, FILE={a.thermal_job}, BSTEP={k}, BINC=1, ESTEP={k}, EINC=1\n")
    # stresses of the LINE elements at the end of the step, in the odb (single precision, ~7 digits: *EL PRINT has only 4,
    # too coarse for +-1 % finite differences); tools/abaqus_mech_extract.py reads them
    w.write("*OUTPUT, FIELD, FREQUENCY=99999\n*ELEMENT OUTPUT, ELSET=LINE\nS\n*END STEP\n")
w.close()
print(f"written {a.out}: {nv} nodes, {ne} C3D8, {len(elems)} output elements, {len(samples)} static steps "
      f"(thermal steps {samples[0]}..{samples[-1]}); E {Et[0]:.0f}..{Et[-1]:.0f} MPa, yield {sy0}->{sy1} at {ep1}, "
      f"CTE zero {T_zero}, T_init {T_ref}, anneal {T_sol}")
