#!/usr/bin/env python3
"""Laser path table (driver format: time [s], x, y, z [mm], power [mW]; a row's power applies to the segment that starts
there) for one AMB2022-03 single track: +X along y = 0 on the top surface z = 0, from x0 to x1, then off until t_end.
usage: make_laser.py OUT.inp --case 0 [--x0 -2.5 --x1 2.5 --tend <s>]      (P, v from inputs/cases.csv)"""
import argparse, csv, os
P = argparse.ArgumentParser(); P.add_argument("out"); P.add_argument("--case", default="0")
P.add_argument("--x0", type=float, default=-2.5); P.add_argument("--x1", type=float, default=2.5); P.add_argument("--tend", type=float, default=None)
a = P.parse_args()
row = {r["case"]: r for r in csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases.csv")))}[a.case]
Pw, v = float(row["power_W"]), float(row["speed_mm_s"])
t1 = (a.x1 - a.x0) / v; tend = a.tend if a.tend else 1.05 * t1
with open(a.out, "w") as f:
    f.write(f"0,{a.x0},0,0,{Pw * 1e3:.1f}\n{t1:.12g},{a.x1},0,0,0\n{tend:.12g},{a.x1},0,0,0\n")
print(f"{a.out}: case {a.case}, P = {Pw} W, v = {v} mm/s, x {a.x0} -> {a.x1} mm in {t1*1e3:.4f} ms, table end {tend*1e3:.4f} ms")
