#!/usr/bin/env python3
"""Mechanical-stage comparison: final element-mean residual stress and its 20 first-order derivatives, two mech runs (same mesh).
usage: post_mech.py REF_MECH_DIR NEW_MECH_DIR"""
import sys, math
import numpy as np
ref, new = sys.argv[1], sys.argv[2]
A = np.loadtxt(ref + "/final_elem_mean.txt"); B = np.loadtxt(new + "/final_elem_mean.txt")
def vm(S): s11, s22, s33, s12, s13, s23 = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], S[:, 5]; return np.sqrt(0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3 * (s12 ** 2 + s13 ** 2 + s23 ** 2))
names = ["S11", "S22", "S33", "S12", "S13", "S23", "PEEQ"]
print("nominal final element-mean fields (%d elements)" % len(A))
for c in range(7):
    a, b = A[:, 1 + c], B[:, 1 + c]; r = a.max() - a.min(); print("  %-5s range %.4g  NRMSE %.4f %%  max err %.4f %%  mean err %+.3g" % (names[c], r, 100 * np.sqrt(np.mean((a - b) ** 2)) / r, 100 * np.abs(a - b).max() / r, np.mean(a - b)))
va, vb = vm(A[:, 1:7]), vm(B[:, 1:7]); r = va.max() - va.min(); print("  von Mises: max %.2f vs %.2f, NRMSE %.4f %%  max err %.4f %%" % (va.max(), vb.max(), 100 * np.sqrt(np.mean((va - vb) ** 2)) / r, 100 * np.abs(va - vb).max() / r))
try:
    PN = ["k","cp","rho","Lfus","Ts","Tliq","hc","T0","P","Eta","Rx","Ry","Rz","X0","Vx","TL","Tw","hp","kp","Rhop"]
    Da = np.loadtxt(ref + "/final_elem_mean_d1.txt")[:, 1:].reshape(len(A), 20, 6); Db = np.loadtxt(new + "/final_elem_mean_d1.txt")[:, 1:].reshape(len(A), 20, 6)
    print("first-order stress derivatives (per unit relative change), worst of the 6 components")
    for p in range(20):
        m = np.abs(Da[:, p, :]).max()
        if m < 1e-12: print("  %-5s (zero)" % PN[p]); continue
        nr = [100 * np.sqrt(np.mean((Da[:, p, c] - Db[:, p, c]) ** 2)) / (Da[:, p, c].max() - Da[:, p, c].min() + 1e-300) for c in range(6)]
        print("  %-5s max|dS| %.4g  NRMSE per component %s  worst %.4f %%" % (PN[p], m, " ".join("%.3f" % x for x in nr), max(nr)))
except Exception as ex: print("derivatives:", ex)
