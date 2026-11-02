#!/usr/bin/env python3
"""MFEM AD derivatives of the von Mises stress (mechanical stage) against Abaqus central finite differences
(run/abaqus_mech_fd.sh) for s = k, cp, rho, L. Same layout as abaqus_fd_compare.py (temperature).

MFEM: a mechanical run with element-mean temperature (omit -iptemp) and -probeelem <the ELEMS list> (probe_elem_hist_<e>.csv: element-mean stress and d/ds_p).
Abaqus: m_<job>_S.csv (tools/abaqus_mech_extract.py: S of the same elements, element = MFEM id + 1, mean of the 8
integration points) at the end of every static step (one per thermal sample, thermal step = EVERY * step number, the last one = the last thermal step).
d sigma_vM / ds_p = (vM(1 + H) - vM(1 - H)) / (2 H); NRMSE = RMS(AD - FD) / (max FD - min FD).
--state-dir: branch-aware reference (one-sided quotient where the forward and backward quotients disagree, a kink, on the
side of the run that stayed on the nominal plastic/anneal branch, see reference()); fd_MPa in the CSV stays the central
quotient, ref_MPa/ref_kind the reference used.
Figures (in FDDIR): abaqus_mech_fd_line_t<ms>.png (centreline elements), abaqus_mech_fd_probe.png (history at PROBE).
usage: abaqus_mech_fd_compare.py MECH_RUN FDDIR ELEMS_FILE PROBE_ELEM [--t 0.75 1.5 2.4] [--h 0.01] [--every 10]
"""
import argparse, csv, json, os, re, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("mech"); ap.add_argument("fd"); ap.add_argument("elems"); ap.add_argument("probe", type=int)
ap.add_argument("--t", type=float, nargs="+", default=[0.75, 1.5, 2.4]); ap.add_argument("--h", type=float, default=0.01)
ap.add_argument("--every", type=int, default=10); ap.add_argument("--dt", type=float, default=5e-6)
ap.add_argument("--out", help="directory for figures and numerical exports (defaults to FDDIR)")
ap.add_argument("--allow-partial", action="store_true", help="explicitly permit missing jobs or steps")
ap.add_argument("--state-dir", help="branch-aware reference: s_<job>_state.npz or m_<job>_state.npz of tools/abaqus_state_extract.py (all 9 jobs)")
ap.add_argument("--kink-tol", type=float, default=0.03, help="branch-aware: kink where |forward - backward| > KINK_TOL * range of the central quotient")
ap.add_argument("--mesh", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mesh", "track_h0.04_x-0.6_1.5.mesh"))
a = ap.parse_args()
outdir = a.out or a.fd
os.makedirs(outdir, exist_ok=True)
if a.h <= 0 or a.dt <= 0: ap.error("--h and --dt must be positive")
PAR = [("k", 0, r"$\partial \sigma_{vM}/\partial s_k$"), ("cp", 1, r"$\partial \sigma_{vM}/\partial s_{c_p}$"),
       ("rho", 2, r"$\partial \sigma_{vM}/\partial s_{\rho}$"), ("L", 3, r"$\partial \sigma_{vM}/\partial s_L$")]
C = ["S11", "S22", "S33", "S12", "S13", "S23"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.6})


def vm_and_d(s, d):
    q = 0.5 * ((s[0] - s[1]) ** 2 + (s[1] - s[2]) ** 2 + (s[2] - s[0]) ** 2) + 3 * (s[3] ** 2 + s[4] ** 2 + s[5] ** 2)
    vm = np.sqrt(q)
    dq = ((s[0] - s[1]) * (d[0] - d[1]) + (s[1] - s[2]) * (d[1] - d[2]) + (s[2] - s[0]) * (d[2] - d[0])
          + 6 * (s[3] * d[3] + s[4] * d[4] + s[5] * d[5]))
    return vm, np.divide(0.5 * dq, vm, out=np.zeros_like(vm), where=vm > 1e-9)


def read_csv(fn):
    """tools/abaqus_mech_extract.py output: {step: {elem: element-mean S (6,)}}"""
    out = {}
    for r in csv.DictReader(open(fn)):
        out.setdefault(int(r["step"]), {})[int(r["elem"])] = np.array([float(r[c]) for c in C])
    return out


elems = [int(x) for x in open(a.elems).read().replace(",", " ").split()]
if a.probe not in elems: ap.error("probe must be present in the element list")
L = open(a.mesh).read().split("\n")
i = L.index("elements"); ne = int(L[i + 1]); E = np.array([list(map(int, l.split()[2:])) for l in L[i + 2:i + 2 + ne]])
i = L.index("vertices"); nv = int(L[i + 1]); V = np.array([list(map(float, l.split())) for l in L[i + 3:i + 3 + nv]])
xc = V[E[elems]].mean(1)[:, 0]

# MFEM: per element, per sample: t, vM, dvM/ds_p
MF = {}
for e in elems:
    rows = list(csv.DictReader(open(os.path.join(a.mech, f"probe_elem_hist_{e}.csv"))))
    s = np.array([[float(r[c]) for c in C] for r in rows]).T
    vm = vm_and_d(s, s)[0]
    d = [vm_and_d(s, np.array([[float(r[f"d{c}_dp{j}"]) for c in C] for r in rows]).T)[1] for _, j, _ in PAR]
    MF[e] = dict(t=np.array([float(r["t"]) for r in rows]), vm=vm, d=d)
# Abaqus: per job, per static step, per element
D = {j: read_csv(os.path.join(a.fd, f"m_{j}_S.csv")) for j in ["nom"] + [p + s for p, _, _ in PAR for s in ("_p", "_m")]
     if os.path.exists(os.path.join(a.fd, f"m_{j}_S.csv"))}
D = {j: d for j, d in D.items() if d}                                      # jobs with stresses printed so far (partial runs)
missing = [j for j in ["nom"] + [p + suffix for p, _, _ in PAR for suffix in ("_p", "_m")] if j not in D]
if "nom" not in D or (missing and not a.allow_partial):
    ap.error("missing or empty stress jobs: " + ", ".join(missing))
PAR = [x for x in PAR if x[0] + "_p" in D and x[0] + "_m" in D]           # parameters with both FD runs
print("parameters with FD:", [p for p, _, _ in PAR])
if not PAR: ap.error("no complete finite-difference parameter pairs")
steps = sorted(set.intersection(*[set(d) for d in D.values()]))
BS = [int(x) for x in re.findall(r"BSTEP=(\d+)", open(os.path.join(a.fd, "m_nom.inp")).read())]   # thermal step of each static step
if not steps: ap.error("no common Abaqus steps")
expected = set(range(1, len(BS) + 1))
complete = not missing and all(set(d) == expected for d in D.values())
if not complete and not a.allow_partial: ap.error("incomplete Abaqus steps; use --allow-partial to inspect explicitly")
for job, data in D.items():
    for n in steps:
        absent = [e + 1 for e in elems if e + 1 not in data[n]]
        if absent: ap.error(f"{job}, step {n}: missing elements {absent}")
T_ab = {n: BS[n - 1] * a.dt for n in steps}
lo = max(MF[e]["t"].min() for e in elems)
hi = min(MF[e]["t"].max() for e in elems)
covered = [n for n in steps if lo - 1e-12 <= T_ab[n] <= hi + 1e-12]
if len(covered) != len(steps):
    if not a.allow_partial: ap.error("MFEM histories do not cover Abaqus times")
    complete = False
    steps = covered
    T_ab = {n: T_ab[n] for n in steps}
    if not steps: ap.error("no Abaqus times covered by the available MFEM history")
print(len(steps), "common Abaqus steps; last thermal time", max(T_ab.values()))
vmA = lambda j, n, e: vm_and_d(D[j][n][e + 1][:, None], D[j][n][e + 1][:, None])[0][0]

# Branch-aware reference: the stress is not differentiable where a perturbed run leaves the nominal discrete state
# (an integration point switches between plastic flow and elastic unloading, or an element between annealed and not).
# There the forward and backward quotients disagree by O(1) instead of O(h): a kink (|forward - backward| > KINK_TOL *
# range of the central quotient over the line or history). Off kinks: central. On a kink, the run that switched is the
# one with a state flip (vs nom, any earlier step) nearest the element; the other run's one-sided quotient is the
# reference (excluded on a tie). Decided from the Abaqus results only.
DM = {}
if a.state_dir:
    sfile = lambda j: next((f for f in (os.path.join(a.state_dir, f"{x}_{j}_state.npz") for x in ("s", "m")) if os.path.exists(f)), None)
    if any(sfile(j) is None for j in D): ap.error("missing s_<job>_state.npz / m_<job>_state.npz in " + a.state_dir)
    st = {j: np.load(sfile(j)) for j in D}
    cen = V[E].mean(1); xe = cen[elems]
    nomS = st["nom"]; lab = nomS["label"] - 1
    for j in D:
        if j == "nom": continue
        if not (np.array_equal(st[j]["label"], nomS["label"]) and np.array_equal(st[j]["ip"], nomS["ip"])): ap.error(f"state points of {j} differ from nom")
        flip = (st[j]["plastic"] != nomS["plastic"]) | (st[j]["annealed"] != nomS["annealed"])   # nstep x npoint
        d = np.full(len(elems), np.inf); DM[j] = {}
        for k, s in enumerate(nomS["step"]):
            el = np.unique(lab[flip[k]])
            if len(el): d = np.minimum(d, np.linalg.norm(xe[:, None, :] - cen[el][None, :, :], axis=2).min(1))
            DM[j][int(s)] = d.copy()                                   # nearest flip up to and including step s


def reference(p, ns, idx, fd, forward, backward):
    """branch-aware FD reference and its kind per (step, element index) (central FD without --state-dir)"""
    if not DM: return fd, np.array(["central"] * len(fd))
    kink = np.abs(forward - backward) > a.kink_tol * (fd.max() - fd.min())
    dp = np.array([DM[p + "_p"][n][i] for n, i in zip(ns, idx)]); dm = np.array([DM[p + "_m"][n][i] for n, i in zip(ns, idx)])
    kind = np.where(~kink, "central", np.where(dm < dp, "forward", np.where(dp < dm, "backward", "excluded")))
    ref = np.select([kind == "central", kind == "forward", kind == "backward"], [fd, forward, backward], np.nan)
    return ref, kind


def nrmse(ad, fd):
    ok = np.isfinite(fd); ad, fd = ad[ok], fd[ok]                  # excluded points (branch-aware) are NaN
    return 100 * np.sqrt(np.mean((ad - fd) ** 2)) / max(fd.max() - fd.min(), 1e-30)


def panel(xs, ad, fd, ax, lab, xl):
    ax.plot(xs, ad, color="red", lw=3.0, label="MFEM AD")
    ax.plot(xs, fd, color="black", lw=2.0, ls=(0, (4, 3)), label=f"Abaqus {'branch-aware' if DM else 'central'} FD (±{100 * a.h:g}%)")
    ax.set_ylabel(lab + "  [MPa]"); ax.text(0.97, 0.06, f"NRMSE = {nrmse(ad, fd):.2f} %", transform=ax.transAxes, ha="right",
                                           bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
    ax.set_xlabel(xl)


res = {}
asymmetry = {}
kinds = {}
count = lambda k: {c: int(np.sum(k == c)) for c in ("central", "forward", "backward", "excluded")}
records = []
NR = 1 if len(PAR) <= 2 else 2


def grid():                                   # 2 x 2 panels (one row while only k and cp have FD runs)
    fig, axs = plt.subplots(NR, 2, figsize=(13.6, 4.4 * NR), sharex=True, gridspec_kw={"wspace": 0.34}, squeeze=False)
    return fig, axs


# nominal check
for tms in a.t:
    n = min(steps, key=lambda n: abs(T_ab[n] - tms * 1e-3)); tt = T_ab[n]
    if abs(tt - tms * 1e-3) > 30e-6: print(f"no Abaqus step near {tms} ms yet"); continue
    jm = [int(np.argmin(np.abs(MF[e]["t"] - tt))) for e in elems]
    vm_m = np.array([MF[e]["vm"][k] for e, k in zip(elems, jm)]); vm_a = np.array([vmA("nom", n, e) for e in elems])
    print(f"t = {tt * 1e3:.3f} ms (MFEM t = {MF[elems[0]]['t'][jm[0]] * 1e3:.3f}): nominal vM max |MFEM - Abaqus| = "
          f"{np.max(np.abs(vm_m - vm_a)):.2f} MPa (max vM {vm_a.max():.0f})")
    fig, axs = grid()
    for q, (ax, (p, j, lab)) in enumerate(zip(axs.flat, PAR)):
        ad = np.array([MF[e]["d"][j][k] for e, k in zip(elems, jm)])
        fd = np.array([(vmA(p + "_p", n, e) - vmA(p + "_m", n, e)) / (2 * a.h) for e in elems])
        forward = np.array([(vmA(p + "_p", n, e) - vmA("nom", n, e)) / a.h for e in elems])
        backward = np.array([(vmA("nom", n, e) - vmA(p + "_m", n, e)) / a.h for e in elems])
        asymmetry[f"line {tt * 1e3:.2f} ms {p}"] = float(np.max(np.abs(forward - backward)))
        ref, kind = reference(p, [n] * len(elems), range(len(elems)), fd, forward, backward); kinds[f"line {tt * 1e3:.2f} ms {p}"] = count(kind)
        records.extend(dict(kind="line", parameter=p, t_s=float(tt), x_mm=float(x), ad_MPa=float(av), fd_MPa=float(fv),
                            fd_forward_MPa=float(ff), fd_backward_MPa=float(fb), ref_MPa=float(rv), ref_kind=rk)
                       for x, av, fv, ff, fb, rv, rk in zip(xc, ad, fd, forward, backward, ref, kind))
        panel(xc, ad, ref, ax, lab, "x [mm]" if ax in axs[-1] else ""); res[f"line {tt * 1e3:.2f} ms {p}"] = nrmse(ad, ref)
    axs[0, 0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.12 + 0.1 * NR), ncol=2, frameon=False)
    out = os.path.join(outdir, f"abaqus_mech_fd_line_t{tt * 1e3:.2f}.png"); fig.savefig(out, dpi=170, bbox_inches="tight"); print("written", out)
# probe history
e = a.probe; tA = np.array([T_ab[n] for n in steps])
fig, axs = grid()
for q, (ax, (p, j, lab)) in enumerate(zip(axs.flat, PAR)):
    ad = np.interp(tA, MF[e]["t"], MF[e]["d"][j])
    fd = np.array([(vmA(p + "_p", n, e) - vmA(p + "_m", n, e)) / (2 * a.h) for n in steps])
    forward = np.array([(vmA(p + "_p", n, e) - vmA("nom", n, e)) / a.h for n in steps])
    backward = np.array([(vmA("nom", n, e) - vmA(p + "_m", n, e)) / a.h for n in steps])
    asymmetry["probe " + p] = float(np.max(np.abs(forward - backward)))
    ie = elems.index(e); ref, kind = reference(p, steps, [ie] * len(steps), fd, forward, backward); kinds["probe " + p] = count(kind)
    records.extend(dict(kind="probe", parameter=p, t_s=float(t), x_mm=float(xc[ie]), ad_MPa=float(av), fd_MPa=float(fv),
                        fd_forward_MPa=float(ff), fd_backward_MPa=float(fb), ref_MPa=float(rv), ref_kind=rk)
                   for t, av, fv, ff, fb, rv, rk in zip(tA, ad, fd, forward, backward, ref, kind))
    panel(tA * 1e3, ad, ref, ax, lab, "t [ms]" if ax in axs[-1] else ""); res["probe " + p] = nrmse(ad, ref)
axs[0, 0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.12 + 0.1 * NR), ncol=2, frameon=False)
fig.savefig(os.path.join(outdir, "abaqus_mech_fd_probe.png"), dpi=170, bbox_inches="tight"); print("written abaqus_mech_fd_probe.png")
for k, v in res.items(): print(f"  NRMSE {k:22s} {v:6.2f} %" + (f"   reference {kinds[k]}" if DM else ""))

with open(os.path.join(outdir, "abaqus_mech_fd_values.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["kind", "parameter", "t_s", "x_mm", "ad_MPa", "fd_MPa", "fd_forward_MPa", "fd_backward_MPa", "ref_MPa", "ref_kind"])
    writer.writeheader(); writer.writerows(records)
with open(os.path.join(outdir, "abaqus_mech_fd_summary.json"), "w") as f:
    json.dump(dict(relative_step=a.h, complete=complete, missing_jobs=missing, common_steps=len(steps),
                   expected_steps=len(BS), stress_measure="von Mises of element-mean stress",
                   derivative="d sigma_vM / d s_p, material property = nominal property * s_p",
                   nrmse_percent=res, max_abs_one_sided_disagreement_MPa=asymmetry,
                   reference=dict(branch_aware=True, kink_tol=a.kink_tol, kinds=kinds) if DM else "central"), f, indent=2)
