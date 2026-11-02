"""Abaqus Python: discrete material state of a mechanical replica job made with abaqus_mech_deck.py --state-output, for
branch-aware finite differences (tools/abaqus_mech_fd_compare.py --state-dir).
Per integration point and static step: plastic = Mises on the yield surface (relative tolerance TOL; AC YIELD is not
available for this C3D8 static procedure), annealed = element-mean temperature >= the anneal temperature. Yield surface
as the deck's *PLASTIC rows: 433 -> 831 MPa linear to PEEQ 0.43 below Ta - 1, hardening reduced linearly over the last
kelvin, perfectly plastic (433 MPa) at/above Ta. Fields are joined by (element, point): S also carries the deck's LINE
request, so the value sequences are not aligned.
usage (in the job directory): abaqus python tools/abaqus_state_extract.py JOB [JOB ...]
-> JOB_state.npz: label, ip (sorted), step (static step numbers), plastic, annealed (bool, nstep x npoint)
"""
import sys
import numpy as np
from odbAccess import openOdb

TA, S0, S1, E1, TOL = 1533.15, 433.0, 831.0, 0.43, 1e-4

for job in sys.argv[1:]:
    o = openOdb(job + '.odb', readOnly=True); keys = None; steps = []; P = []; A = []
    for n, name in enumerate(o.steps.keys(), 1):
        fr = o.steps[name].frames
        if not fr or 'PEEQ' not in fr[-1].fieldOutputs: continue
        fo = fr[-1].fieldOutputs
        mises = dict(((v.elementLabel, v.integrationPoint), v.mises) for v in fo['S'].values)
        peeq = dict(((v.elementLabel, v.integrationPoint), v.data) for v in fo['PEEQ'].values)
        temp = dict(((v.elementLabel, v.integrationPoint), v.data) for v in fo['TEMP'].values)
        if keys is None: keys = sorted(peeq)
        m = np.array([mises[k] for k in keys]); pe = np.array([peeq[k] for k in keys]); t = np.array([temp[k] for k in keys])
        h = np.where(t <= TA - 1.0, 1.0, np.where(t >= TA, 0.0, TA - t))
        sy = S0 + h * (S1 - S0) * np.minimum(pe, E1) / E1
        steps.append(n); P.append(m >= sy * (1.0 - TOL)); A.append(t >= TA)
        print(job, 'step', n); sys.stdout.flush()
    o.close()
    k = np.array(keys)
    np.savez_compressed(job + '_state.npz', label=k[:, 0], ip=k[:, 1], step=np.array(steps), plastic=np.array(P), annealed=np.array(A))
