"""Abaqus Python (abaqus python tools/abaqus_mech_extract.py JOB [JOB ...], run in the job directory): the stresses of the
mechanical replica (tools/abaqus_mech_deck.py) at the end of every static step, averaged over the 8 integration points of
each output element -> JOB_S.csv (step, element, S11, S22, S33, S12, S13, S23). Element = Abaqus label (MFEM id + 1).
"""
import sys
from odbAccess import openOdb

for job in sys.argv[1:]:
    o = openOdb(job + '.odb', readOnly=True)
    out = open(job + '_S.csv', 'w'); out.write('step,elem,S11,S22,S33,S12,S13,S23\n')
    for n, name in enumerate(o.steps.keys(), 1):
        fr = o.steps[name].frames
        if len(fr) < 2 or 'S' not in fr[-1].fieldOutputs: continue
        acc = {}
        for v in fr[-1].fieldOutputs['S'].values:
            a = acc.setdefault(v.elementLabel, [0.0] * 7)
            for c in range(6): a[c] += v.data[c]
            a[6] += 1
        for e in sorted(acc):
            a = acc[e]; out.write('%d,%d,' % (n, e) + ','.join('%.9g' % (a[c] / a[6]) for c in range(6)) + '\n')
    out.close(); o.close(); print(job, 'steps', n)
