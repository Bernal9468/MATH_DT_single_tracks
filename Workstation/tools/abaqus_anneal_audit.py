"""Abaqus Python: audit element-mean thermal histories and discrete anneal decisions.
usage: abaqus python tools/abaqus_anneal_audit.py MESH THERMAL_FD_DIR MFEM_FIELDS OUT
Works with Abaqus Python 2.7 and its bundled NumPy; reads no mechanical state.
"""
import sys,os,json,struct
import numpy as np
from odbAccess import openOdb
mesh,fd,dumps,out=sys.argv[1:5]
reuse='--reuse' in sys.argv[5:]
lines=open(mesh).read().splitlines();i=lines.index('elements');ne=int(lines[i+1]);E=np.array([list(map(int,l.split()[2:])) for l in lines[i+2:i+2+ne]])
i=lines.index('vertices');nv=int(lines[i+1])
steps=list(range(10,518,10))+[517]
Ta=1533.15
os.makedirs(out) if not os.path.isdir(out) else None
stats={};masks={};temps={}
for job in ['nom','k_p','k_m','cp_p','cp_m','rho_p','rho_m','L_p','L_m']:
    cache=os.path.join(out,job+'_element_T.npy')
    if reuse and os.path.exists(cache):
        temp=np.load(cache)
    else:
        o=openOdb(os.path.join(fd,job+'.odb'),readOnly=True)
        rows=[]
        for k in steps:
            field=o.steps['Step-'+str(k)].frames[-1].fieldOutputs['NT11']
            t=np.zeros(nv)
            for v in field.values:t[v.nodeLabel-1]=v.data
            rows.append(t[E].mean(1))
        o.close()
        temp=np.array(rows)
    temps[job]=temp;masks[job]=temp>=Ta
    np.save(os.path.join(out,job+'_element_T.npy'),temp)
    print(job,'hot element-samples',int(masks[job].sum()));sys.stdout.flush()
nom=temps['nom']
# MFEM dump samples; interpolate the last interval to Abaqus's last full thermal step.
def mf(k):
    fn=os.path.join(dumps,'f_'+str(k)+'.bin')
    with open(fn,'rb') as f:
        hdr=struct.unpack('=5id',f.read(28));v=np.fromfile(f,dtype='<f8',count=nv)
    return v[E].mean(1)
mm=[]
for k in steps:
    if k==517: mm.append(mf(510)/15.0+mf(518)*(14.0/15.0))
    else:mm.append(mf(k))
mm=np.array(mm)
nominal=dict(max_abs_element_temperature_K=float(abs(mm-nom).max()),rms_element_temperature_K=float(np.sqrt(np.mean((mm-nom)**2))),anneal_flag_disagreements=int(np.sum((mm>=Ta)!=masks['nom'])))
for p in ['k','cp','rho','L']:
    plus=masks[p+'_p'];minus=masks[p+'_m'];diff=plus!=minus
    stats[p]=dict(plus_minus_anneal_flag_disagreements=int(diff.sum()),elements_with_different_anneal_history=int(np.any(diff,axis=0).sum()),disagreements_by_step=[int(x) for x in diff.sum(axis=1)])
    # per side: elements (Abaqus labels = MFEM index + 1) whose anneal history leaves the nominal one, with the first sample
    for side in ['p','m']:
        d=masks[p+'_'+side]!=masks['nom'];el=np.where(np.any(d,axis=0))[0]
        stats[p]['changed_'+side]=dict((str(e+1),int(steps[int(np.argmax(d[:,e]))])) for e in el)
with open(os.path.join(out,'audit.json'),'w') as f:json.dump(dict(steps=steps,threshold_K=Ta,nominal=nominal,parameters=stats),f,indent=2)
print('nominal',nominal)
for p in ['k','cp','rho','L']:print(p,stats[p]['plus_minus_anneal_flag_disagreements'],stats[p]['elements_with_different_anneal_history'])
