"""Field dumps of the thermal driver (FIELD_DUMP_DIR): reader and planar grids of the mesh nodes.

File f_<step>.bin: int[5] {ndof, step, n1, n2, nm}, double t, then ndof doubles per field: T [K], the n1 first-order fields
dT/ds_p (driver order, post/analyze_case.PARAMS; s_p = relative change of parameter p), n2 second-order fields, nm more
(ever-melted flag). Node k of the order-1 mesh carries dof k.
"""
import glob, os, re, struct
import numpy as np


def dump_files(folder):
    fs = glob.glob(os.path.join(folder, "f_*.bin"))
    return sorted(fs, key=lambda p: int(re.search(r"f_(\d+)", p).group(1)))


def header(fn):
    with open(fn, "rb") as f:
        ndof, step, n1, n2, nm = struct.unpack("5i", f.read(20)); t = struct.unpack("d", f.read(8))[0]
    return dict(ndof=ndof, step=step, n1=n1, n2=n2, nm=nm, t=t)


def read(fn, fields, nodes):
    """values of the given fields (0 = T, 1 + p = dT/ds_p) at the given nodes: array [len(fields), len(nodes)]"""
    h = header(fn)
    M = np.memmap(fn, dtype=np.float64, mode="r", offset=28, shape=(1 + h["n1"] + h["n2"] + h["nm"], h["ndof"]))
    return np.array(M[np.asarray(fields)][:, nodes]), h["t"]


class Plane:
    """nodes of the mesh on a coordinate plane (axis = 1: y = value, 2: z = value) as a tensor grid (u, w): G[iu, iw]"""
    def __init__(self, V, axis, value=0.0, tol=1e-9):
        V = np.asarray(V); keep = [a for a in range(3) if a != axis]
        self.nodes = np.where(np.abs(V[:, axis] - value) < tol)[0]
        self.u, iu = np.unique(np.round(V[self.nodes, keep[0]], 9), return_inverse=True)
        self.w, iw = np.unique(np.round(V[self.nodes, keep[1]], 9), return_inverse=True)
        self.iu, self.iw = iu, iw

    def grid(self, vals):
        G = np.full((len(self.u), len(self.w)), np.nan); G[self.iu, self.iw] = vals
        return G
