// srck_eval (src_kernel.hpp) split per cell, Goldak mode, for the GPU source (gpu_source.cu).
// srck_eval evaluates all kept cells of a sub-sample in one call; the GPU gives every cell its own thread, so its two parts
// are written out here with the identical expressions: the box normalisation and the value of one cell.
// Templated on the scalar: Dual (nvcc build) or double (clang + Enzyme build, differentiated by __enzyme_fwddiff).
#pragma once
#include "src_kernel.hpp"

// one term of the box normalisation (cell ix,iy,iz); srck_full adds them in (ix, iy, iz) order
template <class T> SRCK_HD T srck_full_term(const SrcSub &r, const T th[6], int ix, int iy, int iz) {
    const T af = r.af*(1.0 + th[0]), ar = r.ar*(1.0 + th[0]);
    const T gb = r.b*(1.0 + th[1]),  gc = r.c*(1.0 + th[2]);
    const int nq = r.nq; const double box = r.box;
    const T dx = box*(af + ar)/nq, dy = 2.0*box*gb/nq, dz = box*gc/nq;
    const double dx0 = box*(r.af + r.ar)/nq;
    const bool front = (-box*r.ar + (ix + 0.5)*dx0) >= 0.0;
    const T lx = -box*ar + (ix + 0.5)*dx, ly = -box*gb + (iy + 0.5)*dy, lz = (iz + 0.5)*dz;
    const T ax = front ? af : ar; const double frac = front ? r.ff : r.fr;
    return frac/ax * srck_exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(gb*gb) + lz*lz/(gc*gc)));
}
template <class T> SRCK_HD T srck_full(const SrcSub &r, const T th[6]) {
    const T af = r.af*(1.0 + th[0]), ar = r.ar*(1.0 + th[0]);
    const T gb = r.b*(1.0 + th[1]),  gc = r.c*(1.0 + th[2]);
    const int nq = r.nq; const double box = r.box;
    const T dx = box*(af + ar)/nq, dy = 2.0*box*gb/nq, dz = box*gc/nq;
    const double dx0 = box*(r.af + r.ar)/nq;
    T full = SrcKMake<T>::c(0.0);
    for (int ix = 0; ix < nq; ix++) for (int iy = 0; iy < nq; iy++) for (int iz = 0; iz < nq; iz++) {
        const bool front = (-box*r.ar + (ix + 0.5)*dx0) >= 0.0;
        const T lx = -box*ar + (ix + 0.5)*dx, ly = -box*gb + (iy + 0.5)*dy, lz = (iz + 0.5)*dz;
        const T ax = front ? af : ar; const double frac = front ? r.ff : r.fr;
        full = full + frac/ax * srck_exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(gb*gb) + lz*lz/(gc*gc)));
    }
    return full;
}

template <class T> SRCK_HD void srck_cell(const SrcSub &r, const double *X, const double xi0[3], int ix, int iy, int iz,
                                          const T th[6], T full, T out[8]) {
    const T tau = r.t_s + (1.0 + th[4])*(r.tm - r.t_s) - th[5];
    T p0[3];
    if (r.seg_linear) {
        const T f = (tau - r.ts0) / (r.ts1 - r.ts0);
        for (int d = 0; d < 3; d++) p0[d] = r.xa[d] + f*(r.xb[d] - r.xa[d]);
    } else {
        for (int d = 0; d < 3; d++) p0[d] = SrcKMake<T>::c(r.xa[d]);
    }
    p0[0] = p0[0] + th[3];
    const T af = r.af*(1.0 + th[0]), ar = r.ar*(1.0 + th[0]);
    const T gb = r.b*(1.0 + th[1]),  gc = r.c*(1.0 + th[2]);
    const int nq = r.nq; const double box = r.box;
    const T dx = box*(af + ar)/nq, dy = 2.0*box*gb/nq, dz = box*gc/nq;
    const double dx0 = box*(r.af + r.ar)/nq;
    const bool front = (-box*r.ar + (ix + 0.5)*dx0) >= 0.0;
    const T lx = -box*ar + (ix + 0.5)*dx, ly = -box*gb + (iy + 0.5)*dy, lz = (iz + 0.5)*dz;
    const T ax = front ? af : ar; const double frac = front ? r.ff : r.fr;
    const T raw = frac/ax * srck_exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(gb*gb) + lz*lz/(gc*gc)));
    const T dE = r.Esub * raw / full;
    T p[3];
    for (int d = 0; d < 3; d++) p[d] = p0[d] + lx*r.e1[d] + ly*r.e2[d] + lz*r.e3[d];
    T xi[3], N[8];
    srck_invmap(X, xi0, p, xi);
    srck_shape(xi, N);
    for (int a = 0; a < 8; a++) out[a] = dE * N[a];
}
