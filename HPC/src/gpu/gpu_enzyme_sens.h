// ============================================================================
//  First-order sensitivity blocks (a), (b), (c) of the thermal driver on the GPU, differentiated by Enzyme in device code
//  (bare_plate_single_layer, 2026-10-23). Built with clang -x cuda + ClangEnzyme-21.so (src/gpu/gpu_enzyme_sens.cu).
//    (a) domain conductivity:  d[k(T_next)]/ds_{k,kp}, one thread per (element, quadrature point)      -> fb_dk1
//    (b) boundary film/rad.:   dq/ds_{hc,T0,hp},       one thread per boundary quadrature point          -> fb_dq1
//    (c) nodal capacity/latent: -d nodal/ds_p for all directions, one thread per node                  -> fb_nodal1
//  The differentiated functions are fb_enzyme_fns.hpp, the same code fb_enzyme_kernels.cpp differentiates on the CPU.
//  Every routine ADDS its contributions into the caller's host arrays, as the CPU loops do. Plain C++ interface.
// ============================================================================
#pragma once

bool gpu_sens_init();
void gpu_sens_set_fl2(double fl_eut, double t_eut);   // two-stage liquid fraction of fL_fn (__constant__ copy)

// (a) geometry of DomSensCache: vd[ne*8], shape[ne*nq*8], gshape[ne*nq*8*3] ([e][q][a][d]), w[ne*nq], attr[ne]
void gpu_sens_a_setup(int ne, int nq, int ndof, const int *vd, const double *shape, const double *gshape, const double *w,
                      const int *attr);
void gpu_sens_a(const double *u, const char *active_attr, int nattr, const double *Mv, double kp_fraction,
                double *rk, double *rkp /* may be null */);

// (b) boundary points: qoff[nqp+1] into dof/shp, w[nqp], kind[nqp]
void gpu_sens_b_setup(int nqp, int ndof, const int *qoff, const int *dof, const double *shp, const double *w, const int *kind);
void gpu_sens_b(const double *u, const double *Mv, double h_conv, double hp_over_hc, double emis, double sigma, double Tamb,
                double sub_scale, double bridge_scale, double if_scale, double *rhc, double *rt0, double *rhp /* may be null */);

// (c) nodal block: pd[p][i] = prev_der[p][i] (nw arrays), slot[nw] (-1: no explicit scale), rp[p][i] += -d nodal/ds_p
void gpu_sens_c(int ndof, int nw, const double *Tnext, const double *Tn, const double *const *pd, const int *slot,
                const double *Mv, double rhop_fraction, const double *V, double idt, double Ts, double Tl, double *const *rp);
