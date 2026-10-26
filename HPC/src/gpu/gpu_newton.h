// ============================================================================
//  GPU Newton assembly of the thermal solver (bare_plate_single_layer, 2026-10-23)
//
//  The three state-dependent updates of FachinottiResidualOp::EnsureUpdated, called several times per Newton iteration:
//    conductivity matrix  PersistentK::RefillInto   (2524 elements x 8 points, k(T) / kp(T) blended by the melt fraction)
//    boundary terms       UpdateBoundary            (film + radiation residual Rb and Jacobian Jb, ~5000 boundary points)
//    lumped capacitance   UpdateCapacitance         (per node: rho, cp, latent heat, dMc/dT)
//  Each GPU routine writes exactly the arrays the CPU loop writes (into the caller's host storage). PARDISO, the residual
//  combine and the sensitivity blocks stay on the CPU. Plain C++ interface: no MFEM or CUDA types.
// ============================================================================
#pragma once

struct GpuMat {          // IN625 runtime parameters (the -kscale/-cpscale/... FD options change them)
    double kscale, cpscale, rhoscale, Lscale, kp_fraction, rhop_fraction, T_solidus, T_liquidus, L_fusion,
           fL_eut, T_eut;   // two-stage liquid fraction (driver IN625::fL_eut / T_eut; fL_eut = 0: single interval)
};

// conductivity: geometry/pattern cache of PersistentK (vd[ne*8], pos[ne*64], attr[ne], w[ne*nq], g[ne*nq*24], sh[nq*8])
bool gpu_newton_k_init(int ne, int nq, int ndof, int nnz, const int *vd, const int *pos, const int *attr,
                       const double *w, const double *g, const double *sh);
// K(x) on the pattern into Ad[nnz]; active_attr[attr] as PersistentK's `active`; ever_melted may be null (pure bulk)
void gpu_newton_k_refill(const double *x, const char *active_attr, int nattr, const double *ever_melted, const GpuMat &m,
                         double *Ad);

// boundary: one record set per residual operator (bdry_qps and the Jb pattern are fixed for its lifetime)
//   qoff[nqp+1] offsets into dof/shp; bpos[] the CSR positions of the nd x nd blocks (offsets qoff2[nqp+1])
void gpu_newton_b_setup(int nqp, int ndof, int jnnz, const int *qoff, const int *dof, const double *shp,
                        const double *w, const int *kind, const int *qoff2, const int *bpos);
void gpu_newton_b_update(const double *x, const double *ever_melted, double hconv, double hp, double emis, double sigma,
                         double Ta, double sub_scale, double bridge_scale, double if_scale, double *Rb, double *Jd);

// lumped capacitance (UpdateCapacitance): Mc, Ml, Lp, Mo [n]
void gpu_newton_cap(int n, const double *T, const double *Tn, const double *V, const double *ever_melted, const GpuMat &m,
                    double idt, double *Mc, double *Ml, double *Lp, double *Mo);
