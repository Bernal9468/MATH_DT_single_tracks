// GPU laser source: see gpu_source.h.
//
// Equivalence with the CPU code (SpatialHashSource::BuildSource, spread_mode 2, and block (e) of the driver):
//  * Goldak cells, weights, normalisation and deposition points use the same formulas in the same order of operations.
//  * Element location: the CPU FindElement returns the first active element of the hash bucket whose bounding box contains
//    the point (inclusive bounds). Buckets are filled in increasing element index and every element containing the point
//    is in the point's bucket, so that is the SMALLEST active element index whose box contains the point. The GPU uses
//    its own uniform grid and returns exactly that element. A point in no element's box is skipped, as on the CPU (the
//    CPU fallback search can only return an element whose box does not contain the point, which BuildSource then skips).
//  * Reference coordinates: Newton on the trilinear map (MFEM uses its own Newton); equal to round-off.
//  * Sums over cells use atomics, so the summation order differs from the CPU: results agree to round-off, and two GPU
//    runs can differ at round-off level.
//
// Two derivative back ends, same kernels otherwise (srck_cell.hpp):
//  * default (nvcc):          forward-mode dual numbers (struct Dual below).
//  * -DGPU_SRC_ENZYME (clang -x cuda -fplugin=ClangEnzyme-21.so): Enzyme forward mode (__enzyme_fwddiff) in device code, as
//                             fb_enzyme_kernels.cpp does on the CPU (srck_curve_d1).
#include "gpu_source.h"
#include "srck_cell.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

// ---------------------------------------------------------------- forward-mode dual number (value + tangent)
struct Dual { double v, d; };
#ifndef GPU_SRC_ENZYME
__host__ __device__ inline Dual operator+(Dual a, Dual b) { return {a.v + b.v, a.d + b.d}; }
__host__ __device__ inline Dual operator+(Dual a, double b) { return {a.v + b, a.d}; }
__host__ __device__ inline Dual operator+(double a, Dual b) { return {a + b.v, b.d}; }
__host__ __device__ inline Dual operator-(Dual a, Dual b) { return {a.v - b.v, a.d - b.d}; }
__host__ __device__ inline Dual operator-(Dual a, double b) { return {a.v - b, a.d}; }
__host__ __device__ inline Dual operator-(double a, Dual b) { return {a - b.v, -b.d}; }
__host__ __device__ inline Dual operator-(Dual a) { return {-a.v, -a.d}; }
__host__ __device__ inline Dual operator*(Dual a, Dual b) { return {a.v * b.v, a.d * b.v + a.v * b.d}; }
__host__ __device__ inline Dual operator*(Dual a, double b) { return {a.v * b, a.d * b}; }
__host__ __device__ inline Dual operator*(double a, Dual b) { return {a * b.v, a * b.d}; }
__host__ __device__ inline Dual operator/(Dual a, Dual b) { return {a.v / b.v, (a.d * b.v - a.v * b.d) / (b.v * b.v)}; }
__host__ __device__ inline Dual operator/(Dual a, double b) { return {a.v / b, a.d / b}; }
__host__ __device__ inline Dual operator/(double a, Dual b) { return {a / b.v, -a * b.d / (b.v * b.v)}; }
__host__ __device__ inline Dual srck_exp(Dual a) { const double e = srck_exp(a.v); return {e, e * a.d}; }
template <> struct SrcKMake<Dual> { __host__ __device__ static Dual c(double v) { return {v, 0.0}; } };
#else
// ---------------------------------------------------------------- Enzyme forward mode in device code
extern __device__ int enzyme_dup, enzyme_const, enzyme_dupnoneed;
template <typename RT, typename... Args> __device__ RT __enzyme_fwddiff(void *, Args...);
// th = s * e_j, as srck_curve in fb_enzyme_kernels.cpp
__device__ __noinline__ double full_curve(double s, const SrcSub *r, int j) {
    double th[6] = {0, 0, 0, 0, 0, 0}; th[j] = s; return srck_full<double>(*r, th);
}
__device__ __noinline__ double full_term_curve(double s, const SrcSub *r, int j, int ix, int iy, int iz) {
    double th[6] = {0, 0, 0, 0, 0, 0}; th[j] = s; return srck_full_term<double>(*r, th, ix, iy, iz);
}
__device__ __noinline__ void cell_curve(double s, double full, const SrcSub *r, const double *X, const double *xi0,
                                        int ix, int iy, int iz, int j, double *out) {
    double th[6] = {0, 0, 0, 0, 0, 0}; th[j] = s; srck_cell<double>(*r, X, xi0, ix, iy, iz, th, full, out);
}
// ---- second order (Enzyme forward over forward). Two-parameter curve th = s e_a + t e_b, as srck_curve2 /
// srck_curve2_ds in fb_enzyme_kernels.cpp (a == b gives the pure second derivative). The box normalisation enters the
// per-cell value as its second-order Taylor polynomial f0 + fa s + fb t + fab s t, whose coefficients are summed from the
// per-cell terms in the CPU's cell order -- exact first and mixed derivatives at s = t = 0.
struct Full2 { double f0, fa, fb, fab; };
__device__ __noinline__ double term2(double s, double t, const SrcSub *r, int a, int b, int ix, int iy, int iz) {
    double th[6] = {0, 0, 0, 0, 0, 0}; th[a] += s; th[b] += t; return srck_full_term<double>(*r, th, ix, iy, iz);
}
__device__ __noinline__ double term2_ds(double t, const SrcSub *r, int a, int b, int ix, int iy, int iz) {
    return __enzyme_fwddiff<double>((void*)term2, enzyme_dup, 0.0, 1.0, enzyme_const, t, enzyme_const, r,
                                    enzyme_const, a, enzyme_const, b, enzyme_const, ix, enzyme_const, iy, enzyme_const, iz);
}
__device__ __noinline__ void cell2(double s, double t, const SrcSub *r, const double *X, const double *xi0, int ix, int iy, int iz,
                                   int a, int b, const Full2 *F, double *out) {
    double th[6] = {0, 0, 0, 0, 0, 0}; th[a] += s; th[b] += t;
    const double full = F->f0 + F->fa * s + F->fb * t + F->fab * s * t;
    srck_cell<double>(*r, X, xi0, ix, iy, iz, th, full, out);
}
__device__ __noinline__ void cell2_ds(double t, const SrcSub *r, const double *X, const double *xi0, int ix, int iy, int iz,
                                      int a, int b, const Full2 *F, double *dout) {
    double tmp[8];
    __enzyme_fwddiff<void>((void*)cell2, enzyme_dup, 0.0, 1.0, enzyme_const, t, enzyme_const, r, enzyme_const, X,
                           enzyme_const, xi0, enzyme_const, ix, enzyme_const, iy, enzyme_const, iz, enzyme_const, a,
                           enzyme_const, b, enzyme_const, F, enzyme_dupnoneed, tmp, dout);
}
__constant__ int kPa[21] = {0,0,0,0,0,0, 1,1,1,1,1, 2,2,2,2, 3,3,3, 4,4, 5};   // pair k21 = (a, b), a <= b, row-major
__constant__ int kPb[21] = {0,1,2,3,4,5, 1,2,3,4,5, 2,3,4,5, 3,4,5, 4,5, 5};
__constant__ int kDiag[21] = {1,0,0,0,0,0, 1,0,0,0,0, 1,0,0,0, 1,0,0, 1,0, 1};
#endif

// ---------------------------------------------------------------- device state
namespace {
#ifdef GPU_SRC_ENZYME
// K2a: normalisation-term coefficients per (sub-sample, cell, pair): value, d/ds, d/dt, d2/dsdt
__global__ void k_full2_terms(const GpuSrcSub *subs, int nsub, bool all_pairs, Full2 *t2) {
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long cell = gid / 21; const int p = (int)(gid % 21);
    const int s = (int)(cell / SRCK_MAXCELL), q = (int)(cell % SRCK_MAXCELL);
    if (s >= nsub || (!all_pairs && !kDiag[p])) return;
    const SrcSub &r = subs[s].rec;
    const int nq = r.nq;
    if (q >= nq*nq*nq) return;
    const int ix = q / (nq*nq), iy = (q / nq) % nq, iz = q % nq, a = kPa[p], b = kPb[p];
    Full2 c;
    c.f0  = term2(0.0, 0.0, &r, a, b, ix, iy, iz);
    c.fa  = term2_ds(0.0, &r, a, b, ix, iy, iz);
    c.fb  = __enzyme_fwddiff<double>((void*)term2, enzyme_const, 0.0, enzyme_dup, 0.0, 1.0, enzyme_const, &r,
                                     enzyme_const, a, enzyme_const, b, enzyme_const, ix, enzyme_const, iy, enzyme_const, iz);
    c.fab = __enzyme_fwddiff<double>((void*)term2_ds, enzyme_dup, 0.0, 1.0, enzyme_const, &r, enzyme_const, a, enzyme_const, b,
                                     enzyme_const, ix, enzyme_const, iy, enzyme_const, iz);
    t2[(size_t)cell * 21 + p] = c;
}
// K2b: in-order sums of the coefficients, one thread per (sub-sample, pair)
__global__ void k_full2_sum(const GpuSrcSub *subs, int nsub, bool all_pairs, const Full2 *t2, Full2 *full2) {
    const int s = blockIdx.x, p = threadIdx.x;
    if (s >= nsub || p >= 21 || (!all_pairs && !kDiag[p])) return;
    const int n = subs[s].rec.nq * subs[s].rec.nq * subs[s].rec.nq;
    Full2 f = {0.0, 0.0, 0.0, 0.0};
    for (int q = 0; q < n; q++) { const Full2 &c = t2[((size_t)s * SRCK_MAXCELL + q) * 21 + p];
        f.f0 = f.f0 + c.f0; f.fa = f.fa + c.fa; f.fb = f.fb + c.fb; f.fab = f.fab + c.fab; }
    full2[s * 21 + p] = f;
}
// K2c: second derivative of the cell's 8 nodal energies along pair p, one thread per (kept cell, pair)
__global__ void k_d2(const GpuSrcSub *subs, int nsub, bool all_pairs, const Full2 *full2, const double *X, const int *dofs,
                     const int *kept, const double *xis, int ndof, double *d2) {
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long cid = gid / 21; const int p = (int)(gid % 21);
    const int s = (int)(cid / SRCK_MAXCELL), q = (int)(cid % SRCK_MAXCELL);
    if (s >= nsub || (!all_pairs && !kDiag[p])) return;
    const SrcSub &r = subs[s].rec;
    const int nq = r.nq;
    if (q >= nq*nq*nq) return;
    const int e = kept[cid];
    if (e < 0) return;
    const int ix = q / (nq*nq), iy = (q / nq) % nq, iz = q % nq, a = kPa[p], b = kPb[p];
    double dtmp[8], d2out[8];
    __enzyme_fwddiff<void>((void*)cell2_ds, enzyme_dup, 0.0, 1.0, enzyme_const, &r, enzyme_const, X + 24*e,
                           enzyme_const, xis + 3*cid, enzyme_const, ix, enzyme_const, iy, enzyme_const, iz,
                           enzyme_const, a, enzyme_const, b, enzyme_const, &full2[s * 21 + p], enzyme_dupnoneed, dtmp, d2out);
    const int *dd = dofs + 8*e;
    for (int k = 0; k < 8; k++) if (d2out[k] != 0.0) atomicAdd(&d2[(long)p*ndof + dd[k]], d2out[k]);
}
Full2 *d_t2 = nullptr, *d_full2 = nullptr; double *d_d2 = nullptr; int cap2 = 0;
#endif
struct Grid { double x0[3], h[3]; int n[3]; };
int g_ne = 0, g_ndof = 0;
double *d_X = nullptr, *d_lo = nullptr, *d_hi = nullptr;
int *d_dofs = nullptr, *d_gstart = nullptr, *d_gelem = nullptr;
Grid g_grid;
// per-call buffers (grown on demand)
GpuSrcSub *d_subs = nullptr; int cap_subs = 0;
char *d_active = nullptr;
double *d_f = nullptr, *d_d1 = nullptr, *d_fk = nullptr, *d_full = nullptr, *d_wt = nullptr; Dual *d_fullD = nullptr, *d_dt = nullptr;
int *d_kept = nullptr; double *d_xi = nullptr; unsigned long long *d_nkept = nullptr;
char g_name[256] = "none";

__device__ int find_element(const Grid &G, const int *gstart, const int *gelem, const double *lo, const double *hi,
                            const char *active, double x, double y, double z) {
    const double p[3] = {x, y, z}; int c[3];
    for (int d = 0; d < 3; d++) {
        const double s = floor((p[d] - G.x0[d]) / G.h[d]);
        if (s < 0.0 || s >= G.n[d]) return -1;          // outside the mesh box: no element contains the point
        c[d] = (int)s;
    }
    const int cell = (c[0]*G.n[1] + c[1])*G.n[2] + c[2];
    for (int k = gstart[cell]; k < gstart[cell + 1]; k++) {   // element ids ascending: the first hit is the smallest index
        const int e = gelem[k];
        if (!active[e]) continue;
        const double *l = lo + 3*e, *h = hi + 3*e;
        if (x>=l[0]&&x<=h[0]&&y>=l[1]&&y<=h[1]&&z>=l[2]&&z<=h[2]) return e;
    }
    return -1;
}

// K1a: the normalisation terms, one thread per (sub-sample, cell, slot): slot 0 = nominal weight (BuildSource's own
//      expression), slots 1..6 = value and tangent of the term w.r.t. th_j (dual numbers or Enzyme)
__global__ void k_full_terms(const GpuSrcSub *subs, int nsub, bool want_d1, double *wt, Dual *dt) {
    const long gid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long cell = gid / 7; const int slot = (int)(gid % 7);
    const int s = (int)(cell / SRCK_MAXCELL), q = (int)(cell % SRCK_MAXCELL);
    if (s >= nsub) return;
    const SrcSub &r = subs[s].rec;
    const int nq = r.nq;
    if (q >= nq*nq*nq) return;
    const int ix = q / (nq*nq), iy = (q / nq) % nq, iz = q % nq;
    if (slot == 0) {
        const double box_factor = r.box;
        const double dx = box_factor*(r.af + r.ar)/nq, dy = 2.0*box_factor*r.b/nq, dz = box_factor*r.c/nq;
        const double lx = -box_factor*r.ar + (ix+0.5)*dx, ly = -box_factor*r.b + (iy+0.5)*dy, lz = (iz+0.5)*dz;
        const bool front = (lx >= 0.0);
        const double ax = front ? r.af : r.ar, frac = front ? r.ff : r.fr;
        wt[cell] = frac/ax*exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(r.b*r.b) + lz*lz/(r.c*r.c)));
    } else if (want_d1) {
        const int j = slot - 1;
#ifndef GPU_SRC_ENZYME
        Dual th[6]; for (int k = 0; k < 6; k++) th[k] = {0.0, 0.0};
        th[j].d = 1.0;
        dt[(size_t)cell * 6 + j] = srck_full_term(r, th, ix, iy, iz);
#else
        dt[(size_t)cell * 6 + j].v = full_term_curve(0.0, &r, j, ix, iy, iz);
        dt[(size_t)cell * 6 + j].d = __enzyme_fwddiff<double>((void*)full_term_curve, enzyme_dup, 0.0, 1.0,
                                                             enzyme_const, &r, enzyme_const, j, enzyme_const, ix,
                                                             enzyme_const, iy, enzyme_const, iz);
#endif
    }
}
// K1b: in-order sums of the terms (thread 0 nominal, threads 1..6 the dual sums), as the CPU loops add them
__global__ void k_full_sum(const GpuSrcSub *subs, int nsub, bool want_d1, const double *wt, const Dual *dt,
                           double *full, Dual *fullD) {
    const int s = blockIdx.x, t = threadIdx.x;
    if (s >= nsub) return;
    const int n = subs[s].rec.nq * subs[s].rec.nq * subs[s].rec.nq;
    const size_t c0 = (size_t)s * SRCK_MAXCELL;
    if (t == 0) { double fw = 0.0; for (int q = 0; q < n; q++) fw += wt[c0 + q]; full[s] = fw; }
    else if (want_d1 && t <= 6) {
        const int j = t - 1; Dual f = {0.0, 0.0};
        for (int q = 0; q < n; q++) { const Dual &a = dt[(c0 + q) * 6 + j]; f.v = f.v + a.v; f.d = f.d + a.d; }
        fullD[s*6 + j] = f;
    }
}
// K1 (single-kernel version, one thread per sum: kept for reference; latency-bound, ~4 ms per call): Goldak normalisation per sub-sample. thread 0: nominal weight sum exactly as BuildSource; threads 1..6: dual sums.
__global__ void k_full(const GpuSrcSub *subs, int nsub, bool want_d1, double *full, Dual *fullD) {
    const int s = blockIdx.x, t = threadIdx.x;
    if (s >= nsub) return;
    const SrcSub &r = subs[s].rec;
    if (t == 0) {
        const int nq = r.nq; const double box_factor = r.box;
        const double dx = box_factor*(r.af + r.ar)/nq, dy = 2.0*box_factor*r.b/nq, dz = box_factor*r.c/nq;
        double full_weight = 0.0;
        for (int ix = 0; ix < nq; ix++) for (int iy = 0; iy < nq; iy++) for (int iz = 0; iz < nq; iz++) {
            const double lx = -box_factor*r.ar + (ix+0.5)*dx, ly = -box_factor*r.b + (iy+0.5)*dy, lz = (iz+0.5)*dz;
            const bool front = (lx >= 0.0);
            const double ax = front ? r.af : r.ar, frac = front ? r.ff : r.fr;
            full_weight += frac/ax*exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(r.b*r.b) + lz*lz/(r.c*r.c)));
        }
        full[s] = full_weight;
    } else if (want_d1 && t <= 6) {
#ifndef GPU_SRC_ENZYME
        Dual th[6]; for (int j = 0; j < 6; j++) th[j] = {0.0, 0.0};
        th[t - 1].d = 1.0;
        fullD[s*6 + t - 1] = srck_full(r, th);
#else
        const int j = t - 1;
        fullD[s*6 + j].v = full_curve(0.0, &r, j);
        fullD[s*6 + j].d = __enzyme_fwddiff<double>((void*)full_curve, enzyme_dup, 0.0, 1.0,
                                                    enzyme_const, &r, enzyme_const, j);
#endif
    }
}

// K2: one thread per (sub-sample, Goldak cell): locate, deposit the nominal energy, record (element, xi0) for K3.
__global__ void k_deposit(const GpuSrcSub *subs, int nsub, const double *full, Grid G, const int *gstart, const int *gelem,
                          const double *lo, const double *hi, const double *X, const int *dofs, const char *active,
                          double *f, int *kept, double *xis, unsigned long long *nkept) {
    const long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;
    const int s = (int)(gid / SRCK_MAXCELL), q = (int)(gid % SRCK_MAXCELL);
    if (s >= nsub) return;
    const GpuSrcSub &S = subs[s]; const SrcSub &r = S.rec;
    const int nq = r.nq;
    if (q >= nq*nq*nq) return;
    kept[gid] = -1;
    const int ix = q / (nq*nq), iy = (q / nq) % nq, iz = q % nq;
    // cell weight and point: BuildSource's own expressions
    const double box_factor = r.box;
    const double dx = box_factor*(r.af + r.ar)/nq, dy = 2.0*box_factor*r.b/nq, dz = box_factor*r.c/nq;
    const double lx = -box_factor*r.ar + (ix+0.5)*dx, ly = -box_factor*r.b + (iy+0.5)*dy, lz = (iz+0.5)*dz;
    const bool front = (lx >= 0.0);
    const double ax = front ? r.af : r.ar, frac = front ? r.ff : r.fr;
    const double w = frac/ax*exp(-3.0*(lx*lx/(ax*ax) + ly*ly/(r.b*r.b) + lz*lz/(r.c*r.c)));
    const double dE = r.Esub*(w/full[s]);
    const double px = S.pL[0] + lx*r.e1[0] + ly*r.e2[0] + lz*r.e3[0];
    const double py = S.pL[1] + lx*r.e1[1] + ly*r.e2[1] + lz*r.e3[1];
    const double pz = S.pL[2] + lx*r.e1[2] + ly*r.e2[2] + lz*r.e3[2];
    const int e = find_element(G, gstart, gelem, lo, hi, active, px, py, pz);
    if (e < 0) return;
    // reference coordinates: Newton on the trilinear map from the element centre (4 steps; the box map is affine)
    const double *Xe = X + 24*e;
    const double c0[3] = {0.5, 0.5, 0.5}, p[3] = {px, py, pz};
    double xa[3], xi[3];
    srck_invmap<double>(Xe, c0, p, xa);
    srck_invmap<double>(Xe, xa, p, xi);
    const double tol = 1e-8;                              // the inside test of MFEM's inverse transformation (reference tolerance)
    for (int d = 0; d < 3; d++) if (!(xi[d] >= -tol && xi[d] <= 1.0 + tol)) return;
    double N[8]; srck_shape<double>(xi, N);
    for (int a = 0; a < 8; a++) atomicAdd(&f[dofs[8*e + a]], dE*N[a]);
    kept[gid] = e;
    for (int d = 0; d < 3; d++) xis[3*gid + d] = xi[d];
    atomicAdd(nkept, 1ull);
}

// K3: one thread per (sub-sample, cell, parameter j): first derivative of the cell's 8 nodal energies w.r.t. th_j.
__global__ void k_d1(const GpuSrcSub *subs, int nsub, const Dual *fullD, const double *X, const int *dofs,
                     const int *kept, const double *xis, int ndof, double *d1, double *fk) {
    const long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;
    const long cid = gid / 6; const int j = (int)(gid % 6);
    const int s = (int)(cid / SRCK_MAXCELL), q = (int)(cid % SRCK_MAXCELL);
    if (s >= nsub) return;
    const SrcSub &r = subs[s].rec;
    const int nq = r.nq;
    if (q >= nq*nq*nq) return;
    const int e = kept[cid];
    if (e < 0) return;
    const int ix = q / (nq*nq), iy = (q / nq) % nq, iz = q % nq;
    const int *dd = dofs + 8*e;
#ifndef GPU_SRC_ENZYME
    Dual th[6]; for (int k = 0; k < 6; k++) th[k] = {0.0, 0.0};
    th[j].d = 1.0;
    Dual out[8];
    srck_cell(r, X + 24*e, xis + 3*cid, ix, iy, iz, th, fullD[s*6 + j], out);
    for (int a = 0; a < 8; a++) {
        if (out[a].d != 0.0) atomicAdd(&d1[(long)j*ndof + dd[a]], out[a].d);
        if (fk && j == 0 && out[a].v != 0.0) atomicAdd(&fk[dd[a]], out[a].v);
    }
#else
    double out[8], dout[8];
    __enzyme_fwddiff<void>((void*)cell_curve, enzyme_dup, 0.0, 1.0, enzyme_dup, fullD[s*6 + j].v, fullD[s*6 + j].d,
                           enzyme_const, &r, enzyme_const, X + 24*e, enzyme_const, xis + 3*cid,
                           enzyme_const, ix, enzyme_const, iy, enzyme_const, iz, enzyme_const, j,
                           enzyme_dup, out, dout);
    for (int a = 0; a < 8; a++) {
        if (dout[a] != 0.0) atomicAdd(&d1[(long)j*ndof + dd[a]], dout[a]);
        if (fk && j == 0 && out[a] != 0.0) atomicAdd(&fk[dd[a]], out[a]);
    }
#endif
}
} // namespace

bool gpu_src_init(int ne, int ndof, const double *X, const int *dofs) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return false;
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    snprintf(g_name, sizeof(g_name), "%.200s (cc %d.%d)", prop.name, prop.major, prop.minor);
    g_ne = ne; g_ndof = ndof;
    std::vector<double> lo(3*ne), hi(3*ne);
    double blo[3] = {1e300, 1e300, 1e300}, bhi[3] = {-1e300, -1e300, -1e300}, hmin[3] = {1e300, 1e300, 1e300};
    for (int e = 0; e < ne; e++) {
        for (int d = 0; d < 3; d++) { lo[3*e+d] = 1e300; hi[3*e+d] = -1e300; }
        for (int a = 0; a < 8; a++) for (int d = 0; d < 3; d++) {
            const double v = X[24*e + 3*a + d];
            lo[3*e+d] = std::min(lo[3*e+d], v); hi[3*e+d] = std::max(hi[3*e+d], v);
        }
        for (int d = 0; d < 3; d++) {
            blo[d] = std::min(blo[d], lo[3*e+d]); bhi[d] = std::max(bhi[d], hi[3*e+d]);
            hmin[d] = std::min(hmin[d], hi[3*e+d] - lo[3*e+d]);
        }
    }
    // uniform grid, cell = 2 x the smallest element size per axis (at most 512 cells per axis)
    for (int d = 0; d < 3; d++) {
        const double span = bhi[d] - blo[d];
        int n = std::max(1, std::min(512, (int)std::ceil(span / (2.0*hmin[d]))));
        g_grid.x0[d] = blo[d]; g_grid.n[d] = n; g_grid.h[d] = span / n * (1.0 + 1e-12);
    }
    const int ncell = g_grid.n[0]*g_grid.n[1]*g_grid.n[2];
    std::vector<std::vector<int>> cell_elems(ncell);
    for (int e = 0; e < ne; e++) {                                   // e ascending -> each list sorted ascending
        int c0[3], c1[3];
        for (int d = 0; d < 3; d++) {
            c0[d] = std::max(0, (int)std::floor((lo[3*e+d] - g_grid.x0[d]) / g_grid.h[d]));
            c1[d] = std::min(g_grid.n[d] - 1, (int)std::floor((hi[3*e+d] - g_grid.x0[d]) / g_grid.h[d]));
        }
        for (int a = c0[0]; a <= c1[0]; a++) for (int b = c0[1]; b <= c1[1]; b++) for (int c = c0[2]; c <= c1[2]; c++)
            cell_elems[(a*g_grid.n[1] + b)*g_grid.n[2] + c].push_back(e);
    }
    std::vector<int> gstart(ncell + 1, 0), gelem;
    for (int c = 0; c < ncell; c++) { gstart[c+1] = gstart[c] + (int)cell_elems[c].size(); gelem.insert(gelem.end(), cell_elems[c].begin(), cell_elems[c].end()); }
    CK(cudaMalloc(&d_X, sizeof(double)*24*ne)); CK(cudaMemcpy(d_X, X, sizeof(double)*24*ne, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_dofs, sizeof(int)*8*ne)); CK(cudaMemcpy(d_dofs, dofs, sizeof(int)*8*ne, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_lo, sizeof(double)*3*ne)); CK(cudaMemcpy(d_lo, lo.data(), sizeof(double)*3*ne, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_hi, sizeof(double)*3*ne)); CK(cudaMemcpy(d_hi, hi.data(), sizeof(double)*3*ne, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_gstart, sizeof(int)*(ncell + 1))); CK(cudaMemcpy(d_gstart, gstart.data(), sizeof(int)*(ncell + 1), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_gelem, sizeof(int)*std::max<size_t>(1, gelem.size())));
    if (!gelem.empty()) CK(cudaMemcpy(d_gelem, gelem.data(), sizeof(int)*gelem.size(), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_active, ne));
    CK(cudaMalloc(&d_f, sizeof(double)*ndof)); CK(cudaMalloc(&d_fk, sizeof(double)*ndof));
    CK(cudaMalloc(&d_d1, sizeof(double)*6*ndof)); CK(cudaMalloc(&d_nkept, sizeof(unsigned long long)));
#ifdef GPU_SRC_ENZYME
    const char *ad = "Enzyme forward mode";
#else
    const char *ad = "dual numbers";
#endif
    printf("  [GPU_SOURCE] derivatives: %s; device %s; %d elements, %d dofs; location grid %dx%dx%d (%zu entries)\n",
           ad, g_name, ne, ndof, g_grid.n[0], g_grid.n[1], g_grid.n[2], gelem.size());
    fflush(stdout);
    return true;
}

const char *gpu_src_device_name() { return g_name; }

bool gpu_src_has_second_order() {
#ifdef GPU_SRC_ENZYME
    return true;
#else
    return false;
#endif
}

void gpu_src_build(const GpuSrcSub *subs, int nsub, const char *elem_active, double *f, double *d1, double *d2, int pairs,
                   double *fk, long *nkept) {
    const int ndof = g_ndof;
#ifndef GPU_SRC_ENZYME
    if (d2) { fprintf(stderr, "gpu_src_build: second order needs the Enzyme build\n"); std::abort(); }
#endif
    if (nsub > cap_subs) {
        if (d_subs) { cudaFree(d_subs); cudaFree(d_full); cudaFree(d_fullD); cudaFree(d_kept); cudaFree(d_xi); cudaFree(d_wt); cudaFree(d_dt); }
        cap_subs = std::max(nsub, 2*cap_subs);
        CK(cudaMalloc(&d_subs, sizeof(GpuSrcSub)*cap_subs));
        CK(cudaMalloc(&d_full, sizeof(double)*cap_subs)); CK(cudaMalloc(&d_fullD, sizeof(Dual)*6*cap_subs));
        CK(cudaMalloc(&d_kept, sizeof(int)*(size_t)cap_subs*SRCK_MAXCELL));
        CK(cudaMalloc(&d_xi, sizeof(double)*3*(size_t)cap_subs*SRCK_MAXCELL));
        CK(cudaMalloc(&d_wt, sizeof(double)*(size_t)cap_subs*SRCK_MAXCELL)); CK(cudaMalloc(&d_dt, sizeof(Dual)*6*(size_t)cap_subs*SRCK_MAXCELL));
    }
    CK(cudaMemset(d_f, 0, sizeof(double)*ndof)); CK(cudaMemset(d_nkept, 0, sizeof(unsigned long long)));
    if (d1) CK(cudaMemset(d_d1, 0, sizeof(double)*6*ndof));
    if (fk) CK(cudaMemset(d_fk, 0, sizeof(double)*ndof));
    if (nsub > 0) {
        CK(cudaMemcpy(d_subs, subs, sizeof(GpuSrcSub)*nsub, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_active, elem_active, g_ne, cudaMemcpyHostToDevice));
        { const long nt = 7L * nsub * SRCK_MAXCELL;
          k_full_terms<<<(unsigned)((nt + 127) / 128), 128>>>(d_subs, nsub, d1 != nullptr, d_wt, d_dt);
          k_full_sum<<<nsub, 7>>>(d_subs, nsub, d1 != nullptr, d_wt, d_dt, d_full, d_fullD); }
        const long ncells = (long)nsub*SRCK_MAXCELL; const int B = 128;
        k_deposit<<<(unsigned)((ncells + B - 1)/B), B>>>(d_subs, nsub, d_full, g_grid, d_gstart, d_gelem, d_lo, d_hi,
                                                         d_X, d_dofs, d_active, d_f, d_kept, d_xi, d_nkept);
        if (d1) k_d1<<<(unsigned)((6*ncells + B - 1)/B), B>>>(d_subs, nsub, d_fullD, d_X, d_dofs, d_kept, d_xi, ndof,
                                                              d_d1, fk ? d_fk : nullptr);
#ifdef GPU_SRC_ENZYME
        if (d2) {
            if (nsub > cap2) { if (d_t2) { cudaFree(d_t2); cudaFree(d_full2); }
                CK(cudaMalloc(&d_t2, sizeof(Full2) * 21 * (size_t)cap_subs * SRCK_MAXCELL)); CK(cudaMalloc(&d_full2, sizeof(Full2) * 21 * cap_subs));
                cap2 = cap_subs; }
            if (!d_d2) CK(cudaMalloc(&d_d2, sizeof(double) * 21 * ndof));
            CK(cudaMemset(d_d2, 0, sizeof(double) * 21 * ndof));
            const bool all = (pairs == 2);
            const long n21 = 21L * ncells;
            k_full2_terms<<<(unsigned)((n21 + 127) / 128), 128>>>(d_subs, nsub, all, d_t2);
            k_full2_sum<<<nsub, 32>>>(d_subs, nsub, all, d_t2, d_full2);
            k_d2<<<(unsigned)((n21 + 127) / 128), 128>>>(d_subs, nsub, all, d_full2, d_X, d_dofs, d_kept, d_xi, ndof, d_d2);
        }
#endif
        CK(cudaGetLastError());
    } else if (d2) { std::fill(d2, d2 + 21 * (size_t)ndof, 0.0); }
    CK(cudaMemcpy(f, d_f, sizeof(double)*ndof, cudaMemcpyDeviceToHost));
    if (d1) CK(cudaMemcpy(d1, d_d1, sizeof(double)*6*ndof, cudaMemcpyDeviceToHost));
#ifdef GPU_SRC_ENZYME
    if (d2 && nsub > 0) CK(cudaMemcpy(d2, d_d2, sizeof(double)*21*ndof, cudaMemcpyDeviceToHost));
#endif
    if (fk) CK(cudaMemcpy(fk, d_fk, sizeof(double)*ndof, cudaMemcpyDeviceToHost));
    unsigned long long nk = 0; CK(cudaMemcpy(&nk, d_nkept, sizeof(nk), cudaMemcpyDeviceToHost));
    if (nkept) *nkept = (long)nk;
}
