// GPU Newton assembly of the thermal solver: see gpu_newton.h.
// Compiled with -fmad=false: the per-element / per-point arithmetic is the CPU's, in the CPU's order. The scatters into Ad,
// Rb and Jd use atomics, as the CPU loops do (OpenMP atomics), so the summation order of shared entries is nondeterministic
// on both, at round-off level.
#include "gpu_newton.h"
#ifdef MATERIAL_IN718
#include "material_in718.hpp"   // IN718 (NIST AMB2022-03): same tables as the driver and fb_enzyme_fns.hpp
#endif
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

namespace {
#ifdef MATERIAL_IN718
__constant__ double cpTa_d[IN718_CP_N] = IN718_CP_TA, cpVa_d[IN718_CP_N] = IN718_CP_VA;
__constant__ double kTa_d[IN718_K_N] = IN718_K_TA, kVa_d[IN718_K_N] = IN718_K_VA;
__device__ inline double pwl_d(const double *Ta, const double *Va, int n, double T) {
    if (T <= Ta[0]) return Va[0];
    if (T >= Ta[n-1]) return Va[n-1];
    int i = 0; while (i < n - 2 && Ta[i+1] < T) ++i;
    return Va[i] + (T - Ta[i]) / (Ta[i+1] - Ta[i]) * (Va[i+1] - Va[i]);
}
__device__ inline double slope_d(const double *Ta, const double *Va, int n, double T) {
    if (T <= Ta[0] || T >= Ta[n-1]) return 0.0;
    int i = 0; while (i < n - 2 && Ta[i+1] < T) ++i;
    return (Va[i+1] - Va[i]) / (Ta[i+1] - Ta[i]);
}
#endif
// ---------------------------------------------------------------- IN625 (same expressions as namespace IN625 in the driver)
__device__ inline double lerp_dev(double T, double T0, double V0, double T1, double V1) { return V0 + (T - T0) / (T1 - T0) * (V1 - V0); }
#ifdef MATERIAL_IN718
__device__ double cp_dev(double T, const GpuMat &m) { return pwl_d(cpTa_d, cpVa_d, IN718_CP_N, T) * m.cpscale; }
#else
__device__ double cp_dev(double T, const GpuMat &m) {
    double v;
    if (T <= 294.0)       v = 4.1e8;
    else if (T < 366.0)   v = lerp_dev(T,294.0,4.1e8,366.0,4.27e8);
    else if (T < 477.0)   v = lerp_dev(T,366.0,4.27e8,477.0,4.56e8);
    else if (T < 589.0)   v = lerp_dev(T,477.0,4.56e8,589.0,4.81e8);
    else if (T < 700.0)   v = lerp_dev(T,589.0,4.81e8,700.0,5.11e8);
    else if (T < 811.0)   v = lerp_dev(T,700.0,5.11e8,811.0,5.36e8);
    else if (T < 922.0)   v = lerp_dev(T,811.0,5.36e8,922.0,5.65e8);
    else if (T < 1033.0)  v = lerp_dev(T,922.0,5.65e8,1033.0,5.90e8);
    else if (T < 1144.0)  v = lerp_dev(T,1033.0,5.90e8,1144.0,6.20e8);
    else if (T < 1255.0)  v = lerp_dev(T,1144.0,6.20e8,1255.0,6.45e8);
    else if (T < 1366.0)  v = lerp_dev(T,1255.0,6.45e8,1366.0,6.70e8);
    else                  v = 6.70e8;
    return v * m.cpscale;
}
#endif
#ifdef MATERIAL_IN718
__device__ double cp_prime(double T, const GpuMat &m) { return slope_d(cpTa_d, cpVa_d, IN718_CP_N, T) * m.cpscale; }
#else
__device__ double cp_prime(double T, const GpuMat &m) {
    double v;
    if (T <= 294.0)        v = 0.0;
    else if (T < 366.0)    v = (4.27e8-4.1e8)/(366.0-294.0);
    else if (T < 477.0)    v = (4.56e8-4.27e8)/(477.0-366.0);
    else if (T < 589.0)    v = (4.81e8-4.56e8)/(589.0-477.0);
    else if (T < 700.0)    v = (5.11e8-4.81e8)/(700.0-589.0);
    else if (T < 811.0)    v = (5.36e8-5.11e8)/(811.0-700.0);
    else if (T < 922.0)    v = (5.65e8-5.36e8)/(922.0-811.0);
    else if (T < 1033.0)   v = (5.90e8-5.65e8)/(1033.0-922.0);
    else if (T < 1144.0)   v = (6.20e8-5.90e8)/(1144.0-1033.0);
    else if (T < 1255.0)   v = (6.45e8-6.20e8)/(1255.0-1144.0);
    else if (T < 1366.0)   v = (6.70e8-6.45e8)/(1366.0-1255.0);
    else                   v = 0.0;
    return v * m.cpscale;
}
#endif
#ifdef MATERIAL_IN718
__device__ inline double rho_dev(double, const GpuMat &m) { return IN718_RHO * m.rhoscale; }
#else
__device__ inline double rho_dev(double, const GpuMat &m) { return 8.44e-9 * m.rhoscale; }
#endif
__device__ inline double rhop_dev(double T, const GpuMat &m) { return rho_dev(T, m) * m.rhop_fraction * (1.0 / m.rhoscale); }
__device__ inline double f_L_dev(double T, const GpuMat &m) {   // as the driver's IN625::f_L_dev (two-stage when fL_eut > 0)
    if (T >= m.T_liquidus) return 1.0;
    if (T <= (m.fL_eut > 0.0 ? m.T_eut : m.T_solidus)) return 0.0;
    const bool up = T > m.T_solidus;
    const double lo = up ? m.T_solidus : m.T_eut, hi = up ? m.T_liquidus : m.T_solidus;
    double xi=(T-lo)/(hi-lo);
    const double s = xi*xi*xi*(10.0-15.0*xi+6.0*xi*xi);
    return up ? m.fL_eut + (1.0 - m.fL_eut)*s : m.fL_eut*s;
}
__device__ inline double df_L_dT_dev(double T, const GpuMat &m) {
    if (T >= m.T_liquidus || T <= (m.fL_eut > 0.0 ? m.T_eut : m.T_solidus)) return 0.0;
    const bool up = T > m.T_solidus;
    const double lo = up ? m.T_solidus : m.T_eut, hi = up ? m.T_liquidus : m.T_solidus;
    double DT=hi-lo;
    double xi=(T-lo)/DT;
    double omx=1.0-xi;
    return (up ? 1.0 - m.fL_eut : m.fL_eut)*30.0*xi*xi*omx*omx/DT;
}
__constant__ double kTa[11] = {294.0,311.0,366.0,477.0,589.0,700.0,811.0,922.0,1033.0,1144.0,1255.0};
__constant__ double kVa[11] = {9.8,10.1,10.8,12.5,14.1,15.7,17.5,19.0,20.8,22.8,25.2};
#ifdef MATERIAL_IN718
__device__ double k_dev(double T, const GpuMat &m) { return pwl_d(kTa_d, kVa_d, IN718_K_N, T) * m.kscale; }
#else
__device__ double k_dev(double T, const GpuMat &m) {
    const double *Ta = kTa, *Va = kVa;
    double v;
    if (T <= Ta[0]) v = Va[0];
    else if (T >= Ta[10]) v = Va[10];
    else { int i = 0; while (i < 9 && Ta[i+1] < T) i++; v = Va[i] + (T-Ta[i])/(Ta[i+1]-Ta[i])*(Va[i+1]-Va[i]); }
    return v * m.kscale;
}
#endif
__device__ inline double kp_dev(double T, const GpuMat &m) { return k_dev(T, m) * m.kp_fraction * (1.0 / m.kscale); }

// ---------------------------------------------------------------- device state
int K_ne = 0, K_nq = 0, K_ndof = 0, K_nnz = 0;
int *dK_vd = nullptr, *dK_pos = nullptr, *dK_attr = nullptr;
double *dK_w = nullptr, *dK_g = nullptr, *dK_sh = nullptr, *dK_A = nullptr, *dK_kw = nullptr;
double *d_x = nullptr, *d_xn = nullptr, *d_V = nullptr, *d_melt = nullptr;
char *d_act = nullptr; int act_cap = 0;
double *d_cap = nullptr;                                  // Mc, Ml, Lp, Mo [4n]
int B_nqp = 0, B_ndof = 0, B_jnnz = 0, B_cap_qp = 0, B_cap_v = 0, B_cap_b = 0, B_cap_j = 0;
int *dB_off = nullptr, *dB_dof = nullptr, *dB_kind = nullptr, *dB_off2 = nullptr, *dB_pos = nullptr;
double *dB_shp = nullptr, *dB_w = nullptr, *dB_R = nullptr, *dB_J = nullptr;

template <class T> void grow(T *&p, int &cap, int n) { if (n <= cap) return; if (p) cudaFree(p); NCK(cudaMalloc(&p, sizeof(T) * (n > 0 ? n : 1))); cap = n; }
template <class T> T *dnew(size_t n) { T *p = nullptr; NCK(cudaMalloc(&p, sizeof(T) * (n ? n : 1))); return p; }

// PersistentK::RefillInto in two kernels (same arithmetic and order as the CPU's per-element loop):
//  k_kw:   one thread per (element, point): Tq, melt fraction, k_eff, kw = w * k_eff
//  k_kel:  one thread per (element, upper entry a<=c): elmat[a][c] = sum over the points, in point order, of kw * sd,
//          scattered to (a,c) and its mirror (c,a)
__global__ void k_kw(int ne, int nq, const int *vd, const int *attr, const char *act, int nattr, const double *w,
                     const double *sh, const double *xp, const double *Mv, GpuMat m, double *kw) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)ne * nq) return;
    const int e = (int)(t / nq), q = (int)(t % nq);
    const int at = attr[e];
    if (at < 0 || at >= nattr || !act[at]) return;
    const int *vde = &vd[(size_t)e * 8];
    const double *shq = &sh[(size_t)q * 8];
    double Tq = 0.0;
    for (int a = 0; a < 8; a++) Tq += shq[a] * xp[vde[a]];
    double k_eff;
    if (Mv) {
        double mf = 0.0;
        for (int a = 0; a < 8; a++) mf += shq[a] * Mv[vde[a]];
        k_eff = mf * k_dev(Tq, m) + (1.0 - mf) * kp_dev(Tq, m);
    } else k_eff = k_dev(Tq, m);
    kw[t] = w[t] * k_eff;
}
__constant__ unsigned char kPairA[36], kPairC[36];
__global__ void k_kel(int ne, int nq, const int *pos, const int *attr, const char *act, int nattr, const double *g,
                      const double *kw, double *Ad) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)ne * 36) return;
    const int e = (int)(t / 36), p = (int)(t % 36), a = kPairA[p], c = kPairC[p];
    const int at = attr[e];
    if (at < 0 || at >= nattr || !act[at]) return;
    double v = 0.0;
    for (int q = 0; q < nq; q++) {
        const double *gq = &g[(((size_t)e * nq + q) * 8) * 3];
        double sd = 0.0;
        for (int d = 0; d < 3; d++) sd += gq[a*3+d] * gq[c*3+d];
        v += kw[(size_t)e * nq + q] * sd;
    }
    if (v == 0.0) return;
    const int *pe = &pos[(size_t)e * 64];
    atomicAdd(&Ad[pe[a*8 + c]], v);
    if (c != a) atomicAdd(&Ad[pe[c*8 + a]], v);
}
// (single-kernel version, one thread per element: kept for reference, latency-bound with 2524 threads)
__global__ void k_refill(int ne, int nq, const int *vd, const int *pos, const int *attr, const char *act, int nattr,
                         const double *w, const double *g, const double *sh, const double *xp, const double *Mv,
                         GpuMat m, double *Ad) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    const int at = attr[e];
    if (at < 0 || at >= nattr || !act[at]) return;
    const int *vde = &vd[(size_t)e * 8];
    double Tloc[8], elmat[64], Mloc[8];
    for (int a = 0; a < 8; a++) Tloc[a] = xp[vde[a]];
    for (int i = 0; i < 64; i++) elmat[i] = 0.0;
    if (Mv) for (int a = 0; a < 8; a++) Mloc[a] = Mv[vde[a]];
    for (int q = 0; q < nq; q++) {
        const double *gq = &g[(((size_t)e * nq + q) * 8) * 3];
        const double *shq = &sh[(size_t)q * 8];
        double Tq = 0.0;
        for (int a = 0; a < 8; a++) Tq += shq[a] * Tloc[a];
        double k_eff;
        if (Mv) {
            double mf = 0.0;
            for (int a = 0; a < 8; a++) mf += shq[a] * Mloc[a];
            k_eff = mf * k_dev(Tq, m) + (1.0 - mf) * kp_dev(Tq, m);
        } else k_eff = k_dev(Tq, m);
        const double kw = w[(size_t)e * nq + q] * k_eff;
        for (int a = 0; a < 8; a++)
            for (int c = a; c < 8; c++) {
                double sd = 0.0;
                for (int d = 0; d < 3; d++) sd += gq[a*3+d] * gq[c*3+d];
                elmat[a*8 + c] += kw * sd;
            }
    }
    const int *pe = &pos[(size_t)e * 64];
    for (int a = 0; a < 8; a++)
        for (int c = 0; c < 8; c++) {
            const double v = (c >= a) ? elmat[a*8 + c] : elmat[c*8 + a];
            if (v != 0.0) atomicAdd(&Ad[pe[a*8 + c]], v);
        }
}

// UpdateBoundary: one thread per boundary quadrature point
__global__ void k_bdry(int nqp, const int *off, const int *dof, const double *shp, const double *wq, const int *kind,
                       const int *off2, const int *bpos, const double *xp, const double *Mv, double hconv, double hp,
                       double emis, double sigma, double Ta, double sub_s, double br_s, double if_s, double *Rb, double *Jd) {
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nqp) return;
    const double qscale = (kind[q] == 1) ? sub_s : ((kind[q] == 2) ? br_s : if_s);
    const int o = off[q], nd = off[q + 1] - o;
    const double Ta2 = Ta*Ta;
    double Tq = 0.0;
    for (int a = 0; a < nd; a++) Tq += shp[o + a] * xp[dof[o + a]];
    double mf_b = 1.0;
    if (Mv) { mf_b = 0.0; for (int a = 0; a < nd; a++) mf_b += shp[o + a] * Mv[dof[o + a]]; }
    const double hconv_eff = mf_b * hconv + (1.0 - mf_b) * hp;
    double Tq3 = Tq*Tq*Tq;
    double qval  = hconv_eff*(Tq - Ta) + emis*sigma*(Tq3*Tq - Ta2*Ta2);
    double qprime = hconv_eff + 4.0*emis*sigma*Tq3;
    const int *bp = &bpos[off2[q]];
    const double w = wq[q];
    for (int a = 0; a < nd; a++) {
        double rr = qscale * qval * shp[o + a] * w;
        atomicAdd(&Rb[dof[o + a]], rr);
        for (int b = 0; b < nd; b++) {
            double jj = qscale * qprime * shp[o + a] * shp[o + b] * w;
            atomicAdd(&Jd[bp[a*nd + b]], jj);
        }
    }
}

// UpdateCapacitance: one thread per node
__global__ void k_cap(int n, const double *Tp, const double *Tn, const double *Vp, const double *Mv, GpuMat m, double idt,
                      double *Mc, double *Ml, double *Lp, double *Mo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double Ti = Tp[i];
    double rho_i = Mv ? (Mv[i] * rho_dev(Ti, m) + (1.0 - Mv[i]) * rhop_dev(Ti, m)) : rho_dev(Ti, m);
    Mc[i] = Vp[i] * rho_i * cp_dev(Ti, m);
    Ml[i] = Vp[i] * rho_i * (m.L_fusion * m.Lscale) * df_L_dT_dev(Ti, m);
    Lp[i] = Vp[i] * rho_i * (m.L_fusion * m.Lscale) * f_L_dev(Ti, m);
    double dMc_dT = Vp[i] * rho_i * cp_prime(Ti, m);
    Mo[i] = (Mc[i] + Ml[i]) * idt + dMc_dT * (Ti - Tn[i]) * idt;
}

void ensure_nodal(int n) {
    static int cap = 0;
    if (n <= cap) return;
    if (d_x) { cudaFree(d_x); cudaFree(d_xn); cudaFree(d_V); cudaFree(d_melt); cudaFree(d_cap); }
    d_x = dnew<double>(n); d_xn = dnew<double>(n); d_V = dnew<double>(n); d_melt = dnew<double>(n); d_cap = dnew<double>(4 * (size_t)n);
    cap = n;
}
void upload_active(const char *a, int n) { grow(d_act, act_cap, n); NCK(cudaMemcpy(d_act, a, n, cudaMemcpyHostToDevice)); }
} // namespace

bool gpu_newton_k_init(int ne, int nq, int ndof, int nnz, const int *vd, const int *pos, const int *attr,
                       const double *w, const double *g, const double *sh) {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return false;
    {   // MATH_DT (2026-10-25): stay on the CPU loops when the arrays below do not fit (1.2M-element mesh on the 8 GB card)
        const size_t need = (size_t)ne * (8 + 64 + 1) * sizeof(int) + (size_t)ne * nq * (1 + 24 + 1) * sizeof(double)
                          + (size_t)nnz * sizeof(double) + (size_t)ndof * 16 * sizeof(double) + ((size_t)512 << 20);
        size_t avail = 0, total = 0;
        if (cudaMemGetInfo(&avail, &total) != cudaSuccess || need > avail) {
            printf("  [GPU_NEWTON] OFF: needs %.2f GB on the GPU, %.2f GB free -> Newton assembly on the CPU\n",
                   need / 1073741824.0, avail / 1073741824.0);
            fflush(stdout);
            return false;
        }
    }
    K_ne = ne; K_nq = nq; K_ndof = ndof; K_nnz = nnz;
    dK_vd = dnew<int>((size_t)ne * 8); NCK(cudaMemcpy(dK_vd, vd, sizeof(int) * ne * 8, cudaMemcpyHostToDevice));
    dK_pos = dnew<int>((size_t)ne * 64); NCK(cudaMemcpy(dK_pos, pos, sizeof(int) * (size_t)ne * 64, cudaMemcpyHostToDevice));
    dK_attr = dnew<int>(ne); NCK(cudaMemcpy(dK_attr, attr, sizeof(int) * ne, cudaMemcpyHostToDevice));
    dK_w = dnew<double>((size_t)ne * nq); NCK(cudaMemcpy(dK_w, w, sizeof(double) * ne * nq, cudaMemcpyHostToDevice));
    dK_g = dnew<double>((size_t)ne * nq * 24); NCK(cudaMemcpy(dK_g, g, sizeof(double) * (size_t)ne * nq * 24, cudaMemcpyHostToDevice));
    dK_sh = dnew<double>((size_t)nq * 8); NCK(cudaMemcpy(dK_sh, sh, sizeof(double) * nq * 8, cudaMemcpyHostToDevice));
    dK_A = dnew<double>(nnz); dK_kw = dnew<double>((size_t)ne * nq);
    { unsigned char pa[36], pc[36]; int k = 0; for (int a = 0; a < 8; a++) for (int c = a; c < 8; c++, k++) { pa[k] = a; pc[k] = c; }
      NCK(cudaMemcpyToSymbol(kPairA, pa, 36)); NCK(cudaMemcpyToSymbol(kPairC, pc, 36)); }
    ensure_nodal(ndof);
    printf("  [GPU_NEWTON] conductivity refill, boundary terms and capacitance on the GPU (%d elements, %d nonzeros)\n", ne, nnz);
    fflush(stdout);
    return true;
}

void gpu_newton_k_refill(const double *x, const char *active_attr, int nattr, const double *ever_melted, const GpuMat &m,
                         double *Ad) {
    NCK(cudaMemcpy(d_x, x, sizeof(double) * K_ndof, cudaMemcpyHostToDevice));
    if (ever_melted) NCK(cudaMemcpy(d_melt, ever_melted, sizeof(double) * K_ndof, cudaMemcpyHostToDevice));
    upload_active(active_attr, nattr);
    NCK(cudaMemset(dK_A, 0, sizeof(double) * K_nnz));
    k_kw<<<(unsigned)(((long)K_ne * K_nq + 127) / 128), 128>>>(K_ne, K_nq, dK_vd, dK_attr, d_act, nattr, dK_w, dK_sh, d_x,
                                                             ever_melted ? d_melt : nullptr, m, dK_kw);
    k_kel<<<(unsigned)(((long)K_ne * 36 + 127) / 128), 128>>>(K_ne, K_nq, dK_pos, dK_attr, d_act, nattr, dK_g, dK_kw, dK_A);
    NCK(cudaGetLastError());
    NCK(cudaMemcpy(Ad, dK_A, sizeof(double) * K_nnz, cudaMemcpyDeviceToHost));
}

void gpu_newton_b_setup(int nqp, int ndof, int jnnz, const int *qoff, const int *dof, const double *shp,
                        const double *w, const int *kind, const int *qoff2, const int *bpos) {
    B_nqp = nqp; B_ndof = ndof; B_jnnz = jnnz;
    const int nv = qoff[nqp], nb = qoff2[nqp];
    if (nqp + 1 > B_cap_qp) { if (dB_off) { cudaFree(dB_off); cudaFree(dB_kind); cudaFree(dB_off2); cudaFree(dB_w); }
        dB_off = dnew<int>(nqp + 1); dB_kind = dnew<int>(nqp); dB_off2 = dnew<int>(nqp + 1); dB_w = dnew<double>(nqp); B_cap_qp = nqp + 1; }
    if (nv > B_cap_v) { if (dB_dof) { cudaFree(dB_dof); cudaFree(dB_shp); } dB_dof = dnew<int>(nv); dB_shp = dnew<double>(nv); B_cap_v = nv; }
    if (nb > B_cap_b) { if (dB_pos) cudaFree(dB_pos); dB_pos = dnew<int>(nb); B_cap_b = nb; }
    if (jnnz > B_cap_j) { if (dB_J) cudaFree(dB_J); dB_J = dnew<double>(jnnz); B_cap_j = jnnz; }
    if (!dB_R) dB_R = dnew<double>(ndof);
    ensure_nodal(ndof);
    NCK(cudaMemcpy(dB_off, qoff, sizeof(int) * (nqp + 1), cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(dB_kind, kind, sizeof(int) * nqp, cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(dB_off2, qoff2, sizeof(int) * (nqp + 1), cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(dB_w, w, sizeof(double) * nqp, cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(dB_dof, dof, sizeof(int) * nv, cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(dB_shp, shp, sizeof(double) * nv, cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(dB_pos, bpos, sizeof(int) * nb, cudaMemcpyHostToDevice));
}

void gpu_newton_b_update(const double *x, const double *ever_melted, double hconv, double hp, double emis, double sigma,
                         double Ta, double sub_scale, double bridge_scale, double if_scale, double *Rb, double *Jd) {
    NCK(cudaMemcpy(d_x, x, sizeof(double) * B_ndof, cudaMemcpyHostToDevice));
    if (ever_melted) NCK(cudaMemcpy(d_melt, ever_melted, sizeof(double) * B_ndof, cudaMemcpyHostToDevice));
    NCK(cudaMemset(dB_R, 0, sizeof(double) * B_ndof)); NCK(cudaMemset(dB_J, 0, sizeof(double) * B_jnnz));
    if (B_nqp > 0)
        k_bdry<<<(B_nqp + 127) / 128, 128>>>(B_nqp, dB_off, dB_dof, dB_shp, dB_w, dB_kind, dB_off2, dB_pos, d_x,
                                             ever_melted ? d_melt : nullptr, hconv, hp, emis, sigma, Ta,
                                             sub_scale, bridge_scale, if_scale, dB_R, dB_J);
    NCK(cudaGetLastError());
    NCK(cudaMemcpy(Rb, dB_R, sizeof(double) * B_ndof, cudaMemcpyDeviceToHost));
    NCK(cudaMemcpy(Jd, dB_J, sizeof(double) * B_jnnz, cudaMemcpyDeviceToHost));
}

void gpu_newton_cap(int n, const double *T, const double *Tn, const double *V, const double *ever_melted, const GpuMat &m,
                    double idt, double *Mc, double *Ml, double *Lp, double *Mo) {
    ensure_nodal(n);
    NCK(cudaMemcpy(d_x, T, sizeof(double) * n, cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(d_xn, Tn, sizeof(double) * n, cudaMemcpyHostToDevice));
    NCK(cudaMemcpy(d_V, V, sizeof(double) * n, cudaMemcpyHostToDevice));
    if (ever_melted) NCK(cudaMemcpy(d_melt, ever_melted, sizeof(double) * n, cudaMemcpyHostToDevice));
    double *c = d_cap;
    k_cap<<<(n + 127) / 128, 128>>>(n, d_x, d_xn, d_V, ever_melted ? d_melt : nullptr, m, idt, c, c + n, c + 2 * (size_t)n, c + 3 * (size_t)n);
    NCK(cudaGetLastError());
    NCK(cudaMemcpy(Mc, c, sizeof(double) * n, cudaMemcpyDeviceToHost));
    NCK(cudaMemcpy(Ml, c + n, sizeof(double) * n, cudaMemcpyDeviceToHost));
    NCK(cudaMemcpy(Lp, c + 2 * (size_t)n, sizeof(double) * n, cudaMemcpyDeviceToHost));
    NCK(cudaMemcpy(Mo, c + 3 * (size_t)n, sizeof(double) * n, cudaMemcpyDeviceToHost));
}
