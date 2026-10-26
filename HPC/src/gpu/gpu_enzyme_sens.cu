// First-order sensitivity blocks (a), (b), (c) on the GPU with Enzyme: see gpu_enzyme_sens.h.
// Each kernel mirrors the driver's CPU loop of the same block; the device functions d_fb_dk1 / d_fb_dq1 / d_fb_nodal1 are
// the bodies of fb_dk1 / fb_dq1 / fb_nodal1 (fb_enzyme_kernels.cpp), differentiating the same fb_enzyme_fns.hpp functions.
// Compiled with -ffp-contract=off (no fused multiply-add, as the CPU object). The scatters use atomics, as the CPU loops
// (OpenMP atomics): the summation order of shared entries is nondeterministic on both, at round-off level.
#include "gpu_enzyme_sens.h"
#include "fb_enzyme_fns.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define SCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::abort(); } } while (0)

extern __device__ int enzyme_dup, enzyme_const;
template <typename RT, typename... Args> __device__ RT __enzyme_fwddiff(void *, Args...);

namespace {
using namespace fbk;
// ---------------------------------------------------------------- Enzyme in device code (bodies of the CPU kernels)
__device__ void d_fb_dk1(double T, double mf, double kpf, double *dk_dK, double *dk_dKP)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1}, dS[FB_NSLOT];
    unit_seed(S_K, dS);
    *dk_dK = __enzyme_fwddiff<double>((void*)kblend_fn, enzyme_const, T,
                 enzyme_dup, (const double*)S, (const double*)dS, enzyme_const, mf, enzyme_const, kpf);
    unit_seed(S_KP, dS);
    *dk_dKP = __enzyme_fwddiff<double>((void*)kblend_fn, enzyme_const, T,
                 enzyme_dup, (const double*)S, (const double*)dS, enzyme_const, mf, enzyme_const, kpf);
}
__device__ void d_fb_dq1(double Tq, double mf, double a, double hpa, double b, double Tamb,
                         double *dq_dhc, double *dq_dT0, double *dq_dhp)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1}, dS[FB_NSLOT];
    double *out[3] = {dq_dhc, dq_dT0, dq_dhp};
    const int sl[3] = {S_HC, S_T0, S_HP};
    for (int j = 0; j < 3; j++) {
        unit_seed(sl[j], dS);
        *out[j] = __enzyme_fwddiff<double>((void*)qblend_fn, enzyme_const, Tq,
                     enzyme_dup, (const double*)S, (const double*)dS,
                     enzyme_const, mf, enzyme_const, a, enzyme_const, hpa, enzyme_const, b, enzyme_const, Tamb);
    }
}
__device__ double d_nodal_dir(double Tnext, double Tn, double pdp, int slot, double Mv, double rpf, double V, double idt,
                              double Ts, double Tl)
{
    double S[FB_NSLOT] = {1,1,1,1,1,1,1,1,1,1,1}, dS[FB_NSLOT];
    unit_seed(slot, dS);   // slot -1 -> all zero
    return -__enzyme_fwddiff<double>((void*)nodal_blend_fn,
                enzyme_const, Tnext, enzyme_dup, Tn, pdp,
                enzyme_dup, (const double*)S, (const double*)dS,
                enzyme_const, Mv, enzyme_const, rpf,
                enzyme_const, V, enzyme_const, idt, enzyme_const, Ts, enzyme_const, Tl);
}

// ---------------------------------------------------------------- kernels
__global__ void k_sens_a(int ne, int nq, const int *vd, const double *shape, const double *gshape, const double *w,
                         const int *attr, const char *act, int nattr, const double *up, const double *Mv, double kpf,
                         bool want_kp, double *rk, double *rkp)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)ne * nq) return;
    const int e = (int)(t / nq);
    const int at = attr[e];
    if (at < 0 || at >= nattr || !act[at]) return;
    const int *vde = vd + (size_t)e * 8;
    const double *sh = shape + (size_t)t * 8, *gs = gshape + (size_t)t * 24;
    double Tloc[8]; for (int a = 0; a < 8; a++) Tloc[a] = up[vde[a]];
    double Tnext_q = 0.0;
    for (int a = 0; a < 8; a++) Tnext_q += sh[a] * Tloc[a];
    double gT[3] = {0, 0, 0};
    for (int a = 0; a < 8; a++) for (int d = 0; d < 3; d++) gT[d] += gs[a*3 + d] * Tloc[a];
    double mf = 1.0;
    if (Mv) { mf = 0.0; for (int a = 0; a < 8; a++) mf += sh[a] * Mv[vde[a]]; }
    double dk_dk, dk_dkp;
    d_fb_dk1(Tnext_q, mf, kpf, &dk_dk, &dk_dkp);
    if (!want_kp) dk_dkp = 0.0;
    if (dk_dk == 0.0 && dk_dkp == 0.0) return;
    const double wq = w[t];
    for (int a = 0; a < 8; a++) {
        double gdot = 0.0;
        for (int d = 0; d < 3; d++) gdot += gs[a*3 + d] * gT[d];
        if (dk_dk != 0.0) atomicAdd(&rk[vde[a]], -(dk_dk * gdot * wq));
        if (dk_dkp != 0.0) atomicAdd(&rkp[vde[a]], -(dk_dkp * gdot * wq));
    }
}

__global__ void k_sens_b(int nqp, const int *off, const int *dof, const double *shp, const double *wq, const int *kind,
                         const double *up, const double *Mv, double h_conv, double hp_over_hc, double emis, double sigma,
                         double Tamb, double sub_s, double br_s, double if_s, bool want_hp, double *rhc, double *rt0, double *rhp)
{
    const int qi = blockIdx.x * blockDim.x + threadIdx.x;
    if (qi >= nqp) return;
    const double qscale = (kind[qi] == 1) ? sub_s : (kind[qi] == 2) ? br_s : if_s;
    const int o = off[qi], nd = off[qi + 1] - o;
    double Tq = 0.0;
    for (int a = 0; a < nd; a++) Tq += shp[o + a] * up[dof[o + a]];
    double mf_c = 1.0;
    if (Mv) { mf_c = 0.0; for (int a = 0; a < nd; a++) mf_c += shp[o + a] * Mv[dof[o + a]]; }
    double dq_dhc, dq_dt0, dq_dhp;
    d_fb_dq1(Tq, mf_c, qscale * h_conv, qscale * h_conv * hp_over_hc, qscale * emis * sigma, Tamb, &dq_dhc, &dq_dt0, &dq_dhp);
    if (!want_hp) dq_dhp = 0.0;
    if (dq_dhc == 0.0 && dq_dt0 == 0.0 && dq_dhp == 0.0) return;
    for (int a = 0; a < nd; a++) {
        double contrib = shp[o + a] * wq[qi];
        if (dq_dhc != 0.0) atomicAdd(&rhc[dof[o + a]], -(dq_dhc * contrib));
        if (dq_dt0 != 0.0) atomicAdd(&rt0[dof[o + a]], -(dq_dt0 * contrib));
        if (dq_dhp != 0.0) atomicAdd(&rhp[dof[o + a]], -(dq_dhp * contrib));
    }
}

// one thread per (node, direction): rhs[p][i] = -d nodal_i / ds_p  (0 when the direction does not reach this node)
__global__ void k_sens_c(int ndof, int nw, const double *Tnext, const double *Tn, const double *pd, const int *slot,
                         const double *Mv, double rpf, const double *V, double idt, double Ts, double Tl, double *rhs)
{
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)ndof * nw) return;
    const int p = (int)(t / ndof), i = (int)(t % ndof);
    const double pdp = pd[t];
    rhs[t] = (pdp == 0.0 && slot[p] < 0) ? 0.0 : d_nodal_dir(Tnext[i], Tn[i], pdp, slot[p], Mv[i], rpf, V[i], idt, Ts, Tl);
}

// ---------------------------------------------------------------- device buffers
int A_ne = 0, A_nq = 0, N = 0;
int *dA_vd = nullptr, *dA_attr = nullptr; double *dA_sh = nullptr, *dA_g = nullptr, *dA_w = nullptr;
int B_nqp = 0; int *dB_off = nullptr, *dB_dof = nullptr, *dB_kind = nullptr; double *dB_shp = nullptr, *dB_w = nullptr;
double *d_u = nullptr, *d_un = nullptr, *d_V = nullptr, *d_Mv = nullptr, *d_r = nullptr;   // d_r: 5 RHS arrays (a, b)
double *d_pd = nullptr, *d_rhs = nullptr; int *d_slot = nullptr; int cap_pd = 0;
char *d_act = nullptr; int cap_act = 0;
template <class T> T *dnew(size_t n) { T *p = nullptr; SCK(cudaMalloc(&p, sizeof(T) * (n ? n : 1))); return p; }
template <class T> void up(T *d, const T *h, size_t n) { if (n) SCK(cudaMemcpy(d, h, sizeof(T) * n, cudaMemcpyHostToDevice)); }
template <class T> void down(T *h, const T *d, size_t n) { if (n) SCK(cudaMemcpy(h, d, sizeof(T) * n, cudaMemcpyDeviceToHost)); }
void ensure_nodal(int n) {
    if (n <= N) return;
    if (d_u) { cudaFree(d_u); cudaFree(d_un); cudaFree(d_V); cudaFree(d_Mv); cudaFree(d_r); }
    d_u = dnew<double>(n); d_un = dnew<double>(n); d_V = dnew<double>(n); d_Mv = dnew<double>(n); d_r = dnew<double>(5 * (size_t)n);
    N = n;
}
// adds the device array to the host array (the CPU loops accumulate into RHS vectors that may already hold values)
void add_down(double *h, const double *d, int n) {
    static std::vector<double> tmp; tmp.resize(n); down(tmp.data(), d, n);
    for (int i = 0; i < n; i++) h[i] += tmp[i];
}
} // namespace

bool gpu_sens_init() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return false;
    printf("  [GPU_SENS] sensitivity blocks (a) k, (b) boundary, (c) nodal on the GPU, Enzyme forward mode in device code\n");
    fflush(stdout);
    return true;
}

void gpu_sens_set_fl2(double fl_eut, double t_eut) {
    SCK(cudaMemcpyToSymbol(fbk::fl_eut_d, &fl_eut, sizeof(double)));
    SCK(cudaMemcpyToSymbol(fbk::t_eut_d, &t_eut, sizeof(double)));
}

void gpu_sens_a_setup(int ne, int nq, int ndof, const int *vd, const double *shape, const double *gshape, const double *w,
                      const int *attr) {
    A_ne = ne; A_nq = nq; ensure_nodal(ndof);
    dA_vd = dnew<int>((size_t)ne * 8); up(dA_vd, vd, (size_t)ne * 8);
    dA_sh = dnew<double>((size_t)ne * nq * 8); up(dA_sh, shape, (size_t)ne * nq * 8);
    dA_g = dnew<double>((size_t)ne * nq * 24); up(dA_g, gshape, (size_t)ne * nq * 24);
    dA_w = dnew<double>((size_t)ne * nq); up(dA_w, w, (size_t)ne * nq);
    dA_attr = dnew<int>(ne); up(dA_attr, attr, ne);
}

void gpu_sens_a(const double *u, const char *active_attr, int nattr, const double *Mv, double kp_fraction,
                double *rk, double *rkp) {
    const int n = N;
    up(d_u, u, n); if (Mv) up(d_Mv, Mv, n);
    if (nattr > cap_act) { if (d_act) cudaFree(d_act); d_act = dnew<char>(nattr); cap_act = nattr; }
    up(d_act, active_attr, nattr);
    SCK(cudaMemset(d_r, 0, sizeof(double) * 2 * n));
    k_sens_a<<<(unsigned)(((long)A_ne * A_nq + 127) / 128), 128>>>(A_ne, A_nq, dA_vd, dA_sh, dA_g, dA_w, dA_attr, d_act, nattr,
                                                                  d_u, Mv ? d_Mv : nullptr, kp_fraction, rkp != nullptr,
                                                                  d_r, d_r + n);
    SCK(cudaGetLastError());
    add_down(rk, d_r, n); if (rkp) add_down(rkp, d_r + n, n);
}

void gpu_sens_b_setup(int nqp, int ndof, const int *qoff, const int *dof, const double *shp, const double *w, const int *kind) {
    ensure_nodal(ndof);
    if (dB_off) { cudaFree(dB_off); cudaFree(dB_dof); cudaFree(dB_kind); cudaFree(dB_shp); cudaFree(dB_w); }
    B_nqp = nqp; const int nv = qoff[nqp];
    dB_off = dnew<int>(nqp + 1); up(dB_off, qoff, nqp + 1);
    dB_dof = dnew<int>(nv); up(dB_dof, dof, nv); dB_shp = dnew<double>(nv); up(dB_shp, shp, nv);
    dB_w = dnew<double>(nqp); up(dB_w, w, nqp); dB_kind = dnew<int>(nqp); up(dB_kind, kind, nqp);
}

void gpu_sens_b(const double *u, const double *Mv, double h_conv, double hp_over_hc, double emis, double sigma, double Tamb,
                double sub_scale, double bridge_scale, double if_scale, double *rhc, double *rt0, double *rhp) {
    const int n = N;
    up(d_u, u, n); if (Mv) up(d_Mv, Mv, n);
    double *r = d_r + 2 * (size_t)n;
    SCK(cudaMemset(r, 0, sizeof(double) * 3 * n));
    if (B_nqp > 0)
        k_sens_b<<<(B_nqp + 127) / 128, 128>>>(B_nqp, dB_off, dB_dof, dB_shp, dB_w, dB_kind, d_u, Mv ? d_Mv : nullptr, h_conv,
                                               hp_over_hc, emis, sigma, Tamb, sub_scale, bridge_scale, if_scale, rhp != nullptr,
                                               r, r + n, r + 2 * (size_t)n);
    SCK(cudaGetLastError());
    add_down(rhc, r, n); add_down(rt0, r + n, n); if (rhp) add_down(rhp, r + 2 * (size_t)n, n);
}

void gpu_sens_c(int ndof, int nw, const double *Tnext, const double *Tn, const double *const *pd, const int *slot,
                const double *Mv, double rhop_fraction, const double *V, double idt, double Ts, double Tl, double *const *rp) {
    ensure_nodal(ndof);
    const size_t m = (size_t)nw * ndof;
    if ((int)m > cap_pd) { if (d_pd) { cudaFree(d_pd); cudaFree(d_rhs); cudaFree(d_slot); }
        d_pd = dnew<double>(m); d_rhs = dnew<double>(m); d_slot = dnew<int>(nw); cap_pd = (int)m; }
    static std::vector<double> buf; buf.resize(m);
    for (int p = 0; p < nw; p++) std::copy(pd[p], pd[p] + ndof, buf.begin() + (size_t)p * ndof);
    up(d_pd, buf.data(), m); up(d_slot, slot, nw);
    up(d_u, Tnext, ndof); up(d_un, Tn, ndof); up(d_V, V, ndof); up(d_Mv, Mv, ndof);
    k_sens_c<<<(unsigned)((m + 127) / 128), 128>>>(ndof, nw, d_u, d_un, d_pd, d_slot, d_Mv, rhop_fraction, d_V, idt, Ts, Tl, d_rhs);
    SCK(cudaGetLastError());
    down(buf.data(), d_rhs, m);
    for (int p = 0; p < nw; p++) { double *r = rp[p]; const double *b = buf.data() + (size_t)p * ndof; for (int i = 0; i < ndof; i++) r[i] += b[i]; }
}
