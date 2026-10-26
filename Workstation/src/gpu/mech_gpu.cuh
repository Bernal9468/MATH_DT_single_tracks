// ============================================================================
//  GPU kernels of the mechanical solver (bare_plate_single_layer, 2026-10-23), included by mech_main_aa.cpp when it is
//  compiled by nvcc with -DUSE_GPU_MECH (order-1 runs). Each kernel mirrors one loop of mech_main_aa.cpp:
//    k_asm   <- eval_residual_and_tangent   (element residual + tangent, scattered into rfull and K.a)
//    k_jets  <- the "nominal pass" after Newton (Hill return map with jets: sigma, new state, tangents G)
//    k_drhs  <- first-order right-hand sides (K du_i = - sum w B^T G_sigma kw_i)
//    k_dupd  <- first-order state update and dsigma_i
//  PARDISO (factorisation and solves) stays on the CPU. The plastic state S/N_ (s, ds1) lives on the GPU between steps.
//  Exactness: compiled with -fmad=false, so the per-point arithmetic is the CPU's (no fused multiply-add; a fused rounding
//  flips the elastic/plastic decision of points exactly on the yield surface). Element sums into rfull, K.a and rhs1 use
//  atomics, as the CPU loops do (OpenMP atomics): both are order-nondeterministic at round-off level.
//  The jet pass uses Jet<8,1>: for order 1 only the value and the gradient of the Jet<8,2> pass are used, and those do not
//  depend on the Hessian part, so they are bitwise the same.
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define MCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(3); } } while (0)

namespace mgpu {
typedef Jet<8, 1> J81g;

struct Dev {
    int ne = 0, nv = 0, nf3 = 0; long nip = 0, nnz = 0;
    int *E = nullptr;                                   // [ne*8]
    double *Bb = nullptr, *wq = nullptr;                // [nip*144], [nip]
    int *mstart = nullptr, *mfid = nullptr; double *mw = nullptr;   // node -> (free node, weight) CSR
    long *Kia = nullptr; long *Kja = nullptr;           // 1-based CSR of the upper triangle (as K.ia/K.ja)
    double *Ka = nullptr, *rfull = nullptr, *ufull = nullptr, *fsum = nullptr;
    double *ipT = nullptr, *ipdT = nullptr; char *powder = nullptr, *hot = nullptr;
    double *Ss = nullptr, *Sds1 = nullptr, *Ns = nullptr, *Nds1 = nullptr;   // state (committed) and new state
    double *F = nullptr, *Gm = nullptr, *sig = nullptr, *dsig1 = nullptr;     // [nip*13], [nip*13*8], [nip*6], [nip*P*6]
    double *rhs1 = nullptr, *du1f = nullptr;            // [nf3*P], [P*3*nv]
    double *sgv = nullptr, *CB = nullptr;               // Newton: per-point stress [nip*6] and C*B [nip*144]
    // order 2: Hessians of the return map, first-order dw, second-order thermal input, state, rhs, update and dsigma
    double *H = nullptr, *dw1 = nullptr, *ipd2T = nullptr, *Sds2 = nullptr, *Nds2 = nullptr, *rhs2 = nullptr, *du2f = nullptr, *dsig2 = nullptr;
    unsigned long long *dmax = nullptr; unsigned long long *dcnt = nullptr;   // max_se, max_peeq bits; nplast
};

__device__ inline void atomicMaxPos(unsigned long long *a, double v) {   // v >= 0: the bit pattern orders like the value
    atomicMax(a, (unsigned long long)__double_as_longlong(v));
}
__device__ inline long kpos_d(const long *ia, const long *ja, long r, long c) {   // 0-based r<=c, as kpos in the CPU code
    long lo = ia[r] - 1, hi = ia[r + 1] - 2;
    while (lo <= hi) { long mid = (lo + hi) / 2; long cj = ja[mid] - 1; if (cj == c) return mid; if (cj < c) lo = mid + 1; else hi = mid - 1; }
    return -1;
}

// ---- Newton: element residual and tangent (eval_residual_and_tangent), in two kernels:
//  k_asm_ip: one thread per integration point -- return map (Jet<8,1>), stress and C*B (C = consistent tangent)
//  k_asm_el: one thread per (element, row d1) -- fe[d1] and the stiffness row Ke[d1][:], summed over the 8 points in the
//            CPU's order (q, then i), then scattered through the tie map into rfull and K.a
__global__ void k_asm_ip(Dev d, bool want_K, double fsoft) {
    const long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= d.nip) return;
    const int e = (int)(ip / 8);
    double ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = d.ufull[3 * (size_t)d.E[e * 8 + a] + c];
    const double gsc = d.powder[e] ? fsoft : 1.0;
    const double *B = &d.Bb[ip * 144];
    J81g wv[8];
    for (int i = 0; i < 6; i++) { double eps = 0; for (int dd = 0; dd < 24; dd++) eps += B[i * 24 + dd] * ue[dd]; wv[i] = J81g::var(eps - d.Ss[ip * 7 + i], i); }
    wv[6] = J81g::var(d.ipT[ip], 6); wv[7] = J81g::var(d.Ss[ip * 7 + 6], 7);
    J81g sg[6], dep[6], da; hill::phi_w<J81g>(wv, gsc, sg, dep, da);
    for (int i = 0; i < 6; i++) d.sgv[ip * 6 + i] = sg[i].v;
    if (want_K) {
        double *CB = &d.CB[ip * 144];
        for (int i = 0; i < 6; i++) for (int dd = 0; dd < 24; dd++) { double s = 0; for (int j = 0; j < 6; j++) s += sg[i].g[j] * B[j * 24 + dd]; CB[i * 24 + dd] = s; }
    }
}
__global__ void k_asm_el(Dev d, bool want_K) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)d.ne * 24) return;
    const int e = (int)(t / 24), d1 = (int)(t % 24);
    double fe = 0, Kr[24]; for (int k = 0; k < 24; k++) Kr[k] = 0.0;
    for (int q = 0; q < 8; q++) {
        const size_t ip = (size_t)e * 8 + q; const double *B = &d.Bb[ip * 144]; const double w = d.wq[ip];
        for (int i = 0; i < 6; i++) fe += w * B[i * 24 + d1] * d.sgv[ip * 6 + i];
        if (want_K) { const double *CB = &d.CB[ip * 144];
            for (int d2 = 0; d2 < 24; d2++) { double s = 0; for (int i = 0; i < 6; i++) s += B[i * 24 + d1] * CB[i * 24 + d2]; Kr[d2] += w * s; } }
    }
    atomicAdd(d.fsum, fe * fe);
    const int a = d1 / 3, ca = d1 % 3, na = d.E[e * 8 + a];
    atomicAdd(&d.rfull[3 * (size_t)na + ca], fe);
    if (want_K) for (int ka = d.mstart[na]; ka < d.mstart[na + 1]; ka++) for (int b = 0; b < 8; b++) { const int nb = d.E[e * 8 + b];
        for (int kb = d.mstart[nb]; kb < d.mstart[nb + 1]; kb++) {
            const double wgt = d.mw[ka] * d.mw[kb];
            for (int cb = 0; cb < 3; cb++) {
                long r = 3L * d.mfid[ka] + ca, c = 3L * d.mfid[kb] + cb; if (r > c) continue;
                atomicAdd(&d.Ka[kpos_d(d.Kia, d.Kja, r, c)], wgt * Kr[b * 3 + cb]);
            } } }
}
// (single-kernel version, one thread per element; kept for reference, slower: 2524 threads with a 24x24 local matrix)
__global__ void k_asm(Dev d, bool want_K, double fsoft) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= d.ne) return;
    double ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = d.ufull[3 * (size_t)d.E[e * 8 + a] + c];
    double fe[24] = {0}, Ke[24][24];
    if (want_K) for (int i = 0; i < 24; i++) for (int j = 0; j < 24; j++) Ke[i][j] = 0.0;
    const double gsc = d.powder[e] ? fsoft : 1.0;
    for (int q = 0; q < 8; q++) {
        const size_t ip = (size_t)e * 8 + q; const double *B = &d.Bb[ip * 144]; const double w = d.wq[ip];
        J81g wv[8];
        for (int i = 0; i < 6; i++) { double eps = 0; for (int dd = 0; dd < 24; dd++) eps += B[i * 24 + dd] * ue[dd]; wv[i] = J81g::var(eps - d.Ss[ip * 7 + i], i); }
        wv[6] = J81g::var(d.ipT[ip], 6); wv[7] = J81g::var(d.Ss[ip * 7 + 6], 7);
        J81g sg[6], dep[6], da; hill::phi_w<J81g>(wv, gsc, sg, dep, da);
        for (int i = 0; i < 6; i++) for (int dd = 0; dd < 24; dd++) fe[dd] += w * B[i * 24 + dd] * sg[i].v;
        if (want_K) {
            double C[6][6]; for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) C[i][j] = sg[i].g[j];
            double CB[6][24]; for (int i = 0; i < 6; i++) for (int dd = 0; dd < 24; dd++) { double s = 0; for (int j = 0; j < 6; j++) s += C[i][j] * B[j * 24 + dd]; CB[i][dd] = s; }
            for (int d1 = 0; d1 < 24; d1++) for (int d2 = 0; d2 < 24; d2++) { double s = 0; for (int i = 0; i < 6; i++) s += B[i * 24 + d1] * CB[i][d2]; Ke[d1][d2] += w * s; }
        }
    }
    double fs = 0; for (int dd = 0; dd < 24; dd++) fs += fe[dd] * fe[dd];
    atomicAdd(d.fsum, fs);
    for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) atomicAdd(&d.rfull[3 * (size_t)d.E[e * 8 + a] + c], fe[a * 3 + c]);
    if (want_K) for (int a = 0; a < 8; a++) { const int na = d.E[e * 8 + a];
        for (int ka = d.mstart[na]; ka < d.mstart[na + 1]; ka++) for (int b = 0; b < 8; b++) { const int nb = d.E[e * 8 + b];
            for (int kb = d.mstart[nb]; kb < d.mstart[nb + 1]; kb++) {
                const double wgt = d.mw[ka] * d.mw[kb];
                for (int ca = 0; ca < 3; ca++) for (int cb = 0; cb < 3; cb++) {
                    long r = 3L * d.mfid[ka] + ca, c = 3L * d.mfid[kb] + cb; if (r > c) continue;
                    long p = kpos_d(d.Kia, d.Kja, r, c);
                    atomicAdd(&d.Ka[p], wgt * Ke[a * 3 + ca][b * 3 + cb]);
                } } } }
}

// ---- after Newton: return map with jets at every integration point (order 1, not the initial step)
__global__ void k_jets(Dev d, double fsoft, bool abqanneal) {
    const long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= d.nip) return;
    const int e = (int)(ip / 8);
    double ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = d.ufull[3 * (size_t)d.E[e * 8 + a] + c];
    const double gsc = d.powder[e] ? fsoft : 1.0;
    const bool reset = (d.hot[e] && !abqanneal) || d.powder[e];
    const bool aa = abqanneal && d.hot[e] && !d.powder[e];
    double e0; { Jet<1, 2> t = Jet<1, 2>::var(d.ipT[ip], 0); auto r = hill::thermal_strain<Jet<1, 2>>(t); e0 = r.v; }
    const double *B = &d.Bb[ip * 144];
    double eps[6]; for (int i = 0; i < 6; i++) { double s = 0; for (int dd = 0; dd < 24; dd++) s += B[i * 24 + dd] * ue[dd]; eps[i] = s; }
    const double *Sp = &d.Ss[ip * 7];
    double wv[8]; for (int i = 0; i < 6; i++) wv[i] = eps[i] - Sp[i]; wv[6] = d.ipT[ip]; wv[7] = Sp[6];
    J81g wj[8]; for (int i = 0; i < 8; i++) wj[i] = J81g::var(wv[i], i);
    J81g sj[6], dj[6], daj; hill::phi_w<J81g>(wj, gsc, sj, dj, daj);
    double *F = &d.F[ip * 13], *G = &d.Gm[ip * 104];
    for (int i = 0; i < 6; i++) { F[i] = sj[i].v; F[6 + i] = dj[i].v; for (int k = 0; k < 8; k++) { G[i * 8 + k] = sj[i].g[k]; G[(6 + i) * 8 + k] = dj[i].g[k]; } }
    F[12] = daj.v; for (int k = 0; k < 8; k++) G[12 * 8 + k] = daj.g[k];
    for (int i = 0; i < 6; i++) d.sig[ip * 6 + i] = F[i];
    double *Np = &d.Ns[ip * 7];
    if (reset) { for (int i = 0; i < 6; i++) Np[i] = eps[i] - (i < 3 ? e0 : 0.0); Np[6] = 0.0; }
    else { for (int i = 0; i < 6; i++) Np[i] = Sp[i] + F[6 + i]; Np[6] = aa ? 0.0 : Sp[6] + F[12]; }
    // diagnostics (max Hill stress, max PEEQ, number of points with growing PEEQ)
    double Pm[6][6]; hill::HillP(Pm);
    double q2 = 0; for (int i = 0; i < 6; i++) { double s = 0; for (int j = 0; j < 6; j++) s += Pm[i][j] * F[j]; q2 += F[i] * s; }
    atomicMaxPos(&d.dmax[0], sqrt(q2 > 0.0 ? q2 : 0.0));
    atomicMaxPos(&d.dmax[1], Np[6] > 0.0 ? Np[6] : 0.0);
    if (Np[6] > Sp[6] + 1e-14) atomicAdd(d.dcnt, 1ull);
}

// ---- first-order right-hand sides, one thread per (element, parameter)
template <int P> __global__ void k_drhs(Dev d) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)d.ne * P) return;
    const int e = (int)(t / P), i = (int)(t % P);
    double fe[24] = {0};
    for (int q = 0; q < 8; q++) {
        const size_t ip = (size_t)e * 8 + q; const double *G = &d.Gm[ip * 104]; const double *B = &d.Bb[ip * 144]; const double w = d.wq[ip];
        double kw[8]; const double *dso = &d.Sds1[(ip * P + i) * 7];
        for (int c = 0; c < 6; c++) kw[c] = -dso[c]; kw[6] = d.ipdT[ip * P + i]; kw[7] = dso[6];
        double fs[6]; for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += G[r * 8 + k] * kw[k]; fs[r] = s; }
        for (int dd = 0; dd < 24; dd++) { double s = 0; for (int r = 0; r < 6; r++) s += B[r * 24 + dd] * fs[r]; fe[dd] += w * s; }
    }
    for (int a = 0; a < 8; a++) { const int na = d.E[e * 8 + a];
        for (int ka = d.mstart[na]; ka < d.mstart[na + 1]; ka++) for (int c = 0; c < 3; c++)
            atomicAdd(&d.rhs1[(size_t)i * d.nf3 + 3 * d.mfid[ka] + c], -d.mw[ka] * fe[a * 3 + c]); }
}

// ---- first-order state update and dsigma, one thread per (integration point, parameter)
template <int P> __global__ void k_dupd(Dev d, bool abqanneal) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= d.nip * P) return;
    const long ip = t / P; const int i = (int)(t % P); const int e = (int)(ip / 8);
    const bool reset = (d.hot[e] && !abqanneal) || d.powder[e];
    const bool aa = abqanneal && d.hot[e] && !d.powder[e];
    double e1; { Jet<1, 2> tt = Jet<1, 2>::var(d.ipT[ip], 0); auto r = hill::thermal_strain<Jet<1, 2>>(tt); e1 = r.g[0]; }
    const double *G = &d.Gm[ip * 104]; const double *B = &d.Bb[ip * 144];
    double deps[6], ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = d.du1f[(size_t)i * 3 * d.nv + 3 * (size_t)d.E[e * 8 + a] + c];
    for (int r = 0; r < 6; r++) { double s = 0; for (int dd = 0; dd < 24; dd++) s += B[r * 24 + dd] * ue[dd]; deps[r] = s; }
    const double *dso = &d.Sds1[(ip * P + i) * 7]; double dw[8];
    for (int c = 0; c < 6; c++) dw[c] = deps[c] - dso[c]; dw[6] = d.ipdT[ip * P + i]; dw[7] = dso[6];
    if (d.dw1) for (int k = 0; k < 8; k++) d.dw1[(ip * P + i) * 8 + k] = dw[k];   // order 2: dw1 of the CPU code
    double *dn = &d.Nds1[(ip * P + i) * 7];
    if (reset) { for (int c = 0; c < 6; c++) dn[c] = deps[c] - (c < 3 ? e1 * dw[6] : 0.0); dn[6] = 0.0; }
    else { for (int c = 0; c < 6; c++) { double s = 0; for (int k = 0; k < 8; k++) s += G[(6 + c) * 8 + k] * dw[k]; dn[c] = dso[c] + s; }
           double s = 0; for (int k = 0; k < 8; k++) s += G[12 * 8 + k] * dw[k]; dn[6] = aa ? 0.0 : dso[6] + s; }
    for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += G[r * 8 + k] * dw[k]; d.dsig1[(ip * P + i) * 6 + r] = s; }
}

// ================= order 2 =================
typedef Jet<8, 2> J82g;
__constant__ int c_pi[210], c_pj[210];   // pair index pr -> (i, j), i <= j (g_pi / g_pj of the CPU code)
__device__ inline double contract_d(const double *Ho, const double *dwb, int i, int j) {   // dw_i^T H_o dw_j (CPU: contract)
    const double *a = dwb + (size_t)i * 8, *b = dwb + (size_t)j * 8; double s = 0; int p = 0;
    for (int k = 0; k < 8; k++) for (int l = k; l < 8; l++, p++) s += Ho[p] * (k == l ? a[k] * b[l] : a[k] * b[l] + a[l] * b[k]);
    return s;
}
// jet pass with second-order jets (order 2): as k_jets, plus the Hessians H[13][36]
__global__ void k_jets2(Dev d, double fsoft, bool abqanneal) {
    const long ip = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (ip >= d.nip) return;
    const int e = (int)(ip / 8);
    double ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = d.ufull[3 * (size_t)d.E[e * 8 + a] + c];
    const double gsc = d.powder[e] ? fsoft : 1.0;
    const bool reset = (d.hot[e] && !abqanneal) || d.powder[e];
    const bool aa = abqanneal && d.hot[e] && !d.powder[e];
    double e0; { Jet<1, 2> t = Jet<1, 2>::var(d.ipT[ip], 0); auto r = hill::thermal_strain<Jet<1, 2>>(t); e0 = r.v; }
    const double *B = &d.Bb[ip * 144];
    double eps[6]; for (int i = 0; i < 6; i++) { double s = 0; for (int dd = 0; dd < 24; dd++) s += B[i * 24 + dd] * ue[dd]; eps[i] = s; }
    const double *Sp = &d.Ss[ip * 7];
    double wv[8]; for (int i = 0; i < 6; i++) wv[i] = eps[i] - Sp[i]; wv[6] = d.ipT[ip]; wv[7] = Sp[6];
    J82g wj[8]; for (int i = 0; i < 8; i++) wj[i] = J82g::var(wv[i], i);
    J82g sj[6], dj[6], daj; hill::phi_w<J82g>(wj, gsc, sj, dj, daj);
    double *F = &d.F[ip * 13], *G = &d.Gm[ip * 104], *H = &d.H[ip * 468];
    for (int i = 0; i < 6; i++) { F[i] = sj[i].v; F[6 + i] = dj[i].v;
        for (int k = 0; k < 8; k++) { G[i * 8 + k] = sj[i].g[k]; G[(6 + i) * 8 + k] = dj[i].g[k]; }
        for (int k = 0; k < 36; k++) { H[i * 36 + k] = sj[i].h[k]; H[(6 + i) * 36 + k] = dj[i].h[k]; } }
    F[12] = daj.v; for (int k = 0; k < 8; k++) G[12 * 8 + k] = daj.g[k]; for (int k = 0; k < 36; k++) H[12 * 36 + k] = daj.h[k];
    for (int i = 0; i < 6; i++) d.sig[ip * 6 + i] = F[i];
    double *Np = &d.Ns[ip * 7];
    if (reset) { for (int i = 0; i < 6; i++) Np[i] = eps[i] - (i < 3 ? e0 : 0.0); Np[6] = 0.0; }
    else { for (int i = 0; i < 6; i++) Np[i] = Sp[i] + F[6 + i]; Np[6] = aa ? 0.0 : Sp[6] + F[12]; }
    double Pm[6][6]; hill::HillP(Pm);
    double q2 = 0; for (int i = 0; i < 6; i++) { double s = 0; for (int j = 0; j < 6; j++) s += Pm[i][j] * F[j]; q2 += F[i] * s; }
    atomicMaxPos(&d.dmax[0], sqrt(q2 > 0.0 ? q2 : 0.0));
    atomicMaxPos(&d.dmax[1], Np[6] > 0.0 ? Np[6] : 0.0);
    if (Np[6] > Sp[6] + 1e-14) atomicAdd(d.dcnt, 1ull);
}
// second-order right-hand sides, one thread per (element, pair): K du_ij = - sum w B^T (G kw_ij + dw_i^T H dw_j)
template <int P, int NP2> __global__ void k_drhs2(Dev d) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long)d.ne * NP2) return;
    const int e = (int)(t / NP2), pr = (int)(t % NP2), i = c_pi[pr], j = c_pj[pr];
    double fe[24] = {0};
    for (int q = 0; q < 8; q++) {
        const size_t ip = (size_t)e * 8 + q; const double *G = &d.Gm[ip * 104], *H = &d.H[ip * 468]; const double *B = &d.Bb[ip * 144];
        const double w = d.wq[ip]; const double *dwb = &d.dw1[ip * P * 8];
        const double *dso = &d.Sds2[(ip * NP2 + pr) * 7];
        double kw[8]; for (int c = 0; c < 6; c++) kw[c] = -dso[c]; kw[6] = d.ipd2T[ip * NP2 + pr]; kw[7] = dso[6];
        double fs[6]; for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += G[r * 8 + k] * kw[k]; fs[r] = s + contract_d(H + r * 36, dwb, i, j); }
        for (int dd = 0; dd < 24; dd++) { double s = 0; for (int r = 0; r < 6; r++) s += B[r * 24 + dd] * fs[r]; fe[dd] += w * s; }
    }
    for (int a = 0; a < 8; a++) { const int na = d.E[e * 8 + a];
        for (int ka = d.mstart[na]; ka < d.mstart[na + 1]; ka++) for (int c = 0; c < 3; c++)
            atomicAdd(&d.rhs2[(size_t)pr * d.nf3 + 3 * d.mfid[ka] + c], -d.mw[ka] * fe[a * 3 + c]); }
}
// second-order state update and dsigma, one thread per (integration point, pair)
template <int P, int NP2> __global__ void k_dupd2(Dev d, bool abqanneal) {
    const long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= d.nip * NP2) return;
    const long ip = t / NP2; const int pr = (int)(t % NP2), i = c_pi[pr], j = c_pj[pr]; const int e = (int)(ip / 8);
    const bool reset = (d.hot[e] && !abqanneal) || d.powder[e];
    const bool aa = abqanneal && d.hot[e] && !d.powder[e];
    double e1, e2; { Jet<1, 2> tt = Jet<1, 2>::var(d.ipT[ip], 0); auto r = hill::thermal_strain<Jet<1, 2>>(tt); e1 = r.g[0]; e2 = r.h[0]; }
    const double *G = &d.Gm[ip * 104], *H = &d.H[ip * 468]; const double *B = &d.Bb[ip * 144]; const double *dwb = &d.dw1[ip * P * 8];
    double ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = d.du2f[(size_t)pr * 3 * d.nv + 3 * (size_t)d.E[e * 8 + a] + c];
    double d2eps[6]; for (int r = 0; r < 6; r++) { double s = 0; for (int dd = 0; dd < 24; dd++) s += B[r * 24 + dd] * ue[dd]; d2eps[r] = s; }
    const double *dso = &d.Sds2[(ip * NP2 + pr) * 7];
    double d2w[8]; for (int c = 0; c < 6; c++) d2w[c] = d2eps[c] - dso[c]; d2w[6] = d.ipd2T[ip * NP2 + pr]; d2w[7] = dso[6];
    double *dn = &d.Nds2[(ip * NP2 + pr) * 7];
    if (reset) { const double dTi = d.ipdT[ip * P + i], dTj = d.ipdT[ip * P + j];
        for (int c = 0; c < 6; c++) dn[c] = d2eps[c] - (c < 3 ? e2 * dTi * dTj + e1 * d2w[6] : 0.0); dn[6] = 0.0; }
    else { for (int c = 0; c < 7; c++) { const int o = (c < 6) ? 6 + c : 12; double s = 0; for (int k = 0; k < 8; k++) s += G[o * 8 + k] * d2w[k];
               dn[c] = dso[c] + s + contract_d(H + o * 36, dwb, i, j); if (aa) dn[6] = 0.0; } }
    if (d.dsig2) for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += G[r * 8 + k] * d2w[k];
                                               d.dsig2[(ip * NP2 + pr) * 6 + r] = s + contract_d(H + r * 36, dwb, i, j); }
}

template <class T> inline T *dalloc(size_t n) { T *p = nullptr; MCK(cudaMalloc(&p, sizeof(T) * (n ? n : 1))); return p; }
template <class T> inline void up(T *d, const T *h, size_t n) { if (n) MCK(cudaMemcpy(d, h, sizeof(T) * n, cudaMemcpyHostToDevice)); }
template <class T> inline void down(T *h, const T *d, size_t n) { if (n) MCK(cudaMemcpy(h, d, sizeof(T) * n, cudaMemcpyDeviceToHost)); }
inline unsigned nb(long n, int b) { return (unsigned)((n + b - 1) / b); }
} // namespace mgpu
