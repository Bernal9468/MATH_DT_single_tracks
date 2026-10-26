// Hill-anisotropic small-strain thermo-plasticity (IN625 deck of the simple_case mechanical module, same tables/ratios as
// bare_plate_single_layer/mechanical/src/hill_thermoplastic_material.hpp) written as ONE differentiable map
//     Phi : z = (eps_bbar[6 Mandel], T, epsp_old[6 Mandel], alpha_old)  ->  (sigma[6], epsp_new[6], alpha_new)
// templated on the scalar type (double or Jet<14,ORD>).  The scalar radial-return unknown beta is found in double and then
// refined with simplified Newton steps carried in the jet arithmetic (error shrinks by one jet order per step, so ORD+1
// steps make the jet coefficients exact -- value, consistent tangent and Hessian of the converged return map).
// reset=true replaces the NEW plastic state by the stress-free one (epsp_new = eps_bbar - eps_th I, alpha_new = 0); the
// stress of the step is still the one computed from the old state (keeps the element stiffness non-singular).
#pragma once
#include "jet.hpp"
#include <cstring>
#include <cstdio>
#include <type_traits>

namespace hill {
// bare_plate_single_layer: device copies of the tables (__constant__) for the GPU mechanical kernels. HILL_TBL(x) is the
// device copy in the device compilation pass and the original host table otherwise; for g++ nothing changes.
#if defined(__CUDA_ARCH__)
#define HILL_TBL(x) x##_d
#else
#define HILL_TBL(x) x
#endif

static constexpr int NZ = 14;       // local inputs
static constexpr int NOUT = 13;     // sigma6 + epsp6 + alpha

static constexpr int NT = 9;
#define HILL_VALS_Tt {294.15, 366.15, 477.15, 589.15, 700.15, 811.15, 922.15, 1033.15, 1144.15}
static const double Tt[NT] = HILL_VALS_Tt;
#if defined(__CUDACC__)
__constant__ double Tt_d[NT] = HILL_VALS_Tt;
#endif
#define HILL_VALS_Et {207500., 204100., 197900., 191700., 185500., 178600., 170300., 160600., 147500.}
static const double Et[NT] = HILL_VALS_Et;
#if defined(__CUDACC__)
__constant__ double Et_d[NT] = HILL_VALS_Et;
#endif
#define HILL_VALS_nut {0.278, 0.28, 0.286, 0.29, 0.295, 0.305, 0.321, 0.34, 0.336}
static const double nut[NT] = HILL_VALS_nut;
#if defined(__CUDACC__)
__constant__ double nut_d[NT] = HILL_VALS_nut;
#endif
static constexpr int NA = 9;
#define HILL_VALS_Ta {366.15, 477.15, 589.15, 700.15, 811.15, 922.15, 1033.15, 1144.15, 1200.15}
static const double Ta[NA] = HILL_VALS_Ta;
#if defined(__CUDACC__)
__constant__ double Ta_d[NA] = HILL_VALS_Ta;
#endif
#define HILL_VALS_alphat {1.28e-05, 1.31e-05, 1.33e-05, 1.37e-05, 1.4e-05, 1.48e-05, 1.53e-05, 1.58e-05, 1.62e-05}
static const double alphat[NA] = HILL_VALS_alphat;
#if defined(__CUDACC__)
__constant__ double alphat_d[NA] = HILL_VALS_alphat;
#endif
static constexpr double T_ref = 303.15, T_zero = 298.15;
static constexpr double sigma_y0 = 725.0, sigma_y1 = 990.0, eps_p1 = 0.35;
static constexpr double R11 = 1.0, R22 = 1.0, R33 = 0.8483, R12 = 1.0, R13 = 1.0, R23 = 1.0;

JET_HD inline void HillP(double P[6][6]) {
    double F = 0.5 * (1.0 / (R22 * R22) + 1.0 / (R33 * R33) - 1.0 / (R11 * R11));
    double G = 0.5 * (1.0 / (R33 * R33) + 1.0 / (R11 * R11) - 1.0 / (R22 * R22));
    double Hh = 0.5 * (1.0 / (R11 * R11) + 1.0 / (R22 * R22) - 1.0 / (R33 * R33));
    double L = 1.5 / (R23 * R23), M = 1.5 / (R13 * R13), N = 1.5 / (R12 * R12);
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) P[i][j] = 0.0;
    P[0][0] = Hh + G; P[1][1] = F + Hh; P[2][2] = F + G;
    P[0][1] = P[1][0] = -Hh; P[0][2] = P[2][0] = -G; P[1][2] = P[2][1] = -F;
    P[3][3] = N; P[4][4] = M; P[5][5] = L;
}

template <class S> JET_HD inline S pw(const double *xt, const double *yt, int n, const S &T) {
    const double t = val(T);
    if (t <= xt[0]) return S(yt[0]);
    if (t >= xt[n - 1]) return S(yt[n - 1]);
    int i = 0;
    while (i < n - 2 && t > xt[i + 1]) i++;
    S f = (T - xt[i]) * (1.0 / (xt[i + 1] - xt[i]));
    return f * (yt[i + 1] - yt[i]) + yt[i];
}
template <class S> JET_HD inline S thermal_strain(const S &T) {
    S a = pw(HILL_TBL(Ta), HILL_TBL(alphat), NA, T);
    // alpha(T_ref) with T_ref below the table start -> first entry
    const double aref = HILL_TBL(alphat)[0];
    return a * (T - T_zero) - aref * (T_ref - T_zero);
}
template <class S> JET_HD inline S yield_of(const S &a) {
    S s = a * ((sigma_y1 - sigma_y0) / eps_p1) + sigma_y0;
    return val(s) > sigma_y1 ? S(sigma_y1) : s;
}

template <class S> JET_HD inline void solve6(S A[6][6], const S b[6], S x[6]) {
    S M[6][7];
    for (int i = 0; i < 6; i++) { for (int j = 0; j < 6; j++) M[i][j] = A[i][j]; M[i][6] = b[i]; }
    for (int col = 0; col < 6; col++) {
        int piv = col; double best = std::fabs(val(M[col][col]));
        for (int r = col + 1; r < 6; r++) { double v = std::fabs(val(M[r][col])); if (v > best) { best = v; piv = r; } }
        if (piv != col) for (int j = 0; j < 7; j++) { S tmp_ = M[col][j]; M[col][j] = M[piv][j]; M[piv][j] = tmp_; }
        S invd = inv(M[col][col]);
        for (int r = 0; r < 6; r++) {
            if (r == col) continue;
            if (val(M[r][col]) == 0.0) continue;
            S f = M[r][col] * invd;
            for (int j = col; j < 7; j++) M[r][j] -= f * M[col][j];
        }
    }
    for (int i = 0; i < 6; i++) x[i] = M[i][6] * inv(M[i][i]);
}

template <class S> JET_HD inline S quad6(const S s[6], const double P[6][6], S Ps[6]) {
    S q(0.0);
    for (int i = 0; i < 6; i++) { S a(0.0); for (int j = 0; j < 6; j++) if (P[i][j] != 0.0) a += s[j] * P[i][j]; Ps[i] = a; q += s[i] * a; }
    return q;
}

// g(b) = se(sigma(b)) - yield(alpha_old + b*se),  sigma(b) = (I + b M)^-1 sigma_trial
template <class S> JET_HD inline S g_of(const S &b, S M[6][6], const S sig_tr[6], const double P[6][6], const S &alpha_old, S sig_out[6], S &se_out) {
    S A[6][6];
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) A[i][j] = M[i][j] * b + (i == j ? 1.0 : 0.0);
    solve6(A, sig_tr, sig_out);
    S Ps[6]; S q = quad6(sig_out, P, Ps);
    se_out = val(q) > 0.0 ? jsqrt(q) : S(0.0);
    return se_out - yield_of(alpha_old + b * se_out);
}

// Core map in the 8 variables w = (e[6] = eps_bbar - epsp_old (Mandel), T, alpha_old):
//   sigma(w), delta_epsp(w) (plastic strain increment, Mandel), delta_alpha(w)
static constexpr int NW8 = 8;
template <class S> JET_HD inline void phi_w(const S w[NW8], double gscale, S sig[6], S dep[6], S &dalpha) {
    double P[6][6]; HillP(P);
    const S &T = w[6];
    S E = pw(HILL_TBL(Tt), HILL_TBL(Et), NT, T), nu = pw(HILL_TBL(Tt), HILL_TBL(nut), NT, T);
    S mu = E * inv(nu + 1.0) * 0.5;
    S lam = E * nu * inv((nu + 1.0) * (1.0 - nu * 2.0));
    S C[6][6];
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) C[i][j] = S(0.0);
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) C[i][j] = lam;
    for (int i = 0; i < 3; i++) C[i][i] = C[i][i] + mu * 2.0;
    for (int i = 3; i < 6; i++) C[i][i] = mu * 2.0;
    S M[6][6];
    for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) { S a(0.0); for (int k = 0; k < 6; k++) if (P[k][j] != 0.0) a += C[i][k] * P[k][j]; M[i][j] = a; }
    S eth = thermal_strain(T);
    S eel[6], sig_tr[6];
    for (int i = 0; i < 6; i++) eel[i] = w[i] - (i < 3 ? eth : S(0.0));
    for (int i = 0; i < 6; i++) { S a(0.0); for (int j = 0; j < 6; j++) a += C[i][j] * eel[j]; sig_tr[i] = a; }
    const S &alpha_old = w[7];
    S sy_old = yield_of(alpha_old);
    S Ps[6]; S qtr = quad6(sig_tr, P, Ps);
    const double se_tr = val(qtr) > 0.0 ? std::sqrt(val(qtr)) : 0.0;

    for (int i = 0; i < 6; i++) { sig[i] = sig_tr[i]; dep[i] = S(0.0); }
    dalpha = S(0.0);
    if (se_tr > val(sy_old)) {
        // real-part solve: safeguarded Newton with a 1-variable dual number (exact dg/db), then G0 = dg/db at the root
        typedef Jet<1, 1> D1;
        D1 Mj[6][6], strj[6]; for (int i = 0; i < 6; i++) { strj[i] = D1(val(sig_tr[i])); for (int j = 0; j < 6; j++) Mj[i][j] = D1(val(M[i][j])); }
        D1 a0j(val(alpha_old)), sej, svj[6];
        double b = 0.0, G0 = 1.0;
        {   // bracket [lo,hi], g(lo)>0>g(hi); g is smooth and decreasing in b
            double lo = 0.0, hi = 1e-6;
            for (int k = 0; k < 200; k++) { D1 gh = g_of<D1>(D1::var(hi, 0), Mj, strj, P, a0j, svj, sej); if (gh.v < 0.0) break; lo = hi; hi *= 2.0; }
            b = 0.5 * (lo + hi);
            for (int it = 0; it < 200; it++) {
                D1 gj = g_of<D1>(D1::var(b, 0), Mj, strj, P, a0j, svj, sej);
                G0 = gj.g[0];
                if (std::fabs(gj.v) < 1e-13 * (val(sy_old) > 1.0 ? val(sy_old) : 1.0)) break;
                if (gj.v > 0.0) lo = b; else hi = b;
                double bn = b - gj.v / G0;
                if (!(bn > lo && bn < hi)) bn = 0.5 * (lo + hi);
                if (hi - lo < 1e-18) { b = bn; break; }
                b = bn;
            }
        }
        { D1 gj = g_of<D1>(D1::var(b, 0), Mj, strj, P, a0j, svj, sej); G0 = gj.g[0]; }
        S bj(b), se_j, sg[6];
        constexpr int NIT = std::is_same<S, double>::value ? 1 : 3;
        for (int it = 0; it < NIT; it++) {
            S gj = g_of<S>(bj, M, sig_tr, P, alpha_old, sg, se_j);
            bj = bj - gj * (1.0 / G0);
        }
        S gj = g_of<S>(bj, M, sig_tr, P, alpha_old, sg, se_j);   // sigma(beta), se(beta) at the refined beta
        S dgam = bj * se_j;
        for (int i = 0; i < 6; i++) sig[i] = sg[i];
        dalpha = dgam;
        if (val(dgam) > 0.0 && val(se_j) > 1e-12) {
            S Pq[6]; quad6(sg, P, Pq);
            S f = dgam * inv(se_j);
            for (int i = 0; i < 6; i++) dep[i] = f * Pq[i];
        }
    }
    if (gscale != 1.0) for (int i = 0; i < 6; i++) sig[i] = sig[i] * gscale;
}

// 14-variable wrapper (the form validated by test_phi against finite differences)
template <class S> JET_HD inline void phi(const S z[NZ], double gscale, bool reset, S sig[6], S epn[6], S &alpha_new) {
    S w[NW8], dep[6], da;
    for (int i = 0; i < 6; i++) w[i] = z[i] - z[7 + i];
    w[6] = z[6]; w[7] = z[13];
    phi_w<S>(w, gscale, sig, dep, da);
    for (int i = 0; i < 6; i++) epn[i] = z[7 + i] + dep[i];
    alpha_new = z[13] + da;
    if (reset) {
        S eth = thermal_strain(z[6]);
        for (int i = 0; i < 6; i++) epn[i] = z[i] - (i < 3 ? eth : S(0.0));
        alpha_new = S(0.0);
    }
}

}  // namespace hill
