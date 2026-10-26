// ============================================================================
//  Differentiable material / boundary / nodal functions of the Enzyme kernels (bare_plate_single_layer, 2026-10-23).
//  Moved out of fb_enzyme_kernels.cpp unchanged, so that ONE definition is differentiated by Enzyme on the CPU
//  (fb_enzyme_kernels.cpp) and on the GPU (src/gpu/gpu_enzyme_sens.cu, clang -x cuda + ClangEnzyme-21.so).
//  FB_HD = __host__ __device__ under CUDA compilation, empty otherwise; the tables have a __constant__ device copy
//  (FB_TBL picks the copy of the compilation pass), exactly as in src/mechanical/hill_phi.hpp.
//  Scale slots: 0 k, 1 cp, 2 rho, 3 Lfus, 4 Ts, 5 Tl, 6 hc, 7 T0, 8 kp, 9 rho_p, 10 hp (unit multiplicative scales).
// ============================================================================
#pragma once

#if defined(__CUDACC__)
#define FB_HD __host__ __device__
#else
#define FB_HD
#endif
#define FB_NSLOT 11

#ifdef MATERIAL_IN718
#include "material_in718.hpp"     // IN718 (NIST AMB2022-03), same data as the driver's IN625 namespace under MATERIAL_IN718
#endif
namespace fbk {
enum { S_K = 0, S_CP, S_RHO, S_L, S_TS, S_TL, S_HC, S_T0, S_KP, S_RHOP, S_HP };
#ifdef MATERIAL_IN718
constexpr double RHO = IN718_RHO, L_FUSION = IN718_LFUS;
#define FB_K_N IN718_K_N
#define FB_CP_N IN718_CP_N
#define FB_VALS_k_Ta  IN718_K_TA
#define FB_VALS_k_Va  IN718_K_VA
#define FB_VALS_cp_Ta IN718_CP_TA
#define FB_VALS_cp_Va IN718_CP_VA
#else
constexpr double RHO = 8.44e-9, L_FUSION = 2.72e11;
// tables identical to the driver's IN625::k / cp
#define FB_K_N 11
#define FB_CP_N 11
#define FB_VALS_k_Ta  {294.,311.,366.,477.,589.,700.,811.,922.,1033.,1144.,1255.}
#define FB_VALS_k_Va  {9.8,10.1,10.8,12.5,14.1,15.7,17.5,19.0,20.8,22.8,25.2}
#define FB_VALS_cp_Ta {294.,366.,477.,589.,700.,811.,922.,1033.,1144.,1255.,1366.}
#define FB_VALS_cp_Va {4.1e8,4.27e8,4.56e8,4.81e8,5.11e8,5.36e8,5.65e8,5.90e8,6.20e8,6.45e8,6.70e8}
#endif
static const double k_Ta[FB_K_N] = FB_VALS_k_Ta, k_Va[FB_K_N] = FB_VALS_k_Va, cp_Ta[FB_CP_N] = FB_VALS_cp_Ta, cp_Va[FB_CP_N] = FB_VALS_cp_Va;
#if defined(__CUDACC__)
__constant__ double k_Ta_d[FB_K_N] = FB_VALS_k_Ta, k_Va_d[FB_K_N] = FB_VALS_k_Va, cp_Ta_d[FB_CP_N] = FB_VALS_cp_Ta, cp_Va_d[FB_CP_N] = FB_VALS_cp_Va;
#endif
// two-stage liquid fraction (MATH_DT, 2026-10-25; driver IN625::fL_eut/T_eut): set once at start-up (fb_set_fl2 on the host,
// gpu_sens_set_fl2 for the __constant__ copy); plain run-time constants, inactive for Enzyme
inline double fl_eut = 0.0, t_eut = 0.0;
#if defined(__CUDACC__)
__constant__ double fl_eut_d = 0.0, t_eut_d = 0.0;
#endif
#if defined(__CUDA_ARCH__)
#define FB_TBL(x) x##_d
#else
#define FB_TBL(x) x
#endif

FB_HD inline double pwlin(const double *Ta, const double *Va, int n, double T)
{
    if (T <= Ta[0])   return Va[0];
    if (T >= Ta[n-1]) return Va[n-1];
    int i = 0; while (i < n - 2 && Ta[i+1] < T) ++i;
    return Va[i] + (T - Ta[i]) / (Ta[i+1] - Ta[i]) * (Va[i+1] - Va[i]);
}
FB_HD inline double k_fn(double T, const double *S)  { return pwlin(FB_TBL(k_Ta), FB_TBL(k_Va), FB_K_N, T) * S[S_K]; }
FB_HD inline double cp_fn(double T, const double *S) { return pwlin(FB_TBL(cp_Ta), FB_TBL(cp_Va), FB_CP_N, T) * S[S_CP]; }
// clamp on the value with nominal Ts/Tl (as f_L_oti); d/d(Ts,Tl) via scaled bounds.
// Two-stage (fl_eut > 0): fl_eut of the latent heat between t_eut and Ts, 1 - fl_eut between Ts and Tl.
FB_HD inline double fL_fn(double T, const double *S, double Ts, double Tl)
{
    const double fe = FB_TBL(fl_eut), te = FB_TBL(t_eut);
    if (T >= Tl) return 1.0;
    if (T <= (fe > 0.0 ? te : Ts)) return 0.0;
    const double ts = Ts * S[S_TS];
    if (T <= Ts) {
        const double xe = (T - te) / (ts - te);
        return fe * xe*xe*xe*(10.0 - 15.0*xe + 6.0*xe*xe);
    }
    const double tl = Tl * S[S_TL];
    const double xi = (T - ts) / (tl - ts);
    return fe + (1.0 - fe) * xi*xi*xi*(10.0 - 15.0*xi + 6.0*xi*xi);
}
// ---- first-order functions with the powder blends (v8 driver) ----
FB_HD inline double kblend_fn(double T, const double *S, double mf, double kpf)
{
    return mf * (pwlin(FB_TBL(k_Ta), FB_TBL(k_Va), FB_K_N, T) * S[S_K]) + (1.0 - mf) * (pwlin(FB_TBL(k_Ta), FB_TBL(k_Va), FB_K_N, T) * kpf * S[S_KP]);
}
FB_HD inline double qblend_fn(double T, const double *S, double mf, double a, double hpa, double b, double Tamb)
{
    const double Tinf = Tamb * S[S_T0];
    const double hconv = mf * (a * S[S_HC]) + (1.0 - mf) * (hpa * S[S_HP]);
    return hconv * (T - Tinf) + b * (T*T*T*T - Tinf*Tinf*Tinf*Tinf);
}
FB_HD inline double nodal_blend_fn(double Tnext, double Tn, const double *S, double Mv, double rpf,
                                   double V, double idt, double Ts, double Tl)
{
    const double rho = Mv * (RHO * S[S_RHO]) + (1.0 - Mv) * (RHO * rpf * S[S_RHOP]);
    const double Lf = L_FUSION * S[S_L];
    const double Mc = V * rho * cp_fn(Tnext, S);
    const double L  = V * rho * Lf * fL_fn(Tnext, S, Ts, Tl);
    const double Ln = V * rho * Lf * fL_fn(Tn, S, Ts, Tl);
    return Mc * idt * (Tnext - Tn) + (L - Ln) * idt;
}
// ---- second-order functions: reference blocks (no powder blend, slots 0..7) ----
FB_HD inline double nodal_fn(double Tnext, double Tn, const double *S, double V, double idt, double Ts, double Tl)
{
    const double rho = RHO * S[S_RHO], Lf = L_FUSION * S[S_L];
    const double Mc = V * rho * cp_fn(Tnext, S);
    const double L  = V * rho * Lf * fL_fn(Tnext, S, Ts, Tl);
    const double Ln = V * rho * Lf * fL_fn(Tn, S, Ts, Tl);
    return Mc * idt * (Tnext - Tn) + (L - Ln) * idt;
}
FB_HD inline double q_fn(double T, const double *S, double a, double b, double Tamb)
{
    const double Tinf = Tamb * S[S_T0];
    return a * S[S_HC] * (T - Tinf) + b * (T*T*T*T - Tinf*Tinf*Tinf*Tinf);
}
FB_HD inline void unit_seed(int p, double *dS)
{ for (int i = 0; i < FB_NSLOT; i++) dS[i] = (i == p) ? 1.0 : 0.0; }
} // namespace fbk
