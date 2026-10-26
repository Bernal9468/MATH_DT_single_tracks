// ============================================================================
//  Jacobi-preconditioned BiCGSTAB for the exact (unsymmetric) sensitivity system J_r u_p = b_p (MATH_DT, 2026-10-26).
//  At the AM time steps J_r is strongly diagonally dominant (h 20 um, dt 5 us: |a_ii| >= 6.5 sum_j!=i |a_ij|, asymmetry
//  3e-4, measured on the exported matrix): BiCGSTAB with a diagonal preconditioner converges in ~5 iterations, so the
//  per-step PARDISO factorisation can be replaced by a few sparse matrix-vector products (src/gpu/gpu_krylov.cu on the
//  GPU, this OpenMP version on the CPU). Without the preconditioner it does not converge (rows of very different scale).
//  Convergence test on the preconditioned residual ||D^-1 r|| <= tol ||D^-1 b|| (D^-1 r approximates the error).
//  Returns the iteration count, or -1 if not converged within maxit (the caller falls back to PARDISO).
// ============================================================================
#pragma once
#include <cmath>
#include <vector>

namespace ksens {

struct Csr { int n; const int *I, *J; const double *A; };

inline void diag_inverse(const Csr &M, std::vector<double> &dinv)
{
    dinv.assign(M.n, 0.0);
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < M.n; r++) {
        double d = 0.0;
        for (int k = M.I[r]; k < M.I[r + 1]; k++) if (M.J[k] == r) { d = M.A[k]; break; }
        dinv[r] = d != 0.0 ? 1.0 / d : 1.0;
    }
}

inline void spmv(const Csr &M, const double *x, double *y)
{
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < M.n; r++) {
        double s = 0.0;
        for (int k = M.I[r]; k < M.I[r + 1]; k++) s += M.A[k] * x[M.J[k]];
        y[r] = s;
    }
}

inline double dot(int n, const double *a, const double *b)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+ : s) schedule(static)
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

// x: initial guess on entry (0 is fine), solution on exit
inline int bicgstab(const Csr &M, const double *dinv, const double *b, double *x, double tol, int maxit)
{
    const int n = M.n;
    std::vector<double> r(n), rh(n), p(n, 0.0), v(n, 0.0), y(n), s(n), z(n), t(n), w(n);
    spmv(M, x, t.data());
    double nb = 0.0;
    #pragma omp parallel for reduction(+ : nb) schedule(static)
    for (int i = 0; i < n; i++) { r[i] = b[i] - t[i]; rh[i] = r[i]; nb += (dinv[i] * b[i]) * (dinv[i] * b[i]); }
    nb = std::sqrt(nb);
    if (nb == 0.0) { for (int i = 0; i < n; i++) x[i] = 0.0; return 0; }
    auto pres = [&](const std::vector<double> &q) {             // ||D^-1 q||
        double e = 0.0;
        #pragma omp parallel for reduction(+ : e) schedule(static)
        for (int i = 0; i < n; i++) e += (dinv[i] * q[i]) * (dinv[i] * q[i]);
        return std::sqrt(e);
    };
    if (pres(r) <= tol * nb) return 0;
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    for (int it = 1; it <= maxit; it++) {
        const double rho1 = dot(n, rh.data(), r.data());
        if (rho1 == 0.0) return -1;
        const double beta = (rho1 / rho) * (alpha / omega); rho = rho1;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { p[i] = r[i] + beta * (p[i] - omega * v[i]); y[i] = dinv[i] * p[i]; }
        spmv(M, y.data(), v.data());
        const double rv = dot(n, rh.data(), v.data());
        if (rv == 0.0) return -1;
        alpha = rho / rv;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];
        if (pres(s) <= tol * nb) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) x[i] += alpha * y[i];
            return it;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) z[i] = dinv[i] * s[i];
        spmv(M, z.data(), t.data());
        const double tt = dot(n, t.data(), t.data());
        if (tt == 0.0) return -1;
        omega = dot(n, t.data(), s.data()) / tt;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { x[i] += alpha * y[i] + omega * z[i]; r[i] = s[i] - omega * t[i]; }
        if (pres(r) <= tol * nb) return it;
        if (omega == 0.0) return -1;
    }
    return -1;
}

} // namespace ksens
