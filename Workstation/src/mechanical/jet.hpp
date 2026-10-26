// Truncated second-order jet: value + gradient + (upper-triangular) Hessian w.r.t. N local variables.
// ORD=1 keeps value+gradient only (cheap, used for the Newton tangent); ORD=2 adds the full Hessian.
// h[idx(k,l)], k<=l, holds the FULL second derivative d2f/dz_k dz_l (not half of it).
#pragma once
#include <cmath>
#include <algorithm>
// bare_plate_single_layer: JET_HD = __host__ __device__ under nvcc (GPU mechanical kernels), empty for g++.
#if defined(__CUDACC__)
#define JET_HD __host__ __device__
#else
#define JET_HD
#endif

template <int N, int ORD>
struct Jet {
    static constexpr int NH = (ORD == 2) ? N * (N + 1) / 2 : 1;
    double v;
    double g[N];
    double h[NH];

    static constexpr int idx(int k, int l) { return k * N - k * (k - 1) / 2 + (l - k); }   // k <= l

    JET_HD Jet() : v(0.0) { for (int i = 0; i < N; i++) g[i] = 0.0; for (int i = 0; i < NH; i++) h[i] = 0.0; }
    JET_HD Jet(double c) : v(c) { for (int i = 0; i < N; i++) g[i] = 0.0; for (int i = 0; i < NH; i++) h[i] = 0.0; }
    JET_HD static Jet var(double val, int i) { Jet r(val); r.g[i] = 1.0; return r; }

    JET_HD Jet &operator+=(const Jet &b) { v += b.v; for (int i = 0; i < N; i++) g[i] += b.g[i]; if (ORD == 2) for (int i = 0; i < NH; i++) h[i] += b.h[i]; return *this; }
    JET_HD Jet &operator-=(const Jet &b) { v -= b.v; for (int i = 0; i < N; i++) g[i] -= b.g[i]; if (ORD == 2) for (int i = 0; i < NH; i++) h[i] -= b.h[i]; return *this; }
};

template <int N, int ORD> JET_HD inline Jet<N, ORD> operator+(Jet<N, ORD> a, const Jet<N, ORD> &b) { a += b; return a; }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator-(Jet<N, ORD> a, const Jet<N, ORD> &b) { a -= b; return a; }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator-(const Jet<N, ORD> &a) { Jet<N, ORD> r; r.v = -a.v; for (int i = 0; i < N; i++) r.g[i] = -a.g[i]; if (ORD == 2) for (int i = 0; i < Jet<N, ORD>::NH; i++) r.h[i] = -a.h[i]; return r; }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator+(Jet<N, ORD> a, double c) { a.v += c; return a; }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator+(double c, Jet<N, ORD> a) { a.v += c; return a; }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator-(Jet<N, ORD> a, double c) { a.v -= c; return a; }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator-(double c, const Jet<N, ORD> &a) { Jet<N, ORD> r = -a; r.v += c; return r; }

template <int N, int ORD> JET_HD inline Jet<N, ORD> operator*(const Jet<N, ORD> &a, double c) {
    Jet<N, ORD> r; r.v = a.v * c; for (int i = 0; i < N; i++) r.g[i] = a.g[i] * c;
    if (ORD == 2) for (int i = 0; i < Jet<N, ORD>::NH; i++) r.h[i] = a.h[i] * c;
    return r;
}
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator*(double c, const Jet<N, ORD> &a) { return a * c; }

template <int N, int ORD> JET_HD inline Jet<N, ORD> operator*(const Jet<N, ORD> &a, const Jet<N, ORD> &b) {
    Jet<N, ORD> r;
    r.v = a.v * b.v;
    for (int i = 0; i < N; i++) r.g[i] = a.v * b.g[i] + a.g[i] * b.v;
    if (ORD == 2) {
        int p = 0;
        for (int k = 0; k < N; k++)
            for (int l = k; l < N; l++, p++)
                r.h[p] = a.v * b.h[p] + a.h[p] * b.v + a.g[k] * b.g[l] + a.g[l] * b.g[k];
    }
    return r;
}

// f(a) given f, f', f'' at a.v
template <int N, int ORD> JET_HD inline Jet<N, ORD> jet_apply(const Jet<N, ORD> &a, double f0, double f1, double f2) {
    Jet<N, ORD> r; r.v = f0;
    for (int i = 0; i < N; i++) r.g[i] = f1 * a.g[i];
    if (ORD == 2) {
        int p = 0;
        for (int k = 0; k < N; k++)
            for (int l = k; l < N; l++, p++) r.h[p] = f1 * a.h[p] + f2 * a.g[k] * a.g[l];
    }
    return r;
}
template <int N, int ORD> JET_HD inline Jet<N, ORD> inv(const Jet<N, ORD> &a) { double f = 1.0 / a.v; return jet_apply(a, f, -f * f, 2.0 * f * f * f); }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator/(const Jet<N, ORD> &a, const Jet<N, ORD> &b) { return a * inv(b); }
template <int N, int ORD> JET_HD inline Jet<N, ORD> operator/(const Jet<N, ORD> &a, double c) { return a * (1.0 / c); }
template <int N, int ORD> JET_HD inline Jet<N, ORD> jsqrt(const Jet<N, ORD> &a) {
    double s = std::sqrt(a.v);
    return jet_apply(a, s, 0.5 / s, -0.25 / (s * a.v));
}

// scalar helpers shared by the double and Jet code paths
JET_HD inline double val(double x) { return x; }
template <int N, int ORD> JET_HD inline double val(const Jet<N, ORD> &a) { return a.v; }
JET_HD inline double jsqrt(double x) { return std::sqrt(x); }
JET_HD inline double inv(double x) { return 1.0 / x; }
