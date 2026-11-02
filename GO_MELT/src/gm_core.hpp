// gm_core.hpp -- GO_MELT core: one structured-hex thermal "level" (IN718, AMB2022-03 single tracks).
// Clean, non-OTI implementation of the physics of the verified single-level baseline
// (HPC/src/thermal/full_bridge_activation_tier1_oti.cpp, nominal path):
//   r(T) = K(T) T + R_bdry(T) + Mc(T)(T - Tn)/dt + (L(T) - Ln)/dt - f_src,   backward Euler, lumped cp and latent heat,
//   fully implicit k(T) (Jacobian without dk/dT, as the baseline), Goldak double ellipsoid deposited with trilinear
//   shape functions at the cell centroids, two-stage liquid fraction. Units: mm, tonne, s, mJ, K.
// Axis-aligned tensor-product hexes only (mesh read with MFEM; MFEM is used for mesh I/O only).
#pragma once
#include "mfem.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <omp.h>
#include "material_in718.hpp"
#include "src_kernel.hpp"      // SrcSub / SrcCell records of the differentiable source kernel (Enzyme, built in fb_enzyme_kernels.o)

namespace gm {
using std::vector; using std::array;

// ---------------- material (IN718, same tables and smoothing as the baseline) ----------------
struct Material {
    double Ts = IN718_T_SOLIDUS, Tl = IN718_T_LIQUIDUS, fleut = 0.23, Teut = 1423.15, Lf = IN718_LFUS, rho = IN718_RHO;
    double kT[IN718_K_N] = IN718_K_TA, kV[IN718_K_N] = IN718_K_VA;
    double cT[IN718_CP_N] = IN718_CP_TA, cV[IN718_CP_N] = IN718_CP_VA;
    static double pwl(const double *Ta, const double *Va, int n, double T) {
        if (T <= Ta[0]) return Va[0];
        if (T >= Ta[n - 1]) return Va[n - 1];
        int i = 0; while (i < n - 2 && Ta[i + 1] < T) ++i;
        return Va[i] + (T - Ta[i]) / (Ta[i + 1] - Ta[i]) * (Va[i + 1] - Va[i]);
    }
    static double slope(const double *Ta, const double *Va, int n, double T) {
        if (T <= Ta[0] || T >= Ta[n - 1]) return 0.0;
        int i = 0; while (i < n - 2 && Ta[i + 1] < T) ++i;
        return (Va[i + 1] - Va[i]) / (Ta[i + 1] - Ta[i]);
    }
    double k(double T) const { return pwl(kT, kV, IN718_K_N, T); }
    double dk(double T) const { return slope(kT, kV, IN718_K_N, T); }
    double cp(double T) const { return pwl(cT, cV, IN718_CP_N, T); }
    double dcp(double T) const { return slope(cT, cV, IN718_CP_N, T); }
    double fL(double T) const {
        if (T >= Tl) return 1.0;
        if (T <= (fleut > 0.0 ? Teut : Ts)) return 0.0;
        const bool up = T > Ts; const double lo = up ? Ts : Teut, hi = up ? Tl : Ts;
        const double xi = (T - lo) / (hi - lo), s = xi * xi * xi * (10.0 - 15.0 * xi + 6.0 * xi * xi);
        return up ? fleut + (1.0 - fleut) * s : fleut * s;
    }
    double dfL(double T) const {
        if (T >= Tl || T <= (fleut > 0.0 ? Teut : Ts)) return 0.0;
        const bool up = T > Ts; const double lo = up ? Ts : Teut, hi = up ? Tl : Ts;
        const double DT = hi - lo, xi = (T - lo) / DT, omx = 1.0 - xi;
        return (up ? 1.0 - fleut : fleut) * 30.0 * xi * xi * omx * omx / DT;
    }
};

// ---------------- laser path table (time, x, y, z, power[mW]); zero-order-hold power, linear position ----------------
struct Laser {
    vector<double> t, x, y, z, p; double absorp = 0.75;
    void load(const std::string &fn) {
        std::ifstream f(fn); std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            for (char &c : line) if (c == ',') c = ' ';
            std::istringstream ss(line); double a, b, c, d, e;
            if (ss >> a >> b >> c >> d >> e) { t.push_back(a); x.push_back(b); y.push_back(c); z.push_back(d); p.push_back(e); }
        }
    }
    int bracket(double tt) const {
        if (t.empty() || tt <= t.front()) return 0;
        if (tt >= t.back()) return (int)t.size() - 2;
        int lo = 0, hi = (int)t.size() - 1;
        while (hi - lo > 1) { int m = (lo + hi) / 2; if (t[m] <= tt) lo = m; else hi = m; }
        return lo;
    }
    void position(double tt, double &xL, double &yL, double &zL) const {
        int i = bracket(tt);
        double f = (t[i + 1] > t[i]) ? (tt - t[i]) / (t[i + 1] - t[i]) : 0.0; f = std::max(0.0, std::min(1.0, f));
        xL = x[i] + f * (x[i + 1] - x[i]); yL = y[i] + f * (y[i + 1] - y[i]); zL = z[i] + f * (z[i + 1] - z[i]);
    }
};

struct SrcTape { vector<SrcSub> subs; vector<SrcCell> cells; double t0 = 0, t1 = 0; };
struct GoldakParams { double af = 0.04, ar = 0.04, b = 0.04, c = 0.18, ff = 1.0, fr = 1.0, cutoff = 1.5; int nq = 12; int nseg = 20; };

// ---------------- one structured level ----------------
struct Level {
    // geometry
    int ne = 0, nv = 0;
    vector<array<int, 8>> en;                 // element -> vertices
    vector<array<double, 3>> X;               // vertex coordinates
    vector<array<double, 3>> lo, hi;          // element boxes
    vector<array<signed char, 24>> sg;        // local node signs (a*3+d) = +-1
    vector<double> ax[3];                     // sorted unique coordinates per axis
    vector<int> cellmap; int ncx = 0, ncy = 0, ncz = 0;
    // CSR (full) of the Jacobian
    vector<int> I, J; vector<double> A; vector<int> epos;   // epos[e*64 + a*8 + b]
    vector<double> Vn;                        // lumped nodal volume
    // boundary-loss faces
    struct Face { array<int, 4> v; double area; };
    vector<Face> faces; vector<int> fpos;
    // state
    vector<char> ess; vector<double> essval;  // Dirichlet
    vector<double> T, Tn, Ln, Tpeak;
    Material mat; double h_conv = 0.018, emis = 0.45, sigma = 5.670373e-11, Tamb = 296.65;
    bool toploss = true;   // upstream fix 2026-10-27 (thermo repo 233dc94): the top face z = 0 loses heat (baseline before the fix: false)
    int kq = 3;            // Gauss points per direction of the conduction term: 3 = baseline default (MFEM 2*OrderGrad), 2 = Abaqus DC3D8 rule
    double zmax = 0, zmin = 0;

    static void uniq(vector<double> &v) {
        std::sort(v.begin(), v.end()); vector<double> u;
        for (double a : v) if (u.empty() || a - u.back() > 1e-9) u.push_back(a);
        v.swap(u);
    }
    void init(mfem::Mesh &m) {
        ne = m.GetNE(); nv = m.GetNV(); en.resize(ne); X.resize(nv); lo.resize(ne); hi.resize(ne); sg.resize(ne);
        for (int v = 0; v < nv; v++) for (int d = 0; d < 3; d++) X[v][d] = m.GetVertex(v)[d];
        mfem::Array<int> vs;
        for (int e = 0; e < ne; e++) {
            m.GetElementVertices(e, vs);
            double l[3] = {1e30, 1e30, 1e30}, h[3] = {-1e30, -1e30, -1e30};
            for (int a = 0; a < 8; a++) { en[e][a] = vs[a]; for (int d = 0; d < 3; d++) { l[d] = std::min(l[d], X[vs[a]][d]); h[d] = std::max(h[d], X[vs[a]][d]); } }
            for (int d = 0; d < 3; d++) { lo[e][d] = l[d]; hi[e][d] = h[d]; }
            for (int a = 0; a < 8; a++) for (int d = 0; d < 3; d++) sg[e][a * 3 + d] = (X[vs[a]][d] > 0.5 * (l[d] + h[d])) ? 1 : -1;
        }
        for (int d = 0; d < 3; d++) { ax[d].clear(); for (int v = 0; v < nv; v++) ax[d].push_back(X[v][d]); uniq(ax[d]); }
        ncx = (int)ax[0].size() - 1; ncy = (int)ax[1].size() - 1; ncz = (int)ax[2].size() - 1;
        cellmap.assign((size_t)ncx * ncy * ncz, -1);
        for (int e = 0; e < ne; e++) {
            int c[3];
            for (int d = 0; d < 3; d++) c[d] = (int)(std::lower_bound(ax[d].begin(), ax[d].end(), lo[e][d] - 1e-9) - ax[d].begin());
            cellmap[c[0] + (size_t)ncx * (c[1] + (size_t)ncy * c[2])] = e;
        }
        zmin = ax[2].front(); zmax = ax[2].back();
        // CSR pattern
        vector<vector<int>> nb(nv);
        for (int e = 0; e < ne; e++) for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++) nb[en[e][a]].push_back(en[e][b]);
        I.assign(nv + 1, 0);
        for (int v = 0; v < nv; v++) { std::sort(nb[v].begin(), nb[v].end()); nb[v].erase(std::unique(nb[v].begin(), nb[v].end()), nb[v].end()); I[v + 1] = I[v] + (int)nb[v].size(); }
        J.resize(I[nv]); A.assign(I[nv], 0.0);
        for (int v = 0; v < nv; v++) std::copy(nb[v].begin(), nb[v].end(), J.begin() + I[v]);
        epos.resize((size_t)ne * 64);
        for (int e = 0; e < ne; e++) for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++) epos[(size_t)e * 64 + a * 8 + b] = pos(en[e][a], en[e][b]);
        // lumped nodal volumes: integral of N_a = V/8 (axis-aligned hex)
        Vn.assign(nv, 0.0);
        for (int e = 0; e < ne; e++) { double V = (hi[e][0] - lo[e][0]) * (hi[e][1] - lo[e][1]) * (hi[e][2] - lo[e][2]); for (int a = 0; a < 8; a++) Vn[en[e][a]] += V / 8.0; }
        // boundary faces for convection + radiation: sides (and optionally top); the bottom (z = zmin) is Dirichlet
        for (int be = 0; be < m.GetNBE(); be++) {
            m.GetBdrElementVertices(be, vs); if (vs.Size() != 4) continue;
            bool top = true, bot = true;
            for (int k = 0; k < 4; k++) { if (std::fabs(X[vs[k]][2] - zmax) > 1e-6) top = false; if (std::fabs(X[vs[k]][2] - zmin) > 1e-6) bot = false; }
            if (bot || (top && !toploss)) continue;
            Face f; for (int k = 0; k < 4; k++) f.v[k] = vs[k];
            double u[3], w[3]; for (int d = 0; d < 3; d++) { u[d] = X[vs[1]][d] - X[vs[0]][d]; w[d] = X[vs[3]][d] - X[vs[0]][d]; }
            double cx = u[1] * w[2] - u[2] * w[1], cy = u[2] * w[0] - u[0] * w[2], cz = u[0] * w[1] - u[1] * w[0];
            f.area = std::sqrt(cx * cx + cy * cy + cz * cz); faces.push_back(f);
        }
        fpos.resize(faces.size() * 16);
        for (size_t q = 0; q < faces.size(); q++) for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) fpos[q * 16 + a * 4 + b] = pos(faces[q].v[a], faces[q].v[b]);
        ess.assign(nv, 0); essval.assign(nv, 0.0);
        T.assign(nv, Tamb); Tn = T; Ln.assign(nv, 0.0); Tpeak.assign(nv, -1e300);
    }
    int pos(int r, int c) const {
        int lo_ = I[r], hi_ = I[r + 1] - 1;
        while (lo_ < hi_) { int m = (lo_ + hi_) / 2; if (J[m] < c) lo_ = m + 1; else hi_ = m; }
        return lo_;
    }
    int nearest_node(double px, double py, double z) const {
        int best = -1; double bd = 1e30;
        for (int v = 0; v < nv; v++) { if (std::fabs(X[v][2] - z) > 1e-6) continue; double d = (X[v][0] - px) * (X[v][0] - px) + (X[v][1] - py) * (X[v][1] - py); if (d < bd) { bd = d; best = v; } }
        return best;
    }
    void set_bottom_dirichlet(double Tb) { for (int v = 0; v < nv; v++) if (std::fabs(X[v][2] - zmin) < 1e-6) { ess[v] = 1; essval[v] = Tb; T[v] = Tn[v] = Tb; } }


    // ---- moving box (phase 3): translate the whole uniform grid by m cells of size h along +x; state follows the nodes.
    // Node index = i + NX*(j + NY*k) (gen_box.py order). Returns the nodes that are new (leading slab, no state yet).
    int NXn() const { return (int)ax[0].size(); } int NYn() const { return (int)ax[1].size(); }
    vector<int> shift_x(int m, double h, vector<vector<double>> *S = nullptr) {
        const int NX = NXn(), NY = NYn(), NZ = (int)ax[2].size(); const double dx = m * h; vector<int> fresh;
        for (int v = 0; v < nv; v++) X[v][0] += dx;
        for (int e = 0; e < ne; e++) { lo[e][0] += dx; hi[e][0] += dx; }
        for (double &a : ax[0]) a += dx;
        for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
            int dst = i + NX * (j + (size_t)NY * k);
            if (i + m < NX) { int src = dst + m; T[dst] = T[src]; Tn[dst] = Tn[src]; Ln[dst] = Ln[src]; if (S) for (auto &sp : *S) sp[dst] = sp[src]; } else fresh.push_back(dst);
        }
        return fresh;
    }

    // element containing the point (or -1), local trilinear shape values
    int locate(double px, double py, double pz, double *N, int *nodes) const {
        double p[3] = {px, py, pz}; int c[3];
        for (int d = 0; d < 3; d++) {
            if (p[d] < ax[d].front() - 1e-12 || p[d] > ax[d].back() + 1e-12) return -1;
            int i = (int)(std::upper_bound(ax[d].begin(), ax[d].end(), p[d]) - ax[d].begin()) - 1;
            c[d] = std::max(0, std::min((int)ax[d].size() - 2, i));
        }
        int e = cellmap[c[0] + (size_t)ncx * (c[1] + (size_t)ncy * c[2])]; if (e < 0) return -1;
        double xi[3]; for (int d = 0; d < 3; d++) xi[d] = 2.0 * (p[d] - lo[e][d]) / (hi[e][d] - lo[e][d]) - 1.0;
        for (int a = 0; a < 8; a++) {
            N[a] = 0.125 * (1 + sg[e][a * 3] * xi[0]) * (1 + sg[e][a * 3 + 1] * xi[1]) * (1 + sg[e][a * 3 + 2] * xi[2]); nodes[a] = en[e][a];
        }
        return e;
    }

    // Goldak double-ellipsoid nodal power f [mW] for the step (t1-dt, t1]: the baseline's BuildSource (spread_mode 2)
    void build_source(const Laser &L, const GoldakParams &g, double t1, double dt, vector<double> &f, SrcTape *tape = nullptr) const {
        std::fill(f.begin(), f.end(), 0.0);
        double t0 = std::max(0.0, t1 - dt); if (tape) { tape->subs.clear(); tape->cells.clear(); tape->t0 = t0; tape->t1 = t1; } if (t1 <= t0) return;
        const int i0 = L.bracket(t0), i1 = L.bracket(t1), nT = (int)L.t.size(), nsub = std::max(1, g.nseg), nq = std::max(1, g.nq);
        const double dx = g.cutoff * (g.af + g.ar) / nq, dy = 2.0 * g.cutoff * g.b / nq, dz = g.cutoff * g.c / nq;
        static thread_local array<double, 3> last_dir = {1, 0, 0};
        for (int i = i0; i <= i1; i++) {
            double ta = std::max(t0, L.t[i]), tnx = (i + 1 < nT) ? L.t[i + 1] : t1, tb = std::min(t1, tnx);
            if (tb <= ta) continue;
            double energy = L.absorp * L.p[i] * (tb - ta); if (energy <= 0.0) continue;
            double esub = energy / nsub, sdt = (tb - ta) / nsub;
            { double ddx = L.x[i + 1 < nT ? i + 1 : i] - L.x[i], ddy = L.y[i + 1 < nT ? i + 1 : i] - L.y[i], ddz = L.z[i + 1 < nT ? i + 1 : i] - L.z[i];
              double dn = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz); if (dn > 1e-9) last_dir = {ddx / dn, ddy / dn, ddz / dn}; }
            array<double, 3> e1 = last_dir; double n1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
            if (n1 > 1e-12) for (int d = 0; d < 3; d++) e1[d] /= n1; else e1 = {1, 0, 0};
            array<double, 3> e3 = {0, 0, -1}; double pr = e3[0] * e1[0] + e3[1] * e1[1] + e3[2] * e1[2];
            for (int d = 0; d < 3; d++) e3[d] -= pr * e1[d];
            double n3 = std::sqrt(e3[0] * e3[0] + e3[1] * e3[1] + e3[2] * e3[2]);
            if (n3 < 1e-12) e3 = {0, -1, 0}; else for (int d = 0; d < 3; d++) e3[d] /= n3;
            array<double, 3> e2 = {e3[1] * e1[2] - e3[2] * e1[1], e3[2] * e1[0] - e3[0] * e1[2], e3[0] * e1[1] - e3[1] * e1[0]};
            double n2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]); for (int d = 0; d < 3; d++) e2[d] /= n2;
            e3 = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
            // Goldak cell weights (identical for all sub-samples of the segment)
            struct Cell { double lx, ly, lz, w; int ix, iy, iz; }; vector<Cell> cells; cells.reserve((size_t)nq * nq * nq); double wsum = 0;
            for (int ix = 0; ix < nq; ix++) for (int iy = 0; iy < nq; iy++) for (int iz = 0; iz < nq; iz++) {
                double lx = -g.cutoff * g.ar + (ix + 0.5) * dx, ly = -g.cutoff * g.b + (iy + 0.5) * dy, lz = (iz + 0.5) * dz;
                bool front = lx >= 0.0; double ax_ = front ? g.af : g.ar, fr = front ? g.ff : g.fr;
                double raw = fr / ax_ * std::exp(-3.0 * (lx * lx / (ax_ * ax_) + ly * ly / (g.b * g.b) + lz * lz / (g.c * g.c)));
                cells.push_back({lx, ly, lz, raw, ix, iy, iz}); wsum += raw;
            }
            for (int s = 0; s < nsub; s++) {
                double tm = ta + (s + 0.5) * sdt, xL, yL, zL; L.position(tm, xL, yL, zL);
                SrcSub rec{};
                if (tape) {   // discrete choices of this sub-sample, exactly as the baseline's tape (frozen segment, frame, kept cells)
                    rec.mode = 2; rec.tm = tm; rec.t_s = 0.0; rec.Esub = esub; rec.first_cell = (int)tape->cells.size(); rec.ncell = 0;
                    const int ii = L.bracket(tm); const bool lin = L.t[ii + 1] > L.t[ii];
                    double fr_ = lin ? (tm - L.t[ii]) / (L.t[ii + 1] - L.t[ii]) : 0.0;
                    rec.seg_linear = (lin && fr_ > 0.0 && fr_ < 1.0) ? 1 : 0; rec.ts0 = L.t[ii]; rec.ts1 = lin ? L.t[ii + 1] : L.t[ii] + 1.0;
                    const double fc = std::max(0.0, std::min(1.0, fr_));
                    rec.xa[0] = L.x[ii]; rec.xa[1] = L.y[ii]; rec.xa[2] = L.z[ii]; rec.xb[0] = L.x[ii + 1]; rec.xb[1] = L.y[ii + 1]; rec.xb[2] = L.z[ii + 1];
                    if (!rec.seg_linear) for (int d = 0; d < 3; d++) rec.xa[d] = rec.xa[d] + fc * (rec.xb[d] - rec.xa[d]);
                    for (int d = 0; d < 3; d++) { rec.e1[d] = e1[d]; rec.e2[d] = e2[d]; rec.e3[d] = e3[d]; }
                    rec.af = g.af; rec.ar = g.ar; rec.b = g.b; rec.c = g.c; rec.ff = g.ff; rec.fr = g.fr; rec.box = g.cutoff; rec.nq = nq;
                }
                for (const Cell &c : cells) {
                    double N[8]; int nd[8];
                    int e = locate(xL + c.lx * e1[0] + c.ly * e2[0] + c.lz * e3[0], yL + c.lx * e1[1] + c.ly * e2[1] + c.lz * e3[1],
                                   zL + c.lx * e1[2] + c.ly * e2[2] + c.lz * e3[2], N, nd);
                    if (e < 0) continue;
                    double dE = esub * (c.w / wsum);
                    for (int a = 0; a < 8; a++) f[nd[a]] += dE * N[a];
                    if (tape) {
                        SrcCell sc{}; sc.ix = c.ix; sc.iy = c.iy; sc.iz = c.iz; sc.const_shape = 0;
                        const double pt[3] = {xL + c.lx * e1[0] + c.ly * e2[0] + c.lz * e3[0], yL + c.lx * e1[1] + c.ly * e2[1] + c.lz * e3[1], zL + c.lx * e1[2] + c.ly * e2[2] + c.lz * e3[2]};
                        for (int d = 0; d < 3; d++) sc.xi0[d] = (pt[d] - lo[e][d]) / (hi[e][d] - lo[e][d]);
                        for (int a = 0; a < 8; a++) { for (int d = 0; d < 3; d++) sc.X[a * 3 + d] = X[en[e][a]][d]; sc.dofs[a] = en[e][a]; }
                        tape->cells.push_back(sc); rec.ncell++;
                    }
                }
                if (tape && rec.ncell > 0) tape->subs.push_back(rec);
            }
        }
        for (double &v : f) v /= (t1 - t0);   // mJ -> mW
    }

    // residual r (ess rows zeroed) and Jacobian J = K + Jb + diag(Meff) into A; returns ||r||
    double eval(const vector<double> &Tv, const vector<double> &f, double dt, vector<double> &r, bool want_J, bool exact = false) {
        std::fill(r.begin(), r.end(), 0.0); if (want_J) std::fill(A.begin(), A.end(), 0.0);
        const int nq1 = kq; const double gx[3] = {nq1 == 2 ? -0.5773502691896257 : -0.7745966692414834, nq1 == 2 ? 0.5773502691896257 : 0.0, 0.7745966692414834};
        const double gwt[3] = {nq1 == 2 ? 1.0 : 5.0 / 9.0, nq1 == 2 ? 1.0 : 8.0 / 9.0, 5.0 / 9.0};
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < ne; e++) {
            double h[3] = {hi[e][0] - lo[e][0], hi[e][1] - lo[e][1], hi[e][2] - lo[e][2]};
            double Te[8]; for (int a = 0; a < 8; a++) Te[a] = Tv[en[e][a]];
            double Ke[64] = {0}, Kx[64] = {0}, wv = h[0] * h[1] * h[2] / 8.0;
            for (int qz = 0; qz < nq1; qz++) for (int qy = 0; qy < nq1; qy++) for (int qx = 0; qx < nq1; qx++) {
                const int ii[3] = {qx, qy, qz};
                double xi[3]; for (int d = 0; d < 3; d++) xi[d] = (nq1 == 2) ? (ii[d] ? gx[1] : gx[0]) : gx[ii[d]];
                const double wq = (nq1 == 2) ? 1.0 : gwt[qx] * gwt[qy] * gwt[qz];
                double gN[8][3], Nq[8], Tq = 0;
                for (int a = 0; a < 8; a++) {
                    double f0 = 1 + sg[e][a * 3] * xi[0], f1 = 1 + sg[e][a * 3 + 1] * xi[1], f2 = 1 + sg[e][a * 3 + 2] * xi[2];
                    Nq[a] = 0.125 * f0 * f1 * f2; Tq += Nq[a] * Te[a];
                    gN[a][0] = sg[e][a * 3] * 0.125 * f1 * f2 * 2.0 / h[0]; gN[a][1] = sg[e][a * 3 + 1] * 0.125 * f0 * f2 * 2.0 / h[1]; gN[a][2] = sg[e][a * 3 + 2] * 0.125 * f0 * f1 * 2.0 / h[2];
                }
                double kqv = mat.k(Tq) * wv * wq;
                if (exact && want_J) {   // dk/dT part of the exact Jacobian (sensitivity solves): sum_q w k'(Tq) N_b (gradN_a . gradT)
                    double gT[3] = {0, 0, 0}; for (int b = 0; b < 8; b++) for (int d = 0; d < 3; d++) gT[d] += gN[b][d] * Te[b];
                    const double kd = mat.dk(Tq) * wv * wq;
                    for (int a = 0; a < 8; a++) { const double ga = gN[a][0] * gT[0] + gN[a][1] * gT[1] + gN[a][2] * gT[2]; for (int b = 0; b < 8; b++) Kx[a * 8 + b] += kd * Nq[b] * ga; }
                }
                for (int a = 0; a < 8; a++) for (int b = 0; b < 8; b++) Ke[a * 8 + b] += kqv * (gN[a][0] * gN[b][0] + gN[a][1] * gN[b][1] + gN[a][2] * gN[b][2]);
            }
            for (int a = 0; a < 8; a++) {
                double y = 0; for (int b = 0; b < 8; b++) y += Ke[a * 8 + b] * Te[b];
                #pragma omp atomic
                r[en[e][a]] += y;
            }
            if (want_J) for (int q = 0; q < 64; q++) {
                #pragma omp atomic
                A[epos[(size_t)e * 64 + q]] += Ke[q] + Kx[q];
            }
        }
        // convection + radiation on the side faces (consistent 2x2 quadrature)
        const double Ta4 = Tamb * Tamb * Tamb * Tamb;
        for (size_t q = 0; q < faces.size(); q++) {
            const Face &F = faces[q]; double w = F.area / 4.0;
            for (int gq = 0; gq < 4; gq++) {
                double u = 0.5 + ((gq & 1) ? 0.2886751345948129 : -0.2886751345948129), v = 0.5 + ((gq & 2) ? 0.2886751345948129 : -0.2886751345948129);
                double Nf[4] = {(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v}, Tq = 0;
                for (int a = 0; a < 4; a++) Tq += Nf[a] * Tv[F.v[a]];
                double qv = h_conv * (Tq - Tamb) + emis * sigma * (Tq * Tq * Tq * Tq - Ta4), qp = h_conv + 4.0 * emis * sigma * Tq * Tq * Tq;
                for (int a = 0; a < 4; a++) {
                    r[F.v[a]] += qv * Nf[a] * w;
                    if (want_J) for (int b = 0; b < 4; b++) A[fpos[q * 16 + a * 4 + b]] += qp * Nf[a] * Nf[b] * w;
                }
            }
        }
        // capacitance, latent heat, source
        const double idt = 1.0 / dt; double nrm2 = 0;
        #pragma omp parallel for schedule(static) reduction(+ : nrm2)
        for (int i = 0; i < nv; i++) {
            if (ess[i]) { r[i] = 0.0; if (want_J) A[pos(i, i)] += 1.0e0; continue; }
            double Ti = Tv[i], V = Vn[i], rho = mat.rho;
            double Mc = V * rho * mat.cp(Ti), Ml = V * rho * mat.Lf * mat.dfL(Ti), Lnow = V * rho * mat.Lf * mat.fL(Ti), dMc = V * rho * mat.dcp(Ti);
            r[i] += Mc * idt * (Ti - Tn[i]) + idt * (Lnow - Ln[i]) - f[i];
            if (want_J) A[pos(i, i)] += (Mc + Ml) * idt + dMc * (Ti - Tn[i]) * idt;
            nrm2 += r[i] * r[i];
        }
        return std::sqrt(nrm2);
    }
};

// OpenMP Jacobi-preconditioned CG on the free dofs: A x = b with x[ess] = 0 (ess rows/cols decoupled by masking)
struct PCG {
    static double dot(const vector<double> &a, const vector<double> &b) { double s = 0; const int n = (int)a.size();
        #pragma omp parallel for reduction(+ : s) schedule(static)
        for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
    static int solve(const Level &L, const vector<double> &b, vector<double> &x, double rtol, int maxit) {
        const int n = L.nv; vector<double> r(n), z(n), p(n), Ap(n), dinv(n);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { dinv[i] = L.ess[i] ? 0.0 : 1.0 / L.A[L.pos(i, i)]; x[i] = 0.0; r[i] = L.ess[i] ? 0.0 : b[i]; z[i] = dinv[i] * r[i]; p[i] = z[i]; }
        double rz = dot(r, z), bn = std::sqrt(dot(r, r)); if (bn == 0) return 0;
        for (int it = 1; it <= maxit; it++) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) { double s = 0; if (!L.ess[i]) for (int k = L.I[i]; k < L.I[i + 1]; k++) if (!L.ess[L.J[k]]) s += L.A[k] * p[L.J[k]]; Ap[i] = s; }
            double al = rz / dot(p, Ap);
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) { x[i] += al * p[i]; r[i] -= al * Ap[i]; z[i] = dinv[i] * r[i]; }
            double rzn = dot(r, z); if (std::sqrt(dot(r, r)) <= rtol * bn) return it;
            double be = rzn / rz; rz = rzn;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; i++) p[i] = z[i] + be * p[i];
        }
        return maxit;
    }
};

// one implicit step on a level: Newton with backtracking; returns true on convergence. T is the iterate (initial guess = current T).
struct NewtonOpts { double rtol = 1e-6, atol = 1e-3; int maxit = 200; double lin_rtol = 1e-10; };
inline bool newton_step(Level &L, const vector<double> &f, double dt, const NewtonOpts &o, int &iters, int &lin_total) {
    const int n = L.nv; vector<double> r(n), d(n), Tt(n), rt(n);
    double rn = L.eval(L.T, f, dt, r, true), r0 = rn; iters = 0; lin_total = 0;
    while (rn > std::max(o.rtol * r0, o.atol) && iters < o.maxit) {
        vector<double> b(n); for (int i = 0; i < n; i++) b[i] = -r[i];
        lin_total += PCG::solve(L, b, d, o.lin_rtol, 2000);
        double al = 1.0, rnew = 0; bool ok = false;
        for (int ls = 0; ls < 10; ls++) {
            for (int i = 0; i < n; i++) Tt[i] = L.T[i] + al * d[i];
            rnew = L.eval(Tt, f, dt, rt, false);
            if (rnew < rn || rnew <= std::max(o.rtol * r0, o.atol)) { ok = true; break; }
            al *= 0.5;
        }
        if (!ok) return false;
        L.T = Tt; rn = L.eval(L.T, f, dt, r, true); iters++;
        if (!std::isfinite(rn)) return false;
    }
    return rn <= std::max(o.rtol * r0, o.atol);
}
}   // namespace gm
