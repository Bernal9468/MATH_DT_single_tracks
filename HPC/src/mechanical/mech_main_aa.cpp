// Residual-stress stage with FULL-HESSIAN sensitivities w.r.t. the 20 thermal parameters.
//   * reads the thermal field dumps f_<step>.bin written by the Enzyme thermal driver (T, dT/dp_i, d2T/dp_i dp_j at every node)
//   * small-strain hex8 B-bar (centroid dilatation) Hill thermo-plasticity, element-average temperature (same formulation as the
//     validated bare_plate_single_layer/mechanical module), bottom face fixed, powder-layer/tie handled as in the thermal mesh
//   * history = sequence of dumps; each dump is one quasi-static step (Newton, damped by interval splitting)
//   * derivatives by forward propagation through the history: exact implicit-function solves with the converged consistent tangent,
//     local first/second derivatives of the return map from second-order jets (hill_phi.hpp)
// Mechanical material parameters are NOT differentiated; the derivative parameters are the 20 thermal ones (relative perturbations,
// exactly as in the thermal driver).  Hot elements (T >= T_s) and not-yet-consolidated powder elements have their plastic state reset
// to the stress-free one; powder elements are soft (stiffness factor FSOFT).
#include "hill_phi.hpp"
#ifdef USE_SOLVER_SWITCH
#include "solver_switch.h"          // bare_plate_single_layer: CPU/GPU solver switch for the stiffness system
#include "cudss_backend.h"
#include <memory>
#endif
#ifdef USE_GPU_MECH
#include "mech_gpu.cuh"            // bare_plate_single_layer: GPU kernels (nvcc build, order 1)
#endif
#include <mkl_pardiso.h>
#include <mkl_types.h>
#include <omp.h>
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <dirent.h>
#include <chrono>
#include <functional>
#include <cstring>

using std::vector; using std::array; using std::string;
typedef Jet<8, 1> J81;
typedef Jet<8, 2> J82;
static constexpr int P = 20;                 // thermal parameters
static constexpr int NP2 = P * (P + 1) / 2;  // pairs i<=j
static inline int pk(int i, int j) { if (i > j) std::swap(i, j); return i * P - i * (i - 1) / 2 + (j - i); }
static inline int kmix(int i, int j) { return i * (2 * P - i - 1) / 2 + (j - i - 1); }   // i<j, thermal driver order
static int g_pi[NP2], g_pj[NP2];
static const char *kName[P] = {"k","cp","rho","Lfus","Ts","Tliq","hc","T0","P","Eta","Rx","Ry","Rz","X0","Vx","TL","Tw","hp","kp","Rhop"};
static bool FREEZEHOT = false;   // -freezehot: in -pert (finite-difference) runs the hot/consolidated decisions use the UNPERTURBED temperature (the derivative of the continuous problem, as AD computes)
static bool FULLINT = false;   // -fullint: standard 2x2x2 full-integration strains (no B-bar centroid dilatation)
static vector<int> PROBELIST; static int PROBEEL = -1; static string PROBEFILE = "probe_elem_hist.csv";   // -probeelem E: per-step element-mean stress, d1 and diagonal d2 of that element
static bool ABQANNEAL = false;   // -abqanneal: hot elements (T >= T_solidus) reset ONLY the plastic strain and hardening (Abaqus *ANNEAL TEMPERATURE semantics), the elastic strain is kept
static bool ALLBULK = false;   // -bulk: every element is consolidated from the start (no soft stress-free powder phase) -> comparable with a standard FE code
static bool IPTEMP = false;   // -iptemp: thermal strain from the nodal temperature interpolated to each integration point (default: element-average temperature)
static constexpr double T_SOLIDUS = 1563.0;   // reset temperature (IN625 solidus of the thermal model); not differentiated
static const bool JET2_ALWAYS = std::getenv("MECH_JET2") != nullptr;   // MECH_JET2=1: order 1 also uses Jet<8,2> (the shipped behaviour)
static double FSOFT = 1e-3, NTOL = 1e-10;                  // stiffness factor of unconsolidated powder
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ------------------------------------------------------------------ mesh / tie / reduction
struct Mesh { int nv = 0, ne = 0; vector<array<double, 3>> X; vector<array<int, 8>> E; vector<int> attr; };
static Mesh read_mesh(const string &fn) {
    std::ifstream f(fn); string line; Mesh m;
    while (std::getline(f, line)) {
        if (line == "elements") { std::getline(f, line); m.ne = std::stoi(line); m.E.resize(m.ne); m.attr.resize(m.ne);
            for (int e = 0; e < m.ne; e++) { int g; f >> m.attr[e] >> g; for (int k = 0; k < 8; k++) f >> m.E[e][k]; std::getline(f, line); } }
        else if (line == "vertices") { std::getline(f, line); m.nv = std::stoi(line); std::getline(f, line); m.X.resize(m.nv);
            for (int v = 0; v < m.nv; v++) f >> m.X[v][0] >> m.X[v][1] >> m.X[v][2]; }
    }
    return m;
}
struct Red {   // node -> list of (free node, weight)
    int nfree = 0; vector<int> fid; vector<vector<std::pair<int, double>>> map; vector<char> fixed, slave;
};
static Red build_red(const Mesh &m, const string &tiefile) {
    Red r; r.fid.assign(m.nv, -1); r.map.resize(m.nv); r.fixed.assign(m.nv, 0); r.slave.assign(m.nv, 0);
    double zmin = 1e30; for (auto &x : m.X) zmin = std::min(zmin, x[2]);
    for (int v = 0; v < m.nv; v++) if (std::fabs(m.X[v][2] - zmin) < 1e-9) r.fixed[v] = 1;
    std::ifstream f(tiefile); int ns; f >> ns; vector<array<int, 4>> mas(ns); vector<array<double, 4>> w(ns); vector<int> sv(ns);
    for (int s = 0; s < ns; s++) { f >> sv[s]; for (int k = 0; k < 4; k++) f >> mas[s][k] >> w[s][k]; r.slave[sv[s]] = 1; }
    for (int v = 0; v < m.nv; v++) if (!r.slave[v] && !r.fixed[v]) r.fid[v] = r.nfree++;
    for (int v = 0; v < m.nv; v++) if (r.fid[v] >= 0) r.map[v].push_back({r.fid[v], 1.0});
    for (int s = 0; s < ns; s++) for (int k = 0; k < 4; k++) if (std::fabs(w[s][k]) > 1e-14 && r.fid[mas[s][k]] >= 0) r.map[sv[s]].push_back({r.fid[mas[s][k]], w[s][k]});
    return r;
}

// ------------------------------------------------------------------ hex8 B-bar operator
struct Geo {
    int ne; vector<double> Bb;   // [e][q][6][24]
    vector<double> wq;           // [e][q]
    vector<double> qx;           // [e][q][3] physical position
};
static void hex_shape(double xi, double eta, double ze, double N[8], double dN[8][3]) {
    static const int s[8][3] = {{-1,-1,-1},{1,-1,-1},{1,1,-1},{-1,1,-1},{-1,-1,1},{1,-1,1},{1,1,1},{-1,1,1}};
    for (int a = 0; a < 8; a++) {
        double fx = 1 + s[a][0] * xi, fy = 1 + s[a][1] * eta, fz = 1 + s[a][2] * ze;
        N[a] = 0.125 * fx * fy * fz;
        dN[a][0] = 0.125 * s[a][0] * fy * fz; dN[a][1] = 0.125 * s[a][1] * fx * fz; dN[a][2] = 0.125 * s[a][2] * fx * fy;
    }
}
static void phys_grad(const Mesh &m, int e, double xi, double eta, double ze, double g[8][3], double &detJ, double *xq = nullptr) {
    double N[8], dN[8][3]; hex_shape(xi, eta, ze, N, dN);
    double J[3][3] = {{0}};
    for (int a = 0; a < 8; a++) for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) J[i][j] += m.X[m.E[e][a]][i] * dN[a][j];
    detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    double inv[3][3]; double id = 1.0 / detJ;
    inv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * id; inv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * id; inv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * id;
    inv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * id; inv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * id; inv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * id;
    inv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * id; inv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * id; inv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * id;
    for (int a = 0; a < 8; a++) for (int i = 0; i < 3; i++) { double s = 0; for (int j = 0; j < 3; j++) s += dN[a][j] * inv[j][i]; g[a][i] = s; }
    if (xq) for (int i = 0; i < 3; i++) { xq[i] = 0; for (int a = 0; a < 8; a++) xq[i] += N[a] * m.X[m.E[e][a]][i]; }
}
static Geo build_geo(const Mesh &m) {
    Geo G; G.ne = m.ne; G.Bb.assign((size_t)m.ne * 8 * 144, 0.0); G.wq.assign((size_t)m.ne * 8, 0.0); G.qx.assign((size_t)m.ne * 8 * 3, 0.0);
    const double gp = 1.0 / std::sqrt(3.0), is2 = 1.0 / std::sqrt(2.0);
    #pragma omp parallel for
    for (int e = 0; e < m.ne; e++) {
        double c[8][3], dj; phys_grad(m, e, 0, 0, 0, c, dj);
        int q = 0;
        for (int iz = 0; iz < 2; iz++) for (int iy = 0; iy < 2; iy++) for (int ix = 0; ix < 2; ix++, q++) {
            double g[8][3], detJ, xq[3]; phys_grad(m, e, (ix ? gp : -gp), (iy ? gp : -gp), (iz ? gp : -gp), g, detJ, xq);
            G.wq[(size_t)e * 8 + q] = detJ;
            for (int i = 0; i < 3; i++) G.qx[((size_t)e * 8 + q) * 3 + i] = xq[i];
            double *B = &G.Bb[((size_t)e * 8 + q) * 144];
            for (int a = 0; a < 8; a++) {
                const int d0 = a * 3;
                for (int r = 0; r < 3; r++) B[r * 24 + d0 + r] += g[a][r];
                if (!FULLINT) for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) B[r * 24 + d0 + k] += (c[a][k] - g[a][k]) / 3.0;
                B[3 * 24 + d0 + 0] += g[a][1] * is2; B[3 * 24 + d0 + 1] += g[a][0] * is2;
                B[4 * 24 + d0 + 0] += g[a][2] * is2; B[4 * 24 + d0 + 2] += g[a][0] * is2;
                B[5 * 24 + d0 + 1] += g[a][2] * is2; B[5 * 24 + d0 + 2] += g[a][1] * is2;
            }
        }
    }
    return G;
}

// ------------------------------------------------------------------ thermal samples
struct Sample {
    int step = 0; double t = 0; int n1 = 0, n2 = 0, nm = 0, nv = 0;
    vector<double> T, d1, d2;     // d1[i*nv+v], d2[pk*nv+v] (pair-packed, i<=j)
    vector<double> ever;
    vector<double> Tflag;   // unperturbed temperature (kept only with -freezehot)
};
static bool read_sample(const string &fn, Sample &s, int want_order) {
    std::ifstream f(fn, std::ios::binary); if (!f) return false;
    int hdr[5]; f.read((char *)hdr, sizeof(hdr)); double t; f.read((char *)&t, sizeof(double));
    s.nv = hdr[0]; s.step = hdr[1]; s.n1 = hdr[2]; s.n2 = hdr[3]; s.nm = hdr[4]; s.t = t;
    const size_t nv = s.nv; s.T.resize(nv); f.read((char *)s.T.data(), nv * 8);
    s.d1.assign(want_order >= 1 ? (size_t)P * nv : 0, 0.0); s.d2.assign(want_order >= 2 ? (size_t)NP2 * nv : 0, 0.0);
    vector<double> tmp(nv);
    for (int i = 0; i < s.n1; i++) { f.read((char *)tmp.data(), nv * 8); if (want_order >= 1) std::copy(tmp.begin(), tmp.end(), s.d1.begin() + (size_t)i * nv); }
    for (int i = 0; i < s.n2; i++) { f.read((char *)tmp.data(), nv * 8); if (want_order >= 2) std::copy(tmp.begin(), tmp.end(), s.d2.begin() + (size_t)pk(i, i) * nv); }
    for (int i = 0, k = 0; i < P && s.nm > 0; i++) for (int j = i + 1; j < P; j++, k++) { f.read((char *)tmp.data(), nv * 8); if (want_order >= 2) std::copy(tmp.begin(), tmp.end(), s.d2.begin() + (size_t)pk(i, j) * nv); }
    s.ever.resize(nv); f.read((char *)s.ever.data(), nv * 8);
    return true;
}
static Sample interp(const Sample &a, const Sample &b, double f) {
    Sample r = a; r.t = a.t + f * (b.t - a.t); r.step = -1;
    auto lin = [&](vector<double> &x, const vector<double> &y) { for (size_t i = 0; i < x.size(); i++) x[i] = (1 - f) * x[i] + f * y[i]; };
    lin(r.T, b.T); lin(r.d1, b.d1); lin(r.d2, b.d2); return r;
}
// Taylor-perturbed copy (validation): T(s) = T + sum s_i dT_i + 1/2 sum s_i s_j d2T_ij ; derivatives of the perturbed field are reported
// at the unperturbed point only, so the perturbed copy keeps just T (order 0 run).
static void perturb(Sample &x, const vector<std::pair<int, double>> &sv, bool keep_d1) {
    if (FREEZEHOT) x.Tflag = x.T;
    const size_t nv = x.nv; vector<double> T(x.T); vector<double> d1n;
    if (keep_d1) { d1n = x.d1; for (int i = 0; i < P; i++) for (auto &a : sv) for (size_t v = 0; v < nv; v++) d1n[(size_t)i * nv + v] += a.second * x.d2[(size_t)pk(i, a.first) * nv + v]; }
    for (auto &a : sv) for (size_t v = 0; v < nv; v++) T[v] += a.second * x.d1[(size_t)a.first * nv + v];
    for (auto &a : sv) for (auto &b : sv) { double c = (a.first == b.first) ? 0.5 : (a.first < b.first ? 1.0 : 0.0);   // 1/2 sum_ij = sum_i<j + 1/2 sum_i
        if (c == 0.0) continue; for (size_t v = 0; v < nv; v++) T[v] += c * a.second * b.second * x.d2[(size_t)pk(a.first, b.first) * nv + v]; }
    x.T = T; if (keep_d1) x.d1 = d1n; else x.d1.clear(); x.d2.clear();
}

// ------------------------------------------------------------------ sparse reduced system + PARDISO
struct Sys {
    MKL_INT n = 0; vector<MKL_INT> ia, ja; vector<double> a;   // upper triangle CSR, 0-based -> shifted to 1-based for PARDISO
    void *pt[64]; MKL_INT iparm[64]; MKL_INT mtype = -2; bool analysed = false; vector<MKL_INT> perm;
    inline long pos(MKL_INT r, MKL_INT c) const {   // r<=c
        long lo = ia[r], hi = ia[r + 1] - 1;
        while (lo <= hi) { long mid = (lo + hi) / 2; if (ja[mid] == c) return mid; if (ja[mid] < c) lo = mid + 1; else hi = mid - 1; }
        return -1;
    }
    void init_pardiso() {
        for (int i = 0; i < 64; i++) { pt[i] = 0; iparm[i] = 0; }
        iparm[0] = 1; iparm[1] = 2; iparm[9] = 13; iparm[34] = 0; iparm[26] = 0;   // METIS ordering, 1-based, default pivoting perturbation
        perm.assign(n, 0);
    }
    void call(MKL_INT phase, double *b, double *x, MKL_INT nrhs) {
        MKL_INT maxfct = 1, mnum = 1, msg = 0, err = 0, nn = n; double dd;
        pardiso(pt, &maxfct, &mnum, &mtype, &phase, &nn, a.data(), ia.data(), ja.data(), perm.data(), &nrhs, iparm, &msg, b ? b : &dd, x ? x : &dd, &err);
        if (err) { fprintf(stderr, "PARDISO phase %d error %d\n", (int)phase, (int)err); exit(1); }
    }
#ifndef USE_SOLVER_SWITCH
    void factor() { if (!analysed) { call(12, nullptr, nullptr, 0); analysed = true; } else call(22, nullptr, nullptr, 0); }
    void solve(vector<double> &b, vector<double> &x, int nrhs) { call(33, b.data(), x.data(), nrhs); }
#else
    // CPU/GPU solver switch (src/gpu/solver_switch.h): the stiffness system runs PARDISO and/or cuDSS (symmetric, upper,
    // 1-based CSR as PARDISO's) and keeps whichever is faster on its own steps (factorisation + 1, 20 or 210 right-hand sides).
    std::unique_ptr<CudssSolver> gpu; std::unique_ptr<SolverSwitch> sw; vector<double> xg;
    void gpu_down(const std::string &why) { if (sw->GpuFailed(why)) call(22, nullptr, nullptr, 0); gpu.reset(); }
    void factor() {
        if (!sw) sw.reset(new SolverSwitch("mechanical stiffness (PARDISO mtype -2)"));
        if (!analysed) {
            call(12, nullptr, nullptr, 0); analysed = true;
            bool ready = false;
            static_assert(sizeof(MKL_INT) == sizeof(int), "32-bit CSR indices");
            if (sw->GpuWanted((int)n, CudssSolver::DeviceAvailable())) {
                gpu.reset(new CudssSolver);
                ready = gpu->Analyze((int)n, ia.data(), ja.data(), a.data(), CudssSolver::SYMMETRIC_UPPER, 1, false);
                if (!ready && gpu->Error().find("status 2") != std::string::npos)
                    ready = gpu->Analyze((int)n, ia.data(), ja.data(), a.data(), CudssSolver::SYMMETRIC_UPPER, 1, true);
                if (!ready) { sw->GpuFailed(gpu->Error()); gpu.reset(); }
            }
            sw->Start((int)n, (long)ja.size(), ready);
            return;
        }
        sw->BeginStep();
        if (sw->RunCpu()) { const double t0 = SolverSwitch::Now(); call(22, nullptr, nullptr, 0); sw->AddCpu(SolverSwitch::Now() - t0); }
        if (sw->RunGpu()) { const double t0 = SolverSwitch::Now();
            if (!gpu->Refactor(a.data())) gpu_down(gpu->Error()); else sw->AddGpu(SolverSwitch::Now() - t0); }
    }
    void solve(vector<double> &b, vector<double> &x, int nrhs) {
        if (sw->UseGpuResult()) {
            const double t0 = SolverSwitch::Now();
            if (gpu->Solve(b.data(), x.data(), nrhs)) { sw->AddGpu(SolverSwitch::Now() - t0); return; }
            gpu_down(gpu->Error());
        }
        { const double t0 = SolverSwitch::Now(); call(33, b.data(), x.data(), nrhs); sw->AddCpu(SolverSwitch::Now() - t0); }
        if (sw->Calibrating()) {
            xg.resize((size_t)n * nrhs);
            const double t0 = SolverSwitch::Now();
            if (!gpu->Solve(b.data(), xg.data(), nrhs)) { gpu_down(gpu->Error()); return; }
            sw->AddGpu(SolverSwitch::Now() - t0);
            sw->SolutionDiff(SolverSwitch::RelDiff(xg.data(), x.data(), (size_t)n * nrhs));
        }
    }
#endif
};

struct State {   // per-IP plastic state and its derivatives (old = at start of step)
    vector<double> s;     // [ne*8][7]  epsp(6 Mandel), alpha
    vector<double> ds1;   // [ne*8][P][7]
    vector<double> ds2;   // [ne*8][NP2][7]
};

int main(int argc, char **argv) {
    string meshf = "plate.mesh", tief = "tie_weights.txt", dumpdir, outdir = ".";
    int order = 2, pertorder = 0, maxsteps = 1 << 30, probe_node = -1; bool fd_mode = false; vector<std::pair<int, double>> pert;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "-m") meshf = argv[++i]; else if (a == "-tie") tief = argv[++i]; else if (a == "-dump") dumpdir = argv[++i];
        else if (a == "-o") outdir = argv[++i]; else if (a == "-order") order = atoi(argv[++i]); else if (a == "-nsamp") maxsteps = atoi(argv[++i]);
        else if (a == "-pertorder") pertorder = atoi(argv[++i]); else if (a == "-iptemp") IPTEMP = true; else if (a == "-bulk") ALLBULK = true; else if (a == "-abqanneal") ABQANNEAL = true; else if (a == "-probeelem") { std::stringstream ss(argv[++i]); string tok; while (std::getline(ss, tok, ',')) PROBELIST.push_back(atoi(tok.c_str())); PROBEEL = PROBELIST.empty() ? -1 : PROBELIST[0]; } else if (a == "-fullint") FULLINT = true; else if (a == "-freezehot") FREEZEHOT = true; else if (a == "-ntol") NTOL = atof(argv[++i]); else if (a == "-fsoft") FSOFT = atof(argv[++i]);
        else if (a == "-pert") { string s = argv[++i]; std::stringstream ss(s); string tok; while (std::getline(ss, tok, ',')) { size_t c = tok.find(':'); pert.push_back({atoi(tok.substr(0, c).c_str()), atof(tok.substr(c + 1).c_str())}); } fd_mode = true; }
    }
    if (fd_mode) order = pertorder;
    { int k = 0; for (int r = 0; r < P; r++) for (int c2 = r; c2 < P; c2++, k++) { g_pi[k] = r; g_pj[k] = c2; } }
    printf("mech: order=%d threads=%d\n", order, omp_get_max_threads());
    Mesh m = read_mesh(meshf); printf("mesh: %d elements %d nodes\n", m.ne, m.nv);
    Red R = build_red(m, tief); const int nf3 = 3 * R.nfree; printf("reduced nodes %d (dofs %d)\n", R.nfree, nf3);
    Geo G = build_geo(m);
    // sample list
    vector<std::pair<int, string>> files; { DIR *d = opendir(dumpdir.c_str()); dirent *de; while ((de = readdir(d))) { string n = de->d_name; if (n.size() > 6 && n.substr(0, 2) == "f_" && n.substr(n.size() - 4) == ".bin") files.push_back({atoi(n.substr(2, n.size() - 6).c_str()), dumpdir + "/" + n}); } closedir(d); }
    std::sort(files.begin(), files.end()); if ((int)files.size() > maxsteps) files.resize(maxsteps);
    printf("%zu thermal samples\n", files.size());

    // ---- reduced sparsity (node-block graph)
    vector<vector<int>> enodes(m.ne);
    for (int e = 0; e < m.ne; e++) { vector<int> nl; for (int a = 0; a < 8; a++) for (auto &p : R.map[m.E[e][a]]) nl.push_back(p.first); std::sort(nl.begin(), nl.end()); nl.erase(std::unique(nl.begin(), nl.end()), nl.end()); enodes[e] = nl; }
    Sys K; K.n = nf3;
    {
        vector<vector<int>> adj(R.nfree);
        for (int e = 0; e < m.ne; e++) for (int I : enodes[e]) for (int J : enodes[e]) if (J >= I) adj[I].push_back(J);
        for (auto &v : adj) { std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); }
        K.ia.assign(nf3 + 1, 0);
        for (int I = 0; I < R.nfree; I++) for (int c = 0; c < 3; c++) {
            long cnt = 0; for (int J : adj[I]) { if (J == I) cnt += 3 - c; else cnt += 3; }
            K.ia[3 * I + c + 1] = K.ia[3 * I + c] + cnt;
        }
        K.ja.resize(K.ia[nf3]); K.a.assign(K.ia[nf3], 0.0);
        for (int I = 0; I < R.nfree; I++) for (int c = 0; c < 3; c++) {
            long p = K.ia[3 * I + c];
            for (int J : adj[I]) { if (J == I) { for (int cc = c; cc < 3; cc++) K.ja[p++] = 3 * J + cc; } else for (int cc = 0; cc < 3; cc++) K.ja[p++] = 3 * J + cc; }
        }
        for (auto &x : K.ia) x += 1; for (auto &x : K.ja) x += 1;   // 1-based; pos() below works on 1-based indices
        K.init_pardiso();
    }
    printf("K nnz(upper) = %ld\n", (long)K.ja.size());
    auto kpos = [&](long r, long c) -> long {   // 0-based dof indices r<=c
        long lo = K.ia[r] - 1, hi = K.ia[r + 1] - 2;
        while (lo <= hi) { long mid = (lo + hi) / 2; long cj = K.ja[mid] - 1; if (cj == c) return mid; if (cj < c) lo = mid + 1; else hi = mid - 1; }
        return -1;
    };

    const long NIP = (long)m.ne * 8;
    State S; S.s.assign(NIP * 7, 0.0); if (order >= 1) S.ds1.assign(NIP * P * 7, 0.0); if (order >= 2) S.ds2.assign(NIP * (size_t)NP2 * 7, 0.0);
    State S0;   // state right after the initial (stress-free) step: the Abaqus-style anneal returns hot points to it (plastic strain 0, initial-temperature thermal-strain reference kept)
    State N_; N_.s.assign(NIP * 7, 0.0); if (order >= 1) N_.ds1.assign(NIP * P * 7, 0.0); if (order >= 2) N_.ds2.assign(NIP * (size_t)NP2 * 7, 0.0);
    vector<char> consolidated(m.ne, ALLBULK ? 1 : 0);
    vector<double> ufull(3 * (size_t)m.nv, 0.0), ufree(nf3, 0.0), ufree_last(nf3, 0.0);
    vector<double> du1, du2;   // [nf3*P], [nf3*NP2]
    if (order >= 1) du1.assign((size_t)nf3 * P, 0.0); if (order >= 2) du2.assign((size_t)nf3 * NP2, 0.0);
    // IP outputs (kept for the last step): sigma + derivatives
    vector<double> sig_ip(NIP * 6), dsig1, dsig2, hillse_ip(NIP), vm_ip(NIP);
    // element-average thermal inputs
    vector<double> eT(m.ne), edT, ed2T, ipT, ipdT, ipd2T;   // element averages (flags) + per-integration-point temperature inputs
#ifdef USE_GPU_MECH
    // GPU path: order-1 runs (not -pert); GPU_MECH=0 keeps the CPU loops in this binary
    const bool gpu = (order == 1 || order == 2) && !fd_mode && !(std::getenv("GPU_MECH") && string(std::getenv("GPU_MECH")) == "0");
    mgpu::Dev D;
    if (gpu) {
        using namespace mgpu;
        D.ne = m.ne; D.nv = m.nv; D.nf3 = nf3; D.nip = NIP; D.nnz = (long)K.ja.size();
        vector<int> Eh((size_t)m.ne * 8); for (int e = 0; e < m.ne; e++) for (int a = 0; a < 8; a++) Eh[(size_t)e * 8 + a] = m.E[e][a];
        vector<int> ms(m.nv + 1, 0), mf; vector<double> mwv;
        for (int v = 0; v < m.nv; v++) { for (auto &pr : R.map[v]) { mf.push_back(pr.first); mwv.push_back(pr.second); } ms[v + 1] = (int)mf.size(); }
        vector<long> ia(K.ia.begin(), K.ia.end()), ja(K.ja.begin(), K.ja.end());
        D.E = dalloc<int>(Eh.size()); up(D.E, Eh.data(), Eh.size());
        D.Bb = dalloc<double>(G.Bb.size()); up(D.Bb, G.Bb.data(), G.Bb.size());
        D.wq = dalloc<double>(G.wq.size()); up(D.wq, G.wq.data(), G.wq.size());
        D.mstart = dalloc<int>(ms.size()); up(D.mstart, ms.data(), ms.size());
        D.mfid = dalloc<int>(mf.size()); up(D.mfid, mf.data(), mf.size());
        D.mw = dalloc<double>(mwv.size()); up(D.mw, mwv.data(), mwv.size());
        D.Kia = dalloc<long>(ia.size()); up(D.Kia, ia.data(), ia.size());
        D.Kja = dalloc<long>(ja.size()); up(D.Kja, ja.data(), ja.size());
        D.Ka = dalloc<double>(D.nnz); D.rfull = dalloc<double>(3 * (size_t)m.nv); D.ufull = dalloc<double>(3 * (size_t)m.nv);
        D.fsum = dalloc<double>(1); D.ipT = dalloc<double>(NIP); D.ipdT = dalloc<double>(NIP * P);
        D.powder = dalloc<char>(m.ne); D.hot = dalloc<char>(m.ne);
        D.Ss = dalloc<double>(NIP * 7); D.Ns = dalloc<double>(NIP * 7); D.Sds1 = dalloc<double>(NIP * P * 7); D.Nds1 = dalloc<double>(NIP * P * 7);
        D.F = dalloc<double>(NIP * 13); D.Gm = dalloc<double>(NIP * 104); D.sig = dalloc<double>(NIP * 6); D.dsig1 = dalloc<double>(NIP * P * 6);
        D.rhs1 = dalloc<double>((size_t)nf3 * P); D.du1f = dalloc<double>((size_t)P * 3 * m.nv);
        D.dmax = dalloc<unsigned long long>(2); D.dcnt = dalloc<unsigned long long>(1);
        D.sgv = dalloc<double>(NIP * 6); D.CB = dalloc<double>(NIP * 144);
        if (order >= 2) {   // order 2: Hessians, dw1, second-order thermal input, state, rhs, update, dsigma (~0.8 GB)
            D.H = dalloc<double>(NIP * 468); D.dw1 = dalloc<double>(NIP * P * 8); D.ipd2T = dalloc<double>(NIP * NP2);
            D.Sds2 = dalloc<double>(NIP * NP2 * 7); D.Nds2 = dalloc<double>(NIP * NP2 * 7); D.dsig2 = dalloc<double>(NIP * NP2 * 6);
            D.rhs2 = dalloc<double>((size_t)nf3 * NP2); D.du2f = dalloc<double>((size_t)NP2 * 3 * m.nv);
            MCK(cudaMemcpyToSymbol(c_pi, g_pi, sizeof(int) * NP2)); MCK(cudaMemcpyToSymbol(c_pj, g_pj, sizeof(int) * NP2));
        }
        MCK(cudaHostRegister(K.a.data(), sizeof(double) * K.a.size(), cudaHostRegisterDefault));   // pinned: K.a goes to the host every Newton iteration
        cudaDeviceProp prop; MCK(cudaGetDeviceProperties(&prop, 0));
        printf("GPU mechanical kernels: %s (assembly, jet pass, derivative RHS and update; PARDISO on the CPU)\n", prop.name);
    }
#else
    const bool gpu = false;
#endif

    auto expand = [&](const vector<double> &xf, vector<double> &xfull, int ncol, size_t strideFull_nv, size_t strideFree) {
        // xf[col*strideFree + 3*fid + c] -> xfull[col*strideFull + 3*node + c]
        #pragma omp parallel for
        for (int col = 0; col < ncol; col++) for (int v = 0; v < m.nv; v++) for (int c = 0; c < 3; c++) {
            double s = 0; for (auto &p : R.map[v]) s += p.second * xf[(size_t)col * strideFree + 3 * p.first + c];
            xfull[(size_t)col * strideFull_nv + 3 * v + c] = s; }
    };

    // ---- per-element nodal gather for u (full space)
    auto elem_u = [&](int e, const vector<double> &uf, double ue[24]) { for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = uf[3 * (size_t)m.E[e][a] + c]; };

    // thermal eth derivatives (scalar, per element)
    auto eth_derivs = [&](double T, double &e0, double &e1, double &e2) {
        Jet<1, 2> t = Jet<1, 2>::var(T, 0); auto r = hill::thermal_strain<Jet<1, 2>>(t); e0 = r.v; e1 = r.g[0]; e2 = r.h[0]; };

    Sample cur, prev; bool have_prev = false;
    double t_nom = 0, t_der = 0, t_ass = 0; long nnewt_tot = 0; int nsplit = 0;
    // bare_plate_single_layer: finer timers (timing only)
    double tm_in = 0, tm_asm = 0, tm_kfs = 0, tm_jet = 0, tm_rhs = 0, tm_dsol = 0, tm_upd = 0, tm_out = 0;
    double tm_upd1 = 0, tm_rhs2 = 0, tm_sol2 = 0, tm_upd2 = 0;   // order-2 split of d_update
    vector<FILE *> fpes; for (int pe : PROBELIST) { FILE *fpe = fopen((outdir + "/probe_elem_hist_" + std::to_string(pe) + ".csv").c_str(), "w"); fprintf(fpe, "t,step,T_elem"); const char *cn[7] = {"S11","S22","S33","S12","S13","S23","PEEQ"}; for (int c = 0; c < 7; c++) fprintf(fpe, ",%s", cn[c]); for (int c = 0; c < 6; c++) for (int i = 0; i < 20; i++) fprintf(fpe, ",d%s_dp%d", cn[c], i); if (order >= 2) for (int c = 0; c < 6; c++) for (int i = 0; i < 20; i++) fprintf(fpe, ",d2%s_dp%d_dp%d", cn[c], i, i); fprintf(fpe, "\n"); fpes.push_back(fpe); }
    FILE *fprobe = fopen((outdir + "/mech_history.csv").c_str(), "w");
    fprintf(fprobe, "t,step,newton,Tmax,max_hill_se,max_peeq,n_plastic_ip\n");

    // ================= one load step: applies sample `smp` (nodal T + derivatives) =================
    // returns false if Newton failed (state untouched)
    auto step = [&](const Sample &smp, bool init_step, bool want_out) -> bool {
        if (PROBEEL >= 0 && !init_step) want_out = true;
        double t0 = now();
        const size_t nv = m.nv;
        // element-average thermal inputs
        #pragma omp parallel for
        for (int e = 0; e < m.ne; e++) { double s = 0; for (int a = 0; a < 8; a++) s += smp.T[m.E[e][a]]; eT[e] = s / 8.0; }
        if (order >= 1) { edT.assign((size_t)m.ne * P, 0.0);
            #pragma omp parallel for
            for (int e = 0; e < m.ne; e++) for (int i = 0; i < P; i++) { double s = 0; for (int a = 0; a < 8; a++) s += smp.d1[(size_t)i * nv + m.E[e][a]]; edT[(size_t)e * P + i] = s / 8.0; } }
        if (order >= 2) { ed2T.assign((size_t)m.ne * NP2, 0.0);
            #pragma omp parallel for
            for (int e = 0; e < m.ne; e++) for (int i = 0; i < NP2; i++) { double s = 0; for (int a = 0; a < 8; a++) s += smp.d2[(size_t)i * nv + m.E[e][a]]; ed2T[(size_t)e * NP2 + i] = s / 8.0; } }
        // per-integration-point thermal inputs (IPTEMP: nodal interpolation N_a(xi_q) T_a ; else the element average, i.e. the legacy behaviour)
        {   const size_t NIPl = (size_t)m.ne * 8; ipT.assign(NIPl, 0.0);
            if (order >= 1) ipdT.assign(NIPl * P, 0.0); if (order >= 2) ipd2T.assign(NIPl * NP2, 0.0);
            const double gq = 1.0 / std::sqrt(3.0);
            #pragma omp parallel for
            for (int e = 0; e < m.ne; e++) for (int q = 0; q < 8; q++) {
                const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2);
                if (!IPTEMP) { ipT[ip] = eT[e]; if (order >= 1) for (int i = 0; i < P; i++) ipdT[ip * P + i] = edT[(size_t)e * P + i]; if (order >= 2) for (int i = 0; i < NP2; i++) ipd2T[ip * NP2 + i] = ed2T[(size_t)e * NP2 + i]; continue; }
                double Nn[8], dNn[8][3]; hex_shape(((q & 1) ? gq : -gq), (((q >> 1) & 1) ? gq : -gq), (((q >> 2) & 1) ? gq : -gq), Nn, dNn);
                double s = 0; for (int a = 0; a < 8; a++) s += Nn[a] * smp.T[m.E[e][a]]; ipT[ip] = s;
                if (order >= 1) for (int i = 0; i < P; i++) { double t = 0; for (int a = 0; a < 8; a++) t += Nn[a] * smp.d1[(size_t)i * nv + m.E[e][a]]; ipdT[ip * P + i] = t; }
                if (order >= 2) for (int i = 0; i < NP2; i++) { double t = 0; for (int a = 0; a < 8; a++) t += Nn[a] * smp.d2[(size_t)i * nv + m.E[e][a]]; ipd2T[ip * NP2 + i] = t; }
            }
        }
        // element flags from the nominal temperature (unperturbed one with -freezehot)
        vector<double> eTf(eT); if (!smp.Tflag.empty()) for (int e = 0; e < m.ne; e++) { double s = 0; for (int a = 0; a < 8; a++) s += smp.Tflag[m.E[e][a]]; eTf[e] = s / 8.0; }
        vector<char> hot(m.ne), powder(m.ne);
        for (int e = 0; e < m.ne; e++) { hot[e] = eTf[e] >= T_SOLIDUS; powder[e] = (m.attr[e] == 2) && !consolidated[e]; }
        const bool gstep = gpu && !init_step;   // this step runs its element/point loops on the GPU
#ifdef USE_GPU_MECH
        if (gstep) { mgpu::up(D.ipT, ipT.data(), ipT.size()); mgpu::up(D.ipdT, ipdT.data(), ipdT.size());
                     mgpu::up(D.powder, powder.data(), powder.size()); mgpu::up(D.hot, hot.data(), hot.size());
                     if (order >= 2) mgpu::up(D.ipd2T, ipd2T.data(), ipd2T.size()); }
#endif

        tm_in += now() - t0;
        // ---------------- nominal solve (Newton) ----------------
        vector<double> rfree(nf3), dx(nf3), rfull(3 * nv);
        auto eval_residual_and_tangent = [&](const vector<double> &uf, bool want_K, double &rnorm, double &fref) {
            std::fill(rfull.begin(), rfull.end(), 0.0); if (want_K) std::fill(K.a.begin(), K.a.end(), 0.0);
            double fsum = 0;
#ifdef USE_GPU_MECH
            if (gstep) {
                using namespace mgpu;
                up(D.ufull, uf.data(), uf.size());
                MCK(cudaMemset(D.rfull, 0, sizeof(double) * 3 * nv)); MCK(cudaMemset(D.fsum, 0, sizeof(double)));
                if (want_K) MCK(cudaMemset(D.Ka, 0, sizeof(double) * D.nnz));
                k_asm_ip<<<nb(NIP, 64), 64>>>(D, want_K, FSOFT); k_asm_el<<<nb((long)m.ne * 24, 64), 64>>>(D, want_K); MCK(cudaGetLastError());
                down(rfull.data(), D.rfull, rfull.size()); down(&fsum, D.fsum, 1);
                if (want_K) down(K.a.data(), D.Ka, K.a.size());
            } else
#endif
            {
            #pragma omp parallel for reduction(+ : fsum) schedule(dynamic, 64)
            for (int e = 0; e < m.ne; e++) {
                double ue[24]; elem_u(e, uf, ue);
                double fe[24] = {0}, Ke[24][24]; if (want_K) std::memset(Ke, 0, sizeof(Ke));
                const double gsc = powder[e] ? FSOFT : 1.0;
                for (int q = 0; q < 8; q++) {
                    const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const double *B = &G.Bb[ip * 144]; const double w = G.wq[ip];
                    J81 wv[8];
                    for (int i = 0; i < 6; i++) { double eps = 0; for (int d = 0; d < 24; d++) eps += B[i * 24 + d] * ue[d]; wv[i] = J81::var(eps - S.s[ip * 7 + i], i); }
                    wv[6] = J81::var(ipT[ip], 6); wv[7] = J81::var(S.s[ip * 7 + 6], 7);
                    J81 sg[6], dep[6], da; hill::phi_w<J81>(wv, gsc, sg, dep, da);
                    for (int i = 0; i < 6; i++) for (int d = 0; d < 24; d++) fe[d] += w * B[i * 24 + d] * sg[i].v;
                    if (want_K) {
                        double C[6][6]; for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) C[i][j] = sg[i].g[j];
                        double CB[6][24]; for (int i = 0; i < 6; i++) for (int d = 0; d < 24; d++) { double s = 0; for (int j = 0; j < 6; j++) s += C[i][j] * B[j * 24 + d]; CB[i][d] = s; }
                        for (int d1 = 0; d1 < 24; d1++) for (int d2 = 0; d2 < 24; d2++) { double s = 0; for (int i = 0; i < 6; i++) s += B[i * 24 + d1] * CB[i][d2]; Ke[d1][d2] += w * s; }
                    }
                }
                for (int d = 0; d < 24; d++) fsum += fe[d] * fe[d];
                // scatter (through the node maps)
                for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) {
                    const int na = m.E[e][a]; const double v = fe[a * 3 + c];
                    #pragma omp atomic
                    rfull[3 * (size_t)na + c] += v;
                }
                if (want_K) for (int a = 0; a < 8; a++) for (auto &pa : R.map[m.E[e][a]]) for (int b = 0; b < 8; b++) for (auto &pb : R.map[m.E[e][b]]) {
                    const double wgt = pa.second * pb.second;
                    for (int ca = 0; ca < 3; ca++) for (int cb = 0; cb < 3; cb++) {
                        long r = 3 * pa.first + ca, c = 3 * pb.first + cb; if (r > c) continue;
                        long p = kpos(r, c); double v = wgt * Ke[a * 3 + ca][b * 3 + cb];
                        #pragma omp atomic
                        K.a[p] += v;
                    }
                }
            }
            }
            fref = std::sqrt(fsum);
            std::fill(rfree.begin(), rfree.end(), 0.0);
            for (int v = 0; v < (int)nv; v++) for (auto &p : R.map[v]) for (int c = 0; c < 3; c++) rfree[3 * p.first + c] += p.second * rfull[3 * (size_t)v + c];
            double s = 0; for (double x : rfree) s += x * x; rnorm = std::sqrt(s);
        };
        int nit = 0; bool conv = false;
        if (init_step) { conv = true; std::fill(ufree.begin(), ufree.end(), 0.0); }
        else {
            ufree = ufree_last;
            for (nit = 0; nit < 30; nit++) {
                expand(ufree, ufull, 1, 3 * nv, nf3);
                double ta_ = now(); double rn, fr; eval_residual_and_tangent(ufull, true, rn, fr); tm_asm += now() - ta_;
                if (rn <= std::max(NTOL * fr, 1e-14)) { conv = true; break; }
                ta_ = now(); K.factor();
                if (const char *md = std::getenv("MATDUMP_DIR")) {   // bare_plate_single_layer: write K (upper, solver benchmark) once
                    static const int at = std::getenv("MATDUMP_STEP") ? std::atoi(std::getenv("MATDUMP_STEP")) : 100; static bool done = false;
                    if (!done && smp.step >= at) { done = true;
                        std::ofstream o(string(md) + "/mech_K_step" + std::to_string(smp.step) + ".mtx");
                        o << "%%MatrixMarket matrix coordinate real symmetric\n" << K.n << " " << K.n << " " << K.ja.size() << "\n" << std::setprecision(17);
                        for (MKL_INT r = 0; r < K.n; r++) for (MKL_INT k = K.ia[r] - 1; k < K.ia[r + 1] - 1; k++) o << r + 1 << " " << K.ja[k] << " " << K.a[k] << "\n"; } }
                for (int i = 0; i < nf3; i++) rfree[i] = -rfree[i];
                K.solve(rfree, dx, 1); tm_kfs += now() - ta_;
                double dn = 0; for (int i = 0; i < nf3; i++) { ufree[i] += dx[i]; dn = std::max(dn, std::fabs(dx[i])); }
                if (nit > 12 && !std::isfinite(rn)) break;
            }
        }
        if (!conv) { return false; }
        nnewt_tot += nit; expand(ufree, ufull, 1, 3 * nv, nf3);
        double t1 = now(); t_nom += t1 - t0;

        // ---------------- final evaluation: second-order jets, states, derivative propagation ----------------
        // per-IP storage
        struct IPd { double F[13]; double Gm[13][8]; double H[13][36]; double sig[6]; };
        // bare_plate_single_layer: kept across steps (was a fresh, zero-filled 94 MB vector every step); every entry the
        // derivative passes read is written by the pass below first
        static vector<IPd> ipd_keep; if (order >= 1 && !gstep && (long)ipd_keep.size() != NIP) ipd_keep.resize(NIP);
        vector<IPd> &ipd = ipd_keep;
        vector<double> dw1; if (order >= 1 && !gstep) dw1.assign(NIP * P * 8, 0.0);
        vector<double> dsig1_loc, dsig2_loc;
        if (want_out && order >= 1) { if (!gstep) dsig1.assign(NIP * P * 6, 0.0); else if (dsig1.size() != (size_t)NIP * P * 6) dsig1.assign(NIP * P * 6, 0.0);
                                      if (order >= 2) { if (!gstep) dsig2.assign(NIP * (size_t)NP2 * 6, 0.0);
                                                        else if (dsig2.size() != (size_t)NIP * NP2 * 6) dsig2.assign(NIP * (size_t)NP2 * 6, 0.0); } }
        double eth_e[1];
        // nominal pass (+ jets if derivatives are needed)
        double max_se = 0, max_peeq = 0; long nplast = 0;
        double Pm[6][6]; hill::HillP(Pm);
        static const std::string dump_wv_ = std::getenv("MECH_DUMP_WV") ? std::string(",") + std::getenv("MECH_DUMP_WV") + "," : std::string();
        const bool dump_now_ = !init_step && !dump_wv_.empty() && dump_wv_.find("," + std::to_string(smp.step) + ",") != std::string::npos;
        vector<double> wv_all_(dump_now_ ? NIP * 9 : 0);
#ifdef USE_GPU_MECH
        if (gstep) {
            using namespace mgpu;
            up(D.ufull, ufull.data(), ufull.size());
            MCK(cudaMemset(D.dmax, 0, 2 * sizeof(unsigned long long))); MCK(cudaMemset(D.dcnt, 0, sizeof(unsigned long long)));
            if (order >= 2) k_jets2<<<nb(NIP, 64), 64>>>(D, FSOFT, ABQANNEAL); else k_jets<<<nb(NIP, 64), 64>>>(D, FSOFT, ABQANNEAL);
            MCK(cudaGetLastError());
            unsigned long long mx[2], cnt; down(mx, D.dmax, 2); down(&cnt, D.dcnt, 1);
            std::memcpy(&max_se, &mx[0], 8); std::memcpy(&max_peeq, &mx[1], 8); nplast = (long)cnt;
        } else
#endif
        {
        #pragma omp parallel for schedule(dynamic, 64) reduction(max : max_se, max_peeq) reduction(+ : nplast)
        for (int e = 0; e < m.ne; e++) {
            double ue[24]; elem_u(e, ufull, ue);
            const double gsc = powder[e] ? FSOFT : 1.0; const bool reset = init_step || (hot[e] && !ABQANNEAL) || powder[e]; const bool aa = ABQANNEAL && !init_step && hot[e] && !powder[e];
            
            for (int q = 0; q < 8; q++) {
                const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const double *B = &G.Bb[ip * 144];
                double eps[6]; for (int i = 0; i < 6; i++) { double s = 0; for (int d = 0; d < 24; d++) s += B[i * 24 + d] * ue[d]; eps[i] = s; }
                double wv[8]; for (int i = 0; i < 6; i++) wv[i] = eps[i] - S.s[ip * 7 + i]; wv[6] = ipT[ip]; wv[7] = S.s[ip * 7 + 6];
                if (dump_now_) { for (int i = 0; i < 8; i++) wv_all_[ip * 9 + i] = wv[i]; wv_all_[ip * 9 + 8] = gsc; }
                double sg[6], dep[6], da;
                if (init_step) { for (int i = 0; i < 6; i++) sg[i] = 0.0; }
                if (order == 0 || init_step) {
                    if (!init_step) hill::phi_w<double>(wv, gsc, sg, dep, da);
                    for (int i = 0; i < 6; i++) sig_ip[ip * 6 + i] = sg[i];
                    // new state
                    if (init_step) { for (int i = 0; i < 6; i++) N_.s[ip * 7 + i] = -(i < 3 ? e0 : 0.0); N_.s[ip * 7 + 6] = 0.0; }
                    else if (reset) { for (int i = 0; i < 6; i++) N_.s[ip * 7 + i] = eps[i] - (i < 3 ? e0 : 0.0); N_.s[ip * 7 + 6] = 0.0; }
                    else { for (int i = 0; i < 6; i++) N_.s[ip * 7 + i] = S.s[ip * 7 + i] + dep[i]; N_.s[ip * 7 + 6] = aa ? 0.0 : S.s[ip * 7 + 6] + da; }
                }
                if (order >= 1 && !init_step) {
                    IPd &d = ipd[ip];
                    if (order == 1 && !JET2_ALWAYS) {
                        // bare_plate_single_layer: order 1 uses only F (values) and Gm (gradient) -- Jet<8,1> gives them bitwise
                        // identical to Jet<8,2> (the gradient part does not depend on the Hessian part) at a fraction of the cost
                        J81 wj[8]; for (int i = 0; i < 8; i++) wj[i] = J81::var(wv[i], i);
                        J81 sj[6], dj[6], daj; hill::phi_w<J81>(wj, gsc, sj, dj, daj);
                        for (int i = 0; i < 6; i++) { d.F[i] = sj[i].v; d.F[6 + i] = dj[i].v; for (int k = 0; k < 8; k++) d.Gm[i][k] = sj[i].g[k], d.Gm[6 + i][k] = dj[i].g[k]; }
                        d.F[12] = daj.v; for (int k = 0; k < 8; k++) d.Gm[12][k] = daj.g[k];
                    } else {
                    J82 wj[8]; for (int i = 0; i < 8; i++) wj[i] = J82::var(wv[i], i);
                    J82 sj[6], dj[6], daj; hill::phi_w<J82>(wj, gsc, sj, dj, daj);
                    for (int i = 0; i < 6; i++) { d.F[i] = sj[i].v; d.F[6 + i] = dj[i].v; for (int k = 0; k < 8; k++) d.Gm[i][k] = sj[i].g[k], d.Gm[6 + i][k] = dj[i].g[k]; for (int k = 0; k < 36; k++) d.H[i][k] = sj[i].h[k], d.H[6 + i][k] = dj[i].h[k]; }
                    d.F[12] = daj.v; for (int k = 0; k < 8; k++) d.Gm[12][k] = daj.g[k]; for (int k = 0; k < 36; k++) d.H[12][k] = daj.h[k];
                    }
                    for (int i = 0; i < 6; i++) { sig_ip[ip * 6 + i] = d.F[i]; d.sig[i] = d.F[i]; }
                    if (reset) { for (int i = 0; i < 6; i++) N_.s[ip * 7 + i] = eps[i] - (i < 3 ? e0 : 0.0); N_.s[ip * 7 + 6] = 0.0; }
                    else { for (int i = 0; i < 6; i++) N_.s[ip * 7 + i] = S.s[ip * 7 + i] + d.F[6 + i]; N_.s[ip * 7 + 6] = aa ? 0.0 : S.s[ip * 7 + 6] + d.F[12]; }
                } else if (order >= 1 && init_step) { /* handled below */ }
                // diagnostics
                { double sv[6]; for (int i = 0; i < 6; i++) sv[i] = sig_ip[ip * 6 + i]; double Ps[6]; double q2 = 0; for (int i = 0; i < 6; i++) { double s = 0; for (int j = 0; j < 6; j++) s += Pm[i][j] * sv[j]; Ps[i] = s; q2 += sv[i] * s; }
                  double se = std::sqrt(std::max(0.0, q2)); hillse_ip[ip] = se; max_se = std::max(max_se, se);
                  double mises = 0; { double a = sv[0] - sv[1], b = sv[1] - sv[2], c = sv[2] - sv[0]; mises = std::sqrt(0.5 * (a * a + b * b + c * c) + 3.0 * (sv[3] * sv[3] + sv[4] * sv[4] + sv[5] * sv[5]) * 0.5 * 1.0); }
                  vm_ip[ip] = mises; max_peeq = std::max(max_peeq, N_.s[ip * 7 + 6]); if (N_.s[ip * 7 + 6] > S.s[ip * 7 + 6] + 1e-14) nplast++; }
            }
        }
        }
        double t2 = now(); tm_jet += t2 - t1;
        if (dump_now_) { FILE *fw = fopen((outdir + "/wv_step" + std::to_string(smp.step) + ".bin").c_str(), "wb"); fwrite(wv_all_.data(), 8, wv_all_.size(), fw); fclose(fw);
                         printf("  [dump] jet-pass inputs of step %d (%.1f ms)\n", smp.step, 1e3 * (t2 - t1)); }

        double tc_ = t2;
        // ---------------- derivative propagation ----------------
        if (order >= 1) {
            // initial step: states are functions of T only
            if (init_step) {
                #pragma omp parallel for
                for (int e = 0; e < m.ne; e++) { 
                    for (int q = 0; q < 8; q++) { const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2);
                        for (int i = 0; i < P; i++) { double dT = ipdT[ip * P + i]; double *o = &N_.ds1[(ip * P + i) * 7]; for (int c = 0; c < 6; c++) o[c] = -(c < 3 ? e1 * dT : 0.0); o[6] = 0.0; }
                        if (order >= 2) for (int i = 0; i < P; i++) for (int j = i; j < P; j++) { double *o = &N_.ds2[(ip * NP2 + pk(i, j)) * 7];
                            for (int c = 0; c < 6; c++) o[c] = -(c < 3 ? e2 * ipdT[ip * P + i] * ipdT[ip * P + j] + e1 * ipd2T[ip * NP2 + pk(i, j)] : 0.0); o[6] = 0.0; } } }
            } else {
                // first-order right-hand sides: K du_i = - sum w B^T G_sigma kw_i
                vector<double> rhs1((size_t)nf3 * P, 0.0);
#ifdef USE_GPU_MECH
                if (gstep) {
                    using namespace mgpu;
                    MCK(cudaMemset(D.rhs1, 0, sizeof(double) * rhs1.size()));
                    k_drhs<P><<<nb((long)m.ne * P, 64), 64>>>(D); MCK(cudaGetLastError());
                    down(rhs1.data(), D.rhs1, rhs1.size());
                } else
#endif
                {
                #pragma omp parallel for schedule(dynamic, 64)
                for (int e = 0; e < m.ne; e++) {
                    vector<double> fe((size_t)24 * P, 0.0);
                    for (int q = 0; q < 8; q++) {
                        const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const IPd &d = ipd[ip]; const double *B = &G.Bb[ip * 144]; const double w = G.wq[ip];
                        for (int i = 0; i < P; i++) {
                            double kw[8]; const double *dso = &S.ds1[(ip * P + i) * 7];
                            for (int c = 0; c < 6; c++) kw[c] = -dso[c]; kw[6] = ipdT[ip * P + i]; kw[7] = dso[6];
                            double fs[6]; for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[r][k] * kw[k]; fs[r] = s; }
                            for (int dd = 0; dd < 24; dd++) { double s = 0; for (int r = 0; r < 6; r++) s += B[r * 24 + dd] * fs[r]; fe[(size_t)i * 24 + dd] += w * s; }
                        }
                    }
                    for (int i = 0; i < P; i++) for (int a = 0; a < 8; a++) for (auto &pa : R.map[m.E[e][a]]) for (int c = 0; c < 3; c++) {
                        double v = -pa.second * fe[(size_t)i * 24 + a * 3 + c];
                        #pragma omp atomic
                        rhs1[(size_t)i * nf3 + 3 * pa.first + c] += v;
                    }
                }
                }
                double tb_ = now(); tm_rhs += tb_ - t2;
                K.factor();
                K.solve(rhs1, du1, P);
                tc_ = now(); tm_dsol += tc_ - tb_;
                vector<double> du1f((size_t)3 * nv * P); expand(du1, du1f, P, 3 * nv, nf3);
                // dw_i, new first-order states, dsigma_i
#ifdef USE_GPU_MECH
                if (gstep) {
                    using namespace mgpu;
                    up(D.du1f, du1f.data(), du1f.size());
                    k_dupd<P><<<nb(NIP * P, 128), 128>>>(D, ABQANNEAL); MCK(cudaGetLastError());
                } else
#endif
                {
                #pragma omp parallel for schedule(dynamic, 64)
                for (int e = 0; e < m.ne; e++) {
                    
                    const bool reset = (hot[e] && !ABQANNEAL) || powder[e]; const bool aa = ABQANNEAL && hot[e] && !powder[e];
                    for (int q = 0; q < 8; q++) {
                        const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const IPd &d = ipd[ip]; const double *B = &G.Bb[ip * 144];
                        for (int i = 0; i < P; i++) {
                            double deps[6], ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = du1f[(size_t)i * 3 * nv + 3 * (size_t)m.E[e][a] + c];
                            for (int r = 0; r < 6; r++) { double s = 0; for (int dd = 0; dd < 24; dd++) s += B[r * 24 + dd] * ue[dd]; deps[r] = s; }
                            const double *dso = &S.ds1[(ip * P + i) * 7]; double *dw = &dw1[(ip * P + i) * 8];
                            for (int c = 0; c < 6; c++) dw[c] = deps[c] - dso[c]; dw[6] = ipdT[ip * P + i]; dw[7] = dso[6];
                            double *dn = &N_.ds1[(ip * P + i) * 7];
                            if (reset) { for (int c = 0; c < 6; c++) dn[c] = deps[c] - (c < 3 ? e1 * dw[6] : 0.0); dn[6] = 0.0; }
                            else { for (int c = 0; c < 6; c++) { double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[6 + c][k] * dw[k]; dn[c] = dso[c] + s; } double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[12][k] * dw[k]; dn[6] = aa ? 0.0 : dso[6] + s; }
                            if (want_out) for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[r][k] * dw[k]; dsig1[(ip * P + i) * 6 + r] = s; }
                        }
                    }
                }
                }
                // ---------------- second order ----------------
                const double t2a_ = now(); tm_upd1 += t2a_ - tc_;
                if (order >= 2) {
                    vector<double> rhs2((size_t)nf3 * NP2, 0.0);
                    // per IP: Hv[o][i][8] = H_o dw_i computed on the fly
                    auto contract = [&](const IPd &d, const double *dwb, int o, int i, int j) -> double {   // dw_i^T H_o dw_j
                        const double *a = dwb + (size_t)i * 8, *b = dwb + (size_t)j * 8; double s = 0; int p = 0;
                        for (int k = 0; k < 8; k++) for (int l = k; l < 8; l++, p++) s += d.H[o][p] * (k == l ? a[k] * b[l] : a[k] * b[l] + a[l] * b[k]);
                        return s; };
#ifdef USE_GPU_MECH
                    if (gstep) {
                        using namespace mgpu;
                        MCK(cudaMemset(D.rhs2, 0, sizeof(double) * rhs2.size()));
                        k_drhs2<P, NP2><<<nb((long)m.ne * NP2, 64), 64>>>(D); MCK(cudaGetLastError());
                        down(rhs2.data(), D.rhs2, rhs2.size());
                    } else
#endif
                    {
                    #pragma omp parallel for schedule(dynamic, 32)
                    for (int e = 0; e < m.ne; e++) {
                        vector<double> fe((size_t)24 * NP2, 0.0);
                        for (int q = 0; q < 8; q++) {
                            const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const IPd &d = ipd[ip]; const double *B = &G.Bb[ip * 144]; const double w = G.wq[ip];
                            const double *dwb = &dw1[ip * P * 8];
                            for (int i = 0; i < P; i++) for (int j = i; j < P; j++) {
                                const int pr = pk(i, j); const double *dso = &S.ds2[(ip * NP2 + pr) * 7];
                                double kw[8]; for (int c = 0; c < 6; c++) kw[c] = -dso[c]; kw[6] = ipd2T[ip * NP2 + pr]; kw[7] = dso[6];
                                double fs[6]; for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[r][k] * kw[k]; fs[r] = s + contract(d, dwb, r, i, j); }
                                for (int dd = 0; dd < 24; dd++) { double s = 0; for (int r = 0; r < 6; r++) s += B[r * 24 + dd] * fs[r]; fe[(size_t)pr * 24 + dd] += w * s; }
                            }
                        }
                        for (int pr = 0; pr < NP2; pr++) for (int a = 0; a < 8; a++) for (auto &pa : R.map[m.E[e][a]]) for (int c = 0; c < 3; c++) {
                            double v = -pa.second * fe[(size_t)pr * 24 + a * 3 + c];
                            #pragma omp atomic
                            rhs2[(size_t)pr * nf3 + 3 * pa.first + c] += v;
                        }
                    }
                    }
                    const double t2b_ = now(); tm_rhs2 += t2b_ - t2a_;
                    K.solve(rhs2, du2, NP2);
                    vector<double> du2f((size_t)3 * nv * NP2); expand(du2, du2f, NP2, 3 * nv, nf3);
                    const double t2c_ = now(); tm_sol2 += t2c_ - t2b_;
#ifdef USE_GPU_MECH
                    if (gstep) {
                        using namespace mgpu;
                        up(D.du2f, du2f.data(), du2f.size());
                        k_dupd2<P, NP2><<<nb(NIP * NP2, 128), 128>>>(D, ABQANNEAL); MCK(cudaGetLastError());
                    } else
#endif
                    {
                    #pragma omp parallel for schedule(dynamic, 32)
                    for (int e = 0; e < m.ne; e++) {
                        
                        const bool reset = (hot[e] && !ABQANNEAL) || powder[e]; const bool aa = ABQANNEAL && hot[e] && !powder[e];
                        for (int q = 0; q < 8; q++) {
                            const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const IPd &d = ipd[ip]; const double *B = &G.Bb[ip * 144];
                            const double *dwb = &dw1[ip * P * 8];
                            for (int i = 0; i < P; i++) for (int j = i; j < P; j++) {
                                const int pr = pk(i, j);
                                double ue[24]; for (int a = 0; a < 8; a++) for (int c = 0; c < 3; c++) ue[a * 3 + c] = du2f[(size_t)pr * 3 * nv + 3 * (size_t)m.E[e][a] + c];
                                double d2eps[6]; for (int r = 0; r < 6; r++) { double s = 0; for (int dd = 0; dd < 24; dd++) s += B[r * 24 + dd] * ue[dd]; d2eps[r] = s; }
                                const double *dso = &S.ds2[(ip * NP2 + pr) * 7];
                                double d2w[8]; for (int c = 0; c < 6; c++) d2w[c] = d2eps[c] - dso[c]; d2w[6] = ipd2T[ip * NP2 + pr]; d2w[7] = dso[6];
                                double *dn = &N_.ds2[(ip * NP2 + pr) * 7];
                                if (reset) { const double dTi = ipdT[ip * P + i], dTj = ipdT[ip * P + j];
                                    for (int c = 0; c < 6; c++) dn[c] = d2eps[c] - (c < 3 ? e2 * dTi * dTj + e1 * d2w[6] : 0.0); dn[6] = 0.0; }
                                else { for (int c = 0; c < 7; c++) { const int o = (c < 6) ? 6 + c : 12; double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[o][k] * d2w[k]; dn[c] = dso[c] + s + contract(d, dwb, o, i, j); if (aa) dn[6] = 0.0; } }
                                if (want_out) for (int r = 0; r < 6; r++) { double s = 0; for (int k = 0; k < 8; k++) s += d.Gm[r][k] * d2w[k]; dsig2[(ip * NP2 + pr) * 6 + r] = s + contract(d, dwb, r, i, j); }
                            }
                        }
                    }
                    }
                }
            }
        }
        double t3 = now(); t_der += t3 - t2;
        if (order >= 2 && !init_step) tm_upd2 += t3 - tc_ - 0.0;
        if (order >= 1 && !init_step) tm_upd += t3 - tc_;
        // commit
        const double td_ = now();
        std::swap(S.s, N_.s); if (order >= 1) std::swap(S.ds1, N_.ds1); if (order >= 2) std::swap(S.ds2, N_.ds2);
#ifdef USE_GPU_MECH
        if (gstep) { std::swap(D.Ss, D.Ns); std::swap(D.Sds1, D.Nds1); if (order >= 2) std::swap(D.Sds2, D.Nds2);
            for (int pe : PROBELIST) { const size_t ip0 = (size_t)pe * 8;   // probe elements: the host copies the output below reads
                mgpu::down(&sig_ip[ip0 * 6], D.sig + ip0 * 6, 48); mgpu::down(&S.s[ip0 * 7], D.Ss + ip0 * 7, 56);
                mgpu::down(&dsig1[ip0 * P * 6], D.dsig1 + ip0 * P * 6, 8 * P * 6);
                if (order >= 2) mgpu::down(&dsig2[ip0 * NP2 * 6], D.dsig2 + ip0 * NP2 * 6, 8 * NP2 * 6); } }
        else if (gpu && init_step) { mgpu::up(D.Ss, S.s.data(), S.s.size()); mgpu::up(D.Sds1, S.ds1.data(), S.ds1.size());
                                     if (order >= 2) mgpu::up(D.Sds2, S.ds2.data(), S.ds2.size()); }
#endif
        ufree_last = ufree;
        for (int e = 0; e < m.ne; e++) if (eTf[e] >= T_SOLIDUS) consolidated[e] = 1;
        double Tmax = 0; for (int v = 0; v < (int)nv; v++) Tmax = std::max(Tmax, smp.T[v]);
        if (!init_step) for (size_t pi__ = 0; pi__ < PROBELIST.size(); pi__++) { FILE *fpe = fpes[pi__]; const int e = PROBELIST[pi__]; const double s2_ = std::sqrt(2.0); double Te = 0; for (int a = 0; a < 8; a++) Te += smp.T[m.E[e][a]] / 8.0;
            fprintf(fpe, "%.9e,%d,%.4f", smp.t, smp.step, Te);
            for (int c = 0; c < 6; c++) { double v = 0; for (int q = 0; q < 8; q++) v += sig_ip[((size_t)e * 8 + q) * 6 + c] * (c < 3 ? 1.0 : 1.0 / s2_) / 8.0; fprintf(fpe, ",%.12g", v); }
            { double v = 0; for (int q = 0; q < 8; q++) v += S.s[((size_t)e * 8 + q) * 7 + 6] / 8.0; fprintf(fpe, ",%.12g", v); }
            if (order >= 1) for (int c = 0; c < 6; c++) for (int i = 0; i < P; i++) { double v = 0; for (int q = 0; q < 8; q++) v += dsig1[(((size_t)e * 8 + q) * P + i) * 6 + c] * (c < 3 ? 1.0 : 1.0 / s2_) / 8.0; fprintf(fpe, ",%.12g", v); }
            if (order >= 2) for (int c = 0; c < 6; c++) for (int i = 0; i < P; i++) { double v = 0; for (int q = 0; q < 8; q++) v += dsig2[(((size_t)e * 8 + q) * NP2 + pk(i, i)) * 6 + c] * (c < 3 ? 1.0 : 1.0 / s2_) / 8.0; fprintf(fpe, ",%.12g", v); }
            fprintf(fpe, "\n"); fflush(fpe); }
        fprintf(fprobe, "%.9e,%d,%d,%.2f,%.3f,%.6e,%ld\n", smp.t, smp.step, nit, Tmax, max_se, max_peeq, nplast); fflush(fprobe);
        printf("  t=%.6e step=%d newton=%d Tmax=%.0f max_hill=%.1f peeq=%.3e plastic_ip=%ld | nom %.2fs der %.2fs\n", smp.t, smp.step, nit, Tmax, max_se, max_peeq, nplast, t1 - t0, t3 - t2); fflush(stdout);
        tm_out += now() - td_;
        return true;
    };

    // ----- driver over the samples (interval splitting on Newton failure)
    std::function<void(const Sample &, const Sample &, int, bool)> run_interval = [&](const Sample &a, const Sample &b, int depth, bool out) {
        if (step(b, false, out)) return;
        if (depth > 6) { fprintf(stderr, "Newton failed even at depth 6 (t=%g)\n", b.t); exit(2); }
        nsplit++; Sample mid = interp(a, b, 0.5);
        printf("  [split] interval t=%.6e..%.6e (depth %d)\n", a.t, b.t, depth + 1);
        run_interval(a, mid, depth + 1, false); run_interval(mid, b, depth + 1, out);
    };
    double tm_read = 0; const double tm_loop0 = now();
    for (size_t k = 0; k < files.size(); k++) {
        const double tr_ = now();
        Sample s; const bool ok_ = read_sample(files[k].second, s, fd_mode ? 2 : order); tm_read += now() - tr_;
        if (!ok_) { fprintf(stderr, "cannot read %s\n", files[k].second.c_str()); return 1; }
        if (fd_mode) { perturb(s, pert, pertorder >= 1); if (pertorder == 1) { s.d2.clear(); } }
        if (k == 0) { step(s, true, false); prev = s; S0 = S; continue; }
        const bool last = (k + 1 == files.size());
        run_interval(prev, s, 0, last);
        prev = s;
    }
    printf("done: newton iterations %ld, interval splits %d, nominal %.1fs, derivative %.1fs\n", nnewt_tot, nsplit, t_nom, t_der);
    printf("timers [s]: loop=%.2f read=%.2f inputs=%.2f newton_asm=%.2f newton_pardiso=%.2f jets=%.2f d_rhs=%.2f d_pardiso=%.2f d_update=%.2f output=%.2f\n",
           now() - tm_loop0, tm_read, tm_in, tm_asm, tm_kfs, tm_jet, tm_rhs, tm_dsol, tm_upd, tm_out);
    if (order >= 2) printf("order-2 split [s]: d_update1=%.2f d2_rhs=%.2f d2_pardiso+expand=%.2f d2_update=%.2f\n",
                           tm_upd1, tm_rhs2, tm_sol2, tm_upd2 - tm_upd1 - tm_rhs2 - tm_sol2);
    fclose(fprobe); for (FILE *f_ : fpes) fclose(f_);

#ifdef USE_GPU_MECH
    if (gpu) { mgpu::down(S.s.data(), D.Ss, S.s.size()); mgpu::down(S.ds1.data(), D.Sds1, S.ds1.size());
               mgpu::down(sig_ip.data(), D.sig, sig_ip.size()); dsig1.resize((size_t)NIP * P * 6); mgpu::down(dsig1.data(), D.dsig1, dsig1.size());
               if (order >= 2) { mgpu::down(S.ds2.data(), D.Sds2, S.ds2.size()); dsig2.resize((size_t)NIP * NP2 * 6); mgpu::down(dsig2.data(), D.dsig2, dsig2.size()); } }
#endif
    // ---------------- final-step nodal output ----------------
    {
        const size_t nv = m.nv; const int NF = 12;   // Ux Uy Uz Sxx Syy Szz Sxy Sxz Syz HillEff VonMises PEEQ
        const int nd1 = order >= 1 ? P : 0, nd2 = order >= 2 ? NP2 : 0;
        // per-node accumulation (volume-weighted average of IP values of adjacent elements)
        vector<double> wsum(nv, 0.0);
        vector<double> val((size_t)NF * nv, 0.0), v1((size_t)NF * nd1 * nv, 0.0), v2((size_t)NF * nd2 * nv, 0.0);
        const double s2 = std::sqrt(2.0);
        // IP-level quantities as functions: sigma tensor comps (Mandel->tensor), Hill eff, von Mises, peeq
        auto vm_P = [&](double Q[6][6]) { for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) Q[i][j] = 0;
            Q[0][0] = Q[1][1] = Q[2][2] = 1; Q[0][1] = Q[1][0] = Q[0][2] = Q[2][0] = Q[1][2] = Q[2][1] = -0.5; Q[3][3] = Q[4][4] = Q[5][5] = 1.5; Q[0][0] = 1; };
        vector<double> emean((size_t)m.ne * 7, 0.0), emd1((size_t)m.ne * P * 6, 0.0);   // + element mean of d(sigma)/dp_i (order>=1)   // element mean of IP S11 S22 S33 S12 S13 S23 PEEQ (for Abaqus comparison)
        double PH[6][6]; hill::HillP(PH); double PV[6][6]; vm_P(PV);
        // sigma_vm^2 = sig^T PV sig with PV = [[1,-.5,-.5],..] diag(3/2 shear Mandel)  (checked against the closed form above)
        for (int e = 0; e < m.ne; e++) {
            double ue[24]; elem_u(e, ufull, ue);
            for (int q = 0; q < 8; q++) {
                const size_t ip = (size_t)e * 8 + q; [[maybe_unused]] double e0, e1, e2; eth_derivs(ipT[ip], e0, e1, e2); const double w = G.wq[ip];
                // IP values
                double sv[6]; for (int i = 0; i < 6; i++) sv[i] = sig_ip[ip * 6 + i];
                for (int i = 0; i < 6; i++) emean[(size_t)e * 7 + i] += sv[i] * (i < 3 ? 1.0 : 1.0 / s2) / 8.0; emean[(size_t)e * 7 + 6] += S.s[ip * 7 + 6] / 8.0;
                if (order >= 1) for (int i = 0; i < P; i++) for (int c = 0; c < 6; c++) emd1[((size_t)e * P + i) * 6 + c] += dsig1[(ip * P + i) * 6 + c] * (c < 3 ? 1.0 : 1.0 / s2) / 8.0;
                auto qf = [&](double Q[6][6], const double a[6], const double b[6]) { double s = 0; for (int i = 0; i < 6; i++) for (int j = 0; j < 6; j++) s += a[i] * Q[i][j] * b[j]; return s; };
                const double qh = qf(PH, sv, sv), qv = qf(PV, sv, sv); const double se_h = std::sqrt(std::max(qh, 1e-30)), se_v = std::sqrt(std::max(qv, 1e-30));
                double gh[6], gv[6]; for (int i = 0; i < 6; i++) { double a = 0, b = 0; for (int j = 0; j < 6; j++) { a += PH[i][j] * sv[j]; b += PV[i][j] * sv[j]; } gh[i] = a / se_h; gv[i] = b / se_v; }
                const double fac[6] = {1, 1, 1, 1 / s2, 1 / s2, 1 / s2};   // Mandel -> tensor
                for (int a = 0; a < 8; a++) {
                    const int n = m.E[e][a]; const double wt = w * 0.125; wsum[n] += wt;
                    for (int c = 0; c < 3; c++) val[(size_t)c * nv + n] += wt * 0;   // displacement handled separately
                    for (int c = 0; c < 6; c++) val[(size_t)(3 + c) * nv + n] += wt * sv[c] * fac[c];
                    val[(size_t)9 * nv + n] += wt * se_h; val[(size_t)10 * nv + n] += wt * se_v; val[(size_t)11 * nv + n] += wt * S.s[ip * 7 + 6];
                    for (int i = 0; i < nd1; i++) {
                        const double *ds = &dsig1[(ip * P + i) * 6]; double *o = &v1[(size_t)i * NF * nv];
                        for (int c = 0; c < 6; c++) o[(size_t)(3 + c) * nv + n] += wt * ds[c] * fac[c];
                        double a1 = 0, b1 = 0; for (int c = 0; c < 6; c++) { a1 += gh[c] * ds[c]; b1 += gv[c] * ds[c]; }
                        o[(size_t)9 * nv + n] += wt * a1; o[(size_t)10 * nv + n] += wt * b1; o[(size_t)11 * nv + n] += wt * S.ds1[(ip * P + i) * 7 + 6];
                    }
                    for (int i = 0; i < nd2; i++) {
                        const int pi_ = g_pi[i], pj_ = g_pj[i];
                        const double *ds2v = &dsig2[(ip * NP2 + i) * 6]; double *o = &v2[(size_t)i * NF * nv];
                        for (int c = 0; c < 6; c++) o[(size_t)(3 + c) * nv + n] += wt * ds2v[c] * fac[c];
                        // d2(se) = g.d2sig + dsig_i^T (Q - g g^T)/se dsig_j
                        const double *da = &dsig1[(ip * P + pi_) * 6], *db = &dsig1[(ip * P + pj_) * 6];
                        double ah = 0, av = 0, ch = 0, cv = 0, Pa[6], Pb[6], Va[6], Vb[6];
                        for (int c = 0; c < 6; c++) { ah += gh[c] * ds2v[c]; av += gv[c] * ds2v[c]; double pa = 0, pb = 0, va = 0, vb = 0; for (int r = 0; r < 6; r++) { pa += PH[c][r] * da[r]; pb += PH[c][r] * db[r]; va += PV[c][r] * da[r]; vb += PV[c][r] * db[r]; } Pa[c] = pa; Pb[c] = pb; Va[c] = va; Vb[c] = vb; }
                        double qa = 0, qb = 0, ra = 0, rb = 0; for (int c = 0; c < 6; c++) { ch += da[c] * Pb[c]; cv += da[c] * Vb[c]; qa += gh[c] * da[c]; qb += gh[c] * db[c]; ra += gv[c] * da[c]; rb += gv[c] * db[c]; }
                        ah += (ch - qa * qb) / se_h; av += (cv - ra * rb) / se_v;
                        o[(size_t)9 * nv + n] += wt * ah; o[(size_t)10 * nv + n] += wt * av; o[(size_t)11 * nv + n] += wt * S.ds2[(ip * NP2 + i) * 7 + 6];
                    }
                }
            }
        }
        for (size_t n = 0; n < nv; n++) { const double iw = 1.0 / std::max(wsum[n], 1e-300); for (int f = 0; f < NF; f++) val[(size_t)f * nv + n] *= iw; for (int i = 0; i < nd1; i++) for (int f = 0; f < NF; f++) v1[((size_t)i * NF + f) * nv + n] *= iw; for (int i = 0; i < nd2; i++) for (int f = 0; f < NF; f++) v2[((size_t)i * NF + f) * nv + n] *= iw; }
        for (size_t n = 0; n < nv; n++) for (int c = 0; c < 3; c++) val[(size_t)c * nv + n] = ufull[3 * n + c];
        if (order >= 1) { vector<double> du1f((size_t)3 * nv * P); expand(du1, du1f, P, 3 * nv, nf3); for (int i = 0; i < P; i++) for (size_t n = 0; n < nv; n++) for (int c = 0; c < 3; c++) v1[((size_t)i * NF + c) * nv + n] = du1f[(size_t)i * 3 * nv + 3 * n + c]; }
        if (order >= 2) { vector<double> du2f((size_t)3 * nv * NP2); expand(du2, du2f, NP2, 3 * nv, nf3); for (int i = 0; i < NP2; i++) for (size_t n = 0; n < nv; n++) for (int c = 0; c < 3; c++) v2[((size_t)i * NF + c) * nv + n] = du2f[(size_t)i * 3 * nv + 3 * n + c]; }
        { FILE *fe = fopen((outdir + "/final_elem_mean.txt").c_str(), "w"); fprintf(fe, "# element S11 S22 S33 S12 S13 S23 PEEQ(mean) nip\n"); for (int e = 0; e < m.ne; e++) { fprintf(fe, "%d", e + 1); for (int c = 0; c < 7; c++) fprintf(fe, " %.12g", emean[(size_t)e * 7 + c]); fprintf(fe, " 8\n"); } fclose(fe); }
        if (order >= 1) { FILE *fe = fopen((outdir + "/final_elem_mean_d1.txt").c_str(), "w"); fprintf(fe, "# element then for each parameter i=0..19 the element-mean dS11 dS22 dS33 dS12 dS13 dS23 (per unit relative change)\n"); for (int e = 0; e < m.ne; e++) { fprintf(fe, "%d", e + 1); for (int i = 0; i < P; i++) for (int c = 0; c < 6; c++) fprintf(fe, " %.12g", emd1[((size_t)e * P + i) * 6 + c]); fprintf(fe, "\n"); } fclose(fe); }
        std::ofstream of(outdir + "/final_fields.bin", std::ios::binary);
        int hd[5] = {(int)nv, NF, nd1, nd2, P}; of.write((char *)hd, sizeof(hd));
        of.write((char *)val.data(), val.size() * 8); of.write((char *)v1.data(), v1.size() * 8); of.write((char *)v2.data(), v2.size() * 8);
        std::ofstream ox(outdir + "/final_nodes.bin", std::ios::binary); for (size_t n = 0; n < nv; n++) ox.write((char *)m.X[n].data(), 24);
        printf("wrote final_fields.bin (%d fields x (1+%d+%d) columns x %zu nodes)\n", NF, nd1, nd2, nv);
    }
    return 0;
}
