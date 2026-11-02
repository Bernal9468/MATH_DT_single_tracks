// gm_moving.cpp -- GO_MELT phase 3: nested levels with MOVING L2/L3 boxes following the laser along +x (IN718 single tracks).
// L1 = static plate. L2/L3 are uniform boxes whose x-offset to the laser is fixed; they translate in whole elements of their own size, the
// nodal state follows the nodes (exact coincidence), the new leading slab is seeded from the parent, the trailing slab is dropped.
// Dirichlet values of a child come from its parent (one-way). A global peak-temperature record on a fixed fine grid (-peakmesh) keeps the
// maximum over time of the finest level, for the melt-pool depth/width of the whole track.
#include "gm_core.hpp"
#include "gm_sens.hpp"
#include <chrono>
using namespace gm;
struct Link { vector<int> node; vector<array<int, 8>> pn; vector<array<double, 8>> pw; };
static Link make_link(const Level &c, const Level &p) {
    Link k; const double tol = 1e-9;
    for (int v = 0; v < c.nv; v++) {
        const auto &x = c.X[v];
        bool b = std::fabs(x[0] - c.ax[0].front()) < tol || std::fabs(x[0] - c.ax[0].back()) < tol || std::fabs(x[1] - c.ax[1].front()) < tol ||
                 std::fabs(x[1] - c.ax[1].back()) < tol || std::fabs(x[2] - c.zmin) < tol;
        if (!b) continue;
        double N[8]; int nd[8]; int e = p.locate(x[0], x[1], std::min(x[2], p.zmax), N, nd);
        if (e < 0) { printf("child node %d (%g,%g,%g) outside the parent level\n", v, x[0], x[1], x[2]); exit(3); }
        k.node.push_back(v); array<int, 8> a; array<double, 8> w; for (int i = 0; i < 8; i++) { a[i] = nd[i]; w[i] = N[i]; } k.pn.push_back(a); k.pw.push_back(w);
    }
    return k;
}
static double interp(const Level &L, double x, double y, double z, bool &ok) {
    double N[8]; int nd[8]; int e = L.locate(x, y, z, N, nd); ok = e >= 0; if (!ok) return 0; double s = 0; for (int a = 0; a < 8; a++) s += N[a] * L.T[nd[a]]; return s;
}
static double interpS(const Level &L, const vector<double> &S, double x, double y, double z, bool &ok) {
    double N[8]; int nd[8]; int e = L.locate(x, y, z, N, nd); ok = e >= 0; if (!ok) return 0; double s = 0; for (int a = 0; a < 8; a++) s += N[a] * S[nd[a]]; return s;
}
int main(int argc, char **argv) {
    const char *m1 = "", *m2 = "", *m3 = "", *pkm = "", *laser_file = "", *out = "out"; int halfat = -1; double dt = 5e-6, tf = 2.5875e-3, absorp = 0.748695, gb = 0.041029, gc = 0.182604;
    double probex = 0, probey = 0, tamb_c = 23.5, fleut = 0.23, teut = 1423.15, gcut = 1.5, h2 = 0.04, h3 = 0.02; int fbskip_ic = 1, gsub = 12, nseg = 20, nmax = 200, cuts = 6, toploss = 1, kq = 3, mv2 = 1, mv3 = 1, fb = 1, pkevery = 10, dorder = 0, dumpevery = 20, dumpevery2 = 200, seedcol = 1; const char *dumpdir = "", *dumpmesh = "";
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&m1, "-l1", "--l1", "level-1 mesh (static plate)"); args.AddOption(&m2, "-l2", "--l2", "level-2 box mesh at the laser start position"); args.AddOption(&m3, "-l3", "--l3", "level-3 box mesh at the laser start position");
    args.AddOption(&h2, "-h2", "--h2", "L2 element size"); args.AddOption(&h3, "-h3", "--h3", "L3 element size"); args.AddOption(&mv2, "-mv2", "--mv2", "L2 follows the laser"); args.AddOption(&mv3, "-mv3", "--mv3", "L3 follows the laser");
    args.AddOption(&fbskip_ic, "-fbskipic", "--fbskipic", "1 = the feedback injection skips the T0 and Tw (initial-condition) sensitivities: they are a decaying thin-sheet field that the parent carries accurately, and injecting the child sheet creates a bump at the child box faces");
    args.AddOption(&fb, "-fb", "--fb", "1 = two-way: after each step the child solution is injected into the parent nodes inside the child box");
    args.AddOption(&dorder, "-dorder", "--dorder", "0 = temperatures, 1 = + 20 first-order Enzyme sensitivities carried through the levels");
    args.AddOption(&dumpdir, "-dumpdir", "--dumpdir", "write mechanical-stage field dumps f_<step>.bin (T + 20 first derivatives on the nodes of -dumpmesh, baseline format)");
    args.AddOption(&dumpmesh, "-dumpmesh", "--dumpmesh", "mesh on whose nodes the dumps are written (the baseline single-level mesh)");
    args.AddOption(&dumpevery, "-dumpevery", "--dumpevery", "dump every n steps while the laser scans"); args.AddOption(&dumpevery2, "-dumpevery2", "--dumpevery2", "dump every n steps during the cool-down");
    args.AddOption(&seedcol, "-seedcol", "--seedcol", "1 = new leading slab of a moving box copies the (virgin) front column of the same level (T, L and all sensitivities, so the IC seeds keep their thin-layer structure); 0 = interpolate the parent");
    args.AddOption(&pkevery, "-pkevery", "--pkevery", "every n steps the peak record also takes the coarser levels (L2, then L1) at the peak-grid nodes outside the fine box (the wake keeps heating after the laser has passed)");
    args.AddOption(&pkm, "-peakmesh", "--peakmesh", "fixed fine grid for the global peak-T record (same y,z nodes and spacing as L3)");
    args.AddOption(&laser_file, "-laserfile", "--laser", "laser table"); args.AddOption(&out, "-o", "--out", "output directory"); args.AddOption(&dt, "-fdt", "--dt", "time step [s]");
    args.AddOption(&tf, "-tf", "--tf", "final time [s]"); args.AddOption(&halfat, "-halfat", "--halfat", "step index (0-based) taken with dt/2, -1 none (reproduces the baseline Newton cut)"); args.AddOption(&absorp, "-absorp", "--absorp", "absorptivity"); args.AddOption(&gb, "-gb", "--gb", "Goldak a=b [mm]");
    args.AddOption(&gc, "-gc", "--gc", "Goldak depth c [mm]"); args.AddOption(&gsub, "-gsub", "--gsub", "Goldak cells per axis"); args.AddOption(&gcut, "-gcut", "--gcut", "Goldak box factor");
    args.AddOption(&nseg, "-nseg", "--nseg", "sub-samples per segment"); args.AddOption(&probex, "-probex", "--probex", "probe x"); args.AddOption(&probey, "-probey", "--probey", "probe y");
    args.AddOption(&tamb_c, "-tambc", "--tambc", "ambient/bottom [C]"); args.AddOption(&fleut, "-fleut", "--fleut", "eutectic fraction"); args.AddOption(&teut, "-teut", "--teut", "eutectic start [K]");
    args.AddOption(&nmax, "-nmax", "--nmax", "Newton max iters"); args.AddOption(&toploss, "-toploss", "--toploss", "top-face convection/radiation"); args.AddOption(&kq, "-kq", "--kq", "Gauss points");
    args.Parse(); if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
    system((std::string("mkdir -p ") + out).c_str());
    vector<std::string> files = {m1}; if (*m2) files.push_back(m2); if (*m3) files.push_back(m3);
    const int NL = (int)files.size(); vector<Level> L(NL); vector<double> hl = {0.0, h2, h3}; if (NL == 2) hl[1] = h3;   // two levels: L1 + one box at h3
    vector<int> moving(NL, 0); if (NL == 3) { moving[1] = mv2; moving[2] = mv3; } else if (NL == 2) moving[1] = mv3;
    Laser las; las.load(laser_file); las.absorp = absorp; const double xL0 = las.x[0];
    for (int l = 0; l < NL; l++) {
        mfem::Mesh mesh(files[l].c_str(), 1, 1); L[l].toploss = toploss; L[l].kq = kq; L[l].Tamb = tamb_c + 273.15; L[l].mat.fleut = fleut; L[l].mat.Teut = teut;
        L[l].init(mesh); if (l == 0) L[l].set_bottom_dirichlet(tamb_c + 273.15);
        printf("level %d: %d nodes %d elems, x [%g,%g] y [%g,%g] z [%g,%g]%s\n", l + 1, L[l].nv, L[l].ne, L[l].ax[0].front(), L[l].ax[0].back(), L[l].ax[1].front(), L[l].ax[1].back(), L[l].zmin, L[l].zmax, moving[l] ? " (moving)" : "");
    }
    vector<double> back(NL, 0.0), org(NL, 0.0); vector<long> kcur(NL, 0);
    for (int l = 1; l < NL; l++) { back[l] = xL0 - L[l].ax[0].front(); org[l] = L[l].ax[0].front(); }
    vector<Link> link(NL);
    auto relink = [&](int l) { link[l] = make_link(L[l], L[l - 1]); for (int v : link[l].node) L[l].ess[v] = 1; };
    for (int l = 1; l < NL; l++) relink(l);
    vector<Sens> sd(NL); vector<SrcTape> tapes(NL); vector<vector<vector<double>>> prevS(NL); long gits = 0; double wsens = 0; std::ofstream ps;
    if (dorder) { fb_set_fl2(fleut, teut); for (int l = 0; l < NL; l++) sd[l].init(L[l], tamb_c + 273.15); }
    GoldakParams g; g.af = g.ar = g.b = gb; g.c = gc; g.nq = gsub; g.cutoff = gcut; g.nseg = nseg;
    // global peak record
    Level &F = L[NL - 1]; vector<double> gpeak; vector<vector<double>> gpeak_d; int gNX = 0, gNY = 0, gNZ = 0; double gx0 = 0, gy0 = 0, gz0 = 0;
    if (*pkm) {
        mfem::Mesh pm(pkm, 1, 1); int nv = pm.GetNV(); vector<double> xs, ys, zs; for (int v = 0; v < nv; v++) { xs.push_back(pm.GetVertex(v)[0]); ys.push_back(pm.GetVertex(v)[1]); zs.push_back(pm.GetVertex(v)[2]); }
        Level::uniq(xs); Level::uniq(ys); Level::uniq(zs); gNX = (int)xs.size(); gNY = (int)ys.size(); gNZ = (int)zs.size(); gx0 = xs[0]; gy0 = ys[0]; gz0 = zs[0]; gpeak.assign((size_t)nv, -1e300); if (dorder) gpeak_d.assign(NPAR, vector<double>((size_t)nv, 0.0));
        printf("global peak grid %d x %d x %d from x=%g\n", gNX, gNY, gNZ, gx0);
    }
    vector<std::string> names = {"L1", "L2", "L3"}; 
    if (dorder) { ps.open(std::string(out) + "/sens.csv"); ps.precision(12); ps << "time_native"; for (int p = 0; p < NPAR; p++) ps << ",dT_d" << PNAME[p] << "_row1"; ps << "\n"; }
    std::ofstream pc(std::string(out) + "/probe.csv"); pc << "time_native,layer1_top,T_max,substrate_top\n"; pc.precision(12);
    vector<array<double, 3>> DX; if (*dumpdir) { system((std::string("mkdir -p ") + dumpdir).c_str()); mfem::Mesh dm(dumpmesh, 1, 1); DX.resize(dm.GetNV()); for (int v = 0; v < dm.GetNV(); v++) for (int d = 0; d < 3; d++) DX[v][d] = dm.GetVertex(v)[d]; printf("dumps on %zu nodes of %s\n", DX.size(), dumpmesh); }
    auto write_dump = [&](int step, double tt) {
        const int nd = (int)DX.size(); vector<double> Tn_(nd), Sd((size_t)(dorder ? NPAR : 0) * nd, 0.0), ev(nd, 1.0);
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < nd; v++) {
            const auto &x = DX[v]; bool done = false;
            for (int l = NL - 1; l >= 0 && !done; l--) { double N[8]; int ndn[8]; int e = L[l].locate(x[0], x[1], std::min(x[2], L[l].zmax), N, ndn); if (e < 0) continue;
                double Tv = 0; for (int a = 0; a < 8; a++) Tv += N[a] * L[l].T[ndn[a]]; Tn_[v] = Tv;
                if (dorder) for (int p = 0; p < NACT; p++) { double sv = 0; for (int a = 0; a < 8; a++) sv += N[a] * sd[l].S[p][ndn[a]]; Sd[(size_t)p * nd + v] = sv; }
                done = true; }
            if (!done) Tn_[v] = tamb_c + 273.15;
        }
        std::ofstream of(std::string(dumpdir) + "/f_" + std::to_string(step) + ".bin", std::ios::binary); int hdr[5] = {nd, step, dorder ? NPAR : 0, 0, 0}; of.write((char *)hdr, sizeof(hdr)); of.write((char *)&tt, 8);
        of.write((char *)Tn_.data(), 8 * (size_t)nd); of.write((char *)Sd.data(), 8 * Sd.size()); of.write((char *)ev.data(), 8 * (size_t)nd);
    };
    const double t_scan_end = las.t.size() >= 2 ? las.t[las.t.size() - 2] : las.t.back();
    NewtonOpts no; no.maxit = nmax; vector<vector<double>> f(NL); for (int l = 0; l < NL; l++) f[l].assign(L[l].nv, 0.0);
    vector<vector<double>> prev(NL); vector<double> wall(NL, 0.0); vector<long> nit(NL, 0); long nshift = 0;
    double t = 0; int steps = 0; if (*dumpdir) write_dump(0, 0.0); auto w0 = std::chrono::steady_clock::now();
    while (t < tf - 1e-15) {
        double dtn = std::min(steps == halfat ? 0.5 * dt : dt, tf - t), xL, yL, zL; las.position(t, xL, yL, zL);
        for (int l = 1; l < NL; l++) if (moving[l]) {            // translate box l (and re-seed) when the laser has advanced by whole elements
            const double h = L[l].ax[0][1] - L[l].ax[0][0]; long ktar = (long)std::floor((xL - back[l] - org[l]) / h + 1e-9); int m = (int)(ktar - kcur[l]);
            if (m > 0) {
                vector<int> fresh = L[l].shift_x(m, h, dorder ? &sd[l].S : nullptr); kcur[l] += m; nshift++;
                const int NXl = L[l].NXn();
                for (int v : fresh) {
                    if (seedcol) { const int i = v % NXl, src = v - (i - (NXl - 1 - m));    // front column that survived the shift
                        L[l].T[v] = L[l].T[src]; L[l].Tn[v] = L[l].Tn[src]; L[l].Ln[v] = L[l].Ln[src]; if (dorder) for (int p = 0; p < NACT; p++) sd[l].S[p][v] = sd[l].S[p][src]; continue; }
                    const auto &x = L[l].X[v]; bool ok; double Tv = interp(L[l - 1], x[0], x[1], std::min(x[2], L[l - 1].zmax), ok);
                    if (!ok) { printf("fresh node outside parent\n"); return 5; }
                    L[l].T[v] = L[l].Tn[v] = Tv; L[l].Ln[v] = L[l].Vn[v] * L[l].mat.rho * L[l].mat.Lf * L[l].mat.fL(Tv);
                    if (dorder) for (int p = 0; p < NACT; p++) sd[l].S[p][v] = interpS(L[l - 1], sd[l - 1].S[p], x[0], x[1], std::min(x[2], L[l - 1].zmax), ok); }
                relink(l); if (l + 1 < NL) relink(l + 1);
            }
        }
        for (int l = 0; l < NL; l++) {
            auto s0 = std::chrono::steady_clock::now(); Level &Lv = L[l]; prev[l] = Lv.T; if (dorder) prevS[l] = sd[l].S;
            double dtt = dtn; bool done = false;
            for (int c = 0; c <= cuts && !done; c++) {
                vector<double> Ts = Lv.T, Tns = Lv.Tn, Lns = Lv.Ln; vector<vector<double>> Ssv; if (dorder) Ssv = sd[l].S; int nsub = (int)std::lround(dtn / dtt); bool ok = true; double ts = t;
                for (int s = 0; s < nsub && ok; s++) {
                    ts += dtt;
                    if (l > 0) { double th = (ts - t) / dtn; const Link &k = link[l];
                        for (size_t i = 0; i < k.node.size(); i++) { double a = 0, b = 0; for (int q = 0; q < 8; q++) { a += k.pw[i][q] * prev[l - 1][k.pn[i][q]]; b += k.pw[i][q] * L[l - 1].T[k.pn[i][q]]; }
                            Lv.T[k.node[i]] = (1 - th) * a + th * b; } }
                    Lv.build_source(las, g, ts, dtt, f[l], dorder ? &tapes[l] : nullptr); int i1, l1; ok = newton_step(Lv, f[l], dtt, no, i1, l1); nit[l] += i1;
                    if (ok && dorder) { auto q0 = std::chrono::steady_clock::now(); int gi = 0; vector<vector<double>> sess;
                        if (l > 0) { sess.assign(NACT, vector<double>(Lv.nv, 0.0)); double th = (ts - t) / dtn; const Link &k = link[l];
                            for (int p = 0; p < NACT; p++) for (size_t i = 0; i < k.node.size(); i++) { double a = 0, b = 0; for (int q = 0; q < 8; q++) { a += k.pw[i][q] * prevS[l - 1][p][k.pn[i][q]]; b += k.pw[i][q] * sd[l - 1].S[p][k.pn[i][q]]; } sess[p][k.node[i]] = (1 - th) * a + th * b; } }
                        sens_step(Lv, sd[l], f[l], tapes[l], dtt, l > 0 ? &sess : nullptr, tamb_c + 273.15, gi); gits += gi; wsens += std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count(); }
                    if (ok) for (int i = 0; i < Lv.nv; i++) { Lv.Tn[i] = Lv.T[i]; Lv.Ln[i] = Lv.Vn[i] * Lv.mat.rho * Lv.mat.Lf * Lv.mat.fL(Lv.T[i]); }
                }
                if (ok) done = true; else { Lv.T = Ts; Lv.Tn = Tns; Lv.Ln = Lns; if (dorder) sd[l].S = Ssv; dtt *= 0.5; }
            }
            if (!done) { printf("Newton failed on level %d at t=%g\n", l + 1, t); return 2; }
            wall[l] += std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        }
        if (fb) for (int l = NL - 1; l >= 1; l--) {     // two-way feedback by injection: child -> parent (finest first, so L1 receives L3 information through L2)
            Level &P = L[l - 1], &C = L[l]; const double tol = 1e-9;
            for (int v = 0; v < P.nv; v++) {
                if (P.ess[v]) continue; const auto &x = P.X[v];
                if (x[0] < C.ax[0].front() - tol || x[0] > C.ax[0].back() + tol || x[1] < C.ax[1].front() - tol || x[1] > C.ax[1].back() + tol || x[2] < C.zmin - tol) continue;
                bool ok; double Tv = interp(C, x[0], x[1], std::min(x[2], C.zmax), ok); if (!ok) continue;
                P.T[v] = P.Tn[v] = Tv; P.Ln[v] = P.Vn[v] * P.mat.rho * P.mat.Lf * P.mat.fL(Tv);
                if (dorder) for (int p = 0; p < NACT; p++) { if (fbskip_ic && (p == P_T0 || p == P_TW)) continue; bool ok2; sd[l - 1].S[p][v] = interpS(C, sd[l].S[p], x[0], x[1], std::min(x[2], C.zmax), ok2); }
            }
        }
        t += dtn; steps++;
        double Tm = 0; for (int i = 0; i < F.nv; i++) Tm = std::max(Tm, F.T[i]);
        if (!gpeak.empty()) { const int NX = F.NXn(), NY = F.NYn(), NZ = (int)F.ax[2].size(); const double hh = hl[NL - 1];
            for (int k = 0; k < NZ; k++) for (int j = 0; j < NY; j++) for (int i = 0; i < NX; i++) {
                int ig = (int)std::lround((F.ax[0][i] - gx0) / hh), jg = (int)std::lround((F.ax[1][j] - gy0) / hh), kg = (int)std::lround((F.ax[2][k] - gz0) / hh);
                if (ig < 0 || ig >= gNX || jg < 0 || jg >= gNY || kg < 0 || kg >= gNZ) continue;
                size_t gi = ig + (size_t)gNX * (jg + (size_t)gNY * kg); double Tv = F.T[i + NX * (j + (size_t)NY * k)]; if (Tv > gpeak[gi]) { gpeak[gi] = Tv; if (dorder) for (int p = 0; p < NACT; p++) gpeak_d[p][gi] = sd[NL - 1].S[p][i + NX * (j + (size_t)NY * k)]; } } }
        if (!gpeak.empty() && NL > 1 && (steps % pkevery == 0 || t >= tf - 1e-15)) {   // wake: nodes behind/outside the fine box take the finest level that contains them
            const double hh = hl[NL - 1];
            #pragma omp parallel for schedule(static)
            for (int kg = 0; kg < gNZ; kg++) for (int jg = 0; jg < gNY; jg++) for (int ig = 0; ig < gNX; ig++) {
                double x = gx0 + ig * hh, y = gy0 + jg * hh, z = gz0 + kg * hh; size_t gi = ig + (size_t)gNX * (jg + (size_t)gNY * kg);
                if (x >= F.ax[0].front() - 1e-9 && x <= F.ax[0].back() + 1e-9 && y >= F.ax[1].front() - 1e-9 && y <= F.ax[1].back() + 1e-9 && z >= F.zmin - 1e-9) continue;   // inside the fine box: the fine value was recorded
                for (int l = NL - 2; l >= 0; l--) { bool ok = false; double N[8]; int nd[8]; int e = L[l].locate(x, y, std::min(z, L[l].zmax), N, nd); if (!e && false) {} 
                    if (e >= 0) { double Tv = 0; for (int a = 0; a < 8; a++) Tv += N[a] * L[l].T[nd[a]]; if (Tv > gpeak[gi]) { gpeak[gi] = Tv; if (dorder) for (int p = 0; p < NACT; p++) { double sv = 0; for (int a = 0; a < 8; a++) sv += N[a] * sd[l].S[p][nd[a]]; gpeak_d[p][gi] = sv; } } ok = true; } if (ok) break; }
            }
        }
        if (*dumpdir) { const bool in_scan = (t <= t_scan_end + 1e-12), last = (t >= tf - 1e-9); if (last || (in_scan && steps % dumpevery == 0) || (!in_scan && steps % dumpevery2 == 0)) write_dump(steps, t); }
        double Tp = 0; bool ok = false; for (int l = NL - 1; l >= 0 && !ok; l--) Tp = interp(L[l], probex, probey, L[l].zmax, ok);
        if (dorder) { ps << t; for (int p = 0; p < NPAR; p++) { double sv = 0; bool o2 = false; if (p < NACT) for (int l = NL - 1; l >= 0 && !o2; l--) sv = interpS(L[l], sd[l].S[p], probex, probey, L[l].zmax, o2); ps << "," << sv; } ps << "\n"; }
        pc << t << "," << Tp << "," << Tm << "," << 296.65 << "\n";
        if (steps % 50 == 0 || t >= tf - 1e-15) { printf("step %d t=%.4e xL=%.3f Tprobe=%.1f Tmax=%.1f shifts=%ld wall=%.0fs\n", steps, t, xL, Tp, Tm, nshift, std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count()); fflush(stdout); }
    }
    if (!gpeak.empty()) { std::ofstream pf(std::string(out) + "/peakT.bin", std::ios::binary); int n32 = (int)gpeak.size(), n1 = dorder ? NPAR : 0; pf.write((char *)&n32, 4); pf.write((char *)&n1, 4); pf.write((char *)gpeak.data(), 8 * gpeak.size()); for (int p = 0; p < n1; p++) pf.write((char *)gpeak_d[p].data(), 8 * gpeak.size()); }
    for (int l = 0; l < NL; l++) printf("level %d: wall %.1f s, %ld Newton iters (%.2f/step)\n", l + 1, wall[l], nit[l], (double)nit[l] / steps);
    if (dorder) printf("sensitivities: %.1f s, %ld GMRES its\n", wsens, gits);
    printf("done: %d steps, %ld box shifts, total wall %.1f s\n", steps, nshift, std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
    return 0;
}
