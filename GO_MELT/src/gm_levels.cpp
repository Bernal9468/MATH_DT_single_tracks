// gm_levels.cpp -- GO_MELT phase 1: nested multi-level thermal solve (L1 coarse plate -> L2 meso box -> L3 fine box), IN718 single tracks.
// Every level solves the full nonlinear heat equation with its own source deposition; a child's Dirichlet boundary values
// (all box faces except the top surface z = ztop) are interpolated from its parent (trilinear, linear in time inside a step).
// One-way coupling (parent -> child); the child does not feed back yet (phase 2). Static boxes (phase 3 moves them).
#include "gm_core.hpp"
#include <chrono>
using namespace gm;
struct Link { vector<int> node; vector<array<int, 8>> pn; vector<array<double, 8>> pw; };   // child boundary node -> parent interpolation
static Link make_link(const Level &child, const Level &parent, bool box_top_free) {
    Link k; const double tol = 1e-9;
    for (int v = 0; v < child.nv; v++) {
        const auto &x = child.X[v];
        bool b = std::fabs(x[0] - child.ax[0].front()) < tol || std::fabs(x[0] - child.ax[0].back()) < tol ||
                 std::fabs(x[1] - child.ax[1].front()) < tol || std::fabs(x[1] - child.ax[1].back()) < tol || std::fabs(x[2] - child.zmin) < tol;
        if (!b) continue;
        double N[8]; int nd[8]; int e = parent.locate(x[0], x[1], std::min(x[2], parent.zmax), N, nd);
        if (e < 0) { printf("child node %d (%g,%g,%g) outside the parent level\n", v, x[0], x[1], x[2]); exit(3); }
        k.node.push_back(v); array<int, 8> a; array<double, 8> w; for (int i = 0; i < 8; i++) { a[i] = nd[i]; w[i] = N[i]; } k.pn.push_back(a); k.pw.push_back(w);
    }
    return k;
}
int main(int argc, char **argv) {
    const char *m1 = "", *m2 = "", *m3 = "", *laser_file = "", *out = "out"; double dt = 5e-6, tf = 2.5875e-3, absorp = 0.748695, gb = 0.041029, gc = 0.182604;
    double probex = 0, probey = 0, tamb_c = 23.5, fleut = 0.23, teut = 1423.15, gcut = 1.5; int gsub = 12, nseg = 20, nmax = 200, cuts = 6, toploss = 1, kq = 3;
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&m1, "-l1", "--l1", "level-1 mesh (whole plate)"); args.AddOption(&m2, "-l2", "--l2", "level-2 mesh (box, optional)"); args.AddOption(&m3, "-l3", "--l3", "level-3 mesh (box, optional)");
    args.AddOption(&laser_file, "-laserfile", "--laser", "laser table"); args.AddOption(&out, "-o", "--out", "output directory"); args.AddOption(&dt, "-fdt", "--dt", "time step [s]");
    args.AddOption(&tf, "-tf", "--tf", "final time [s]"); args.AddOption(&absorp, "-absorp", "--absorp", "absorptivity"); args.AddOption(&gb, "-gb", "--gb", "Goldak a=b [mm]");
    args.AddOption(&gc, "-gc", "--gc", "Goldak depth c [mm]"); args.AddOption(&gsub, "-gsub", "--gsub", "Goldak cells per axis"); args.AddOption(&gcut, "-gcut", "--gcut", "Goldak box factor");
    args.AddOption(&nseg, "-nseg", "--nseg", "sub-samples per segment"); args.AddOption(&probex, "-probex", "--probex", "probe x"); args.AddOption(&probey, "-probey", "--probey", "probe y");
    args.AddOption(&tamb_c, "-tambc", "--tambc", "ambient/bottom [C]"); args.AddOption(&fleut, "-fleut", "--fleut", "eutectic fraction"); args.AddOption(&teut, "-teut", "--teut", "eutectic start [K]");
    args.AddOption(&nmax, "-nmax", "--nmax", "Newton max iters"); args.AddOption(&toploss, "-toploss", "--toploss", "top-face convection/radiation"); args.AddOption(&kq, "-kq", "--kq", "Gauss points");
    args.Parse(); if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
    system((std::string("mkdir -p ") + out).c_str());
    vector<std::string> files = {m1}; if (*m2) files.push_back(m2); if (*m3) files.push_back(m3);
    const int NL = (int)files.size(); vector<Level> L(NL);
    for (int l = 0; l < NL; l++) {
        mfem::Mesh mesh(files[l].c_str(), 1, 1); L[l].toploss = toploss; L[l].kq = kq; L[l].Tamb = tamb_c + 273.15; L[l].mat.fleut = fleut; L[l].mat.Teut = teut;
        L[l].init(mesh); if (l == 0) L[l].set_bottom_dirichlet(tamb_c + 273.15);
        printf("level %d: %d nodes %d elems, x [%g,%g] y [%g,%g] z [%g,%g]\n", l + 1, L[l].nv, L[l].ne, L[l].ax[0].front(), L[l].ax[0].back(), L[l].ax[1].front(), L[l].ax[1].back(), L[l].zmin, L[l].zmax);
    }
    vector<Link> link(NL);
    for (int l = 1; l < NL; l++) { link[l] = make_link(L[l], L[l - 1], true); for (int v : link[l].node) L[l].ess[v] = 1; printf("level %d: %zu Dirichlet nodes from level %d\n", l + 1, link[l].node.size(), l); }
    Laser las; las.load(laser_file); las.absorp = absorp;
    GoldakParams g; g.af = g.ar = g.b = gb; g.c = gc; g.nq = gsub; g.cutoff = gcut; g.nseg = nseg;
    Level &F = L[NL - 1];
    int pn = F.nearest_node(probex, probey, F.zmax);
    std::ofstream pc(std::string(out) + "/probe.csv"); pc << "time_native,layer1_top,T_max,substrate_top\n"; pc.precision(12);
    NewtonOpts no; no.maxit = nmax; vector<vector<double>> f(NL); for (int l = 0; l < NL; l++) f[l].assign(L[l].nv, 0.0);
    vector<vector<double>> prev(NL); vector<double> wall(NL, 0.0); vector<long> nit(NL, 0);
    double t = 0; int steps = 0; auto w0 = std::chrono::steady_clock::now();
    while (t < tf - 1e-15) {
        double dtn = std::min(dt, tf - t);
        for (int l = 0; l < NL; l++) {
            auto s0 = std::chrono::steady_clock::now(); Level &Lv = L[l]; prev[l] = Lv.T;
            double dtt = dtn; bool done = false;
            for (int c = 0; c <= cuts && !done; c++) {
                vector<double> Ts = Lv.T, Tns = Lv.Tn, Lns = Lv.Ln; int nsub = (int)std::lround(dtn / dtt); bool ok = true; double ts = t;
                for (int s = 0; s < nsub && ok; s++) {
                    ts += dtt;
                    if (l > 0) { double th = (ts - t) / dtn; const Link &k = link[l];
                        for (size_t i = 0; i < k.node.size(); i++) { double a = 0, b = 0; for (int q = 0; q < 8; q++) { a += k.pw[i][q] * prev[l - 1][k.pn[i][q]]; b += k.pw[i][q] * L[l - 1].T[k.pn[i][q]]; }
                            Lv.T[k.node[i]] = (1 - th) * a + th * b; } }
                    Lv.build_source(las, g, ts, dtt, f[l]); int i1, l1; ok = newton_step(Lv, f[l], dtt, no, i1, l1); nit[l] += i1;
                    if (ok) for (int i = 0; i < Lv.nv; i++) { Lv.Tn[i] = Lv.T[i]; Lv.Ln[i] = Lv.Vn[i] * Lv.mat.rho * Lv.mat.Lf * Lv.mat.fL(Lv.T[i]); }
                }
                if (ok) done = true; else { Lv.T = Ts; Lv.Tn = Tns; Lv.Ln = Lns; dtt *= 0.5; }
            }
            if (!done) { printf("Newton failed on level %d at t=%g\n", l + 1, t); return 2; }
            wall[l] += std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
        }
        t += dtn; steps++;
        double Tm = 0; for (int i = 0; i < F.nv; i++) { Tm = std::max(Tm, F.T[i]); F.Tpeak[i] = std::max(F.Tpeak[i], F.T[i]); }
        pc << t << "," << F.T[pn] << "," << Tm << "," << F.T[F.nearest_node(probex, probey, F.zmin)] << "\n";
        if (steps % 50 == 0 || t >= tf - 1e-15) { printf("step %d t=%.4e Tprobe=%.1f Tmax(L%d)=%.1f wall=%.0fs\n", steps, t, F.T[pn], NL, Tm, std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count()); fflush(stdout); }
    }
    { std::ofstream pf(std::string(out) + "/peakT.bin", std::ios::binary); int n32 = F.nv, n1 = 0; pf.write((char *)&n32, 4); pf.write((char *)&n1, 4); pf.write((char *)F.Tpeak.data(), 8 * F.nv); }
    for (int l = 0; l < NL; l++) printf("level %d: wall %.1f s, %ld Newton iters (%.2f/step)\n", l + 1, wall[l], nit[l], (double)nit[l] / steps);
    printf("done: %d steps, total wall %.1f s\n", steps, std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
    return 0;
}
