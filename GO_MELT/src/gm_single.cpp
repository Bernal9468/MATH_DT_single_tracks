// gm_single.cpp -- GO_MELT phase 0: single-level nominal thermal driver (non-OTI), same inputs/outputs as the baseline
// (probe.csv, peakT.bin). Must reproduce arc/results/all/case*/ before any multi-level work.
#include "gm_core.hpp"
#include "gm_sens.hpp"
#include <chrono>
using namespace gm;
int main(int argc, char **argv) {
    const char *mesh_file = "", *laser_file = "", *out = "out"; double dt = 5e-6, tf = 2.5875e-3, absorp = 0.748695, gb = 0.041029, gc = 0.182604;
    double probex = 0, probey = 0, tamb_c = 23.5, fleut = 0.23, teut = 1423.15; int gsub = 12, nseg = 20, nmax = 200, cuts = 6, toploss = 1, kq = 3, dorder = 0; double gcut = 1.5;
    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh", "mesh"); args.AddOption(&laser_file, "-laserfile", "--laser", "laser table");
    args.AddOption(&out, "-o", "--out", "output directory"); args.AddOption(&dt, "-fdt", "--dt", "time step [s]"); args.AddOption(&tf, "-tf", "--tf", "final time [s]");
    args.AddOption(&absorp, "-absorp", "--absorp", "absorptivity"); args.AddOption(&gb, "-gb", "--gb", "Goldak a=b [mm]"); args.AddOption(&gc, "-gc", "--gc", "Goldak depth c [mm]");
    args.AddOption(&gsub, "-gsub", "--gsub", "Goldak cells per axis"); args.AddOption(&gcut, "-gcut", "--gcut", "Goldak box factor"); args.AddOption(&nseg, "-nseg", "--nseg", "sub-samples per segment");
    args.AddOption(&probex, "-probex", "--probex", "probe x"); args.AddOption(&probey, "-probey", "--probey", "probe y"); args.AddOption(&tamb_c, "-tambc", "--tambc", "ambient/bottom [C]");
    args.AddOption(&fleut, "-fleut", "--fleut", "eutectic latent fraction"); args.AddOption(&teut, "-teut", "--teut", "eutectic start [K]");
    args.AddOption(&dorder, "-dorder", "--dorder", "0 = temperatures, 1 = + 20 first-order Enzyme sensitivities");
    args.AddOption(&nmax, "-nmax", "--nmax", "Newton max iters"); args.AddOption(&toploss, "-toploss", "--toploss", "1 = convection/radiation on the top face (default; the pre-2026-10-27 baseline had 0)"); args.AddOption(&kq, "-kq", "--kq", "Gauss points per direction in the conduction term (3 = baseline, 2 = Abaqus)");
    args.Parse(); if (!args.Good()) { args.PrintUsage(std::cout); return 1; }
    system((std::string("mkdir -p ") + out).c_str());
    mfem::Mesh mesh(mesh_file, 1, 1); Level L; L.toploss = toploss; L.kq = kq; L.Tamb = tamb_c + 273.15; L.mat.fleut = fleut; L.mat.Teut = teut;
    L.init(mesh); L.set_bottom_dirichlet(tamb_c + 273.15);
    Laser las; las.load(laser_file); las.absorp = absorp;
    GoldakParams g; g.af = g.ar = g.b = gb; g.c = gc; g.nq = gsub; g.cutoff = gcut; g.nseg = nseg;
    int pn = L.nearest_node(probex, probey, L.zmax), sn = L.nearest_node(probex, probey, L.zmin);
    printf("nodes %d elems %d threads %d probe node %d at (%g,%g,%g)\n", L.nv, L.ne, omp_get_max_threads(), pn, L.X[pn][0], L.X[pn][1], L.X[pn][2]);
    std::ofstream pc(std::string(out) + "/probe.csv"); pc << "time_native,layer1_top,T_max,substrate_top\n"; pc.precision(12);
    vector<double> f(L.nv); NewtonOpts no; no.maxit = nmax;
    Sens sd; SrcTape tape; vector<vector<double>> Tpk_d; long gits = 0; double wsens = 0; std::ofstream ps;
    if (dorder) { fb_set_fl2(fleut, teut); sd.init(L, tamb_c + 273.15); Tpk_d.assign(NPAR, vector<double>(L.nv, 0.0)); ps.open(std::string(out) + "/sens.csv"); ps.precision(12); ps << "time_native"; for (int p = 0; p < NPAR; p++) ps << ",dT_d" << PNAME[p] << "_row1"; ps << "\n"; }
    double t = 0; int steps = 0, tot_it = 0; auto w0 = std::chrono::steady_clock::now();
    while (t < tf - 1e-15) {
        double dtn = std::min(dt, tf - t), dtt = dtn; bool done = false; int it = 0, lt = 0;
        for (int c = 0; c <= cuts && !done; c++) {
            // substeps of size dtt covering dtn (state restored on failure)
            vector<double> Tsave = L.T, Tnsave = L.Tn, Lnsave = L.Ln; vector<vector<double>> Ssave; if (dorder) Ssave = sd.S; int nsub = (int)std::lround(dtn / dtt); bool ok = true; double ts = t;
            for (int s = 0; s < nsub && ok; s++) {
                ts += dtt; L.build_source(las, g, ts, dtt, f, dorder ? &tape : nullptr); int i1, l1; ok = newton_step(L, f, dtt, no, i1, l1); it += i1; lt += l1;
                if (ok && dorder) { auto q0 = std::chrono::steady_clock::now(); int gi = 0; sens_step(L, sd, f, tape, dtt, nullptr, tamb_c + 273.15, gi); gits += gi; wsens += std::chrono::duration<double>(std::chrono::steady_clock::now() - q0).count(); }
                if (ok) { for (int i = 0; i < L.nv; i++) { L.Tn[i] = L.T[i]; L.Ln[i] = L.Vn[i] * L.mat.rho * L.mat.Lf * L.mat.fL(L.T[i]); } }
            }
            if (ok) done = true; else { L.T = Tsave; L.Tn = Tnsave; L.Ln = Lnsave; if (dorder) sd.S = Ssave; dtt *= 0.5; }
        }
        if (!done) { printf("Newton failed at t=%g\n", t); return 2; }
        t += dtn; steps++; tot_it += it;
        double Tm = 0; for (int i = 0; i < L.nv; i++) { Tm = std::max(Tm, L.T[i]); if (L.T[i] > L.Tpeak[i]) { L.Tpeak[i] = L.T[i]; if (dorder) for (int p = 0; p < NACT; p++) Tpk_d[p][i] = sd.S[p][i]; } }
        if (dorder) { ps << t; for (int p = 0; p < NPAR; p++) ps << "," << (p < NACT ? sd.S[p][pn] : 0.0); ps << "\n"; }
        pc << t << "," << L.T[pn] << "," << Tm << "," << L.T[sn] << "\n";
        if (steps % 20 == 0 || t >= tf - 1e-15) { double w = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(); printf("step %d t=%.4e Tprobe=%.1f Tmax=%.1f newton=%d lin=%d wall=%.0fs\n", steps, t, L.T[pn], Tm, it, lt, w); fflush(stdout); }
    }
    { std::ofstream pf(std::string(out) + "/peakT.bin", std::ios::binary); int n32 = L.nv, n1 = dorder ? NPAR : 0; pf.write((char *)&n32, 4); pf.write((char *)&n1, 4); pf.write((char *)L.Tpeak.data(), 8 * L.nv); for (int p = 0; p < n1; p++) pf.write((char *)Tpk_d[p].data(), 8 * L.nv); }
    if (dorder) printf("sensitivities: %.1f s, %ld GMRES its\n", wsens, gits);
    printf("done: %d steps, %d Newton iters (%.2f/step), wall %.1f s\n", steps, tot_it, (double)tot_it / steps, std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count());
    return 0;
}
