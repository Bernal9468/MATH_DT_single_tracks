// gm_sens.hpp -- first-order sensitivities (forward / implicit-function) of the GO_MELT level solve w.r.t. the 20 thermal parameters of the
// baseline (relative scales: p -> p (1 + s_p)): k cp rho Lfus Ts Tliq hc T0 P Eta Rx Ry Rz X0 Vx TL Tw hp kp Rhop (the last three are
// identically zero on the bare plate and are carried as zeros).  Parameter derivatives of every residual term come from the Enzyme kernels
// of fb_enzyme_kernels.o (the baseline's OTI-free kernels: fb_dk1, fb_dq1, fb_nodal1, ak_src_d1); no finite differences, no OTI.
//   after a converged step:   J_exact s^{n+1} = -dR/dp|_T  -  (Mc/dt) s^n - d(L^n)/dp ,   J_exact = K + dK/dT + boundary + nodal (exact Jacobian).
#pragma once
#include "gm_core.hpp"
#include "fb_enzyme_kernels.h"
namespace gm {
constexpr int NPAR = 20, NACT = 17;          // carried parameters / parameters with a non-zero derivative on the bare plate
static const char *PNAME[NPAR] = {"k","cp","rho","Lfus","Ts","Tliq","hc","T0","P","Eta","Rx","Ry","Rz","X0","Vx","TL","Tw","hp","kp","Rhop"};
enum { P_K = 0, P_CP, P_RHO, P_LFUS, P_TS, P_TL, P_HC, P_T0, P_P, P_ETA, P_RX, P_RY, P_RZ, P_X0, P_VX, P_TLASER, P_TW };

struct Sens {
    vector<vector<double>> S;                // [p][node] current s^n
    double Tsub = 296.65;
    // IC seeds of the baseline: T0 on the top-layer nodes, Tw on the rest.  The baseline's top layer is one node thick on its 20 um mesh, i.e. the seed carries the
    // heat content of a layer of thickness href/2; a coarser level represents that layer by its own, thicker, top node, so the seed is scaled by r = href/hz (hz = top element
    // thickness of the level) and the complement goes to Tw (S[T0] + S[Tw] = Tamb everywhere at t = 0, as in the baseline).  r = 1 on the 20 um level.
    void init(const Level &L, double Tamb, double href = 0.02) {
        S.assign(NACT, vector<double>(L.nv, 0.0));
        const int nz = (int)L.ax[2].size(); const double hz = L.ax[2][nz - 1] - L.ax[2][nz - 2], r = std::min(1.0, href / hz);
        for (int v = 0; v < L.nv; v++) { const bool top = std::fabs(L.X[v][2] - L.zmax) < 1e-9; if (top) { S[P_T0][v] = Tamb * r; S[P_TW][v] = Tamb * (1.0 - r); } else S[P_TW][v] = Tamb; }
        // (child levels overwrite their Dirichlet nodes from the parent every step; their top-ring nodes keep the thin-layer seed so the t = 0 dump is right.  The root level has no ess node in its top layer.)
    }
};

// restarted GMRES(m) on the free dofs, left Jacobi preconditioner; A = L.A (exact Jacobian), unknown masked by L.ess
inline int gmres(const Level &L, const vector<double> &b, vector<double> &x, double rtol, int m = 30, int maxrest = 20) {
    const int n = L.nv; vector<double> dinv(n), r(n), w(n), z(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) dinv[i] = L.ess[i] ? 0.0 : 1.0 / L.A[L.pos(i, i)];
    auto Aop = [&](const vector<double> &u, vector<double> &y) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) { double s = 0; if (!L.ess[i]) for (int k = L.I[i]; k < L.I[i + 1]; k++) if (!L.ess[L.J[k]]) s += L.A[k] * u[L.J[k]]; y[i] = s; } };
    auto dot = [&](const vector<double> &a, const vector<double> &c) { double s = 0;
        #pragma omp parallel for reduction(+ : s) schedule(static)
        for (int i = 0; i < n; i++) s += a[i] * c[i]; return s; };
    std::fill(x.begin(), x.end(), 0.0); int total = 0;
    vector<double> bp(n); for (int i = 0; i < n; i++) bp[i] = dinv[i] * (L.ess[i] ? 0.0 : b[i]); const double bn = std::sqrt(dot(bp, bp)); if (bn == 0) return 0;
    for (int rs = 0; rs < maxrest; rs++) {
        Aop(x, w); for (int i = 0; i < n; i++) r[i] = dinv[i] * ((L.ess[i] ? 0.0 : b[i]) - w[i]);
        double beta = std::sqrt(dot(r, r)); if (beta <= rtol * bn) break;
        vector<vector<double>> V(m + 1, vector<double>(n)); vector<vector<double>> H(m + 1, vector<double>(m, 0.0)); vector<double> cs(m), sn(m), g(m + 1, 0.0);
        for (int i = 0; i < n; i++) V[0][i] = r[i] / beta; g[0] = beta; int j = 0;
        for (; j < m; j++) {
            Aop(V[j], w); for (int i = 0; i < n; i++) z[i] = dinv[i] * w[i];
            for (int k = 0; k <= j; k++) { H[k][j] = dot(z, V[k]); for (int i = 0; i < n; i++) z[i] -= H[k][j] * V[k][i]; }
            H[j + 1][j] = std::sqrt(dot(z, z)); if (H[j + 1][j] > 0) for (int i = 0; i < n; i++) V[j + 1][i] = z[i] / H[j + 1][j];
            for (int k = 0; k < j; k++) { double t = cs[k] * H[k][j] + sn[k] * H[k + 1][j]; H[k + 1][j] = -sn[k] * H[k][j] + cs[k] * H[k + 1][j]; H[k][j] = t; }
            double den = std::hypot(H[j][j], H[j + 1][j]); cs[j] = H[j][j] / den; sn[j] = H[j + 1][j] / den; H[j][j] = den; H[j + 1][j] = 0.0;
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j]; total++;
            if (std::fabs(g[j + 1]) <= rtol * bn) { j++; break; }
        }
        vector<double> y(j, 0.0); for (int k = j - 1; k >= 0; k--) { double s = g[k]; for (int l = k + 1; l < j; l++) s -= H[k][l] * y[l]; y[k] = s / H[k][k]; }
        for (int k = 0; k < j; k++) for (int i = 0; i < n; i++) x[i] += y[k] * V[k][i];
    }
    return total;
}

// One sensitivity step on a level.  Call right after newton_step converged (L.T = T^{n+1}, L.Tn = T^n, L.Ln = L^n) and BEFORE Tn/Ln are advanced.
// sess[p][i]: prescribed s at the Dirichlet nodes (nullptr: zero, and Tw = Tsub on the bottom nodes of the plate).
inline void sens_step(Level &L, Sens &sn, const vector<double> &f, const SrcTape &tape, double dt, const vector<vector<double>> *sess, double Tsub, int &gm_its) {
    const int n = L.nv; const double idt = 1.0 / dt; vector<vector<double>> R(NACT, vector<double>(n, 0.0));
    // (c) capacitance + latent heat (carries s^n through T^n and L^n): rhs = -d nodal / ds at fixed T^{n+1}
    int slot[NACT]; for (int p = 0; p < NACT; p++) slot[p] = (p <= P_T0) ? p : -1;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (L.ess[i]) continue; double pd[NACT], rr[NACT]; for (int p = 0; p < NACT; p++) pd[p] = sn.S[p][i];
        fb_nodal1(L.T[i], L.Tn[i], pd, NACT, slot, 1.0, 0.6, L.Vn[i], idt, IN718_T_SOLIDUS, IN718_T_LIQUIDUS, rr);
        for (int p = 0; p < NACT; p++) R[p][i] = rr[p];
    }
    // (a) conduction: only k enters explicitly; rhs_k[a] = - sum_q w dk/dS_k (gradN_a . gradT)
    {
        const int nq1 = L.kq; const double gx[3] = {nq1 == 2 ? -0.5773502691896257 : -0.7745966692414834, nq1 == 2 ? 0.5773502691896257 : 0.0, 0.7745966692414834};
        const double gwt[3] = {nq1 == 2 ? 1.0 : 5.0 / 9.0, nq1 == 2 ? 1.0 : 8.0 / 9.0, 5.0 / 9.0};
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < L.ne; e++) {
            double h[3] = {L.hi[e][0] - L.lo[e][0], L.hi[e][1] - L.lo[e][1], L.hi[e][2] - L.lo[e][2]}, Te[8], wv = h[0] * h[1] * h[2] / 8.0;
            for (int a = 0; a < 8; a++) Te[a] = L.T[L.en[e][a]];
            for (int qz = 0; qz < nq1; qz++) for (int qy = 0; qy < nq1; qy++) for (int qx = 0; qx < nq1; qx++) {
                const int ii[3] = {qx, qy, qz}; double xi[3]; for (int d = 0; d < 3; d++) xi[d] = (nq1 == 2) ? (ii[d] ? gx[1] : gx[0]) : gx[ii[d]];
                const double wq = (nq1 == 2) ? 1.0 : gwt[qx] * gwt[qy] * gwt[qz];
                double gN[8][3], Tq = 0, gT[3] = {0, 0, 0};
                for (int a = 0; a < 8; a++) {
                    double f0 = 1 + L.sg[e][a * 3] * xi[0], f1 = 1 + L.sg[e][a * 3 + 1] * xi[1], f2 = 1 + L.sg[e][a * 3 + 2] * xi[2];
                    Tq += 0.125 * f0 * f1 * f2 * Te[a];
                    gN[a][0] = L.sg[e][a * 3] * 0.125 * f1 * f2 * 2.0 / h[0]; gN[a][1] = L.sg[e][a * 3 + 1] * 0.125 * f0 * f2 * 2.0 / h[1]; gN[a][2] = L.sg[e][a * 3 + 2] * 0.125 * f0 * f1 * 2.0 / h[2];
                }
                for (int a = 0; a < 8; a++) for (int d = 0; d < 3; d++) gT[d] += gN[a][d] * Te[a];
                double dkK, dkKP; fb_dk1(Tq, 1.0, 0.2, &dkK, &dkKP);
                for (int a = 0; a < 8; a++) { double v = -wv * wq * dkK * (gN[a][0] * gT[0] + gN[a][1] * gT[1] + gN[a][2] * gT[2]);
                    #pragma omp atomic
                    R[P_K][L.en[e][a]] += v; }
            }
        }
    }
    // (b) boundary film/radiation: hc and T0 (ambient) enter q explicitly
    for (size_t q = 0; q < L.faces.size(); q++) {
        const auto &F = L.faces[q]; double w = F.area / 4.0;
        for (int gq = 0; gq < 4; gq++) {
            double u = 0.5 + ((gq & 1) ? 0.2886751345948129 : -0.2886751345948129), v = 0.5 + ((gq & 2) ? 0.2886751345948129 : -0.2886751345948129);
            double Nf[4] = {(1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v}, Tq = 0; for (int a = 0; a < 4; a++) Tq += Nf[a] * L.T[F.v[a]];
            double dhc, dt0, dhp; fb_dq1(Tq, 1.0, L.h_conv, L.h_conv * 0.1, L.emis * L.sigma, L.Tamb, &dhc, &dt0, &dhp);
            for (int a = 0; a < 4; a++) { R[P_HC][F.v[a]] -= dhc * Nf[a] * w; R[P_T0][F.v[a]] -= dt0 * Nf[a] * w; }
        }
    }
    // (d) P, Eta multiply the deposited power;  (e) Goldak radii, X0, Vx, tL through the AD source kernel
    for (int i = 0; i < n; i++) { R[P_P][i] += f[i]; R[P_ETA][i] += f[i]; }
    {
        const double scl = 1.0 / (tape.t1 - tape.t0); vector<vector<double>> D(6, vector<double>(n, 0.0)); const int nsub = (int)tape.subs.size(); const SrcCell *cells = tape.cells.data();
        #pragma omp parallel
        { std::vector<double> buf(SRCK_MAXOUT);
          #pragma omp for schedule(dynamic, 4)
          for (int si = 0; si < nsub; si++) {
              const SrcSub &r = tape.subs[si]; const int nout = r.ncell * 8; const SrcCell *rc = cells + r.first_cell;
              for (int j = 0; j < 6; j++) { ak_src_d1(&r, cells, j, buf.data());
                  for (int k = 0; k < nout; k++) { const double v = buf[k]; if (v == 0.0) continue;
                      #pragma omp atomic
                      D[j][rc[k / 8].dofs[k % 8]] += v; } }
          } }
        for (int j = 0; j < 6; j++) for (int i = 0; i < n; i++) R[P_RX + j][i] += D[j][i] * scl;
    }
    // Dirichlet rows and right-hand side, exact Jacobian, solves
    vector<double> rt(n); L.eval(L.T, f, dt, rt, true, true);
    vector<vector<double>> Snew(NACT, vector<double>(n, 0.0));
    for (int p = 0; p < NACT; p++) {
        vector<double> se(n, 0.0);
        for (int i = 0; i < n; i++) if (L.ess[i]) se[i] = sess ? (*sess)[p][i] : ((p == P_TW && std::fabs(L.X[i][2] - L.zmin) < 1e-9) ? Tsub : 0.0);
        vector<double> b(n, 0.0);
        for (int i = 0; i < n; i++) if (!L.ess[i]) { double s = R[p][i]; for (int k = L.I[i]; k < L.I[i + 1]; k++) if (L.ess[L.J[k]]) s -= L.A[k] * se[L.J[k]]; b[i] = s; }
        vector<double> x(n, 0.0); gm_its += gmres(L, b, x, 1e-11);
        for (int i = 0; i < n; i++) Snew[p][i] = L.ess[i] ? se[i] : x[i];
    }
    sn.S.swap(Snew);
}
}   // namespace gm
