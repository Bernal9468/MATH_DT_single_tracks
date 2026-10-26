// ============================================================================
//  CPU/GPU solver switch (bare_plate_single_layer, 2026-10-24): decides, per linear system, whether PARDISO (CPU) or
//  cuDSS (GPU) does its factorisations and solves, by measuring both on the system's own matrices. Header-only, no CUDA.
//
//  A "step" is one numerical (re)factorisation plus every solve until the next one (as the time loop uses a system).
//    1. Exclusions: SOLVER=pardiso, no GPU, n < SOLVER_MIN_N, or an earlier GPU error -> PARDISO, no calibration.
//    2. Calibration: after a new pattern (analysis), SOLVER_WARMUP steps are discarded (first GPU calls, analysis), then for
//       SOLVER_CAL_STEPS steps BOTH solvers run on the same matrix and right-hand sides; the PARDISO result is the one used,
//       the cuDSS result is compared with it. Measured: refactorisation + solves, cuDSS including the host<->GPU copies.
//    3. Decision: GPU only if its total over the window is below (1 - SOLVER_MARGIN) x PARDISO's and every solution agreed
//       to SOLVER_TOL (max |x_gpu - x_cpu| / max |x_cpu|); otherwise PARDISO. Sticky.
//    4. Re-evaluation: a new pattern restarts the calibration; so does a slowdown of the chosen solver (mean of the last
//       SOLVER_DRIFT_N steps > SOLVER_DRIFT x its calibrated mean). Any GPU error -> PARDISO for the rest of the run.
//  SOLVER=cudss forces cuDSS (no calibration; still falls back to PARDISO on error).
//  Environment: SOLVER=auto|pardiso|cudss (auto), SOLVER_MIN_N (2000), SOLVER_WARMUP (1), SOLVER_CAL_STEPS (8),
//               SOLVER_MARGIN (0.15), SOLVER_TOL (1e-8), SOLVER_DRIFT (1.5), SOLVER_DRIFT_N (20), SOLVER_VERBOSE (1).
// ============================================================================
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>

class SolverSwitch {
public:
    enum Mode { AUTO, FORCE_CPU, FORCE_GPU };
    enum State { CPU_ONLY, CAL, CPU, GPU };

    explicit SolverSwitch(std::string name) : name_(std::move(name)) {
        const char *m = std::getenv("SOLVER");
        mode_ = !m ? AUTO : std::string(m) == "pardiso" ? FORCE_CPU : std::string(m) == "cudss" ? FORCE_GPU : AUTO;
        min_n_ = envi("SOLVER_MIN_N", 2000); warm_ = envi("SOLVER_WARMUP", 1); cal_ = envi("SOLVER_CAL_STEPS", 8);
        margin_ = envd("SOLVER_MARGIN", 0.15); tol_ = envd("SOLVER_TOL", 1e-8); drift_ = envd("SOLVER_DRIFT", 1.5);
        drift_n_ = envi("SOLVER_DRIFT_N", 20); verbose_ = envi("SOLVER_VERBOSE", 1);
    }
    static double Now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

    // should a GPU solver be set up for a system of size n?
    bool GpuWanted(int n, bool device_available) const { return mode_ != FORCE_CPU && !gpu_dead_ && device_available && n >= min_n_; }
    // new pattern: gpu_ready = the GPU solver analysed and factorised it successfully
    void Start(int n, long nnz, bool gpu_ready) {
        n_ = n; nnz_ = nnz; step_ = -1; closed_ = 0; tc_ = tg_ = 0; sc_ = sg_ = 0; worst_diff_ = 0; recent_.clear();
        state_ = !gpu_ready ? CPU_ONLY : (mode_ == FORCE_GPU ? GPU : CAL);
        if (verbose_ && gpu_ready) printf("  [solver switch] %s: n=%d nnz=%ld -> %s\n", name_.c_str(), n, nnz,
                                         state_ == GPU ? "cuDSS (forced by SOLVER=cudss)" : "calibrating PARDISO vs cuDSS");
        fflush(stdout);
        BeginStep();
    }
    bool RunCpu() const { return state_ != GPU; }          // PARDISO factorises/solves in this step
    bool RunGpu() const { return state_ == CAL || state_ == GPU; }
    bool Calibrating() const { return state_ == CAL; }
    bool UseGpuResult() const { return state_ == GPU; }
    State GetState() const { return state_; }

    // a (re)factorisation opens a new step and closes the previous one
    void BeginStep() {
        if (step_ >= 0) close_step();
        ++step_; sc_ = sg_ = 0;
    }
    void AddCpu(double s) { sc_ += s; }
    void AddGpu(double s) { sg_ += s; }
    void SolutionDiff(double rel) { if (step_ >= warm_) worst_diff_ = std::max(worst_diff_, rel); }
    // GPU error: PARDISO from now on. Returns true if the GPU was the active solver (the caller must refactor with PARDISO).
    bool GpuFailed(const std::string &why) {
        const bool was = (state_ == GPU);
        gpu_dead_ = true; state_ = CPU_ONLY;
        printf("  [solver switch] %s: cuDSS error (%s) -> PARDISO for the rest of the run\n", name_.c_str(), why.c_str()); fflush(stdout);
        return was;
    }
    static double RelDiff(const double *x, const double *y, size_t m) {
        double d = 0, s = 0; for (size_t i = 0; i < m; i++) { d = std::max(d, std::fabs(x[i] - y[i])); s = std::max(s, std::fabs(y[i])); }
        return s > 0 ? d / s : d;
    }

private:
    static int envi(const char *k, int d) { const char *e = std::getenv(k); return e ? std::atoi(e) : d; }
    static double envd(const char *k, double d) { const char *e = std::getenv(k); return e ? std::atof(e) : d; }
    void close_step() {
        if (state_ == CAL) {
            if (step_ >= warm_) { tc_ += sc_; tg_ += sg_; ++closed_; }
            if (closed_ >= cal_) decide();
        } else if (state_ == CPU || state_ == GPU) {
            recent_.push_back(state_ == GPU ? sg_ : sc_);
            if ((int)recent_.size() > drift_n_) recent_.pop_front();
            if ((int)recent_.size() == drift_n_ && mode_ == AUTO) {
                double m = 0; for (double v : recent_) m += v; m /= drift_n_;
                const double ref = (state_ == GPU ? tg_ : tc_) / std::max(1, closed_);
                if (ref > 0 && m > drift_ * ref) {
                    if (verbose_) printf("  [solver switch] %s: %s slowed down (%.3g s/step vs %.3g calibrated) -> recalibrating\n",
                                         name_.c_str(), state_ == GPU ? "cuDSS" : "PARDISO", m, ref);
                    state_ = CAL; closed_ = 0; tc_ = tg_ = 0; worst_diff_ = 0; recent_.clear(); step_ = 0;
                }
            }
        }
    }
    void decide() {
        const bool agree = worst_diff_ <= tol_;
        state_ = (agree && tg_ < (1.0 - margin_) * tc_) ? GPU : CPU;
        if (verbose_) printf("  [solver switch] %s: n=%d, %d steps: PARDISO %.4g s/step, cuDSS %.4g s/step, max solution diff %.1e -> %s\n",
                             name_.c_str(), n_, closed_, tc_ / closed_, tg_ / closed_, worst_diff_,
                             state_ == GPU ? "cuDSS (GPU)" : (agree ? "PARDISO (CPU)" : "PARDISO (CPU; solutions disagree)"));
        fflush(stdout);
    }
    std::string name_; Mode mode_; State state_ = CPU_ONLY;
    int min_n_, warm_, cal_, drift_n_, verbose_; double margin_, tol_, drift_;
    int n_ = 0; long nnz_ = 0; int step_ = -1, closed_ = 0; double tc_ = 0, tg_ = 0, sc_ = 0, sg_ = 0, worst_diff_ = 0;
    bool gpu_dead_ = false; std::deque<double> recent_;
};
