// ============================================================================
//  cuDSS backend of the CPU/GPU solver switch (bare_plate_single_layer, 2026-10-24)
//
//  NVIDIA cuDSS (GPU sparse direct solver) behind a plain C++ interface: no CUDA types here, so the drivers include it
//  with g++. Host CSR in, host solutions out (the drivers' matrices live on the host); the copies are part of every call
//  and of every time the switch measures. Implementation: cudss_backend.cpp (CUDA 12 runtime + libcudss, toolchain in
//  MFEM/toolchain/cuda12). Every call returns false on any CUDA/cuDSS error (message in Error()); the switch then falls
//  back to PARDISO for good.
// ============================================================================
#pragma once
#include <memory>
#include <string>

class CudssSolver {
public:
    enum Kind { GENERAL, SYMMETRIC_UPPER, SPD_UPPER };   // GENERAL: full CSR (LU); *_UPPER: upper triangle (LDL^T / Cholesky)
    CudssSolver();
    ~CudssSolver();
    static bool DeviceAvailable();
    static size_t FreeDeviceMemory();
    // analysis + first numerical factorisation of a new pattern. base = 0 or 1 (index base of ia/ja).
    // hybrid: keep the factors in host memory (CUDSS_CONFIG_HYBRID_MEMORY_MODE), for systems larger than the GPU memory.
    bool Analyze(int n, const int *ia, const int *ja, const double *a, Kind kind, int base, bool hybrid = false);
    bool Refactor(const double *a);                     // same pattern, new values
    bool Solve(const double *b, double *x, int nrhs);   // b, x: n x nrhs column-major, host
    int N() const;
    const std::string &Error() const;
private:
    struct Impl; std::unique_ptr<Impl> p;
};
