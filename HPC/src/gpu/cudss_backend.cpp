// cuDSS backend of the CPU/GPU solver switch: see cudss_backend.h.
#include "cudss_backend.h"
#include <cuda_runtime_api.h>
#include <cudss.h>
#include <cstdio>
#include <vector>

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { char m_[256]; snprintf(m_, sizeof(m_), "CUDA %s at %s", cudaGetErrorString(e_), #x); p->err = m_; return false; } } while (0)
#define DS(x) do { cudssStatus_t s_ = (x); if (s_ != CUDSS_STATUS_SUCCESS) { char m_[256]; snprintf(m_, sizeof(m_), "cuDSS status %d at %s", (int)s_, #x); p->err = m_; return false; } } while (0)

struct CudssSolver::Impl {
    int n = 0; long nnz = 0; int nrhs_cap = 0;
    int *d_ia = nullptr, *d_ja = nullptr; double *d_a = nullptr, *d_b = nullptr, *d_x = nullptr;
    cudssHandle_t h = nullptr; cudssConfig_t cfg = nullptr; cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr, B = nullptr, X = nullptr; int bx_nrhs = 0;
    std::string err;
    void release() {
        if (A) cudssMatrixDestroy(A); if (B) cudssMatrixDestroy(B); if (X) cudssMatrixDestroy(X);
        if (data) cudssDataDestroy(h, data); if (cfg) cudssConfigDestroy(cfg); if (h) cudssDestroy(h);
        cudaFree(d_ia); cudaFree(d_ja); cudaFree(d_a); cudaFree(d_b); cudaFree(d_x);
        A = B = X = nullptr; data = nullptr; cfg = nullptr; h = nullptr; d_ia = d_ja = nullptr; d_a = d_b = d_x = nullptr;
        nrhs_cap = bx_nrhs = 0;
    }
};

CudssSolver::CudssSolver() : p(new Impl) {}
CudssSolver::~CudssSolver() { p->release(); }
int CudssSolver::N() const { return p->n; }
const std::string &CudssSolver::Error() const { return p->err; }

bool CudssSolver::DeviceAvailable() { int n = 0; return cudaGetDeviceCount(&n) == cudaSuccess && n > 0; }
size_t CudssSolver::FreeDeviceMemory() { size_t f = 0, t = 0; return cudaMemGetInfo(&f, &t) == cudaSuccess ? f : 0; }

bool CudssSolver::Analyze(int n, const int *ia, const int *ja, const double *a, Kind kind, int base, bool hybrid) {
    p->release(); p->err.clear();
    p->n = n; p->nnz = ia[n] - base;
    CU(cudaMalloc(&p->d_ia, sizeof(int) * (n + 1))); CU(cudaMalloc(&p->d_ja, sizeof(int) * p->nnz)); CU(cudaMalloc(&p->d_a, sizeof(double) * p->nnz));
    CU(cudaMemcpy(p->d_ia, ia, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    CU(cudaMemcpy(p->d_ja, ja, sizeof(int) * p->nnz, cudaMemcpyHostToDevice));
    CU(cudaMemcpy(p->d_a, a, sizeof(double) * p->nnz, cudaMemcpyHostToDevice));
    DS(cudssCreate(&p->h)); DS(cudssConfigCreate(&p->cfg)); DS(cudssDataCreate(p->h, &p->data));
    if (hybrid) { int one = 1; DS(cudssConfigSet(p->cfg, CUDSS_CONFIG_HYBRID_MEMORY_MODE, &one, sizeof(one))); }
    const cudssMatrixType_t mt = kind == GENERAL ? CUDSS_MTYPE_GENERAL : kind == SPD_UPPER ? CUDSS_MTYPE_SPD : CUDSS_MTYPE_SYMMETRIC;
    const cudssMatrixViewType_t mv = kind == GENERAL ? CUDSS_MVIEW_FULL : CUDSS_MVIEW_UPPER;
    DS(cudssMatrixCreateCsr(&p->A, n, n, p->nnz, p->d_ia, nullptr, p->d_ja, p->d_a, CUDA_R_32I, CUDA_R_32I, CUDA_R_64F, mt, mv,
                            base == 0 ? CUDSS_BASE_ZERO : CUDSS_BASE_ONE));
    p->nrhs_cap = 1; CU(cudaMalloc(&p->d_b, sizeof(double) * n)); CU(cudaMalloc(&p->d_x, sizeof(double) * n));
    DS(cudssMatrixCreateDn(&p->B, n, 1, n, p->d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    DS(cudssMatrixCreateDn(&p->X, n, 1, n, p->d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)); p->bx_nrhs = 1;
    DS(cudssExecute(p->h, CUDSS_PHASE_ANALYSIS, p->cfg, p->data, p->A, p->X, p->B));
    DS(cudssExecute(p->h, CUDSS_PHASE_FACTORIZATION, p->cfg, p->data, p->A, p->X, p->B));
    CU(cudaDeviceSynchronize());
    return true;
}

bool CudssSolver::Refactor(const double *a) {
    if (!p->A) { p->err = "Refactor before Analyze"; return false; }
    CU(cudaMemcpy(p->d_a, a, sizeof(double) * p->nnz, cudaMemcpyHostToDevice));
    DS(cudssExecute(p->h, CUDSS_PHASE_REFACTORIZATION, p->cfg, p->data, p->A, p->X, p->B));
    CU(cudaDeviceSynchronize());
    return true;
}

bool CudssSolver::Solve(const double *b, double *x, int nrhs) {
    if (!p->A) { p->err = "Solve before Analyze"; return false; }
    const int n = p->n;
    if (nrhs > p->nrhs_cap) {
        cudaFree(p->d_b); cudaFree(p->d_x); p->d_b = p->d_x = nullptr;
        CU(cudaMalloc(&p->d_b, sizeof(double) * (size_t)n * nrhs)); CU(cudaMalloc(&p->d_x, sizeof(double) * (size_t)n * nrhs));
        p->nrhs_cap = nrhs; p->bx_nrhs = 0;
    }
    if (nrhs != p->bx_nrhs) {
        if (p->B) cudssMatrixDestroy(p->B); if (p->X) cudssMatrixDestroy(p->X); p->B = p->X = nullptr;
        DS(cudssMatrixCreateDn(&p->B, n, nrhs, n, p->d_b, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));
        DS(cudssMatrixCreateDn(&p->X, n, nrhs, n, p->d_x, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR)); p->bx_nrhs = nrhs;
    }
    CU(cudaMemcpy(p->d_b, b, sizeof(double) * (size_t)n * nrhs, cudaMemcpyHostToDevice));
    DS(cudssExecute(p->h, CUDSS_PHASE_SOLVE, p->cfg, p->data, p->A, p->X, p->B));
    CU(cudaMemcpy(x, p->d_x, sizeof(double) * (size_t)n * nrhs, cudaMemcpyDeviceToHost));
    return true;
}
