// GPU Jacobi-preconditioned BiCGSTAB for the exact sensitivity system -- see gpu_krylov.h and src/thermal/krylov_sens.hpp.
#include "gpu_krylov.h"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

#define KCK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return false; } } while (0)

namespace {
int N = 0, NNZ = 0;
int *dI = nullptr, *dJ = nullptr;
double *dA = nullptr, *dD = nullptr;
double *dr = nullptr, *drh = nullptr, *dp = nullptr, *dv = nullptr, *dy = nullptr, *ds = nullptr, *dz = nullptr, *dt = nullptr,
       *dx = nullptr, *db = nullptr, *dred = nullptr;
const int TB = 256, NRED = 4;      // reduction slots: up to 4 dot products per call

__global__ void k_spmv(int n, const int *I, const int *J, const double *A, const double *x, double *y)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double s = 0.0;
    for (int k = I[r]; k < I[r + 1]; k++) s += A[k] * x[J[k]];
    y[r] = s;
}
// out[slot] += sum a.*b  (block reduction + one atomic per block)
__global__ void k_dot(int n, const double *a, const double *b, double *out)
{
    __shared__ double sh[TB];
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) s += a[i] * b[i];
    sh[threadIdx.x] = s; __syncthreads();
    for (int w = TB / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}
__global__ void k_pdot(int n, const double *d, const double *a, double *out)    // sum (d a)^2
{
    __shared__ double sh[TB];
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) { const double q = d[i] * a[i]; s += q * q; }
    sh[threadIdx.x] = s; __syncthreads();
    for (int w = TB / 2; w > 0; w >>= 1) { if (threadIdx.x < w) sh[threadIdx.x] += sh[threadIdx.x + w]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}
__global__ void k_init(int n, const double *b, const double *Ax, double *r, double *rh)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { r[i] = b[i] - Ax[i]; rh[i] = r[i]; } }
__global__ void k_p(int n, double beta, double omega, const double *r, const double *v, const double *d, double *p, double *y)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { p[i] = r[i] + beta * (p[i] - omega * v[i]); y[i] = d[i] * p[i]; } }
__global__ void k_s(int n, double alpha, const double *r, const double *v, const double *d, double *s, double *z)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { s[i] = r[i] - alpha * v[i]; z[i] = d[i] * s[i]; } }
__global__ void k_x1(int n, double alpha, const double *y, double *x)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += alpha * y[i]; }
__global__ void k_x2(int n, double alpha, double omega, const double *y, const double *z, const double *s, const double *t,
                     double *x, double *r)
{ const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { x[i] += alpha * y[i] + omega * z[i]; r[i] = s[i] - omega * t[i]; } }

inline int nb(int n) { return (n + TB - 1) / TB; }
inline int nr(int n) { const int b = nb(n); return b < 1024 ? b : 1024; }
// dot products into dred[0..k-1], copied back
bool fetch(double *h, int k) { return cudaMemcpy(h, dred, sizeof(double) * k, cudaMemcpyDeviceToHost) == cudaSuccess; }
double ddot(const double *a, const double *b) { double h = 0; cudaMemset(dred, 0, sizeof(double)); k_dot<<<nr(N), TB>>>(N, a, b, dred); fetch(&h, 1); return h; }
double dpnorm(const double *a) { double h = 0; cudaMemset(dred, 0, sizeof(double)); k_pdot<<<nr(N), TB>>>(N, dD, a, dred); fetch(&h, 1); return std::sqrt(h); }
} // namespace

bool gpu_krylov_setup(int n, int nnz, const int *I, const int *J)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) return false;
    size_t need = (size_t)nnz * (sizeof(int) + sizeof(double)) + (size_t)(n + 1) * sizeof(int) + (size_t)n * 12 * sizeof(double) + ((size_t)256 << 20);
    size_t avail = 0, total = 0;
    if (cudaMemGetInfo(&avail, &total) != cudaSuccess || need > avail) {
        printf("  [GPU_KRYLOV] OFF: needs %.2f GB, %.2f GB free -> sensitivity Krylov solves on the CPU\n", need / 1073741824.0, avail / 1073741824.0);
        fflush(stdout); return false;
    }
    if (n != N || nnz != NNZ) {
        for (void *q : {(void*)dI, (void*)dJ, (void*)dA, (void*)dD, (void*)dr, (void*)drh, (void*)dp, (void*)dv, (void*)dy, (void*)ds,
                        (void*)dz, (void*)dt, (void*)dx, (void*)db, (void*)dred}) if (q) cudaFree(q);
        N = n; NNZ = nnz;
        KCK(cudaMalloc(&dI, sizeof(int) * (n + 1))); KCK(cudaMalloc(&dJ, sizeof(int) * nnz)); KCK(cudaMalloc(&dA, sizeof(double) * nnz));
        for (double **q : {&dD, &dr, &drh, &dp, &dv, &dy, &ds, &dz, &dt, &dx, &db}) KCK(cudaMalloc(q, sizeof(double) * n));
        KCK(cudaMalloc(&dred, sizeof(double) * NRED));
    }
    KCK(cudaMemcpy(dI, I, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    KCK(cudaMemcpy(dJ, J, sizeof(int) * nnz, cudaMemcpyHostToDevice));
    printf("  [GPU_KRYLOV] sensitivity system on the GPU: Jacobi-BiCGSTAB, %d unknowns, %d nonzeros\n", n, nnz); fflush(stdout);
    return true;
}

void gpu_krylov_values(const double *A, const double *dinv)
{
    cudaMemcpy(dA, A, sizeof(double) * NNZ, cudaMemcpyHostToDevice);
    cudaMemcpy(dD, dinv, sizeof(double) * N, cudaMemcpyHostToDevice);
}

int gpu_krylov_solve(const double *b, double *x, double tol, int maxit)
{
    const int n = N;
    if (cudaMemcpy(db, b, sizeof(double) * n, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    if (cudaMemcpy(dx, x, sizeof(double) * n, cudaMemcpyHostToDevice) != cudaSuccess) return -1;
    k_spmv<<<nb(n), TB>>>(n, dI, dJ, dA, dx, dt);
    k_init<<<nb(n), TB>>>(n, db, dt, dr, drh);
    cudaMemset(dp, 0, sizeof(double) * n); cudaMemset(dv, 0, sizeof(double) * n);
    const double nrm_b = dpnorm(db);
    int its = -1;
    if (nrm_b == 0.0) { cudaMemset(dx, 0, sizeof(double) * n); its = 0; }
    else if (dpnorm(dr) <= tol * nrm_b) its = 0;
    else {
        double rho = 1.0, alpha = 1.0, omega = 1.0;
        for (int it = 1; it <= maxit; it++) {
            const double rho1 = ddot(drh, dr);
            if (rho1 == 0.0) break;
            const double beta = (rho1 / rho) * (alpha / omega); rho = rho1;
            k_p<<<nb(n), TB>>>(n, beta, omega, dr, dv, dD, dp, dy);
            k_spmv<<<nb(n), TB>>>(n, dI, dJ, dA, dy, dv);
            const double rv = ddot(drh, dv);
            if (rv == 0.0) break;
            alpha = rho / rv;
            k_s<<<nb(n), TB>>>(n, alpha, dr, dv, dD, ds, dz);
            if (dpnorm(ds) <= tol * nrm_b) { k_x1<<<nb(n), TB>>>(n, alpha, dy, dx); its = it; break; }
            k_spmv<<<nb(n), TB>>>(n, dI, dJ, dA, dz, dt);
            const double tt = ddot(dt, dt);
            if (tt == 0.0) break;
            omega = ddot(dt, ds) / tt;
            k_x2<<<nb(n), TB>>>(n, alpha, omega, dy, dz, ds, dt, dx, dr);
            if (dpnorm(dr) <= tol * nrm_b) { its = it; break; }
            if (omega == 0.0) break;
        }
    }
    if (its >= 0 && cudaMemcpy(x, dx, sizeof(double) * n, cudaMemcpyDeviceToHost) != cudaSuccess) return -1;
    return its;
}
