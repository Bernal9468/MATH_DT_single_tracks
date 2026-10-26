// ============================================================================
//  GPU version of ksens::bicgstab (src/thermal/krylov_sens.hpp): Jacobi-preconditioned BiCGSTAB for the exact
//  sensitivity system on the GPU (MATH_DT, 2026-10-26). Same algorithm and stopping test as the CPU version; one thread
//  per row for the CSR products, block reductions for the dot products. Plain C++ interface.
// ============================================================================
#pragma once

// pattern of the matrix (n rows, nnz entries, CSR I[n+1], J[nnz]); false if no device or not enough memory
bool gpu_krylov_setup(int n, int nnz, const int *I, const int *J);
// values of the current matrix (same pattern) and the inverse diagonal
void gpu_krylov_values(const double *A, const double *dinv);
// solve one right-hand side; x is the initial guess on entry. Returns the iterations, or -1 if not converged
int gpu_krylov_solve(const double *b, double *x, double tol, int maxit);
