// ============================================================================
//  GPU laser source (bare_plate_single_layer, 2026-10-23)
//
//  Moves SpatialHashSource::BuildSource (Goldak mode, -spread 2) and the source-parameter AD of block (e) to the GPU.
//  The host still walks the toolpath (a few dozen sub-samples per step) and fills one GpuSrcSub per sub-sample; the GPU
//  does the per-cell work (nq^3 Goldak cells per sub-sample): element location, reference coordinates, nominal
//  deposition, and the first derivatives w.r.t. th = [s_rx, s_ry, s_rz, x0, s_Vx, tL] by forward-mode dual numbers
//  evaluated on the same src_kernel.hpp templates that Enzyme differentiates on the CPU.
//  Plain C++ interface: no MFEM or CUDA types, so the driver is still compiled by g++.
// ============================================================================
#pragma once
#include "src_kernel.hpp"

struct GpuSrcSub {
    SrcSub rec;       // AD record of the sub-sample (frozen segment, Goldak frame and axes, Esub, nq, box), as in the tape
    double pL[3];     // nominal laser position laser.position(tm), the point BuildSource deposits around
};

// Element data of the (fixed) mesh: X[e*24 + a*3 + d] vertex coordinates in element-vertex order, dofs[e*8 + a]
// the matching (sign-fixed) H1 dofs. Returns false if no CUDA device is usable.
bool gpu_src_init(int ne, int ndof, const double *X, const int *dofs);

// One source evaluation for the sub-samples subs[0..nsub). elem_active[e] = 1 for active elements.
//   f[ndof]      nominal deposited ENERGY per dof [mJ] (BuildSource divides by the window length afterwards)
//   d1[6*ndof]   if not null: d(energy)/d th_j per dof, j = 0..5 (block (e)'s unscaled D1[j])
//   fk[ndof]     if not null: the kernel's own value (block (e)'s consistency check against f)
//   nkept        number of (sub-sample, cell) pairs that deposited energy
//   d2[21*ndof]  if not null (Enzyme build only): second derivatives along the 21 pairs (a<=b, row-major: 00 01 .. 05 11 ..
//                55) of th; pairs = 1: only the 6 diagonal pairs are computed (the others stay 0), pairs = 2: all 21
void gpu_src_build(const GpuSrcSub *subs, int nsub, const char *elem_active,
                   double *f, double *d1, double *d2, int pairs, double *fk, long *nkept);
bool gpu_src_has_second_order();   // true for the Enzyme build

const char *gpu_src_device_name();
