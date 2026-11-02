# GO_MELT: 3-level multi-mesh port of the AMB2022-03 single tracks (IN718)

All files for this work live in this folder (user rule, 2026-11-01). Enzyme for derivatives only; NO OTI code.

## Task 1 (current): replicate the single-track problem with 3 mesh levels and verify against the single-level baseline

Baseline (already verified, h = 20 um, dt = 5 us, calibrated eta 0.749, C0 0.183, n 0.986):
`../../arc/results/all/case{0,1.1,1.2,2.1,2.2,3.1,3.2}/` : probe.csv, peakT.bin, result.json (depth, width, TTAM, cooling rates).
Case 0: depth 139.6 um, width 135.1 um, TTAM 1.28 ms (baseline run: 518 steps, 102 min on 40 cores with dorder=1).
Verification target: GO_MELT nominal (dorder 0) matches probe T(t), peak-T field, depth/width, TTAM, cooling rates to <= 1%.

## Findings so far
- Baseline driver (`../HPC/src/thermal/full_bridge_activation_tier1_oti.cpp`, 6695 lines) uses Enzyme for AD but is still
  compiled with OTI headers (-DOTI_ORDER_VAL=1, otinum). It cannot be reused as is -> GO_MELT needs a clean non-OTI driver.
- Reusable (no AD in it): mesh/box generation, tie weights, window clipping, volumetric coarse seed, state transfer
  (from Full_bridge/full_bridge_alternative, see docs/DESIGN_PLAN.md there). Laser axis there is +y; here +x.
- Native build env exists on the cluster (Makefile in full_bridge_alternative/thermal_enzyme/src: MFEM_DIR, clang+Enzyme plugin).

## Physics to reproduce (from the baseline)
IN718 tables (material_in718.hpp), constant rho, lumped cp and latent-heat capacitance, two-stage liquid fraction
(FL_EUT 0.23 at T_EUT 1423.15 K, quintic smoothstep), fully implicit k(T), Goldak double ellipsoid (a=b=sqrt(6)sigma,
c=C0(X/X0)^n, gsub 12, gcut 1.5, spread 2), convection+radiation on top, bottom T = 23.5 C, backward Euler dt 5 us.

## Phases
0. Clean non-OTI nominal driver (single level) -> must reproduce case 0 baseline (this proves the new code).
1. Level hierarchy L1 plate (coarse) / L2 meso box / L3 20 um box at the laser; static nesting, one-way coarse->fine Dirichlet.
2. Two-way (subgrid) feedback, lagged/Picard.
3. Moving L2/L3 boxes along +x (exact-node-coincidence window clips + seed transfer).
4. Verify all 7 cases vs baseline; then the 10 mm track (speed-up).
5. Enzyme first-order sensitivities (20 params) through the coupled scheme. First-order only; Abaqus check once at the end.

## Status
- 2026-11-01 Phase 0 DONE for case 0 (job 854426, baseline mesh h=20 um, 518 steps, 245 s on 24 cores vs 102 min for the baseline dorder=1 run):
  probe NRMSE 0.0063%, peak-T field NRMSE 0.0035% (max 3.7 K at the 5690 K peak), depth 139.564 vs 139.563 um, width 135.19 vs 135.12 um,
  TTAM +0.007%, TLCR -0.01%, TTCR +3.1%, TSCR -1.8% (slope fits over a few points amplify the <=1.6 K pointwise differences).
  Code: src/gm_core.hpp, src/gm_single.cpp (make in src/). Baseline quirk reproduced: no convection/radiation on the top face.
- Other 6 cases: job phase0_all.sbatch (compare with post_compare.py).
- Next: Phase 1 (levels).
- 2026-11-01 Phase 0 vs the FIXED upstream baseline (top face loses heat; job 854479, case 0): probe NRMSE 0.0019% (max 0.25 K), peak field 0.0027%,
  depth 139.564 vs 139.563, width 135.03 vs 135.02, TTAM +0.012%, cooling rates <= 0.022%. (gm_single defaults -toploss 1 -kq 3.) Cases 1.1/3.1 vs the old
  baseline looked off only because the baseline had a Newton cut at t=3e-5 that shifted its time stamps by dt/2 (post_compare.py now interpolates).
- 2026-11-01 PHASE 1 DONE (job 854487; static L1 0.2 mm plate -> L2 box -> L3 20 um box = baseline fine box, one-way Dirichlet from parent), case 0 vs fixed baseline:
    L1+L3             : probe NRMSE 0.052%, peak field (38976 matching nodes) 0.67% / max 164 K, TTAM +0.38%, cooling rates -0.6..-1.5%
    L1+L2(0.08)+L3    : probe 0.015%, peak field 0.16% / 53 K, TTAM +0.10%, cooling rates -0.14..-0.49%
    L1+L2(0.04)+L3    : probe 0.0047%, peak field 0.045% / 23 K, TTAM +0.03%, cooling rates -0.04..-0.12%, depth identical, width +0.012 um
  All within the 1% target. Wall (static boxes, short track): 3 levels 325 s on 6 cores vs 245 s single level on 24 cores: no gain yet at 2.1 mm (expected;
  gain needs moving boxes on the 10 mm track).
- Next: Phase 2 (two-way feedback) is optional given the accuracy; Phase 3 (moving L2/L3 boxes along +x) is what delivers the speed-up. Code: src/gm_levels.cpp, post_levels.py.
- 2026-11-01 PHASE 3 (moving boxes) on the 2.1 mm track, case 0, vs the fixed baseline (job 854489; src/gm_moving.cpp; boxes follow the laser in whole elements, state follows the nodes, leading slab seeded from the parent, one-way Dirichlet):
    L1 0.2 + moving L2 0.04 + moving L3 0.02: probe NRMSE 0.0047%, peak field 0.0975% (max 27 K), TTAM +0.03%, cooling rates -0.04..-0.12%, depth identical, width +0.012 um; 157 shifts; 199 s on 8 cores
    L1 0.2 + moving L2 0.08 + moving L3 0.02: probe 0.015%, peak field 0.17%, TTAM +0.10%, cooling rates -0.14..-0.49%; 117 s on 8 cores
  L3 box = 105 x 20 x 15 elements (1.8 mm behind the laser, 0.3 ahead: the probe cools below 1150 C ~2 mm behind), L2 = 85 x 30 x 20 (2.6 behind, 0.8 ahead).
- Next: full 10 mm track (job phase3_full10.sbatch; baseline arc/results/case0_full10 = 490 min on 40 cores, pre-fix physics, so run -toploss 0).
- 2026-11-01 Phase 3, FULL 10 mm track, one-way (job 854490, toploss 0 vs baseline case0_full10 = 490 min/40 cores): 13 min on 16 cores (2164 steps, 4.4 min/1000 steps at 16 cores?). Metrics inside L3 match
  (TTAM +0.032%, cooling rates -0.04..-0.13%, depth identical, width +0.011 um, T_max(t) 0.0018%), BUT: (a) probe after it leaves L3 (t > 7.08 ms) is read from the
  coarse levels and is 160-225 K too hot (probe NRMSE 2.0%); (b) global peak field NRMSE 0.55%, bias +21 K, concentrated at the L3 boundary (z=-0.3: +65 K, |y|=0.18: +41 K).
  Cause: one-way coupling -- the coarse levels never see the fine pool heat, so the quasi-steady wake/periphery on L1/L2 (0.2 / 0.04 mm) is wrong; the 2.1 mm tests were too short to show it.
  Fix being tested: two-way feedback by injection (child solution copied into the parent nodes inside the child box after every step, -fb 1) = Phase 2 pulled forward (job phase3_fb_full10.sbatch).
- 2026-11-01 Phase 3 + two-way injection (-fb 1), full 10 mm (job 854495, vs case0_full10): moving L2 0.04 / L3 0.02: probe NRMSE 0.50% (was 2.0%; max 65 K vs 225 K), TTAM +0.021%, cooling rates -0.03..-0.09%,
  depth identical, width +0.009 um; peak field NRMSE 0.51%, bias +14 K, max 87 K (still on the L3 boundary). L2 0.08 variant: probe 0.38%, peak field 0.47% but TSCR -0.55%.
  Next test (job 854496): larger L3/L2 cross-sections (y +-0.3/z -0.4 and y +-0.4/z -0.5) to see if the boundary error shrinks.
- 2026-11-01 DIAGNOSIS of the peak-field/probe-tail error (long track): bigger L3/L2 boxes did not change it (jobs 854496: +15.4 K bias either way). Cause: the periphery nodes (|y|~0.2, z~-0.3) reach their peak LONG after the laser has passed (heat from the whole wake keeps arriving; baseline long vs short track differs by +61 K mean at x in [0,1.4]), when they are no longer inside L3, so the L3-only peak record misses the rise, and the wake itself is carried by the coarse L1/L2. Not a box-boundary problem. Fix under test: peak record takes L2/L1 at nodes outside the fine box (-pkevery) + finer L1 (0.1) (job 854498).
- ROADMAP to the full goal (user, 2026-11-01): (1) fix/verify thermal nominal, (2) thermal first-order derivatives with Enzyme (20-parameter set, reference = baseline AD in arc/results/all/case*/sens.csv + peakT.bin derivative arrays) through the coupled levels (child Dirichlet + injection also act on the sensitivities), (3) mechanical stage: feed multilevel thermal dumps (T + dT/dp every 20 steps, interpolated to the baseline mesh) to the existing mech_main_aa and compare residual stress + first-order derivatives, (4) wall-time bookkeeping for every stage vs the single-level baseline (baseline full 10 mm thermal dorder=0: 490 min on 40 cores).
- 2026-11-01 PEAK RECORD FIX CONFIRMED (full 10 mm, case 0, vs case0_full10, toploss 0; jobs 854498): with the coarse levels in the peak record and two-way injection:
    L1 0.2 / L2 0.04 / L3 0.02 : probe NRMSE 0.50% (tail 65 K), peak field NRMSE 0.118%, bias -0.017 K (was +14 K), TTAM +0.021%, cooling rates -0.03..-0.09%, depth identical, width +0.009 um; wall 933 s on 8 cores
    L1 0.1  / L2 0.04 / L3 0.02 : probe NRMSE 0.066% (max 8 K), peak field NRMSE 0.071%, bias -0.13 K, same fine-box metrics; wall 1411 s on 8 cores (L1 634 s, L2 397 s, L3 375 s)
  Remaining max error ~42 K sits on the L3 bottom face (z = -0.3 is L3's own Dirichlet boundary). Baseline: 490 min on 40 cores (= 327 core-h) -> 1411 s x 8 cores = 3.1 core-h (about 100x fewer core-hours, 21x less wall).
- 2026-11-01 FIRST-ORDER ENZYME SENSITIVITIES VALIDATED (single level, job 854500, h = 0.04 test mesh, case 0, vs the baseline's AD = fixed upstream binary dorder 1):
    probe dT/ds over time: NRMSE <= 0.11% for all 17 non-zero parameters (k .006, cp .007, rho .007, Lfus .008, Ts .108, Tliq .026, hc .017, T0 .018, P/Eta .007, Rx .016, Ry .006, Rz .005, X0 .017, Vx .016, TL .017, Tw .046); hp/Rhop zero in both, kp noise 5e-15 vs 0
    peak-field dTpeak/ds at 74200 nodes: NRMSE <= 0.07% (k .001 ... T0 .053, Tw .070)
  gm_single -dorder 1: 179 s on 7 cores (sensitivities 101 s of it).
- 2026-11-01 MULTI-LEVEL FIRST-ORDER SENSITIVITIES (case 0, 2.1 mm, toploss 0, vs baseline AD arc/results/all/case0; gm_moving -dorder 1, L1 0.1 / L2 0.04 / L3 0.02, injection feedback):
    moving boxes, seeds from the parent (job 854502): probe NRMSE k .20, cp .057, rho .034, Lfus .21, Ts .28, Tliq .22, P/Eta .11, Rx .005, Ry .005, Rz .031, X0 .0055, Vx .011, TL .0055 % -- BUT T0 13.3 % and Tw 7.6 %.
       nominal: probe 0.0036 %, peak field 0.053 % (max 12 K), TTAM +0.021 %, cooling rates <= 0.1 %, depth identical; wall 544 s on 14 cores, of which sensitivities 335 s (baseline: 6145 s on 40 cores).
    static boxes (job 854503, diagnostic): T0 0.40 %, Tw 0.27 %, k .24, cp .071, rho .044, Lfus .25, Ts .37, Tliq .27, P/Eta .13, radii/X0/Vx/TL <= .037 % -> the T0/Tw error was the SEEDING of the new leading slab of a moving box
       (T0 / Tw are IC seeds of a one-node-thick top layer; interpolating them from a coarse parent smears that layer). Fix: -seedcol 1 (default): new slab copies the virgin front column of the same level (T, L and all S). Re-test = job 854505 (also dumps every 20 steps like the baseline: the first run's post-scan dump interval (200) differed).
- 2026-11-01 FIRST-ORDER THERMAL DERIVATIVES, MULTI-LEVEL, VERIFIED (job 854506; moving L1 0.1 / L2 0.04 / L3 0.02 + injection, seedcol, IC-layer seed scaling r = 0.02/hz per level; case 0, 2.1 mm, toploss 0)
  vs the baseline AD (arc/results/all/case0), probe dT/ds histories from t >= 0.5 ms (when the fine box covers the probe; before that the probe is read from L2, and the T0/Tw IC-layer seeds are
  resolution dependent by construction): k 0.22, cp 0.063, rho 0.038, Lfus 0.23, Ts 0.31, Tliq 0.25, T0 0.53, P/Eta 0.12, Rx 0.006, Ry 0.006, Rz 0.034, X0 0.006, Vx 0.012, TL 0.006, Tw 0.23 % (hc, hp, Rhop: zero in both).
  peak-field dTpeak/ds (38976 matching nodes, NRMSE): k .19, cp .10, rho .10, Lfus .08, Ts .05, Tliq .06, T0 1.55, Tw .46, P/Eta .05, Rx .003, Ry .005, Rz .037, X0 .008, Vx .005, TL .008 %.
  (whole-history NRMSE of T0/Tw is 4.5 / 2.6 % because of the first ~80 steps.)  Nominal: probe 0.0036 %, peak field 0.053 %, wall 544 s / 14 cores incl. sensitivities (baseline 6145 s / 40 cores).
  hc was zero in the (pre-fix) baseline; hc verification uses the fixed binary: single level 0.017 % (job 854500); multi-level toploss 1 not yet run.
