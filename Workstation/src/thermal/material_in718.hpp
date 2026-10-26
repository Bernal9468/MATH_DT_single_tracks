// ============================================================================
//  IN718 thermal properties for the NIST AM Bench AMB2022-03 single tracks (MATH_DT phase 1, 2026-10-25).
//  Active with -DMATERIAL_IN718 (thermal driver, fb_enzyme_fns.hpp, gpu_newton.cu); without it the IN625 data are used.
//  Model units: mm, tonne, s, mJ, K  ->  k [W/(m K)] = [mW/(mm K)];  cp [J/(kg K)] x 1e6 = [mJ/(tonne K)];
//  rho [kg/m^3] x 1e-12 = [tonne/mm^3];  L [kJ/kg] x 1e9 = [mJ/tonne].
//
//  Sources (copies in MATH_DT/single_tracks/references/):
//  [A] A.Sh. Agazhanov, D.A. Samoshkin, Yu.M. Kozlovskii, "Thermophysical properties of Inconel 718 alloy",
//      J. Phys.: Conf. Ser. 1382 (2019) 012175 (open access) -- measured on IN718: Table 2 (k, cp, 298-1400 K),
//      eq. (10) k(T) = 11.75 + 0.011 T - 9.327e-7 T^2 (fit range 1173-1375 K), rho0 = 8.17 g/cm^3,
//      Table 1 solid-liquid transition enthalpy 196.85 J/g (DSC).
//  [H] H. Hosaeus, A. Seifter, E. Kaschnitz, G. Pottlacher, "Thermophysical properties of solid and liquid Inconel 718
//      alloy", High Temp. High Press. 33 (2001) 405-410 -- liquid cp = 775 J/(kg K); Table 2 corrected resistivity
//      1.472 / 1.480 / 1.493 / 1.507 / 1.520 / 1.533 uOhm m at 1337 / 1400 / 1500 / 1600 / 1700 / 1800 C, converted to
//      k = L T / rho_el with L = 2.45e-8 W Ohm/K^2 (Wiedemann-Franz, as in the paper).
//  [N] NIST AMB2022-03 challenge description: nominal solidus 1260 C, liquidus 1336 C.
//
//  Choices (documented, open to change):
//  - k between 1400 K and the solidus: [A] eq. (10) extrapolated (26.42 at 1533.15 K; the fit reproduces [A] Table 2 at
//    1400 K); mushy zone: linear to the liquid value; above 2073.15 K: constant (table end).
//  - cp between 800 and 1100 K: linear between [A] Table 2 points (the solid-state transformation peaks, 4.45 + 23.27 J/g
//    in [A] Table 1, are not represented); solid above 1400 K: [A] 0.634 J/(g K); liquid: [H] 775 J/(kg K).
//  - density: constant [A] rho0 = 8170 kg/m^3 (the framework has a constant density; liquid 7578 kg/m^3 at the liquidus [H]).
//  - latent heat: [A] 196.85 kJ/kg (DSC). [H]'s enthalpy step between 1255 and 1337 C implies ~231 kJ/kg after removing
//    the sensible part; L is one of the sensitivity parameters (Lfus).
// ============================================================================
#pragma once

#define IN718_T_SOLIDUS 1533.15     // K  [N] 1260 C
#define IN718_T_LIQUIDUS 1609.15    // K  [N] 1336 C
#define IN718_RHO 8.17e-9           // tonne/mm^3  [A]
#define IN718_LFUS 1.9685e11        // mJ/tonne    [A]

// conductivity table [K] -> [W/(m K)]
#define IN718_K_N 17
#define IN718_K_TA {298.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1100.0, 1200.0, 1300.0, 1400.0, 1533.15, \
                    1610.15, 1673.15, 1773.15, 1873.15, 1973.15, 2073.15}
#define IN718_K_VA {9.94, 11.59, 13.24, 14.91, 16.61, 18.34, 22.72, 23.61, 24.47, 25.32, 26.42, \
                    26.80, 27.70, 29.10, 30.45, 31.80, 33.13}
// specific heat table [K] -> [mJ/(tonne K)]  (J/(kg K) x 1e6)
#define IN718_CP_N 13
#define IN718_CP_TA {298.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1100.0, 1200.0, 1300.0, 1400.0, 1533.15, \
                     1609.15, 2500.0}
#define IN718_CP_VA {4.25e8, 4.47e8, 4.68e8, 4.89e8, 5.10e8, 5.31e8, 6.35e8, 6.35e8, 6.35e8, 6.34e8, 6.34e8, \
                     7.75e8, 7.75e8}
