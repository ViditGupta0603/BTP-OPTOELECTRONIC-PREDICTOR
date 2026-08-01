# OptoStack negative-case test report

**Date:** 2026-08-01  
**Mode:** `predict_stack(..., use_llm=False)`  
**Totals:** 39 PASS / 0 FAIL / 39 cases  

## Summary table

| # | Case | Category | Stack | Expected | Got | Types | Eg | Result |
|---|------|----------|-------|----------|-----|-------|----|--------|
| 1 | `block_contact_ZnO` | should_block | ZnO / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 2 | `block_contact_TiO2` | should_block | TiO2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 3 | `block_contact_SnO2` | should_block | SnO2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 4 | `block_contact_MoO3` | should_block | MoO3 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 5 | `block_contact_NiO` | should_block | NiO / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 6 | `block_pv_CZTS` | should_block | CZTS / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 7 | `block_pv_CIGS` | should_block | CIGS / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 8 | `block_pv_GaAs` | should_block | GaAs / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 9 | `block_pv_CdTe` | should_block | CdTe / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 10 | `block_pv_Si` | should_block | Si / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 11 | `block_pv_graphene` | should_block | graphene / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 12 | `block_2d_1T-PbI2` | should_block | 1T-PbI2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 13 | `block_2d_2H-MoS2` | should_block | 2H-MoS2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 14 | `block_2d_BA2PbI4` | should_block | BA2PbI4 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 15 | `block_BeSiP2` | should_block | BeSiP2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 16 | `block_garbage_asdf` | should_block | asdf / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 17 | `block_garbage_123` | should_block | 123 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 18 | `block_garbage_H2O` | should_block | H2O / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 19 | `block_garbage_NaCl` | should_block | NaCl / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 20 | `role_misuse_K2TiI6_MoO3_as_ETL_TiO2_as_HTL` | edge | K2TiI6 / MoO3 / TiO2 | not in ['BLOCKED', 'CRASH'] | MARGINAL; types=Type III/Type I; Eg=1.610 | Type III/Type I | 1.610 | **PASS** |
| 21 | `role_misuse_CsPbBr3_PEDOT_as_ETL_TiO2_as_HTL` | edge | CsPbBr3 / PEDOT:PSS / TiO2 | not in ['BLOCKED', 'CRASH'] | YES; types=Type II/Type II; Eg=2.360 | Type II/Type II | 2.360 | **PASS** |
| 22 | `pedot_as_htl_caveat` | edge | CsPbBr3 / TiO2 / PEDOT:PSS | not in ['BLOCKED', 'CRASH']; PEDOT caveat | YES; types=Type II/Type II; Eg=2.360 | Type II/Type II | 2.360 | **PASS** |
| 23 | `halide_FAPbBr3_not_FAPbI3_Eg` | edge | FAPbBr3 / TiO2 / Spiro-OMeTAD | not in ['BLOCKED', 'CRASH']; Eg in (2.05, 2.4); Eg≠1.48 | YES; types=Type II/Type II; Eg=2.230 | Type II/Type II | 2.230 | **PASS** |
| 24 | `halide_Cs2TiI6_not_Br_Eg` | edge | Cs2TiI6 / TiO2 / Spiro-OMeTAD | not in ['BLOCKED', 'CRASH']; Eg in (1.5, 1.7); Eg≠1.8 | YES; types=Type I/Type II; Eg=1.580 | Type I/Type II | 1.580 | **PASS** |
| 25 | `typo_FAPbBr_near_miss` | edge | FAPbBr / TiO2 / NiO | Eg≠1.48 | BLOCKED; types=—; Eg=— | — | — | **PASS** |
| 26 | `typo_Cs2TiI_near_miss` | edge | Cs2TiI / TiO2 / NiO | Eg≠1.8 | BLOCKED; types=—; Eg=— | — | — | **PASS** |
| 27 | `typo_MAPbI_near_miss` | edge | MAPbI / TiO2 / NiO | Eg≠1.55 | BLOCKED; types=—; Eg=— | — | — | **PASS** |
| 28 | `edge_Cs2SnI6_TiO2_P3HT` | edge | Cs2SnI6 / TiO2 / P3HT | verdict in ['YES']; not in ['BLOCKED', 'CRASH']; Eg in (1.3, | YES; types=Type I/Type II; Eg=1.350 | Type I/Type II | 1.350 | **PASS** |
| 29 | `edge_Cs2SnI6_TiO2_CFTS_typeIII` | edge | Cs2SnI6 / TiO2 / CFTS | verdict in ['MARGINAL', 'NO']; not in ['BLOCKED', 'CRASH', ' | MARGINAL; types=Type I/Type III; Eg=1.350 | Type I/Type III | 1.350 | **PASS** |
| 30 | `block_precursor_PbI2` | should_block | PbI2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 31 | `block_precursor_SnI2` | should_block | SnI2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 32 | `block_precursor_GeI2` | should_block | GeI2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 33 | `block_precursor_PbBr2` | should_block | PbBr2 / TiO2 / NiO | BLOCKED | BLOCKED | — | — | **PASS** |
| 34 | `edge_widegap_BaTiO3` | edge | BaTiO3 / TiO2 / NiO | not in ['CRASH'] | YES; types=Type I/Type II; Eg=3.391 | Type I/Type II | 3.391 | **PASS** |
| 35 | `stability_MAPbI3` | stability | MAPbI3 / TiO2 / Spiro-OMeTAD | identical key fields | ('YES', 'Type II', 'Type II', 1.55, 3.2, 3.0, 'compute_from_Eg_chi', F | Type II/Type II | 1.550 | **PASS** |
| 36 | `stability_Cs2TiI6` | stability | Cs2TiI6 / TiO2 / NiO | identical key fields | ('YES', 'Type I', 'Type II', 1.58, 3.2, 3.6, 'compute_from_Eg_chi', Fa | Type I/Type II | 1.580 | **PASS** |
| 37 | `stability_blocked_ZnO` | stability | ZnO / TiO2 / NiO | identical key fields | ('BLOCKED', None, None, None, None, None, 'blocked_non_perovskite', Tr | — | — | **PASS** |
| 38 | `stability_garbage_asdf` | stability | asdf / TiO2 / NiO | identical key fields | ('BLOCKED', None, None, None, None, None, 'blocked_non_perovskite', Tr | — | — | **PASS** |
| 39 | `pedot_as_etl_behavior` | role_misuse | CsPbBr3 / PEDOT:PSS / TiO2 | runs; PEDOT-as-ETL caveat optional (HTL caveat is HTL-only) | YES; types=Type II/Type II | Type II/Type II | 2.360 | **PASS** |

## Failures detail

None — all cases passed.
## Category notes

### Should BLOCK
Contact oxides, thin-film PV, graphene, 2D RP/DJ/monolayers, BeSiP2, and garbage absorbers must return `blocked=True` / verdict BLOCKED with no Type/suitability claim.

### Role misuse
Tool does not enforce ETL/HTL conventions; stacks must still run. PEDOT:PSS as **HTL** must attach degenerate/metallic caveat. PEDOT as **ETL** currently has no dedicated caveat (HTL-only path).

### Halide traps
FAPbBr3 must resolve ~2.23 eV (not FAPbI3 1.48). Cs2TiI6 must resolve ~1.58 eV (not Br ~1.8).

### Edge stacks
Cs2SnI6/TiO2/P3HT should not be a blind YES without Type III somewhere.

### Stability
Identical inputs must yield identical key fields; no crashes.

## Verdict

**39/39 PASS.** Negative-case gate looks healthy.
