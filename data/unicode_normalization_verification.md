# Unicode formula normalization verification

**Date:** 2026-08-01  
**Mode:** `predict_stack(..., use_llm=False)`  
**Contacts:** ETL `TiO2` / HTL `MoO3`

## Equivalent spellings

| Group | Input | Normalized | Eg (eV) | Family | Types | Verdict | Result |
|-------|-------|------------|---------|--------|-------|---------|--------|
| MAPbI3 | `CH₃NH₃PbI₃` | `CH3NH3PbI3` | 1.55 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbI3 | `CH3NH3PbI3` | `CH3NH3PbI3` | 1.55 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbI3 | `MAPbI3` | `CH3NH3PbI3` | 1.55 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbI3 | `MAPbI₃` | `CH3NH3PbI3` | 1.55 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbI3 | `ＣＨ３ＮＨ３ＰｂＩ３` | `CH3NH3PbI3` | 1.55 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbBr3 | `CH₃NH₃PbBr₃` | `CH3NH3PbBr3` | 2.32 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbBr3 | `CH3NH3PbBr3` | `CH3NH3PbBr3` | 2.32 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbBr3 | `MAPbBr3` | `CH3NH3PbBr3` | 2.32 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| MAPbBr3 | `MAPbBr₃` | `CH3NH3PbBr3` | 2.32 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| FAPbI3 | `FAPbI₃` | `HC(NH2)2PbI3` | 1.48 | abx3_halide_3d | Type II/Type I | YES | **PASS** |
| FAPbI3 | `FAPbI3` | `HC(NH2)2PbI3` | 1.48 | abx3_halide_3d | Type II/Type I | YES | **PASS** |
| FAPbI3 | `HC(NH₂)₂PbI₃` | `HC(NH2)2PbI3` | 1.48 | abx3_halide_3d | Type II/Type I | YES | **PASS** |
| FAPbBr3 | `FAPbBr₃` | `HC(NH2)2PbBr3` | 2.23 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| FAPbBr3 | `FAPbBr3` | `HC(NH2)2PbBr3` | 2.23 | abx3_halide_3d | Type II/Type II | YES | **PASS** |
| Cs2TiI6 | `Cs₂TiI₆` | `Cs2TiI6` | 1.58 | vacancy_ordered_a2bx6 | Type I/Type II | YES | **PASS** |
| Cs2TiI6 | `Cs2TiI6` | `Cs2TiI6` | 1.58 | vacancy_ordered_a2bx6 | Type I/Type II | YES | **PASS** |
| Cs3Sb2I9 | `Cs₃Sb₂I₉` | `Cs3Sb2I9` | 2.05 | a3b2x9_0d | Type II/Type II | YES | **PASS** |
| Cs3Sb2I9 | `Cs3Sb2I9` | `Cs3Sb2I9` | 2.05 | a3b2x9_0d | Type II/Type II | YES | **PASS** |
| Cs2AgBiBr6 | `Cs₂AgBiBr₆` | `Cs2AgBiBr6` | 2.19 | halide_double_a2bbx6 | Type II/Type II | YES | **PASS** |
| Cs2AgBiBr6 | `Cs2AgBiBr6` | `Cs2AgBiBr6` | 2.19 | halide_double_a2bbx6 | Type II/Type II | YES | **PASS** |

## Halide / A-site identity kept distinct

| A | B | Eg A | Eg B | Result |
|---|---|------|------|--------|
| MAPbI3 | MAPbBr3 | 1.55 | 2.32 | **PASS** |
| FAPbI3 | FAPbBr3 | 1.48 | 2.23 | **PASS** |
| MAPbI3 | FAPbI3 | 1.55 | 1.48 | **PASS** |

## Blocked cases (all spellings)

| Input | Normalized | Blocked | Result |
|-------|------------|---------|--------|
| `ZnO` | `ZnO` | True | **PASS** |
| `ＺｎＯ` | `ZnO` | True | **PASS** |
| `CZTS` | `CZTS` | True | **PASS** |
| `ＣＺＴＳ` | `CZTS` | True | **PASS** |
| `PbI2` | `PbI2` | True | **PASS** |
| `PbI₂` | `PbI2` | True | **PASS** |
| `ＰｂＩ２` | `PbI2` | True | **PASS** |
| `1T-PbI2` | `1T-PbI2` | True | **PASS** |
| `1T‑PbI₂` | `1T-PbI2` | True | **PASS** |

## Verdict

**All unicode spellings resolve identically to their ASCII form.**
