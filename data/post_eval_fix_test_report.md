# Post-eval fix test report

**Date:** 2026-07-25  
**Branch:** `fix/eval-report-p3ht-tio2-scope`  
**Mode:** `predict_stack(..., use_llm=False)`  
**Models:** `perovskite_eg_regressor.joblib` + `stack_type_classifier.joblib` present; Flask import + POST `/` smoke OK (port default 7860).

## 1. Report stacks (manual)

| Stack | Verdict | Types (ETL/HTL) | Eg (abs/ETL/HTL) | Result | Notes |
|-------|---------|-----------------|------------------|--------|-------|
| K2TiI6 / TiO2 / MoO3 | YES | I / I | 1.61 / 3.4 / 3.0 | **PASS** | `literature_stack_row` (SCAPS TiO2 Eg=3.4) |
| Cs2TiI6 / TiO2 / Spiro-OMeTAD | YES | I / I | 1.80 / 3.2 / 3.0 | **PASS** | |
| Cs2TiI6 / TiO2 / Spiro | YES | I / I | 1.80 / 3.2 / 3.0 | **PASS** | Spiro alias OK |
| Rb2TiI6 / SnO2 / CuSCN | YES | I / I | 1.72 / 3.6 / 3.6 | **PASS** | |
| MAPbI3 / TiO2 / Spiro-OMeTAD | YES | I / I | 1.55 / 3.2 / 3.0 | **PASS** | |
| CH3NH3PbI3 / TiO2 / Spiro-OMeTAD | YES | I / I | 1.55 / 3.2 / 3.0 | **PASS** | |
| CsPbI3 / TiO2 / NiO | YES | I / I | 1.73 / 3.2 / 3.6 | **PASS** | |
| CsPbBr3 / TiO2 / Spiro-OMeTAD | YES | I / I | 2.36 / 3.2 / 3.0 | **PASS** | |
| Cs2AgBiBr6 / TiO2 / Spiro-OMeTAD | YES | I / I | 2.19 / 3.2 / 3.0 | **PASS** | `gap_type: indirect` caveat present |
| Cs2SnI6 / TiO2 / P3HT | MARGINAL | **III** / I | 1.35 / **3.2** / **1.9** | **PASS*** | P3HT Eg≈1.9 + TiO2≈3.2 fixed; Type III is physics |
| CZTS / CdS / MoO3 | BLOCKED | — | — | **PASS** | non-perovskite scope block |
| CIGS / TiO2 / NiO | BLOCKED | — | — | **PASS** | non-perovskite scope block |

\*Functional checks for the P3HT/TiO2 fix pass; suitability is correctly **MARGINAL** because absorber–ETL is Type III (broken gap: χ_abs=4.8, χ_TiO2=4.0, CBO=+0.8 eV).

**Manual stacks: 12/12 PASS**

## 2. `data/optostack_eval_test_set.csv`

| case | Stack | Focus | Verdict | Eg vs lit | Result |
|------|-------|-------|---------|-----------|--------|
| 1 | Rb2TiI6 / SnO2 / CuSCN | A2TiI6 types | YES (I/I) | 1.72 = 1.72 | **PASS** |
| 2 | Cs2TiI6 / SnO2 / CuSCN | A2TiI6 types | YES (I/I) | 1.80 = 1.80 | **PASS** |
| 3 | K2TiI6 / TiO2 / MoO3 | known SCAPS | YES (I/I) | 1.61 = 1.61 | **PASS** |
| 4 | CsPbBr3 / TiO2 / Spiro-OMeTAD | ABX3 | YES (I/I) | 2.36 = 2.36 | **PASS** |
| 5 | Cs2SnI6 / TiO2 / P3HT | P3HT + TiO2 Eg | MARGINAL (III/I) | 1.35 = 1.35; P3HT=1.9; TiO2=3.2 | **PASS** |
| 6 | Cs2AgBiBr6 / TiO2 / NiO | indirect | YES + indirect caveat | 2.19 = 2.19 | **PASS** |
| 7 | CH3NH3PbI3 / TiO2 / Spiro-OMeTAD | MAPbI3 | YES (I/I) | 1.55 = 1.55 | **PASS** |
| 8 | CZTS / TiO2 / NiO | scope block | BLOCKED | — | **PASS** |
| 9 | CIGS / TiO2 / NiO | scope block | BLOCKED | — | **PASS** |
| 10 | Cs2AgBiBr6 / TiO2 / Spiro-OMeTAD | indirect YES caveat | YES + indirect | 2.19 = 2.19 | **PASS** |

**Eval set: 10/10 PASS** (Eg MAE vs labeled lit = 0.0 on labeled rows)

## 3. Literature set (`python scripts/eval_literature_test_set.py`)

Script still works. Summary (tool vs literature, `use_llm=False`):

| Metric | Value |
|--------|-------|
| Scored / predicted | 30 / 30 (100%) |
| Blocked | 0 |
| Tool MAE (all rows) | 0.140 eV |
| Tool MAE (lookup subset, n=24) | 0.117 eV |
| Tool MAE (unseen, n=6) | 0.235 eV |
| Within 0.2 / 0.3 / 0.5 eV | 73% / 77% / 97% |
| Type / suitability accuracy | N/A (no GT labels in CSV) |

Artifacts refreshed: `data/perovskite_test_set_literature_predictions.csv`, `*_accuracy_report.md`, `*_accuracy_report.json`.

## 4. Flask / models

| Check | Result |
|-------|--------|
| `import app` | OK |
| Eg + Type joblibs on disk | OK |
| `GET /` | 200 |
| `POST /` K2TiI6–TiO2–MoO3 | 200, YES / Type I |
| `POST /` CZTS | 200, blocked messaging |
| `POST /` Cs2SnI6–TiO2–P3HT | 200, Type III + MARGINAL + P3HT 1.9 |

## 5. Remaining issues

1. **Cs2SnI6 / TiO2 → Type III (physics, not a regression).** With lookup χ(Cs2SnI6)=4.8 eV and χ(TiO2)=4.0 eV, CBO is positive enough for a broken-gap / Type III classification; tool correctly returns **MARGINAL**. Fixing this would need band-offset policy changes or alternate χ sources — not an Eg-library bug.
2. **K2TiI6 SCAPS stack TiO2 Eg=3.4 vs layer-library TiO2≈3.2.** Literature-row path preserves the paper’s 3.4 eV; other stacks using layer lookup get 3.2. Harmless inconsistency unless UI should normalize contact Eg.
3. **Literature CSV has no Type/suitability ground truth**, so type/suitability accuracy remains unreported there (Eg metrics only).

## Verdict

Eval-report fixes hold: **P3HT optical Eg≈1.9**, **TiO2≈3.2** on lookup path, **CZTS/CIGS blocked**, **indirect-gap caveat** on Cs2AgBiBr6, A2TiI6 / ABX3 stacks YES. Overall **22/22** targeted stack checks pass; remaining open item is Cs2SnI6–TiO2 Type III physics → MARGINAL.
