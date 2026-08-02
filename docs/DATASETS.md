# OptoStack — Dataset catalog

Goal: screen **absorber + ETL + HTL** stacks for **optoelectronic band-alignment suitability** (Type I / II / III), **not** PCE prediction. All stacked values come from peer-reviewed tables (not MP/JARVIS as ground truth).

Project overview: [README.md](../README.md) · [TOOL_WORKFLOW.md](TOOL_WORKFLOW.md) · [TECHNICAL_WORKFLOW.md](TECHNICAL_WORKFLOW.md) · [PROJECT_DEVELOPMENT_LOG.md](PROJECT_DEVELOPMENT_LOG.md)

---

## 1. Primary datasets (use these)

| File | Rows | What it is | Verified |
|------|-----:|------------|----------|
| [`data/perovskite_stack_dataset.csv`](../data/perovskite_stack_dataset.csv) | **~1030** | Perovskite **full stacks** (Eg, CBO/VBO, Type, DOI) — SCAPS pool + expansion absorbers + CdTe training-only corner cases | Yes — physics-labeled Types via `junction_type` |
| [`data/perovskite_absorber_library.csv`](../data/perovskite_absorber_library.csv) | **1763** | Perovskite **absorber Eg+χ** library (Paper5 + Pilania + K₂GeI₆ + verified ABX₃ lead halides) | Eg yes; χ = literature or estimated (`chi_source`) |
| [`data/etl_material_library.csv`](../data/etl_material_library.csv) | ~11–15 | **ETL** Eg+χ (SCAPS contacts) | Literature χ where present |
| [`data/htl_material_library.csv`](../data/htl_material_library.csv) | ~20–25 | **HTL** Eg+χ (SCAPS contacts) | Literature χ where present |
| [`data/layer_properties.csv`](../data/layer_properties.csv) | — | Unified Eg+χ lookup for the pipeline | Built by `enrich_chi_dataset.py` |

### χ enrichment

Run `python scripts/enrich_chi_dataset.py` to (re)fill χ and rebuild contact libraries.

| `chi_source` | Meaning |
|--------------|---------|
| `literature_SCAPS` | From peer-reviewed SCAPS tables in `data/raw/` |
| `ml_plus_family_prior` | Estimated for absorbers (screening only) |
| `ml_plus_contact_prior` | Estimated for rare missing contact χ |
| `family_prior_vacancy_ordered_iodide` | K₂GeI₆ family fill |

**Estimated χ is for stack screening only — not measured literature.**

### Stack provenance (~1030)

Types are **physics-derived** from Eg+χ via `junction_type` (not hand-labeled). Build: `python scripts/build_perovskite_dataset.py`.

| Slice | Approx. rows | Notes |
|-------|-------------:|-------|
| SCAPS perovskite absorbers × pooled contacts | ~897 | K₂TiI₆, CsPbZnIBr₂, Cs₃Sb₂Br₉ × ETL/HTL pool with **physical** MoO₃/V₂O₅ χ (UPS/IPES) and wide-gap **MgO/Al₂O₃** for Type I/III corners |
| Expansion absorbers × curated grid | ~112 | MAPbI₃, FAPbI₃, CsPbI₃, FASnI₃, CsSnBr₃, Cs₂AgBiBr₆, K₂GeI₆ × TiO₂/SnO₂/MgO/Al₂O₃ × NiO/Spiro/MoO₃/V₂O₅ |
| CdTe paper training-only | ~21 | See CdTe policy below |

| Source absorber (SCAPS pool) | DOI | Role |
|------------------------------|-----|------|
| K₂TiI₆ | 10.1038/s41598-025-98351-y | absorber × pooled contacts |
| CsPb₀.₆₂₅Zn₀.₃₇₅IBr₂ | 10.1038/s41598-024-81797-x | absorber × pooled contacts |
| Cs₃Sb₂Br₉ | 10.1016/j.nxmate.2026.102491 | absorber × pooled contacts |

**Limit:** only SCAPS / expansion absorbers with Eg+χ enter the *stack* table. The absorber library still carries many Eg-only rows (Paper5 / Pilania) for Eg regression — do not inflate that library with fake χ.

### CdTe inclusion policy

The CdTe device paper (`pratyush.pdf`, DOI **10.1002/pssb.70269**; PDF removed from `research paper/` earlier) is **ingested for Type-ML training only**:

- Raw layers: `data/raw/paper_cdte_scaps_materials.csv` (CdTe Eg=1.547, χ=3.9; SnO₂/CdS family + corner HTLs)
- Stack rows tagged `perovskite_family=CdTe_chalcogenide_training_only`, `record_type=literature_corner_case_non_perovskite`
- **Product gate unchanged:** `check_absorber_perovskite` still **blocks CdTe** (and CZTS/CIGS/GaAs/…) at inference / UI screening

Rebuild meta: `data/perovskite_dataset_build_meta.json`.

---

### Absorber library provenance (1763)

| Source | DOI | Rows |
|--------|-----|-----:|
| Paper5 halide double perovskites | 10.1038/s41524-019-0177-0 | 441 |
| Pilania oxide double perovskites | 10.1038/srep19375 | 1306 |
| K₂GeI₆ DFT | 10.1007/s44291-026-00245-4 | 1 |
| Verified ABX₃ lead halides (external) | see `data/raw/verified_lead_halide_perovskites.csv` | 15 |

**Verified external additions** (`record_type=verified_external`): literature-cited Eg for classic lead-halide perovskites missing from Paper5 — CsPbX₃, MAPbI₃/Br₃/Cl₃, FAPbI₃/Br₃, RbPbI₃, and mixed-halide CsPb(I/Br/Cl)₃ Vegard-series points (DOIs: 10.3390/physchem5010003, 10.1039/D3CP05956A, 10.1038/nature12340, 10.1038/ncomms7382, 10.1038/ncomms7228, 10.1021/acs.chemmater.5b02716, 10.1039/D5CC00735F). Rebuild with `python scripts/build_perovskite_dataset.py`.

Supporting meta: `perovskite_dataset_build_meta.json`, `perovskite_verification_report.json`, `research_paper_verification_report.json`.

---

## 2. Broader literature stacks (earlier dataset)

Use for general Type / offset learning; **mostly 2D monolayers**, not perovskite-only.

| File | Rows | Notes |
|------|-----:|-------|
| `opto_literature_dataset.csv` | 1781 | Master mixed stack table |
| `opto_literature_dataset_externally_verified.csv` | 1404 | External web audit subset |
| `opto_literature_dataset_hse06_only.csv` | 1675 | Homogeneous HSE06 2D |
| `opto_literature_dataset_scaps_only.csv` | 96 | Homogeneous SCAPS devices |
| `opto_literature_dataset_experimental_only.csv` | 10 | MPS₃ experimental |
| `paper5_absorber_library.csv` | 447 | Paper5 absorber dump (includes near-zero gaps; filtered copy is in perovskite library) |

Verification: `dataset_verification_report.json`, `internet_verification_report.json`, `internet_verification_summary.csv`, `material_literature_audit.json`, `heterogeneous_data_audit.json`.

---

## 3. Raw layer tables (`data/raw/`)

SCAPS / DFT layer **Eg + χ** used to rebuild stacks:

- `paper4_scaps_materials.csv` — K₂TiI₆ family  
- `paper_cs_pb_scaps_materials.csv` — CsPbZnIBr₂ family  
- `paper_cs3sb2br9_scaps_materials.csv` — Cs₃Sb₂Br₉ / TiO₂ / CFTS  
- `paper_k2gei6_dft_absorber.csv` — K₂GeI₆ Eg only  
- `paper_cdte_scaps_materials.csv` — CdTe training-only layers (DOI 10.1002/pssb.70269)  
- `verified_expansion_absorbers.csv` — ABX₃ / tin / double-perovskite Eg+χ for stack expansion  
- `paper_besip2_scaps_materials.csv`, `paper1_*`, `paper2_*`, `ozcelik_*`, `paper5_double_perovskites.csv`, `pilania_double_perovskites_gap.csv`, `verified_lead_halide_perovskites.csv`

---

## 4. Not in product screening (by design)

| Item | Reason |
|------|--------|
| DSSC / GaInP absorbers | Non-perovskite — skipped from product screening |
| CdTe as **absorber** at inference | Blocked by perovskite-only gate; **training stacks** from the CdTe paper are included (see CdTe policy above) |
| Ca₃NI₃ bifacial extract | Failed literature verification |
| PDF text extracts | Removed as regenerable clutter |

Kept perovskite-related PDFs under `research paper/` (kaushiki, Dubey, tilt excluded paper kept for reference only). CdTe PDF was deleted earlier; values live in `data/raw/paper_cdte_scaps_materials.csv`.

---

## 5. Rebuild / re-verify

```bash
pip install -r requirements.txt
python scripts/build_perovskite_dataset.py
python scripts/verify_perovskite_dataset.py
python scripts/enrich_chi_dataset.py         # χ + ETL/HTL libraries
python scripts/build_literature_dataset.py   # optional: full opto master
python scripts/verify_literature_ranges.py
```
